//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (CPU-writable VRAM via Resizable BAR when supported with
//! headroom, or write-back system memory otherwise), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! Ranges draws are known to read are brought up to date (vertex and index
//! buffer views, constant buffers in tables, root descriptors, buffer views
//! in tables, and buffer / constant views reached through
//! `ResourceDescriptorHeap` via [`crate::descriptor::host::HostHeap::refresh_bindless_buffers`]).
//! shortcut: unordered buffer ranges reached only through the heap are
//! refreshed for reading but not copied back after the submission (a
//! full-heap UAV snapshot is too expensive).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::{
    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATES, ID3D12Resource,
};

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// Total VRAM bytes currently allocated to ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`)
/// shadow buffers across live [`Shadow`] instances.
static REBAR_SHADOW_BYTES: AtomicU64 = AtomicU64::new(0);

/// Atomically reserves `size` bytes from `budget`, returning `false` if doing
/// so would exceed `budget`.
fn try_reserve_rebar(used: &AtomicU64, size: u64, budget: u64) -> bool {
    if budget == 0 || size == 0 || size > budget {
        return false;
    }
    used.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
        let next = cur.checked_add(size)?;
        (next <= budget).then_some(next)
    })
    .is_ok()
}

/// Maximum 64 KiB pieces hashed on the GPU for one ReBAR shadow
/// (`65535 * 64 KiB`, ~4 GiB): keeps 32-bit byte offsets in `rebar_hash.hlsl`
/// overflow-free and fits in a single 1D D3D12 `Dispatch(pieces, 1, 1)`.
pub(crate) const REBAR_HASH_MAX_PIECES: u32 = 65535;

/// Compact bitset of 64 KiB pieces whose VRAM contents were modified on the CPU
/// since their GPU hash was last computed.
struct RebarDirtyBits {
    pieces: u32,
    words: Vec<u64>,
    any: bool,
}

impl RebarDirtyBits {
    fn new(pieces: u32) -> Self {
        Self {
            pieces,
            words: vec![0; (pieces as usize).div_ceil(64)],
            any: false,
        }
    }

    /// Marks pieces `[first, first + count)` (clamped to `self.pieces`) dirty.
    fn mark_pieces(&mut self, first: u32, count: u32) {
        let start = first.min(self.pieces);
        let end = start.saturating_add(count).min(self.pieces);
        if start >= end {
            return;
        }
        self.any = true;
        let first_word = (start / 64) as usize;
        let last_word = ((end - 1) / 64) as usize;
        let start_bit = start % 64;
        let end_bit = ((end - 1) % 64) + 1;
        if first_word == last_word {
            let mask = if end_bit == 64 {
                !0u64 << start_bit
            } else {
                ((1u64 << (end_bit - start_bit)) - 1) << start_bit
            };
            self.words[first_word] |= mask;
        } else {
            self.words[first_word] |= !0u64 << start_bit;
            for w in &mut self.words[first_word + 1..last_word] {
                *w = !0u64;
            }
            let tail_mask = if end_bit == 64 {
                !0u64
            } else {
                (1u64 << end_bit) - 1
            };
            self.words[last_word] |= tail_mask;
        }
    }

    /// Marks every 64 KiB piece overlapping buffer byte range `[off, off + len)`.
    fn mark_byte_range(&mut self, off: usize, len: usize) {
        if len == 0 || self.pieces == 0 {
            return;
        }
        let end_byte = off.saturating_add(len);
        let first = (off / PAGE).min(self.pieces as usize) as u32;
        let end = end_byte.div_ceil(PAGE).min(self.pieces as usize) as u32;
        if first < end {
            self.mark_pieces(first, end - first);
        }
    }

    /// Clears all dirty bits and returns coalesced `(first_piece, count)` runs.
    fn take_runs(&mut self) -> Vec<(u32, u32)> {
        if !self.any {
            return Vec::new();
        }
        self.any = false;
        let mut runs: Vec<(u32, u32)> = Vec::new();
        let mut cur: Option<(u32, u32)> = None;
        for (wi, word) in self.words.iter_mut().enumerate() {
            let mut w = std::mem::replace(word, 0);
            let base = (wi as u32) * 64;
            while w != 0 {
                let tz = w.trailing_zeros();
                let shifted = w >> tz;
                let ones = shifted.trailing_ones();
                let run_start = base + tz;
                let run_end = (run_start + ones).min(self.pieces);
                if run_start < run_end {
                    match cur {
                        Some((s, e)) if e == run_start => cur = Some((s, run_end)),
                        Some((s, e)) => {
                            runs.push((s, e - s));
                            cur = Some((run_start, run_end));
                        }
                        None => cur = Some((run_start, run_end)),
                    }
                }
                if tz + ones >= 64 {
                    break;
                }
                w &= !(((1u64 << ones) - 1) << tz);
            }
        }
        if let Some((s, e)) = cur {
            runs.push((s, e - s));
        }
        runs
    }
}

/// Prepared GPU hash state for one ReBAR shadow before a submission.
pub(crate) struct PreparedRebarHashes {
    pub(crate) hashes_gpu: u64,
    pub(crate) aligned_len: u32,
    pub(crate) pre_runs: Vec<(u32, u32)>,
}

/// GPU-side 128-bit per-64-KiB-piece hashes for a UAV-written ReBAR shadow.
///
/// Invariant: for every piece whose bit in `dirty` is clear, `buffer[piece]`
/// holds the 128-bit hash of `Shadow::buffer`'s bytes in that piece.
struct RebarHashes {
    _buffer: ID3D12Resource,
    gpu: u64,
    dirty: RebarDirtyBits,
}

/// Extra state for a [`Shadow`] backed by CPU-writable VRAM
/// (`D3D12_HEAP_TYPE_GPU_UPLOAD`, Resizable BAR).
///
/// `Shadow::data` is a write-combining pointer into VRAM and MUST NEVER be read
/// on the CPU. Read-only buffers keep `mirror = None` and `hashes = None` (zero
/// duplicate RAM or hash VRAM). On the first UAV write, `mirror` is lazily
/// populated via a one-time GPU DMA and kept synced on subsequent
/// `Shadow::copy` / `copy_back_rebar` calls; `hashes` is lazily created as a
/// `DEFAULT` UAV buffer (`16` bytes per 64 KiB piece) so post-submission
/// copy-back only DMAs 64 KiB pieces whose GPU hash actually changed.
struct RebarShadow {
    bytes: u64,
    mirror: Mutex<Option<Vec<u8>>>,
    /// Leaf lock: never acquired while holding `mirror`, `pending`, or `copied`,
    /// and never held across D3D12 or kernel calls.
    hashes: Mutex<Option<RebarHashes>>,
}

impl RebarShadow {
    fn mark_dirty_range(&self, off: usize, len: usize) {
        if len == 0 {
            return;
        }
        let mut guard = self.hashes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = guard.as_mut() {
            h.dirty.mark_byte_range(off, len);
        }
    }
}

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Monotonic resource creation sequence for epoch-based retirement.
    born_seq: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// The copy's bytes before the submission ran, from an offset on,
    /// covering the ranges its shaders write: what the GPU wrote differs.
    before: Mutex<Before>,
    /// Present when `buffer` lives in ReBAR `D3D12_HEAP_TYPE_GPU_UPLOAD` VRAM.
    rebar: Option<RebarShadow>,
}

impl Drop for Shadow {
    fn drop(&mut self) {
        if let Some(rebar) = &self.rebar {
            REBAR_SHADOW_BYTES.fetch_sub(rebar.bytes, Ordering::Relaxed);
        }
    }
}

#[derive(Default)]
struct Before {
    snapshot: Option<(usize, Vec<u8>)>,
    unchanged: UnchangedPages,
}

/// Pages already compared in full against this submission's snapshot.
#[derive(Default)]
struct UnchangedPages {
    start: usize,
    end: usize,
    pages: Vec<bool>,
}

impl UnchangedPages {
    fn begin(&mut self, start: usize, end: usize) {
        if self.pages.is_empty() {
            self.start = start;
            self.end = end;
            self.pages.resize(end.div_ceil(PAGE) - start / PAGE, false);
        }
    }

    fn differs(&mut self, address: usize, current: &[u8], before: &[u8]) -> bool {
        let page = address / PAGE;
        let unchanged = &mut self.pages[page - self.start / PAGE];
        if *unchanged {
            return false;
        }
        if current != before {
            return true;
        }
        // A later range can touch bytes a partial comparison missed. At
        // the edges, only bytes covered by the snapshot can be compared.
        *unchanged = address == (page * PAGE).max(self.start)
            && address + current.len() == ((page + 1) * PAGE).min(self.end);
        false
    }

    fn clear(&mut self) {
        self.pages.clear();
    }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool (at most 8 kept, the largest).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    if pool.len() > 8 {
        pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
        pool.truncate(8);
    }
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Where the copy starts in title memory and how long it is.
    pub(crate) fn span(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// The offset of a title address in the copy, if it lies in it.
    pub(crate) fn offset_in(&self, address: u64) -> Option<u64> {
        address.checked_sub(self.start).filter(|&at| at < self.len)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// `len` bytes at a title address from the host copy, for the log: the
    /// bytes the copy holds, as of its last refresh (None outside the buffer).
    pub(crate) fn bytes(&self, address: u64, len: u64) -> Option<&[u8]> {
        if address < self.start || len == 0 || len > self.left(address) {
            return None;
        }
        let at = usize::try_from(address - self.start).ok()?;
        let len = usize::try_from(len).ok()?;
        // SAFETY: inside the mapping, which is as long as the buffer.
        Some(unsafe { std::slice::from_raw_parts(self.data.add(at), len) })
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset. Cut to the copy.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        self.refresh_as(list, address, len, D3D12_RESOURCE_STATE_GENERIC_READ);
    }

    /// Refresh a range and prepare its host buffer for the specified GPU use.
    pub(crate) fn refresh_as(
        self: &Arc<Self>,
        list: &HostList,
        address: u64,
        len: u64,
        access: D3D12_RESOURCE_STATES,
    ) {
        list.use_shadow(&self.buffer, access);
        let from = address.max(self.start);
        let len = range_inside(
            from,
            len.saturating_sub(from.saturating_sub(address)),
            self.start,
            self.len,
        );
        if len == 0 {
            return;
        }
        list.defer_refresh(self, from as usize, len as usize);
    }

    fn new(
        buffer: ID3D12Resource,
        data: *mut u8,
        gpu: u64,
        start: u64,
        len: u64,
        rebar_bytes: Option<u64>,
    ) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            born_seq: crate::host::birth_seq(),
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            before: Mutex::default(),
            rebar: rebar_bytes.map(|bytes| RebarShadow {
                bytes,
                mirror: Mutex::new(None),
                hashes: Mutex::new(None),
            }),
        }
    }

    /// Whether this copy lives in ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`) VRAM.
    pub(crate) fn is_rebar(&self) -> bool {
        self.rebar.is_some()
    }

    /// Logical byte length of the title buffer.
    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    /// Base host GPU virtual address of the shadow buffer.
    pub(crate) fn base_gpu(&self) -> u64 {
        self.gpu
    }

    /// Lazily allocates the per-64-KiB-piece hash UAV buffer for a ReBAR shadow
    /// and returns [`PreparedRebarHashes`], where `pre_runs` are the
    /// `(first_piece, piece_count)` runs that must be hashed before the
    /// submission executes (all pieces on initial creation, or only
    /// CPU-dirtied pieces thereafter). Returns `None` if this shadow is not
    /// eligible for GPU hashing or allocation fails.
    pub(crate) fn prepare_rebar_hashes(
        &self,
        alloc: impl FnOnce(u64) -> Option<ID3D12Resource>,
    ) -> Option<PreparedRebarHashes> {
        let rebar = self.rebar.as_ref()?;
        if self.len == 0 {
            return None;
        }
        let pieces = u32::try_from(self.len.div_ceil(PAGE as u64)).ok()?;
        if pieces > REBAR_HASH_MAX_PIECES {
            return None;
        }
        let aligned_len = u32::try_from(self.len.next_multiple_of(16)).ok()?;
        if rebar
            .mirror
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_none()
        {
            return None;
        }
        {
            let mut guard = rebar.hashes.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(h) = guard.as_mut() {
                return Some(PreparedRebarHashes {
                    hashes_gpu: h.gpu,
                    aligned_len,
                    pre_runs: h.dirty.take_runs(),
                });
            }
        }
        // Allocate outside the leaf lock.
        let buffer = alloc(u64::from(pieces) * 16)?;
        crate::host::name(&buffer, || {
            format!("ReBAR hashes {:#x}+{:#x}", self.start, self.len)
        });
        // SAFETY: live D3D12 buffer resource.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let mut guard = rebar.hashes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = guard.as_mut() {
            return Some(PreparedRebarHashes {
                hashes_gpu: h.gpu,
                aligned_len,
                pre_runs: h.dirty.take_runs(),
            });
        }
        *guard = Some(RebarHashes {
            _buffer: buffer,
            gpu,
            dirty: RebarDirtyBits::new(pieces),
        });
        Some(PreparedRebarHashes {
            hashes_gpu: gpu,
            aligned_len,
            pre_runs: vec![(0, pieces)],
        })
    }

    /// Marks `(first_piece, piece_count)` runs dirty in this ReBAR shadow's
    /// hash state (e.g. if a pre-hash recording failed after draining dirty
    /// bits).
    pub(crate) fn mark_rebar_dirty_runs(&self, runs: &[(u32, u32)]) {
        let Some(rebar) = &self.rebar else {
            return;
        };
        let mut guard = rebar.hashes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = guard.as_mut() {
            for &(first, count) in runs {
                h.dirty.mark_pieces(first, count);
            }
        }
    }

    /// Marks all 64 KiB pieces overlapping buffer-offset `spans` dirty in this
    /// ReBAR shadow's hash state (used when a submission falls back to
    /// full-range readback without running the post-compare shader).
    pub(crate) fn mark_rebar_dirty_spans(&self, spans: &[(usize, usize)]) {
        let Some(rebar) = &self.rebar else {
            return;
        };
        let mut guard = rebar.hashes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = guard.as_mut() {
            for &(from, to) in spans {
                if from < to {
                    h.dirty.mark_byte_range(from, to - from);
                }
            }
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    pub(crate) fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, grown to cover every range the
    /// submission declares (the world declares overlapping views of its
    /// 174 MB heap dozens of times), in reused memory (fresh snapshots cost
    /// ~2.3 million page faults per world frame).
    pub(crate) fn writing(
        &self,
        address: usize,
        len: usize,
        rebar_init: impl FnOnce(&ID3D12Resource, u64) -> Option<Vec<u8>>,
    ) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address + len));
        if let Some(rebar) = &self.rebar {
            let mut mirror = rebar.mirror.lock().unwrap_or_else(|e| e.into_inner());
            if mirror.is_none() {
                *mirror = rebar_init(&self.buffer, self.len);
            }
            return;
        }
        let (from, to) = self.inside(address, len);
        // SAFETY: inside the mapping (write-back system memory; ReBAR buffers
        // returned early above so write-combining memory is never read).
        let host =
            |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let before = &mut before.snapshot;
        let (start, end) = match &*before {
            Some((at, bytes)) => ((*at).min(from), (at + bytes.len()).max(to)),
            None => (from, to),
        };
        if let Some((at, bytes)) = &*before
            && (*at, at + bytes.len()) == (start, end)
        {
            return;
        }
        // Parts newly covered from the copy, the rest as it was taken.
        let mut grown = pooled(end - start);
        match before.take() {
            Some((at, bytes)) => {
                grown.extend_from_slice(host(start, at));
                grown.extend_from_slice(&bytes);
                grown.extend_from_slice(host(at + bytes.len(), end));
                give_back(bytes);
            }
            None => grown.extend_from_slice(host(start, end)),
        }
        *before = Some((start, grown));
    }

    /// Clears pending UAV-write ranges for a ReBAR shadow if readback failed.
    pub(crate) fn clear_pending_rebar(&self) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// ReBAR counterpart to [`Shadow::copy_back`]: merges post-submission GPU
    /// bytes (`gpu_slices`, already DMAed into cached `READBACK` RAM) for
    /// sorted disjoint buffer-offset `spans` against the cached RAM mirror,
    /// without ever reading write-combining VRAM on the CPU.
    pub(crate) fn copy_back_rebar(&self, spans: &[(usize, usize)], gpu_slices: &[&[u8]]) -> usize {
        let Some(rebar) = &self.rebar else {
            return 0;
        };
        let mut mirror_guard = rebar.mirror.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let mut vram_dirty: Vec<(usize, usize)> = Vec::new();
        let start = self.start as usize;
        if let Some(mirror) = mirror_guard.as_mut() {
            let mut title = None;
            for (&(from, to), &gpu_span) in spans.iter().zip(gpu_slices) {
                if from >= to || to > mirror.len() || gpu_span.len() != to - from {
                    continue;
                }
                let mut synced = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PAGE - (start + at) % PAGE).min(to - at);
                    let gpu = &gpu_span[at - from..at - from + n];
                    let old = &mut mirror[at..at + n];
                    if gpu != old {
                        let title_buf = &mut title.get_or_insert_with(|| vec![0u8; PAGE])[..n];
                        if crate::host::read_into((start + at) as *const u8, title_buf) {
                            let mut changed = false;
                            let mut host_differs = false;
                            for ((t, &g), o) in title_buf.iter_mut().zip(gpu).zip(old.iter_mut()) {
                                if g != *o && *t != g {
                                    *t = g;
                                    changed = true;
                                }
                                if *t != g {
                                    host_differs = true;
                                }
                                *o = *t;
                            }
                            if changed {
                                crate::watch::writing(start + at, n);
                                crate::host::write_mapped(
                                    (start + at) as *mut u8,
                                    title_buf.as_ptr(),
                                    n,
                                );
                                written += n;
                            }
                            if host_differs {
                                // Only push back to WC VRAM when a concurrent CPU
                                // write altered bytes in the same page; record the
                                // range so its GPU piece hash is refreshed next.
                                // SAFETY: `at + n <= self.len`, write-only WC store.
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        title_buf.as_ptr(),
                                        self.data.add(at),
                                        n,
                                    );
                                }
                                vram_dirty.push((at, n));
                            }
                            synced.push((start + at) / PAGE);
                        } else {
                            // Unmapped in title memory: VRAM holds `gpu`, keep mirror in sync.
                            old.copy_from_slice(gpu);
                        }
                    }
                    at += n;
                }
                let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
                if first < last
                    && !synced.is_empty()
                    && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
                    && let Some((now, _)) = crate::watch::writes(first * PAGE, last * PAGE, true)
                {
                    let base = start / PAGE;
                    let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                    for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                        copied[page - base] = now;
                    }
                }
            }
        }
        drop(mirror_guard);
        for (at, n) in vram_dirty {
            rebar.mark_dirty_range(at, n);
        }
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        written
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let Before {
            snapshot,
            unchanged,
        } = &mut *before;
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if let Some((taken, before)) = &*snapshot
            && *taken <= from
            && to <= taken + before.len()
            && from < to
        {
            unchanged.begin(start + taken, start + taken + before.len());
            let before = &before[from - taken..to - taken];
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            // An unchanged page is never modified by copy-back, so later
            // overlapping ranges can skip it for the rest of the submission.
            let mut title = None;
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let n = (PAGE - (start + at) % PAGE).min(to - at);
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                if unchanged.differs(start + at, host, old)
                    && let title = &mut title.get_or_insert_with(|| vec![0u8; PAGE])[..n]
                    && crate::host::read_into((start + at) as *const u8, title)
                {
                    // Title memory with the GPU's bytes put in; the copy
                    // takes the title's bytes in turn.
                    let mut changed = false;
                    for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
                        if *h != *o && *t != *h {
                            *t = *h;
                            changed = true;
                        }
                        *h = *t;
                    }
                    if changed {
                        crate::watch::writing(start + at, n);
                        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                        written += n;
                    }
                    synced.push((start + at) / PAGE);
                }
                at += n;
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if first < last
                && !synced.is_empty()
                && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
                && let Some((now, _)) = crate::watch::writes(first * PAGE, last * PAGE, true)
            {
                let base = start / PAGE;
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                    copied[page - base] = now;
                }
            }
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|&r| r != (address, address + len));
        if pending.is_empty()
            && let Some((_, bytes)) = snapshot.take()
        {
            unchanged.clear();
            give_back(bytes);
        }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            // Only mark the shadow permanently unwatched if it has never been
            // observed as watched. A sub-range that lands entirely in an
            // unmapped hole inside an already-watched view pool must not
            // disable write-watching for the rest of the pool.
            let _ =
                self.watched
                    .compare_exchange(0, NOT_WATCHED, Ordering::Relaxed, Ordering::Relaxed);
        }
        self.copy(from, to);
        to - from
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if from >= to {
            return 0;
        }
        let off = from - start;
        let len = to - from;
        if let Some(rebar) = &self.rebar {
            let mut mirror = rebar.mirror.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(m) = mirror.as_mut()
                && off + len <= m.len()
            {
                // Copy once from guarded title memory into cached RAM, then
                // push straight to write-combining VRAM so both see identical bytes.
                crate::host::copy_mapped(
                    m.as_mut_ptr().wrapping_add(off),
                    from as *const u8,
                    len,
                    false,
                );
                // SAFETY: `off + len <= self.len`, write-only WC store.
                unsafe {
                    std::ptr::copy_nonoverlapping(m.as_ptr().add(off), self.data.add(off), len);
                }
                drop(mirror);
                rebar.mark_dirty_range(off, len);
                return len;
            }
        }
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(off) };
        crate::host::copy_mapped(at, from as *const u8, len, false);
        if let Some(rebar) = &self.rebar {
            rebar.mark_dirty_range(off, len);
        }
        len
    }
}

/// A refresh that copies none of its range is skipped: the writes it was
/// asked about were already in the copy, the range is outside the buffer,
/// or shaders were still writing all of it (that part is left to a later
/// refresh). Counted for the frame-rate line as one skipped refresh of
/// `len` bytes, so `0 skipped` there means no refresh was answered
/// without a copy, not that nothing was checked. A range copied in part is
/// not skipped; of what it did not copy, `unchanged` counts only the parts
/// this refresh checked, not the sub-ranges shaders still write.
#[cfg(any(feature = "diagnostics", test))]
pub(crate) fn skipped(len: usize, copied: usize) -> Option<u64> {
    (len > 0 && copied == 0).then_some(len as u64)
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive,
/// and any active epoch keeps it until that work completes.
pub(crate) fn unregister(resource: usize, start: u64) {
    let shadow = {
        let mut map = buffers();
        map.remove(&(start, resource)).and_then(|e| e.shadow)
    };
    if let Some(shadow) = shadow {
        let born = shadow.born_seq;
        crate::host::retire_born(shadow, born);
    }
}

/// Allocates a shadow backing buffer: prefers ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`)
/// VRAM within the adapter's budget, falling back cleanly to system-RAM
/// [`Host::shared_buffer`] when ReBAR is unavailable, disabled, or full.
fn alloc_shadow_buffer(size: u64) -> Option<(ID3D12Resource, *mut u8, Option<u64>)> {
    let h = host()?;
    let budget = h.rebar_shadow_budget();
    if try_reserve_rebar(&REBAR_SHADOW_BYTES, size, budget) {
        if let Some((buffer, data)) = h.try_rebar_shadow_buffer(size) {
            return Some((buffer, data, Some(size)));
        }
        REBAR_SHADOW_BYTES.fetch_sub(size, Ordering::Relaxed);
    }
    let (buffer, data) = h.shared_buffer(size)?;
    Some((buffer, data, None))
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let alloc_size = len.next_multiple_of(1 << 16);
        let (buffer, data, rebar_bytes) = alloc_shadow_buffer(alloc_size)?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                rebar = rebar_bytes.is_some(),
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                rebar = rebar_bytes.is_some(),
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(
            buffer,
            data,
            gpu,
            key.0,
            len,
            rebar_bytes,
        )));
    }
    entry.shadow.clone()
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, if one has already been made: the same lookup as [`find`] without
/// making a copy, for paths that must not allocate one.
pub(crate) fn peek(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let map = buffers();
    let shadow = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .and_then(|(_, e)| e.shadow.clone());
    if shadow.is_some() {
        return shadow;
    }
    drop(map);
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    LOOSE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&(start, len))
        .cloned()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

/// Copies of just one view's bytes, by (start, length): for a view whose
/// start is not on an element of its host copy (`aligned`). The heap slot
/// holds the copy; this map only finds it again, so it holds weak references
/// and the copy goes when the slot is remade.
static ALIGNED: Mutex<BTreeMap<(u64, u64), std::sync::Weak<Shadow>>> = Mutex::new(BTreeMap::new());

/// A copy of `[address, address + len)` that starts at `address` itself, so a
/// structured view of it can have its elements where the title put them.
///
/// A title aligns a resource; the host copy of a buffer starts where the
/// buffer does, so a view of memory the title placed inside another object
/// (`View::Placed`) starts at an offset that need not be a multiple of the
/// view's stride. This copy starts at the view, and the slot that made it
/// keeps it alive.
///
/// Its own refresh copies the 64 KiB pages the title wrote since they were
/// last copied, straight from title memory: a submission's copy-back runs
/// before the next submission prepares (`Host::execute` waits for the GPU
/// before copying back), so by then title memory holds what the GPU wrote in
/// any copy, and what shaders wrote in this one has gone back there. Copying
/// from the buffer's copy instead would copy the view's whole range again on
/// every submission.
pub(crate) fn aligned(address: u64, len: u64) -> Option<Arc<Shadow>> {
    if len == 0 {
        return None;
    }
    let key = (address, len);
    if let Some(copy) = aligned_existing(address, len) {
        return Some(copy);
    }
    // Views hold hundreds of megabytes in the world; made as a buffer's copy
    // is, so that shaders may write it (unordered access).
    let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(64))?;
    crate::host::name(&buffer, || format!("aligned view {address:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let copy = Arc::new(Shadow::new(buffer, data, gpu, address, len, None));
    // Populate now so a ResourceHeap-only first use (which never walks the
    // table refresh path) still sees title memory rather than zeros (#553).
    // SAFETY: `data` is the mapping of `buffer`; `address` is title memory
    // of `len` bytes the view was created for.
    crate::host::copy_mapped(data, address as *const u8, len as usize, false);
    {
        let mut copies = ALIGNED.lock().unwrap_or_else(|e| e.into_inner());
        // Drop expired weaks so distinct (address, len) churn cannot grow
        // the map without bound (docs/code-standards.md host-cache bound).
        copies.retain(|_, weak| weak.strong_count() > 0);
        // Another recorder may have published the same range while we allocated.
        if let Some(existing) = copies.get(&key).and_then(std::sync::Weak::upgrade) {
            return Some(existing);
        }
        copies.insert(key, Arc::downgrade(&copy));
    }
    tracing::debug!(
        address = format_args!("{address:#x}"),
        len,
        "aligned view copy made"
    );
    Some(copy)
}

/// The copy `aligned` made for this range, if one is still alive.
pub(crate) fn aligned_existing(address: u64, len: u64) -> Option<Arc<Shadow>> {
    ALIGNED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&(address, len))
        .and_then(std::sync::Weak::upgrade)
}

/// How many bytes of `[address, address + len)` lie inside the copy that
/// starts at `start` of `size` bytes: what a refresh of that copy can bring up
/// to date. `0` when the range is not inside it at all.
fn range_inside(address: u64, len: u64, start: u64, size: u64) -> u64 {
    let end = start.saturating_add(size);
    let from = address.max(start);
    let to = address.saturating_add(len).min(end);
    to.saturating_sub(from)
}

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    // Keep unbounded loose 4 MiB windows in system memory so they cannot
    // starve registered title buffers out of the bounded ReBAR budget.
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len, None));
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    all.extend(
        ALIGNED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .filter_map(|w| w.upgrade())
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Range;

    #[test]
    fn concurrent_aligned_views_share_the_copy_refreshed_at_submission() {
        if crate::host::host().is_none() {
            return;
        }

        const THREADS: usize = 8;
        let values = [[19u32; 14]; 8];
        for value in &values {
            let address = value.as_ptr() as u64;
            let len = size_of_val(value) as u64;
            let gate = std::sync::Barrier::new(THREADS);
            let copies = std::thread::scope(|scope| {
                let jobs: Vec<_> = (0..THREADS)
                    .map(|_| {
                        let gate = &gate;
                        scope.spawn(move || {
                            gate.wait();
                            aligned(address, len).expect("aligned host buffer")
                        })
                    })
                    .collect();
                jobs.into_iter()
                    .map(|job| job.join().unwrap())
                    .collect::<Vec<_>>()
            });
            let refreshed = aligned_existing(address, len).expect("cached copy");
            refreshed.update(address as usize, len as usize);
            for copy in copies {
                assert!(
                    Arc::ptr_eq(&copy, &refreshed),
                    "slot retained a different copy"
                );
                // SAFETY: the mapped buffer has 56 bytes and refresh has completed.
                let first = unsafe { copy.data.cast::<u32>().read_unaligned() };
                assert_eq!(first, value[0]);
            }
        }
    }

    /// A refresh of the aligned copy brings up to date the bytes of the view
    /// that lie inside it, cut at both ends, and nothing when the range is
    /// outside it.
    #[test]
    fn a_refresh_covers_the_views_bytes_inside_its_copy() {
        // The hit-detection view: 3136 bytes of 56-byte elements, 48 bytes
        // into its buffer, in a copy that starts at the view.
        assert_eq!(range_inside(0x3000_0030, 3136, 0x3000_0030, 3136), 3136);
        // A view reaching past its copy is cut to it.
        assert_eq!(range_inside(0x4000_0100, 900, 0x4000_0100, 0x100), 0x100);
        // A view starting before its copy loses the part before it.
        assert_eq!(range_inside(0x0ff8, 0x40, 0x1000, 0x1000), 0x38);
        // Outside the copy at either end: nothing to refresh.
        assert_eq!(range_inside(0x1000, 0x10, 0x2000, 0x100), 0);
        assert_eq!(range_inside(0x20f0, 0x20, 0x1000, 0x100), 0);
        // A size that would overflow does not.
        assert_eq!(range_inside(u64::MAX - 8, 16, 0, 16), 0);
    }

    /// A refresh that copied nothing of its range is what the frame-rate
    /// line's `skipped` counts; one that copied any part of it is not, and
    /// an empty range is neither.
    #[test]
    fn a_refresh_that_copies_nothing_is_counted_as_skipped() {
        assert_eq!(skipped(0, 0), None);
        assert_eq!(skipped(4 << 20, 0), Some(4 << 20));
        assert_eq!(skipped(4 << 20, 1), None);
        assert_eq!(skipped(4 << 20, 4 << 20), None);
        // Nothing in the buffer: the range was outside it.
        assert_eq!(skipped(4096, 0), Some(4096));
    }

    #[test]
    fn partial_match_leaves_unchecked_bytes_visible() {
        let mut unchanged = UnchangedPages::default();
        let before = vec![0; PAGE];
        let mut host = before.clone();
        host[PAGE - 1] = 1;
        unchanged.begin(PAGE, 2 * PAGE);

        assert!(!unchanged.differs(PAGE, &host[..4], &before[..4]));
        assert!(!unchanged.pages[0]);
        assert!(unchanged.differs(PAGE, &host, &before));
        assert!(!unchanged.pages[0]);
    }

    #[test]
    fn snapshot_edges_are_cacheable_and_survive_later_ranges() {
        let mut unchanged = UnchangedPages::default();
        unchanged.begin(PAGE - 3, PAGE + 4);
        assert!(!unchanged.differs(PAGE - 3, &[1; 3], &[1; 3]));
        assert!(!unchanged.differs(PAGE, &[2; 4], &[2; 4]));
        assert_eq!(unchanged.pages, [true, true]);

        unchanged.begin(PAGE - 3, PAGE + 4);
        assert!(!unchanged.differs(PAGE - 2, &[1; 2], &[1; 2]));
        assert!(!unchanged.differs(PAGE + 1, &[2; 2], &[2; 2]));
        assert_eq!(unchanged.pages, [true, true]);
    }

    #[test]
    fn next_submission_checks_the_same_page_again() {
        let mut unchanged = UnchangedPages::default();
        unchanged.begin(0, 4);
        assert!(!unchanged.differs(0, &[0; 4], &[0; 4]));
        unchanged.clear();
        unchanged.begin(0, 4);
        assert!(unchanged.differs(0, &[0, 0, 0, 1], &[0; 4]));
    }

    // Both versions walk exactly the same declared ranges, in their
    // original order. Only the page comparison differs.
    fn scan(
        host: &[u8],
        before: &[u8],
        ranges: &[Range<usize>],
        mut differs: impl FnMut(usize, &[u8], &[u8]) -> bool,
    ) -> usize {
        let mut changed = 0;
        for range in ranges {
            let mut at = range.start;
            while at < range.end {
                let n = (PAGE - at % PAGE).min(range.end - at);
                changed += usize::from(differs(at, &host[at..at + n], &before[at..at + n]));
                at += n;
            }
        }
        changed
    }

    #[test]
    fn overlapping_ranges_match_uncached_comparisons() {
        let before = vec![0; 4 * PAGE];
        let mut host = before.clone();
        for at in [2, PAGE - 1, PAGE + 5, 3 * PAGE + 7] {
            host[at] = 1;
        }
        let ranges = [
            0..4,
            2..2 * PAGE + 3,
            PAGE - 2..4 * PAGE,
            2 * PAGE..3 * PAGE,
        ];
        let uncached = scan(&host, &before, &ranges, |_, host, before| host != before);
        let mut unchanged = UnchangedPages::default();
        unchanged.begin(0, before.len());
        let cached = scan(&host, &before, &ranges, |at, host, before| {
            unchanged.differs(at, host, before)
        });
        assert_eq!(uncached, 6);
        assert_eq!(cached, uncached);
        assert_eq!(unchanged.pages, [false, false, true, false]);
    }

    // cargo test --release -p victory-gpu benchmark_overlapping_copy_back -- --ignored --nocapture
    #[test]
    #[ignore = "synthetic CPU release benchmark"]
    fn benchmark_overlapping_copy_back() {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        const SIZE: usize = 128 << 20;
        let before = vec![0xA5; SIZE];
        let mut host = before.clone();
        for page in (0..SIZE / PAGE).step_by(64) {
            host[page * PAGE + PAGE / 2] ^= 0xFF;
        }
        let ranges: Vec<_> = (0..24).map(|i| i * PAGE + i..SIZE - i * 13).collect();
        let mut baseline = [Duration::ZERO; 5];
        let mut cached = [Duration::ZERO; 5];
        let mut unchanged = UnchangedPages::default();
        for (baseline, cached) in baseline.iter_mut().zip(&mut cached) {
            let started = Instant::now();
            let expected = scan(
                black_box(&host),
                black_box(&before),
                black_box(&ranges),
                |_, host, before| host != before,
            );
            *baseline = started.elapsed();

            unchanged.clear();
            let started = Instant::now();
            unchanged.begin(0, before.len());
            let actual = scan(
                black_box(&host),
                black_box(&before),
                black_box(&ranges),
                |at, host, before| unchanged.differs(at, host, before),
            );
            *cached = started.elapsed();
            assert_eq!(actual, expected);
        }
        baseline.sort_unstable();
        cached.sort_unstable();
        println!(
            "median of five runs, 24 overlapping ranges across 128 MiB: {:?} -> {:?} ({:.1}x)",
            baseline[2],
            cached[2],
            baseline[2].as_secs_f64() / cached[2].as_secs_f64(),
        );
    }

    #[cfg(windows)]
    #[test]
    fn unregister_drops_shadow_immediately_when_idle() {
        let _retire = crate::host::isolated_retire_scope();
        if host().is_none() {
            return;
        }
        let data = vec![0u8; 4096];
        let start = data.as_ptr() as u64;
        let resource = 0xdead_beef_0001_usize;
        register(resource, start, data.len() as u64);
        let weak = {
            let shadow = find(start, 64).expect("shadow");
            Arc::downgrade(&shadow)
        };
        assert_eq!(weak.strong_count(), 1);
        unregister(resource, start);
        assert_eq!(weak.strong_count(), 0);
    }

    #[cfg(windows)]
    #[test]
    fn bytes_reads_the_copy_inside_the_buffer_only() {
        let _retire = crate::host::isolated_retire_scope();
        if host().is_none() {
            return;
        }
        let data = vec![0u8; 4096];
        let start = data.as_ptr() as u64;
        let resource = 0xdead_beef_0002_usize;
        register(resource, start, data.len() as u64);
        let shadow = find(start, 64).expect("shadow");
        assert_eq!(shadow.bytes(start, 16).map(<[u8]>::len), Some(16));
        assert_eq!(
            shadow
                .bytes(start + data.len() as u64 - 16, 16)
                .map(<[u8]>::len),
            Some(16),
            "a range ending in the buffer's last bytes is inside it"
        );
        assert!(shadow.bytes(start - 4, 4).is_none(), "before the buffer");
        assert!(
            shadow.bytes(start + data.len() as u64 - 8, 16).is_none(),
            "past the buffer"
        );
        assert!(shadow.bytes(start, 0).is_none(), "nothing to read");
        unregister(resource, start);
    }

    #[cfg(windows)]
    #[test]
    fn unregister_retires_shadow_until_active_epoch_finishes() {
        let _retire = crate::host::isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let data = vec![0u8; 4096];
        let start = data.as_ptr() as u64;
        let resource = 0xdead_beef_0002_usize;
        register(resource, start, data.len() as u64);
        let weak = {
            let shadow = find(start, 64).expect("shadow");
            Arc::downgrade(&shadow)
        };
        assert_eq!(weak.strong_count(), 1);

        let list = host.new_list().expect("host list");
        list.mark_work();

        unregister(resource, start);
        assert_eq!(
            weak.strong_count(),
            1,
            "unregistered shadow must wait for active epochs that recorded after its birth"
        );

        list.close();
        list.reset();
        assert_eq!(weak.strong_count(), 0);
    }

    #[cfg(windows)]
    #[test]
    fn unregister_drops_new_shadow_despite_older_pinned_list() {
        let _retire = crate::host::isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let stale = host.new_list().expect("stale list");
        stale.mark_work();

        let data = vec![0u8; 4096];
        let start = data.as_ptr() as u64;
        let resource = 0xdead_beef_0003_usize;
        register(resource, start, data.len() as u64);
        let weak = {
            let shadow = find(start, 64).expect("shadow");
            Arc::downgrade(&shadow)
        };
        assert_eq!(weak.strong_count(), 1);

        unregister(resource, start);
        assert_eq!(
            weak.strong_count(),
            0,
            "shadow created after stale list's last work must not leak behind it"
        );
        stale.close();
        stale.reset();
    }

    #[test]
    fn rebar_budget_reservation_enforces_cap_and_releases() {
        let used = AtomicU64::new(0);
        assert!(!try_reserve_rebar(&used, 64, 0));
        assert!(!try_reserve_rebar(&used, 0, 1024));
        assert!(!try_reserve_rebar(&used, 2048, 1024));
        assert!(try_reserve_rebar(&used, 600, 1024));
        assert_eq!(used.load(Ordering::Relaxed), 600);
        assert!(!try_reserve_rebar(&used, 500, 1024));
        assert!(try_reserve_rebar(&used, 424, 1024));
        assert_eq!(used.load(Ordering::Relaxed), 1024);
    }

    /// Serializes ReBAR GPU tests that share the process-wide `Host` hash stats.
    #[cfg(windows)]
    static REBAR_GPU_TEST: Mutex<()> = Mutex::new(());

    #[cfg(windows)]
    #[test]
    fn rebar_shadow_gpu_write_round_trips_and_preserves_concurrent_cpu_writes() {
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COPY_DEST;

        let _rebar = REBAR_GPU_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let _retire = crate::host::isolated_retire_scope();
        let Some(h) = host() else {
            return;
        };
        if h.rebar_shadow_budget() == 0 {
            // Requires a discrete GPU with D3D12 GPU upload heap (ReBAR) support.
            return;
        }
        let mut backing = vec![0x11u8; 4096];
        let start = backing.as_ptr() as u64;
        let resource = 0xdead_beef_0005_usize;
        register(resource, start, backing.len() as u64);

        let shadow = find(start, backing.len() as u64).expect("shadow");
        assert!(
            shadow.is_rebar(),
            "expected ReBAR-backed shadow on ReBAR adapter"
        );

        let list = h.new_list().expect("host list");
        // Seed initial title bytes, declare overlapping GPU writes, and overwrite
        // bytes 16..32 on the GPU via CopyBufferRegion while the CPU concurrently
        // writes byte 64 in the same page.
        shadow.refresh_as(
            &list,
            start,
            backing.len() as u64,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        shadow.gpu_writes(&list, start, 128);
        shadow.gpu_writes(&list, start + 16, 128);

        let (src_buf, src_ptr) = h.upload_buffer(16).expect("upload");
        // SAFETY: `src_ptr` is mapped for 16 bytes.
        unsafe {
            std::ptr::write_bytes(src_ptr, 0xAA, 16);
        }
        // SAFETY: live buffers and valid 16-byte copy inside both.
        unsafe {
            list.list
                .CopyBufferRegion(&shadow.buffer, 16, &src_buf, 0, 16);
        }
        // Simulate a concurrent CPU write to byte 64 while the GPU runs.
        let concurrent_addr = backing.as_mut_ptr() as usize;
        list.defer_host(move || {
            // SAFETY: `backing` stays alive for the duration of this test.
            unsafe {
                *(concurrent_addr as *mut u8).add(64) = 0x77;
            }
        });
        list.close();
        h.execute(&[&list]);

        assert!(backing[..16].iter().all(|&b| b == 0x11));
        assert!(backing[16..32].iter().all(|&b| b == 0xAA));
        assert_eq!(backing[64], 0x77);
        assert!(backing[65..128].iter().all(|&b| b == 0x11));
        // Also verify the initial 0x11 bytes ACTUALLY arrived in VRAM (not zeroed
        // by ReadProcessMemory failing on a PCIe BAR destination!).
        let rebar_mirror = shadow
            .rebar
            .as_ref()
            .unwrap()
            .mirror
            .lock()
            .unwrap()
            .clone()
            .expect("mirror populated after gpu_writes");
        assert!(
            rebar_mirror[..16].iter().all(|&b| b == 0x11),
            "initial title bytes did not reach ReBAR VRAM! got: {:02x?}",
            &rebar_mirror[..16]
        );

        list.reset();
        drop(shadow);
        unregister(resource, start);
    }

    #[test]
    fn rebar_dirty_bits_marks_and_coalesces_across_word_boundaries() {
        let mut bits = RebarDirtyBits::new(140);
        assert!(bits.take_runs().is_empty());

        // Single piece, cross-word range (62..67), and tail clamped at 140.
        bits.mark_pieces(3, 1);
        bits.mark_pieces(62, 5);
        bits.mark_pieces(138, 10);
        assert_eq!(bits.take_runs(), vec![(3, 1), (62, 5), (138, 2)]);
        assert!(bits.take_runs().is_empty());

        // Byte range straddling a 64 KiB boundary marks both pieces.
        bits.mark_byte_range(PAGE - 8, 16);
        bits.mark_byte_range(5 * PAGE, PAGE);
        assert_eq!(bits.take_runs(), vec![(0, 2), (5, 1)]);
        assert!(bits.take_runs().is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn rebar_gpu_hash_compare_skips_unchanged_pieces_and_merges_concurrent_writes() {
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COPY_DEST;

        let _rebar = REBAR_GPU_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let _retire = crate::host::isolated_retire_scope();
        let Some(h) = host() else {
            return; // no host GPU here
        };
        if h.rebar_shadow_budget() == 0 {
            // Requires a discrete GPU with D3D12 GPU upload heap (ReBAR) support.
            return;
        }
        let errors_before = h.d3d12_errors();
        let mut backing = vec![0x11u8; 4 * PAGE];
        let start = backing.as_ptr() as u64;
        let resource = 0xdead_beef_0004_usize;
        register(resource, start, backing.len() as u64);

        let shadow = find(start, backing.len() as u64).expect("shadow");
        assert!(
            shadow.is_rebar(),
            "expected ReBAR-backed shadow on ReBAR adapter"
        );

        let (src_aa, ptr_aa) = h.upload_buffer(16).expect("upload AA");
        let (src_bb, ptr_bb) = h.upload_buffer(16).expect("upload BB");
        let (src_11, ptr_11) = h.upload_buffer(16).expect("upload 11");
        // SAFETY: each upload buffer is mapped for 16 bytes.
        unsafe {
            std::ptr::write_bytes(ptr_aa, 0xAA, 16);
            std::ptr::write_bytes(ptr_bb, 0xBB, 16);
            std::ptr::write_bytes(ptr_11, 0x11, 16);
        }

        // Submission 1: declare the entire 256 KiB buffer as UAV-written, but
        // only modify piece 1 (+32..+48) and piece 3 (+64..+80) on the GPU
        // while the CPU concurrently writes piece 1 (+128) and piece 0 (+128).
        let stats0 = h.rebar_hash_stats();
        let list = h.new_list().expect("host list");
        shadow.refresh_as(
            &list,
            start,
            backing.len() as u64,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        shadow.gpu_writes(&list, start, backing.len() as u64);
        // SAFETY: live buffers and in-bounds 16-byte copies.
        unsafe {
            list.list
                .CopyBufferRegion(&shadow.buffer, (PAGE + 32) as u64, &src_aa, 0, 16);
            list.list
                .CopyBufferRegion(&shadow.buffer, (3 * PAGE + 64) as u64, &src_bb, 0, 16);
        }
        let base_addr = backing.as_mut_ptr() as usize;
        list.defer_host(move || {
            // SAFETY: `backing` stays alive for the duration of this test.
            unsafe {
                *(base_addr as *mut u8).add(128) = 0x55;
                *(base_addr as *mut u8).add(PAGE + 128) = 0x77;
            }
        });
        list.close();
        h.execute(&[&list]);
        list.reset();

        let stats1 = h.rebar_hash_stats();
        assert_eq!(stats1.compared_pieces - stats0.compared_pieces, 4);
        assert_eq!(
            stats1.changed_pieces - stats0.changed_pieces,
            2,
            "pieces 0 and 2 were untouched by the GPU and must be skipped"
        );
        assert_eq!(stats1.post_readback_calls - stats0.post_readback_calls, 1);
        assert_eq!(
            stats1.post_readback_bytes - stats0.post_readback_bytes,
            (2 * PAGE) as u64,
            "only the 2 changed 64 KiB pieces should be DMAed back"
        );
        assert_eq!(backing[128], 0x55);
        assert!(backing[PAGE + 32..PAGE + 48].iter().all(|&b| b == 0xAA));
        assert_eq!(backing[PAGE + 128], 0x77);
        assert!(backing[2 * PAGE..3 * PAGE].iter().all(|&b| b == 0x11));
        assert!(
            backing[3 * PAGE + 64..3 * PAGE + 80]
                .iter()
                .all(|&b| b == 0xBB)
        );

        // Submission 2: speculative UAV over all 256 KiB with zero GPU writes.
        // Piece 1 was pushed back to VRAM at the end of submission 1 (due to
        // the concurrent 0x77 write), so pre-hash must re-hash piece 1 and
        // post-compare must report 0 changed pieces and 0 readback DMAs.
        shadow.refresh_as(
            &list,
            start,
            backing.len() as u64,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        shadow.gpu_writes(&list, start, backing.len() as u64);
        list.close();
        h.execute(&[&list]);
        list.reset();

        let stats2 = h.rebar_hash_stats();
        assert_eq!(stats2.compared_pieces - stats1.compared_pieces, 4);
        assert_eq!(stats2.changed_pieces - stats1.changed_pieces, 0);
        assert_eq!(
            stats2.post_readback_calls - stats1.post_readback_calls,
            0,
            "0 changed pieces must trigger 0 readback DMAs"
        );
        assert_eq!(stats2.post_readback_bytes - stats1.post_readback_bytes, 0);
        assert_eq!(backing[128], 0x55);
        assert_eq!(backing[PAGE + 128], 0x77);

        // Submission 3: GPU overwrites byte PAGE + 128 back to 0x11 (replacing
        // the concurrent CPU 0x77 write). Only piece 1 must be flagged & DMAed.
        shadow.refresh_as(
            &list,
            start,
            backing.len() as u64,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        shadow.gpu_writes(&list, start, backing.len() as u64);
        // SAFETY: live buffers and in-bounds 16-byte copy.
        unsafe {
            list.list
                .CopyBufferRegion(&shadow.buffer, (PAGE + 128) as u64, &src_11, 0, 16);
        }
        list.close();
        h.execute(&[&list]);
        list.reset();

        let stats3 = h.rebar_hash_stats();
        assert_eq!(stats3.compared_pieces - stats2.compared_pieces, 4);
        assert_eq!(stats3.changed_pieces - stats2.changed_pieces, 1);
        assert_eq!(
            stats3.post_readback_bytes - stats2.post_readback_bytes,
            PAGE as u64
        );
        assert!(backing[PAGE + 128..PAGE + 144].iter().all(|&b| b == 0x11));

        // Submission 4: CPU modifies piece 2 and refreshes it, then declares a
        // speculative UAV write with no GPU writes. Pre-hash must refresh piece
        // 2's hash so post-compare sees 0 changes and 0 readback DMAs.
        backing[2 * PAGE + 16..2 * PAGE + 32].fill(0xCC);
        shadow.refresh_as(
            &list,
            start,
            backing.len() as u64,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        shadow.gpu_writes(&list, start, backing.len() as u64);
        list.close();
        h.execute(&[&list]);
        list.reset();

        let stats4 = h.rebar_hash_stats();
        assert_eq!(stats4.compared_pieces - stats3.compared_pieces, 4);
        assert_eq!(stats4.changed_pieces - stats3.changed_pieces, 0);
        assert_eq!(stats4.post_readback_calls - stats3.post_readback_calls, 0);
        assert_eq!(stats4.post_readback_bytes - stats3.post_readback_bytes, 0);
        assert!(
            backing[2 * PAGE + 16..2 * PAGE + 32]
                .iter()
                .all(|&b| b == 0xCC)
        );
        assert_eq!(
            h.d3d12_errors(),
            errors_before,
            "unexpected D3D12 debug layer error during ReBAR hash-compare test"
        );

        drop(shadow);
        unregister(resource, start);
    }
}
