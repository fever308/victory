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
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use windows::Win32::Graphics::Direct3D12::{
    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATES, ID3D12Heap, ID3D12Resource,
};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// 64 KiB piece unit for card-copy hashing, compare, and relay merges.
const PIECE: usize = PAGE;

/// Total VRAM bytes currently allocated to ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`)
/// registered shadow buffers across live [`Shadow`] instances.
static REBAR_SHADOW_BYTES: AtomicU64 = AtomicU64::new(0);

/// Total VRAM bytes currently allocated to ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`)
/// `LOOSE` window buffers across live [`Shadow`] instances.
static REBAR_LOOSE_BYTES: AtomicU64 = AtomicU64::new(0);

/// Monotonic identifier for each live [`Shadow`] host copy (used by indirect
/// window and per-buffer gather gating to avoid pointer-reuse ABA).
static NEXT_SHADOW_ID: AtomicU64 = AtomicU64::new(1);

/// Current ReBAR VRAM bytes reserved by live registered [`Shadow`]s.
pub(crate) fn rebar_shadow_bytes() -> u64 {
    REBAR_SHADOW_BYTES.load(Ordering::Relaxed)
}

/// Current ReBAR VRAM bytes reserved by live `LOOSE` window [`Shadow`]s.
pub(crate) fn rebar_loose_bytes() -> u64 {
    REBAR_LOOSE_BYTES.load(Ordering::Relaxed)
}

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
    is_loose: bool,
    mirror: Mutex<Option<Vec<u8>>>,
    /// Leaf lock: never acquired while holding `mirror`, `pending`, or `copied`,
    /// and never held across D3D12 or kernel calls.
    hashes: Mutex<Option<RebarHashes>>,
}

impl RebarShadow {
    fn has_mirror(&self) -> bool {
        self.mirror
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    fn mark_dirty_range(&self, off: usize, len: usize) {
        if len == 0 {
            return;
        }
        let mut guard = self.hashes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = guard.as_mut() {
            h.dirty.mark_byte_range(off, len);
        }
    }

    fn mark_dirty_segs(&self, start: usize, segs: &[CopySeg]) {
        if segs.is_empty() {
            return;
        }
        let mut guard = self.hashes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = guard.as_mut() {
            for seg in segs {
                if seg.len > 0 {
                    h.dirty.mark_byte_range(seg.src - start, seg.len);
                }
            }
        }
    }
}

/// Our own `MapViewOfFile` view backing a `VICTORY_STRETCH` in-place heap.
/// Unmapped on drop after `Shadow::buffer` and `Shadow::place` have released
/// the placed resource and `ID3D12Heap`.
struct StretchView {
    addr: usize,
    section: usize,
}

impl Drop for StretchView {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        // SAFETY: our own view mapped in `through_another_view`, no longer held
        // by any D3D12 resource or heap (`buffer` and `place` drop earlier).
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: self.addr as *mut _,
            });
        }
    }
}

/// Per-send sorted index of ranges uploaded to a card copy (`Card::sent`),
/// so [`Shadow::card_merge_relay`] queries overlapping ranges in
/// `O(in_flight_sends * log(ranges) + k)` instead of `O(pieces * sent.len())`
/// and [`card_dirty`] prunes landed sends in `O(1)`.
#[derive(Default)]
struct CardSent {
    /// Strictly ascending by `sent_as`; each inner `Vec` is non-empty, sorted,
    /// and disjoint (`a < b`, `ranges[i].1 < ranges[i + 1].0`).
    sends: Vec<(u64, Vec<(usize, usize)>)>,
    /// Total range entries across `sends`.
    total: usize,
}

impl CardSent {
    #[cfg(test)]
    fn from_triples(entries: Vec<(usize, usize, u64)>) -> Self {
        let mut by_send: std::collections::BTreeMap<u64, Vec<(usize, usize)>> =
            std::collections::BTreeMap::new();
        for (a, b, sent_as) in entries {
            if a < b {
                by_send.entry(sent_as).or_default().push((a, b));
            }
        }
        let mut out = Self::default();
        for (sent_as, ranges) in by_send {
            out.push_sorted(sent_as, joined(ranges));
        }
        out
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.total
    }

    fn has_after(&self, fetched: u64) -> bool {
        self.sends.last().is_some_and(|s| s.0 > fetched)
    }

    fn needs_prune(&self, fetched: u64) -> bool {
        self.sends.first().is_some_and(|s| s.0 <= fetched)
    }

    fn prune(&mut self, fetched: u64) {
        if !self.needs_prune(fetched) {
            return;
        }
        let keep = self.sends.partition_point(|s| s.0 <= fetched);
        let dropped: usize = self.sends[..keep].iter().map(|s| s.1.len()).sum();
        self.total -= dropped;
        self.sends.drain(..keep);
    }

    fn push_sorted(&mut self, sent_as: u64, ranges: Vec<(usize, usize)>) {
        if ranges.is_empty() {
            return;
        }
        match self.sends.last_mut() {
            Some(last) if last.0 < sent_as => {
                self.total += ranges.len();
                self.sends.push((sent_as, ranges));
            }
            Some(last) if last.0 == sent_as => {
                let prev = last.1.len();
                last.1.extend(ranges);
                last.1 = joined(std::mem::take(&mut last.1));
                self.total = self.total - prev + last.1.len();
            }
            None => {
                self.total += ranges.len();
                self.sends.push((sent_as, ranges));
            }
            Some(_) => match self.sends.binary_search_by_key(&sent_as, |s| s.0) {
                Ok(i) => {
                    let prev = self.sends[i].1.len();
                    self.sends[i].1.extend(ranges);
                    self.sends[i].1 = joined(std::mem::take(&mut self.sends[i].1));
                    self.total = self.total - prev + self.sends[i].1.len();
                }
                Err(i) => {
                    self.total += ranges.len();
                    self.sends.insert(i, (sent_as, ranges));
                }
            },
        }
    }

    fn collect_skip(
        &self,
        lo: usize,
        hi: usize,
        fetched_after: u64,
        out: &mut Vec<(usize, usize)>,
    ) {
        if lo >= hi || !self.has_after(fetched_after) {
            return;
        }
        let start = self.sends.partition_point(|s| s.0 <= fetched_after);
        for (_, ranges) in &self.sends[start..] {
            let first = ranges.partition_point(|&(_, b)| b <= lo);
            for &(a, b) in &ranges[first..] {
                if a >= hi {
                    break;
                }
                out.push((a.max(lo), b.min(hi)));
            }
        }
    }
}

/// What a copy in the card's memory ([`cards`]) has beside the buffer
/// shaders see.
struct Card {
    /// The CPU's side, in the PC's memory: what `Shadow::data` maps.
    staging: ID3D12Resource,
    /// Per-64-KiB-piece 128-bit hashes on the card: allocated lazily on first
    /// declared shader write (`Host::card_before`).
    before: Mutex<Option<ID3D12Resource>>,
    /// Offsets of the copy whose bytes changed on the CPU's side and have not
    /// gone to the card's side yet.
    dirty: RwLock<Vec<(usize, usize)>>,
    /// With [`relay`]: the offsets that went from `dirty` to the card's side,
    /// indexed by the number of the send that took them ([`card_dirty`]), kept
    /// until a fetch sent after them has been merged. Locked inside `dirty`'s lock.
    sent: RwLock<CardSent>,
    /// The buffers' bytes (rounded up to 64 KiB).
    size: u64,
    /// Whether this card copy was allocated from [`WINDOW_CARD_BYTES`].
    is_window: bool,
}

/// With [`relay`]: how many submissions not yet copied back declared each
/// range shaders may be writing `(range, count, last_submission_number)`.
#[derive(Default)]
struct Flights(Vec<((usize, usize), u32, u64)>);

impl Flights {
    /// Pending ranges that can already have GPU writes from before this
    /// submission. A range counted for both this submission and an older
    /// one is still protected, even when its last declaration is current.
    fn older_ranges(&self, preparing: u64, from: usize, to: usize) -> Vec<(usize, usize)> {
        joined(
            self.0
                .iter()
                .filter(|&&((a, b), count, last)| {
                    a < to && b > from && (last != preparing || count > 1)
                })
                .map(|&((a, b), _, _)| (a, b))
                .collect(),
        )
    }

    /// Submission `number` declares `range`: counted once for it, however
    /// often it declares it.
    fn declare(&mut self, range: (usize, usize), number: u64) {
        match self.0.iter_mut().find(|f| f.0 == range) {
            Some(f) => {
                if f.2 != number {
                    f.1 += 1;
                    f.2 = number;
                }
            }
            None => self.0.push((range, 1, number)),
        }
    }

    /// One submission that declared `ranges` (each once) is copied back:
    /// returns the ranges no other submission still holds.
    fn done(&mut self, ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let mut free = Vec::new();
        for range in ranges {
            match self.0.iter().position(|f| f.0 == *range) {
                Some(at) => {
                    self.0[at].1 = self.0[at].1.saturating_sub(1);
                    if self.0[at].1 == 0 {
                        self.0.swap_remove(at);
                        free.push(*range);
                    }
                }
                None => free.push(*range),
            }
        }
        free
    }
}

/// A title buffer's host copy.
pub(crate) struct Shadow {
    id: u64,
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
    /// With [`relay`]: how many submissions not yet copied back declared each
    /// range of `pending` ([`Flights`]). Locked inside `pending`'s lock.
    flights: Mutex<Flights>,
    /// The copy's bytes before the submission ran, from an offset on,
    /// covering the ranges its shaders write: what the GPU wrote differs.
    /// Also used as the CPU-side lock for card copies (`RwLock` so relay
    /// merges take read locks while `merge_under_writes` takes a write lock).
    before: RwLock<Before>,
    /// With [`cards`], for a big title buffer: `buffer` is in the card's own
    /// memory, and this holds its staging side, hashes, and dirty/sent ranges.
    card: Option<Card>,
    /// Whether this shadow is a loose window outside registered title buffers.
    window: bool,
    /// With `crate::inplace`: `buffer` is placed in a heap opened on title
    /// memory itself (or our `StretchView`); holds `(heap, bias)`.
    place: Option<(ID3D12Heap, u64)>,
    /// Present when `buffer` lives in ReBAR `D3D12_HEAP_TYPE_GPU_UPLOAD` VRAM.
    rebar: Option<RebarShadow>,
    /// Monotonic tick of the last loose-cache access (0 for registered buffers).
    last_used: AtomicU64,
    /// Our own `MapViewOfFile` view backing a `VICTORY_STRETCH` in-place heap;
    /// MUST remain the last field so `buffer` and `place` drop before unmapping.
    stretch_view: Option<StretchView>,
}

impl Drop for Shadow {
    fn drop(&mut self) {
        if crate::lazy_back::active() {
            crate::lazy_back::discard_shadow(self as *const Shadow as usize);
        }
        if let Some(card) = &self.card {
            if card.is_window {
                WINDOW_CARD_BYTES.fetch_sub(card.size, Ordering::Relaxed);
            } else {
                CARD_BYTES[0].fetch_sub(card.size, Ordering::Relaxed);
            }
            if card
                .before
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_some()
            {
                CARD_BYTES[1].fetch_sub(card_hash_bytes(card.size), Ordering::Relaxed);
            }
        }
        if self.place.is_some() {
            crate::inplace::gone(self.window, self.len);
        }
        if let Some(rebar) = &self.rebar {
            if rebar.is_loose {
                REBAR_LOOSE_BYTES.fetch_sub(rebar.bytes, Ordering::Relaxed);
            } else {
                REBAR_SHADOW_BYTES.fetch_sub(rebar.bytes, Ordering::Relaxed);
            }
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

/// How many snapshot buffers are kept (`VICTORY_SNAPSHOT_POOL`, default 32;
/// `=8` restores the previous cap).
fn snapshot_pool_cap() -> usize {
    static CAP: OnceLock<usize> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("VICTORY_SNAPSHOT_POOL")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(32)
            .max(1)
    })
}

/// Returns a snapshot buffer to the pool (up to [`snapshot_pool_cap`] kept, the largest).
fn give_back(v: Vec<u8>) {
    let cap = snapshot_pool_cap();
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    if pool.len() > cap {
        pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
        pool.truncate(cap);
    }
}

/// Whether the host copies of big title buffers live in the card's own VRAM
/// (`VICTORY_CARD_BUFFERS=0` disables; on by default).
pub(crate) fn cards() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_BUFFERS").is_none_or(|v| v != "0"))
}

/// Minimum title buffer size and maximum total VRAM budget for card copies,
/// scaling the default cap from 1024 MiB to 2048 MiB on >= 24 GiB GPUs
/// (`VICTORY_CARD_BUFFERS_MIN_MB`, `VICTORY_CARD_BUFFERS_MB`).
fn card_limits_for(
    min_mb: Option<u64>,
    max_mb: Option<u64>,
    local_vram_budget: Option<u64>,
) -> (u64, u64) {
    let default_max_mb = if local_vram_budget.is_some_and(|b| b >= 24 * GIB) {
        2048
    } else {
        1024
    };
    let min_bytes = min_mb.unwrap_or(1).max(1) << 20;
    let max_bytes = max_mb.unwrap_or(default_max_mb) << 20;
    (min_bytes, max_bytes)
}

fn card_limits() -> (u64, u64) {
    static LIMITS: OnceLock<(u64, u64)> = OnceLock::new();
    *LIMITS.get_or_init(|| {
        let parse = |name: &str| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
        };
        card_limits_for(
            parse("VICTORY_CARD_BUFFERS_MIN_MB"),
            parse("VICTORY_CARD_BUFFERS_MB"),
            host().and_then(|h| h.local_vram_budget()),
        )
    })
}

/// Bytes alive of copies' card sides, and of their "before" hash buffers.
static CARD_BYTES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// Title buffers turned down for want of VRAM room (count, bytes).
static CARD_REFUSED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// Live copies in the card's memory.
static CARDED: Mutex<Vec<std::sync::Weak<Shadow>>> = Mutex::new(Vec::new());
/// Since the last frame line: bytes copied to card sides and the ranges they
/// were; pieces compared on the card, pieces fetched and their bytes; the waits
/// for the card the fetches added and their microseconds.
pub(crate) static CARD_STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// Whether the texture queue leaves room on the card for card copies
/// (`VICTORY_CARD_ROOM_FOR_BUFFERS=0` disables; on by default).
fn card_room_kept_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_ROOM_FOR_BUFFERS").is_none_or(|v| v != "0"))
}

/// Bytes of VRAM reserved for card copies not yet allocated.
#[allow(dead_code)]
pub(crate) fn card_room_kept() -> u64 {
    if !card_room_kept_on() || !cards() {
        return 0;
    }
    card_limits().1.saturating_sub(
        CARD_BYTES[0].load(Ordering::Relaxed) + CARD_REFUSED[1].load(Ordering::Relaxed),
    )
}

/// [`CARD_STATS`], taken; then live card-copy count, their bytes, and their
/// hash buffers' bytes.
#[allow(dead_code)]
pub(crate) fn card_stats() -> [u64; 10] {
    let mut out = [0u64; 10];
    for (i, stat) in CARD_STATS.iter().enumerate() {
        out[i] = stat.swap(0, Ordering::Relaxed);
    }
    out[7] = CARDED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .filter(|w| w.strong_count() > 0)
        .count() as u64;
    out[8] = CARD_BYTES[0].load(Ordering::Relaxed) + WINDOW_CARD_BYTES.load(Ordering::Relaxed);
    out[9] = CARD_BYTES[1].load(Ordering::Relaxed);
    out
}

/// Bytes of the 128-bit piece hashes kept on the card for a copy of `size` bytes.
pub(crate) fn card_hash_bytes(size: u64) -> u64 {
    (size / PIECE as u64 + 2) * 16
}

fn card_for(size: u64) -> Option<ID3D12Resource> {
    if !cards() {
        return None;
    }
    let (least, most) = card_limits();
    if size < least || size >= 1 << 30 {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    let hundredths = if card_room_kept_on() { 97 } else { 80 };
    if let Some([budget, used, ..]) = host.video_memory()
        && used + size > budget / 100 * hundredths
    {
        CARD_REFUSED[0].fetch_add(1, Ordering::Relaxed);
        CARD_REFUSED[1].fetch_add(size, Ordering::Relaxed);
        return None;
    }
    if CARD_BYTES[0].fetch_add(size, Ordering::Relaxed) + size > most {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

/// Maximum VRAM bytes of loose windows placed on the card
/// (`VICTORY_CARD_WINDOWS_MB`, default 0 = off).
fn window_card_most() -> u64 {
    static MOST: OnceLock<u64> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_CARD_WINDOWS_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(0)
            << 20
    })
}

static WINDOW_CARD_BYTES: AtomicU64 = AtomicU64::new(0);

fn window_card_for(size: u64) -> Option<ID3D12Resource> {
    let most = window_card_most();
    if most == 0 || !cards() {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    if let Some([budget, used, ..]) = host.video_memory()
        && used + size > budget / 10 * 8
    {
        return None;
    }
    if WINDOW_CARD_BYTES.fetch_add(size, Ordering::Relaxed) + size > most {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

/// Drains dirty CPU-side ranges from all live card-backed shadows and records
/// them in `Card::sent` under `sent_as` when [`relay`] is active.
pub(crate) type CardDirty = (Arc<Shadow>, Vec<(usize, usize)>);

pub(crate) fn card_dirty(sent_as: u64) -> Vec<CardDirty> {
    let mut carded = CARDED.lock().unwrap_or_else(|e| e.into_inner());
    let fetched = RELAY_FETCHED.load(Ordering::Acquire);
    let use_relay = relay();
    card_dirty_with(&mut carded, sent_as, fetched, use_relay)
}

/// Core implementation of [`card_dirty`] parameterized over `carded`, `fetched`,
/// and `use_relay` so tests can exercise the production draining and pruning
/// paths in isolation without mutating process-wide `CARDED` or `RELAY_FETCHED`.
pub(crate) fn card_dirty_with(
    carded: &mut Vec<std::sync::Weak<Shadow>>,
    sent_as: u64,
    fetched: u64,
    use_relay: bool,
) -> Vec<CardDirty> {
    let mut out = Vec::new();
    carded.retain(|weak| {
        let Some(shadow) = weak.upgrade() else {
            return false;
        };
        let ranges = shadow
            .card
            .as_ref()
            .map(|c| {
                // Fast path for untouched shadows: avoid taking a write lock
                // while parallel `card_merge_relay` workers hold read locks.
                // Bind booleans so read guards drop strictly before any write lock.
                let dirty_empty = c.dirty.read().unwrap_or_else(|e| e.into_inner()).is_empty();
                if dirty_empty {
                    let needs_prune = use_relay
                        && c.sent
                            .read()
                            .unwrap_or_else(|e| e.into_inner())
                            .needs_prune(fetched);
                    if needs_prune {
                        c.sent
                            .write()
                            .unwrap_or_else(|e| e.into_inner())
                            .prune(fetched);
                    }
                    return Vec::new();
                }
                let mut dirty = c.dirty.write().unwrap_or_else(|e| e.into_inner());
                let taken = std::mem::take(&mut *dirty);
                let ranges = if taken.is_empty() {
                    Vec::new()
                } else {
                    joined(taken)
                };
                if use_relay {
                    let mut sent = c.sent.write().unwrap_or_else(|e| e.into_inner());
                    sent.prune(fetched);
                    if !ranges.is_empty() {
                        sent.push_sorted(sent_as, ranges.clone());
                    }
                }
                ranges
            })
            .unwrap_or_default();
        if !ranges.is_empty() {
            out.push((shadow, ranges));
        }
        true
    });
    out
}

/// Ranges in order, those that overlap or touch merged as one.
fn joined(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

/// Cuts `runs` `(from, to, known_before)` on previously copied `pages` to only
/// the 4 KiB sub-pages marked written in `parts` (16-bit mask per 64 KiB page).
fn by_parts(
    runs: Vec<(usize, usize, bool)>,
    pages: &[(usize, u64)],
    parts: &[u16],
) -> Vec<(usize, usize, bool)> {
    const PART: usize = PAGE / 16;
    if parts.iter().all(|&p| p == u16::MAX) {
        return runs;
    }
    let mut out: Vec<(usize, usize, bool)> = Vec::with_capacity(runs.len() + pages.len());
    let mut pages = pages.iter().zip(parts).peekable();
    for (from, to, known) in runs {
        if !known {
            out.push((from, to, known));
            continue;
        }
        while let Some((&(page, _), &parts)) = pages.next_if(|((page, _), _)| page * PAGE < to) {
            for k in (0..16).filter(|k| parts >> k & 1 != 0) {
                let (a, b) = (
                    (page * PAGE + k * PART).max(from),
                    (page * PAGE + (k + 1) * PART).min(to),
                );
                if a >= b {
                    continue;
                }
                match out.last_mut() {
                    Some((_, end, true)) if *end == a => *end = b,
                    _ => out.push((a, b, true)),
                }
            }
        }
    }
    out
}

/// Runs of pieces to re-hash (`Host::card_before`), coalescing overlapping or
/// adjacent piece runs.
pub(crate) fn joined_piece_runs(mut runs: Vec<CardSpan>) -> Vec<CardSpan> {
    runs.sort_unstable_by_key(|r| r.first);
    let mut out: Vec<CardSpan> = Vec::with_capacity(runs.len());
    for run in runs {
        match out.last_mut() {
            Some(last) if run.first <= last.first + last.pieces => {
                last.pieces = last.pieces.max(run.first + run.pieces - last.first);
                last.to = last.to.max(run.to);
            }
            _ => out.push(run),
        }
    }
    out
}

/// 64 KiB piece range of offsets `[from, to)` in a copy starting at `start`.
fn pieces_of(start: usize, from: usize, to: usize) -> std::ops::Range<usize> {
    let base = start / PIECE;
    (start + from) / PIECE - base..(start + to).div_ceil(PIECE) - base
}

/// Offsets in a copy of `len` bytes at `start` held by `piece`.
fn piece_at(start: usize, len: usize, piece: usize) -> (usize, usize) {
    let base = start / PIECE;
    let lo = ((base + piece) * PIECE).saturating_sub(start);
    let hi = (((base + piece + 1) * PIECE).saturating_sub(start)).min(len);
    (lo, hi)
}

/// A declared range of a card-backed copy as passed to `CARD_COMPARE` / `CARD_FETCH`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CardSpan {
    pub(crate) from: u32,
    pub(crate) to: u32,
    pub(crate) first: u32,
    pub(crate) pieces: u32,
    pub(crate) skew: u32,
}

fn card_span_of(start: usize, from: usize, to: usize) -> CardSpan {
    let pieces = pieces_of(start, from, to);
    CardSpan {
        from: from as u32,
        to: to as u32,
        first: pieces.start as u32,
        pieces: pieces.len() as u32,
        skew: (start % PIECE) as u32,
    }
}

fn card_piece_of(start: usize, len: usize, span: &CardSpan, i: u32) -> (usize, usize) {
    let (lo, hi) = piece_at(start, len, (span.first + i) as usize);
    (lo.max(span.from as usize), hi.min(span.to as usize))
}

/// Whether asynchronous relay copy-back is enabled (`VICTORY_RELAY=1` or `=2`,
/// default on in production; `VICTORY_RELAY=0` disables; off by default in unit
/// tests unless explicitly set so synchronous tests observe title memory
/// immediately after submission; also disabled when `VICTORY_LATE_WAIT=0` or
/// `VICTORY_GPU_TIMING` is set).
///
/// Delegates to [`crate::host::relay`] so `shadow` and `host_relay` always
/// agree on whether relay mode is active.
pub(crate) fn relay() -> bool {
    crate::host::relay()
}

/// Submission number currently being prepared (under `Host::submitted`).
static PREPARED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn preparing(number: u64) {
    PREPARED.store(number, Ordering::Relaxed);
}

/// Watermark of the oldest `QUEUE_ORDER` send whose ranges may still be needed
/// by an unmerged relay fetch.
static RELAY_FETCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn relay_fetched(number: u64) {
    RELAY_FETCHED.fetch_max(number, Ordering::AcqRel);
}

#[allow(dead_code)]
pub(crate) fn relay_fetched_value() -> u64 {
    RELAY_FETCHED.load(Ordering::Acquire)
}

/// Bytewise three-way merge of GPU writes (`now`) vs pre-submission reference
/// (`held`) into current title memory (`title`), updating `held` in place.
/// Returns `true` if any byte of `title` changed.
fn merge_piece(now: &[u8], held: &mut [u8], title: &mut [u8]) -> bool {
    let mut differs = 0u8;
    for ((n, h), t) in now.iter().zip(held.iter_mut()).zip(title.iter_mut()) {
        let wrote = if *n != *h { 0xffu8 } else { 0 };
        let merged = (*t & !wrote) | (*n & wrote);
        differs |= merged ^ *t;
        *t = merged;
        *h = (*h & !wrote) | (*n & wrote);
    }
    differs != 0
}

/// `[from, to)` minus the intervals in `skip`, in ascending order.
pub(crate) fn outside(
    from: usize,
    to: usize,
    mut skip: Vec<(usize, usize)>,
) -> Vec<(usize, usize)> {
    skip.sort_unstable();
    let mut out = Vec::new();
    let mut at = from;
    for (a, b) in skip {
        if b <= at {
            continue;
        }
        if a >= to {
            break;
        }
        if a > at {
            out.push((at, a));
        }
        at = at.max(b);
    }
    if at < to {
        out.push((at, to));
    }
    out
}

const DIFF_CHUNK: usize = 32;

/// Where `a` and `b` first differ within their common length (32-byte SIMD chunks).
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    let (offset, a, b) = match a_chunks.iter().zip(b_chunks).position(|(x, y)| x != y) {
        Some(i) => (i * DIFF_CHUNK, &a_chunks[i][..], &b_chunks[i][..]),
        None => (len - a_tail.len(), a_tail, b_tail),
    };
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .map(|i| offset + i)
}

/// Where `a` and `b` last differ within their common length (32-byte SIMD chunks).
fn last_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    if let Some(i) = a_tail.iter().zip(b_tail).rposition(|(x, y)| x != y) {
        return Some(len - a_tail.len() + i);
    }
    let chunk = a_chunks.iter().zip(b_chunks).rposition(|(x, y)| x != y)?;
    a_chunks[chunk]
        .iter()
        .zip(&b_chunks[chunk])
        .rposition(|(x, y)| x != y)
        .map(|i| chunk * DIFF_CHUNK + i)
}

/// Statistics for [`Shadow::merge_ranges_under_writes`]:
/// `[0]` pages compared, `[1]` changed runs put in, `[2]` bytes put in, and
/// fallback counts `[3]` unwatched, `[4]` no_watch, `[5]` runs_cap, `[6]` no_ref.
static MERGED: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

pub(crate) fn merged_stats() -> [u64; 7] {
    [0, 1, 2, 3, 4, 5, 6].map(|i| MERGED[i].swap(0, Ordering::Relaxed))
}

/// Runs `(offset, len)` where `now` differs from `was` (8-byte fast skip).
fn changed_runs(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    let n = now.len().min(was.len());
    let mut runs = Vec::new();
    let mut i = 0;
    while i < n {
        while i + 8 <= n && now[i..i + 8] == was[i..i + 8] {
            i += 8;
        }
        while i < n && now[i] == was[i] {
            i += 1;
        }
        if i >= n {
            break;
        }
        let first = i;
        while i < n && now[i] != was[i] {
            i += 1;
        }
        runs.push((first, i - first));
    }
    runs
}

/// Splits a card-backed page slice `[lo, lo + title.len())` against sorted,
/// disjoint `protected` intervals where an older in-flight shader may have
/// written on the card:
/// - Sub-slices outside `protected` have no older writer and upload as one
///   contiguous range if any byte differs from `held`.
/// - Sub-slices inside `protected` upload only exact changed byte runs (joined
///   by `gap`) so older GPU writes on equal bytes stay untouched.
///
/// Returns the number of exact byte-diff runs emitted inside `protected`.
fn card_busy_page_runs(
    lo: usize,
    title: &[u8],
    held: &[u8],
    protected: &[(usize, usize)],
    gap: usize,
    out: &mut Vec<(usize, usize)>,
) -> usize {
    let hi = lo + title.len();
    let mut at = lo;
    let mut exact_runs = 0usize;
    let push_run = |out: &mut Vec<(usize, usize)>, r: usize, m: usize| {
        if m == 0 {
            return;
        }
        match out.last_mut() {
            Some(last) if r <= last.0 + last.1 => {
                last.1 = (last.0 + last.1).max(r + m) - last.0;
            }
            _ => out.push((r, m)),
        }
    };
    for &(pa, pb) in protected {
        if pb <= at {
            continue;
        }
        if pa >= hi {
            break;
        }
        if pa > at {
            let (s0, s1) = (at - lo, pa - lo);
            if title[s0..s1] != held[s0..s1] {
                push_run(out, s0, s1 - s0);
            }
            at = pa;
        }
        let p_end = pb.min(hi);
        if p_end > at {
            let (s0, s1) = (at - lo, p_end - lo);
            let runs = join_runs(changed_runs(&title[s0..s1], &held[s0..s1]), gap);
            exact_runs += runs.len();
            for (r, m) in runs {
                push_run(out, s0 + r, m);
            }
            at = p_end;
        }
    }
    if at < hi {
        let (s0, s1) = (at - lo, hi - lo);
        if title[s0..s1] != held[s0..s1] {
            push_run(out, s0, s1 - s0);
        }
    }
    exact_runs
}

/// Maximum exact byte-diff fragments [`Shadow::update_busy`] emits under an
/// older in-flight shader write before settling overlapping flight(s) so
/// remaining pages upload as single contiguous ranges (`VICTORY_LATE_MERGE_RUNS`,
/// default 512).
fn busy_exact_run_budget() -> usize {
    static RUNS: OnceLock<usize> = OnceLock::new();
    *RUNS.get_or_init(|| {
        std::env::var("VICTORY_LATE_MERGE_RUNS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(512)
    })
}

/// An older flight may have written bytes between title changes. Only
/// send exact changes there; otherwise a changed page slice goes up as one
/// range so adjacent slices can join into a single upload.
#[cfg(test)]
fn card_busy_runs(title: &[u8], held: &[u8], exact: bool) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let protected = if exact { &[(0, title.len())][..] } else { &[] };
    card_busy_page_runs(0, title, held, protected, 0, &mut out);
    out
}

/// Maximum byte gap bridged when joining changed runs in a card copy under
/// pending shader writes (`VICTORY_LATE_MERGE_GAP`, default 0).
fn merge_gap() -> usize {
    static GAP: OnceLock<usize> = OnceLock::new();
    *GAP.get_or_init(|| {
        std::env::var("VICTORY_LATE_MERGE_GAP")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    })
}

fn join_runs(runs: Vec<(usize, usize)>, gap: usize) -> Vec<(usize, usize)> {
    if gap == 0 {
        return runs;
    }
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    for (at, n) in runs {
        match out.last_mut() {
            Some(last) if at <= last.0 + last.1 + gap => last.1 = at + n - last.0,
            _ => out.push((at, n)),
        }
    }
    out
}

enum LockRead<G> {
    Ok(G),
    /// Destination is unmapped/unreadable with no overlapping deferred lazy block.
    Unmapped,
    /// Read failed because an overlapping deferred lazy block (owned by this
    /// shadow or another) failed to materialize or remained armed.
    LazyFailed,
}

#[cfg(test)]
thread_local! {
    static BEFORE_LOCK_AND_READ_RAW_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
    static AFTER_LOCK_AND_READ_FAIL_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

/// Materializes any deferred `lazy_back` blocks in `[addr, addr + buf.len())`
/// outside the per-`Shadow` lock, then acquires `acquire()` and reads via
/// `read_into_raw`. If a concurrent relay thread re-deferred the block (or
/// another thread materialized it between the failed read and the lazy table
/// lookup), drops the lock, materializes as needed, and retries boundedly so a
/// newly readable page is never misclassified as `LockRead::Unmapped`.
fn lock_and_read<G>(acquire: impl Fn() -> G, addr: usize, buf: &mut [u8]) -> LockRead<G> {
    const MAX_LAZY_RETRIES: usize = 8;
    let lazy_possible = crate::lazy_back::can_arm() || crate::lazy_back::active();
    for attempt in 0..=MAX_LAZY_RETRIES {
        if (attempt > 0 || crate::lazy_back::active())
            && !crate::lazy_back::materialize_range(addr, buf.len())
        {
            return LockRead::LazyFailed;
        }
        let gen_before = if lazy_possible {
            crate::lazy_back::remove_generation()
        } else {
            0
        };
        #[cfg(test)]
        if attempt == 0
            && let Some(hook) = BEFORE_LOCK_AND_READ_RAW_HOOK.with(|h| h.borrow_mut().take())
        {
            hook();
        }
        let guard = acquire();
        if crate::host::read_into_raw(addr as *const u8, buf) {
            return LockRead::Ok(guard);
        }
        drop(guard);
        #[cfg(test)]
        if attempt == 0
            && let Some(hook) = AFTER_LOCK_AND_READ_FAIL_HOOK.with(|h| h.borrow_mut().take())
        {
            hook();
        }
        if !lazy_possible {
            return LockRead::Unmapped;
        }
        let has_block = crate::lazy_back::has_block_in_range(addr, buf.len());
        let gen_changed = crate::lazy_back::remove_generation() != gen_before;
        if !has_block && !gen_changed {
            // Confirm once under `acquire()` in case a concurrent arm rolled
            // back between the failed read and `has_block_in_range`.
            let guard = acquire();
            if crate::host::read_into_raw(addr as *const u8, buf) {
                return LockRead::Ok(guard);
            }
            drop(guard);
            if !crate::lazy_back::has_block_in_range(addr, buf.len())
                && crate::lazy_back::remove_generation() == gen_before
            {
                return LockRead::Unmapped;
            }
        }
        if attempt == MAX_LAZY_RETRIES {
            return LockRead::LazyFailed;
        }
    }
    LockRead::LazyFailed
}

/// Scans `work` (`(shadow, from, to)` title-address spans of PC-backed shadows)
/// against their pre-submission snapshots, returning differing `(offset, len)`
/// pieces per entry.
pub(crate) fn find_all(work: &[(Arc<Shadow>, usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    let scan_one = |(shadow, from, to): &(Arc<Shadow>, usize, usize)| -> Vec<(usize, usize)> {
        let (lo, hi) = shadow.inside(*from, to.saturating_sub(*from));
        if lo >= hi {
            return Vec::new();
        }
        let before = shadow.before.read().unwrap_or_else(|e| e.into_inner());
        let Some((taken, snap)) = before.snapshot.as_ref() else {
            return Vec::new();
        };
        if *taken > lo || hi > *taken + snap.len() {
            return Vec::new();
        }
        let start = shadow.start as usize;
        let mut found = Vec::new();
        let mut at = lo;
        while at < hi {
            let n = (PIECE - (start + at) % PIECE).min(hi - at);
            // SAFETY: inside `shadow.data` mapping (`at + n <= shadow.len`).
            let host = unsafe { std::slice::from_raw_parts(shadow.data.add(at), n) };
            let old = &snap[at - *taken..at - *taken + n];
            if host != old {
                found.push((at, n));
            }
            at += n;
        }
        found
    };
    if work.len() < 4 || in_wide() {
        return work.iter().map(scan_one).collect();
    }
    type FoundSlots = Arc<Vec<Mutex<Vec<(usize, usize)>>>>;
    let slots: FoundSlots = Arc::new((0..work.len()).map(|_| Mutex::new(Vec::new())).collect());
    let jobs: Vec<Job> = work
        .iter()
        .enumerate()
        .map(|(i, (shadow, from, to))| {
            let shadow = shadow.clone();
            let (from, to) = (*from, *to);
            let slots = slots.clone();
            Box::new(move || {
                let res = scan_one(&(shadow, from, to));
                *slots[i].lock().unwrap_or_else(|e| e.into_inner()) = res;
            }) as Job
        })
        .collect();
    wide(jobs);
    slots
        .iter()
        .map(|s| std::mem::take(&mut *s.lock().unwrap_or_else(|e| e.into_inner())))
        .collect()
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// Unique identifier for this host copy, stable for its lifetime.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

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
        let bias = self.place.as_ref().map_or(0, |p| p.1);
        address
            .checked_sub(self.start)
            .filter(|&at| at < self.len)
            .map(|at| at + bias)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        let bias = self.place.as_ref().map_or(0, |p| p.1);
        address - self.start + bias
    }

    /// Start address of this shadow in title memory.
    pub(crate) fn start(&self) -> u64 {
        self.start
    }

    /// Marks 64 KiB hash dirty bits overlapping `[off, off + len)` for a ReBAR shadow.
    pub(crate) fn mark_rebar_dirty_range(&self, off: usize, len: usize) {
        if let Some(rebar) = &self.rebar {
            rebar.mark_dirty_range(off, len);
        }
    }

    /// Whether this shadow has any in-flight GPU writes on `Host::queue`
    /// (declared shader writes in `pending` or unlanded card-copy uploads in `card.sent`).
    pub(crate) fn has_inflight_gpu_write(&self) -> bool {
        if !self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
        {
            return true;
        }
        if let Some(card) = &self.card {
            let fetched = RELAY_FETCHED.load(Ordering::Acquire);
            if card
                .sent
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .has_after(fetched)
            {
                return true;
            }
        }
        false
    }

    /// Whether this shadow is known not to be write-watched (or in-place).
    pub(crate) fn unwatched(&self) -> bool {
        self.place.is_some() || self.watched.load(Ordering::Relaxed) == NOT_WATCHED
    }

    /// Whether this copy's GPU buffer lives in VRAM (`Card`).
    pub(crate) fn carded(&self) -> bool {
        self.card.is_some()
    }

    /// Staging buffer, optional hash buffer, and rounded buffer size for a card copy.
    pub(crate) fn card_sides(&self) -> Option<(&ID3D12Resource, Option<ID3D12Resource>, u64)> {
        let card = self.card.as_ref()?;
        let before = card
            .before
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        Some((&card.staging, before, card.size))
    }

    /// Lazily allocates the per-piece hash buffer for a card copy, returning
    /// `true` if newly created (caller must hash the whole buffer once).
    pub(crate) fn card_make_before(
        &self,
        make: impl FnOnce(u64) -> Option<ID3D12Resource>,
    ) -> bool {
        let Some(card) = &self.card else {
            return false;
        };
        let mut before = card.before.lock().unwrap_or_else(|e| e.into_inner());
        if before.is_some() {
            return false;
        }
        let Some(buf) = make(card.size) else {
            return false;
        };
        CARD_BYTES[1].fetch_add(card_hash_bytes(card.size), Ordering::Relaxed);
        *before = Some(buf);
        true
    }

    /// Records that CPU-side offsets `[from, to)` of a card copy changed.
    fn card_changed(&self, from: usize, to: usize) {
        self.card_changed_runs(std::iter::once((from, to)));
    }

    /// Batched [`Shadow::card_changed`] for offset `runs`: acquires `card.dirty` once.
    fn card_changed_runs(&self, runs: impl Iterator<Item = (usize, usize)>) {
        let Some(card) = &self.card else {
            return;
        };
        let mut iter = runs.peekable();
        if iter.peek().is_none() {
            return;
        }
        let mut dirty = card.dirty.write().unwrap_or_else(|e| e.into_inner());
        for (from, to) in iter {
            if from >= to {
                continue;
            }
            match dirty.last_mut() {
                Some(last) if from <= last.1 && to >= last.0 => {
                    *last = (last.0.min(from), last.1.max(to));
                }
                _ => dirty.push((from, to)),
            }
        }
    }

    /// Batched [`Shadow::card_changed`] for `segs`: acquires `card.dirty` once.
    fn card_changed_segs(&self, start: usize, segs: &[CopySeg]) {
        self.card_changed_runs(segs.iter().filter(|s| s.len > 0).map(|s| {
            let from = s.src - start;
            (from, from + s.len)
        }));
    }

    /// [`CardSpan`] for title addresses `[from, to)` clamped to this copy.
    pub(crate) fn card_span(&self, from: usize, to: usize) -> Option<CardSpan> {
        let (lo, hi) = self.inside(from, to.saturating_sub(from));
        (lo < hi).then(|| card_span_of(self.start as usize, lo, hi))
    }

    /// [`CardSpan`] for buffer offsets `[from, to)` clamped to `self.len`.
    pub(crate) fn card_run(&self, from: usize, to: usize) -> CardSpan {
        let len = self.len as usize;
        let (lo, hi) = (from.min(len), to.min(len).max(from.min(len)));
        card_span_of(self.start as usize, lo, hi.max(lo + 1).min(len.max(1)))
    }

    /// Offsets of piece `i` of `span` within `span`.
    pub(crate) fn card_piece(&self, span: &CardSpan, i: u32) -> (usize, usize) {
        card_piece_of(self.start as usize, self.len as usize, span, i)
    }

    /// `len` bytes at a title address from the host copy: the bytes the copy
    /// holds, as of its last refresh (None outside the buffer). Tests only.
    #[cfg(test)]
    pub(crate) fn bytes(&self, address: u64, len: u64) -> Option<&[u8]> {
        if address < self.start || len == 0 || len > self.left(address) {
            return None;
        }
        let at = usize::try_from(address - self.start).ok()?;
        let len = usize::try_from(len).ok()?;
        // SAFETY: inside the mapping, which is as long as the buffer.
        Some(unsafe { std::slice::from_raw_parts(self.data.add(at), len) })
    }

    /// Clamps `[address, address + len)` to this copy's title-address span,
    /// returning `None` if the intersection is empty.
    #[inline]
    pub(crate) fn clamp_range(&self, address: u64, len: u64) -> Option<(u64, u64)> {
        let from = address.max(self.start);
        let clamped = range_inside(
            from,
            len.saturating_sub(from.saturating_sub(address)),
            self.start,
            self.len,
        );
        (clamped > 0).then_some((from, clamped))
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset, without recording a
    /// resource-state transition on `list` itself (used when `list`'s prelude
    /// reads the buffer and restores it to `COMMON`).
    pub(crate) fn refresh_for_prelude(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        list.keep(&self.buffer);
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

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset. Cut to the copy.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        self.refresh_as(list, address, len, D3D12_RESOURCE_STATE_GENERIC_READ);
    }

    /// Whether this buffer is placed directly in title memory (no host copy).
    #[inline]
    pub(crate) fn is_placed(&self) -> bool {
        self.place.is_some()
    }

    /// Refresh a range and prepare its host buffer for the specified GPU use.
    /// Clamped to the copy.
    pub(crate) fn refresh_as(
        self: &Arc<Self>,
        list: &HostList,
        address: u64,
        len: u64,
        access: D3D12_RESOURCE_STATES,
    ) {
        list.use_shadow(&self.buffer, access);
        if self.place.is_some() {
            return;
        }
        let Some((from, len)) = self.clamp_range(address, len) else {
            return;
        };
        list.defer_refresh(self, from as usize, len as usize);
    }

    fn placed(placed: crate::inplace::Placed, start: u64, len: u64, window: bool) -> Shadow {
        Shadow {
            id: NEXT_SHADOW_ID.fetch_add(1, Ordering::Relaxed),
            buffer: placed.buffer,
            data: std::ptr::null_mut(),
            gpu: placed.gpu,
            start,
            len,
            born_seq: crate::host::birth_seq(),
            copied: Mutex::new(Vec::new()),
            watched: AtomicU8::new(NOT_WATCHED),
            pending: Mutex::default(),
            flights: Mutex::default(),
            before: RwLock::default(),
            card: None,
            window,
            place: Some((placed.heap, placed.bias)),
            rebar: None,
            last_used: AtomicU64::new(0),
            stretch_view: None,
        }
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
            id: NEXT_SHADOW_ID.fetch_add(1, Ordering::Relaxed),
            buffer,
            data,
            gpu,
            start,
            len,
            born_seq: crate::host::birth_seq(),
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            flights: Mutex::default(),
            before: RwLock::default(),
            card: None,
            window: false,
            place: None,
            rebar: rebar_bytes.map(|bytes| RebarShadow {
                bytes,
                is_loose: false,
                mirror: Mutex::new(None),
                hashes: Mutex::new(None),
            }),
            last_used: AtomicU64::new(0),
            stretch_view: None,
        }
    }

    fn new_loose(
        buffer: ID3D12Resource,
        data: *mut u8,
        gpu: u64,
        start: u64,
        len: u64,
        rebar_bytes: Option<u64>,
    ) -> Shadow {
        let mut shadow = Self::new(buffer, data, gpu, start, len, rebar_bytes);
        shadow.window = true;
        if let Some(rebar) = &mut shadow.rebar {
            rebar.is_loose = true;
        }
        shadow
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

    fn touch(&self) {
        self.last_used.store(
            LOOSE_TICK.fetch_add(1, Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }

    #[cfg(all(test, windows))]
    pub(crate) fn read_test_bytes(&self, address: u64, len: usize) -> Vec<u8> {
        let (start, end) = self.inside(address as usize, len);
        // SAFETY: `self.data` is persistently mapped for `self.len` bytes and
        // `inside` clamps `start..end` to `0..self.len`.
        unsafe { std::slice::from_raw_parts(self.data.add(start), end - start) }.to_vec()
    }

    #[cfg(all(test, windows))]
    pub(crate) fn new_rebar_from_parts_for_test(
        buffer: ID3D12Resource,
        data: *mut u8,
        gpu: u64,
        start: u64,
        len: u64,
    ) -> Arc<Self> {
        // Balance `Shadow::drop`'s `REBAR_SHADOW_BYTES` decrement even on unwind.
        REBAR_SHADOW_BYTES.fetch_add(len, Ordering::Relaxed);
        Arc::new(Self::new(buffer, data, gpu, start, len, Some(len)))
    }

    #[cfg(all(test, windows))]
    pub(crate) fn has_pending(&self) -> bool {
        !self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    pub(crate) fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (
            from - start,
            address.saturating_add(len).clamp(from, end) - start,
        )
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
        if self.place.is_some() {
            return;
        }
        let end = self.start.saturating_add(self.len);
        let from = address.max(self.start);
        let to = address.saturating_add(len).min(end);
        let len = to.saturating_sub(from);
        if len == 0 {
            return;
        }
        list.defer_gpu_write(self, from as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, grown to cover every range the
    /// submission declares (the world declares overlapping views of its
    /// 174 MB heap dozens of times), in reused memory (fresh snapshots cost
    /// ~2.3 million page faults per world frame).
    ///
    /// Returns `false` if a ReBAR shadow's initial RAM mirror could not be read
    /// back; in that case `pending` stays recorded so refreshes cannot overwrite
    /// the range.
    pub(crate) fn writing(
        &self,
        address: usize,
        len: usize,
        rebar_init: impl FnOnce(&ID3D12Resource, u64) -> Option<Vec<u8>>,
    ) -> bool {
        if self.place.is_some() {
            return true;
        }
        let range = (address, address.saturating_add(len));
        {
            let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            if relay() {
                let n = PREPARED.load(Ordering::Relaxed);
                self.flights
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .declare(range, n);
            }
            if !pending.contains(&range) {
                pending.push(range);
            }
        }
        if self.card.is_some() {
            return true;
        }
        if let Some(rebar) = &self.rebar {
            let mut mirror = rebar.mirror.lock().unwrap_or_else(|e| e.into_inner());
            if mirror.is_none() {
                *mirror = rebar_init(&self.buffer, self.len);
            }
            return mirror.is_some();
        }
        let (from, to) = self.inside(address, len);
        // SAFETY: inside the mapping (write-back system memory; ReBAR and card
        // buffers returned early above).
        let host =
            |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.write().unwrap_or_else(|e| e.into_inner());
        let before = &mut before.snapshot;
        let (start, end) = match &*before {
            Some((at, bytes)) => ((*at).min(from), (at + bytes.len()).max(to)),
            None => (from, to),
        };
        if let Some((at, bytes)) = &*before
            && (*at, at + bytes.len()) == (start, end)
        {
            return true;
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
        true
    }

    /// Clears pending UAV-write ranges for a ReBAR shadow once GPU hash-compare
    /// proves no copy-back is needed (or declared spans are empty).
    pub(crate) fn clear_pending_rebar(&self) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// Notifies the watch journal and `self.copied` after an own-view write at
    /// `[at, at + len)` with alias pages `shown`.
    fn record_own_write(&self, at: usize, shown: &[usize]) -> bool {
        let page = at / PAGE;
        let others: Vec<usize> = shown
            .iter()
            .map(|&a| a / PAGE)
            .filter(|&p| p != page)
            .collect();
        let (before, now) = crate::watch::wrote_own(page, &others);
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(taken) = page.checked_sub(base).and_then(|i| copied.get_mut(i))
            && *taken != 0
            && *taken >= before
        {
            *taken = (*taken).max(now);
            return true;
        }
        false
    }

    /// With `crate::inplace::back_own` (`VICTORY_BACK_OWN_VIEW=1`): stores
    /// merged bytes into title memory through our own view without opening
    /// title guards, noting the page current for this copy.
    fn put_back_own(&self, at: usize, bytes: &[u8]) -> bool {
        static CHECK: OnceLock<bool> = OnceLock::new();
        let check = *CHECK
            .get_or_init(|| std::env::var_os("VICTORY_BACK_OWN_CHECK").is_some_and(|v| v == "1"));
        if !crate::inplace::back_own() {
            return false;
        }
        let Some(shown) = crate::inplace::write_own(at, bytes) else {
            return false;
        };
        if check {
            let mut back = vec![0u8; bytes.len()];
            if crate::host::read_into_raw(at as *const u8, &mut back) && back != bytes {
                tracing::warn!(
                    at = format_args!("{at:#x}"),
                    "VICTORY_BACK_OWN_CHECK mismatch"
                );
            }
        }
        let _ = self.record_own_write(at, &shown);
        true
    }

    /// Pre-lock materialization of deferred `lazy_back` blocks overlapping ReBAR
    /// GPU readback `spans` before taking `rebar.mirror.lock()`.
    /// Returns `false` if lazy materialization failed for any overlapping block.
    fn materialize_rebar_spans(&self, spans: &[(usize, usize)], gpu_slices: &[&[u8]]) -> bool {
        if !crate::lazy_back::active() {
            return true;
        }
        let start = self.start as usize;
        let limit = self.len as usize;
        let ranges: Vec<(usize, usize)> = spans
            .iter()
            .zip(gpu_slices)
            .filter(|&(&(from, to), &gpu_span)| {
                from < to && to <= limit && gpu_span.len() == to - from
            })
            .map(|(&(from, to), _)| (start + from, to - from))
            .collect();
        if ranges.is_empty() {
            return true;
        }
        crate::lazy_back::materialize_ranges(&ranges)
    }

    /// [`relay`] counterpart to [`Shadow::copy_back_rebar`]: merges GPU bytes
    /// DMAed by the relay thread (`gpu_slices`, for sorted disjoint buffer-offset
    /// `spans`) into title memory and the cached RAM mirror under `rebar.mirror`'s
    /// lock. Never writes back to WC VRAM and never clears `pending` (handled
    /// per-flight by [`Shadow::copy_back_done`]).
    ///
    /// Returns `None` without advancing `mirror` for any unreadable span if
    /// pre-pass lazy materialization fails or a title read fails (even after
    /// retrying a concurrently armed lazy block).
    pub(crate) fn relay_merge_rebar(
        &self,
        spans: &[(usize, usize)],
        gpu_slices: &[&[u8]],
    ) -> Option<usize> {
        const PARALLEL_THRESHOLD: usize = 4 << 20;
        const PER_THREAD: usize = 4 << 20;

        let Some(rebar) = &self.rebar else {
            return Some(0);
        };
        let start = self.start as usize;
        if !self.materialize_rebar_spans(spans, gpu_slices) {
            return None;
        }
        let mut mirror_guard = rebar.mirror.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mirror) = mirror_guard.as_mut() else {
            return Some(0);
        };
        let mirror_len = mirror.len();
        let total_bytes: usize = spans
            .iter()
            .zip(gpu_slices)
            .filter(|&(&(from, to), &gpu_span)| {
                from < to && to <= mirror_len && gpu_span.len() == to - from
            })
            .map(|(&(from, to), _)| to - from)
            .sum();
        if total_bytes == 0 {
            return Some(0);
        }

        let mut wrote_fallback = false;
        let mut written = 0;
        let mut title = Vec::new();
        let mut retry_subs: Vec<(usize, usize, usize, usize)> = Vec::new();
        if total_bytes < PARALLEL_THRESHOLD {
            for (slice_idx, (&(from, to), &gpu_span)) in spans.iter().zip(gpu_slices).enumerate() {
                if from >= to || to > mirror_len || gpu_span.len() != to - from {
                    continue;
                }
                let mut at = from;
                while at < to {
                    let n = (PAGE - (start + at) % PAGE).min(to - at);
                    let span_off = at - from;
                    let gpu = &gpu_span[span_off..span_off + n];
                    let old = &mut mirror[at..at + n];
                    if gpu != old
                        && let Some(first) = first_difference(gpu, old)
                    {
                        let last = last_difference(gpu, old).unwrap_or(first);
                        match self.apply_rebar_relay_subrange(
                            start,
                            at + first,
                            &gpu[first..=last],
                            &mut old[first..=last],
                            &mut title,
                            false,
                            &mut wrote_fallback,
                        ) {
                            Some(n) => written += n,
                            None => retry_subs.push((
                                at + first,
                                last + 1 - first,
                                slice_idx,
                                span_off + first,
                            )),
                        }
                    }
                    at += n;
                }
            }
        } else {
            let threads = total_bytes.div_ceil(PER_THREAD).clamp(2, 8);
            let target = total_bytes.div_ceil(threads);
            let mut buckets: Vec<Vec<(usize, usize, usize, usize)>> = Vec::with_capacity(threads);
            let mut cur_bucket: Vec<(usize, usize, usize, usize)> = Vec::new();
            let mut cur_bytes = 0usize;
            for (slice_idx, (&(from, to), &gpu_span)) in spans.iter().zip(gpu_slices).enumerate() {
                if from >= to || to > mirror_len || gpu_span.len() != to - from {
                    continue;
                }
                let mut at = from;
                while at < to {
                    if buckets.len() + 1 >= threads {
                        cur_bucket.push((slice_idx, from, at, to));
                        break;
                    }
                    let room = target.saturating_sub(cur_bytes).max(1);
                    let next = if to - at <= room {
                        to
                    } else {
                        let raw = at + room;
                        let rem = (start + raw) % PAGE;
                        if rem == 0 {
                            raw
                        } else {
                            raw.saturating_add(PAGE - rem).min(to)
                        }
                    };
                    cur_bucket.push((slice_idx, from, at, next));
                    cur_bytes += next - at;
                    at = next;
                    if cur_bytes >= target && buckets.len() + 1 < threads {
                        buckets.push(std::mem::take(&mut cur_bucket));
                        cur_bytes = 0;
                    }
                }
            }
            if !cur_bucket.is_empty() {
                buckets.push(cur_bucket);
            }

            let diffs = {
                let mirror_ro: &[u8] = mirror;
                let scan_bucket = |bucket: Vec<(usize, usize, usize, usize)>|
                 -> Vec<(usize, usize, usize, usize)> {
                    let mut found = Vec::new();
                    for (slice_idx, span_from, seg_from, seg_to) in bucket {
                        let gpu_span = gpu_slices[slice_idx];
                        let mut at = seg_from;
                        while at < seg_to {
                            let n = (PAGE - (start + at) % PAGE).min(seg_to - at);
                            let span_off = at - span_from;
                            let gpu = &gpu_span[span_off..span_off + n];
                            let old = &mirror_ro[at..at + n];
                            if gpu != old
                                && let Some(first) = first_difference(gpu, old)
                            {
                                let last = last_difference(gpu, old).unwrap_or(first);
                                found.push((
                                    at + first,
                                    last + 1 - first,
                                    slice_idx,
                                    span_off + first,
                                ));
                            }
                            at += n;
                        }
                    }
                    found
                };
                std::thread::scope(|s| {
                    let mut iter = buckets.into_iter();
                    let first = iter.next().unwrap_or_default();
                    let scan_bucket = &scan_bucket;
                    let handles: Vec<_> = iter
                        .map(|bucket| s.spawn(move || scan_bucket(bucket)))
                        .collect();
                    let mut out = scan_bucket(first);
                    for handle in handles {
                        match handle.join() {
                            Ok(part) => out.extend(part),
                            Err(panic) => std::panic::resume_unwind(panic),
                        }
                    }
                    out
                })
            };

            for (sub_at, sub_n, slice_idx, slice_off) in diffs {
                let gpu = &gpu_slices[slice_idx][slice_off..slice_off + sub_n];
                let old = &mut mirror[sub_at..sub_at + sub_n];
                match self.apply_rebar_relay_subrange(
                    start,
                    sub_at,
                    gpu,
                    old,
                    &mut title,
                    false,
                    &mut wrote_fallback,
                ) {
                    Some(n) => written += n,
                    None => retry_subs.push((sub_at, sub_n, slice_idx, slice_off)),
                }
            }
        }

        if !retry_subs.is_empty() {
            drop(mirror_guard);
            let retry_ranges: Vec<(usize, usize)> = retry_subs
                .iter()
                .map(|&(sub_at, sub_n, _, _)| (start + sub_at, sub_n))
                .collect();
            let materialized =
                !crate::lazy_back::active() || crate::lazy_back::materialize_ranges(&retry_ranges);
            if !materialized {
                if wrote_fallback {
                    crate::watch::ask_once_void();
                }
                return None;
            }
            let mut mirror_guard = rebar.mirror.lock().unwrap_or_else(|e| e.into_inner());
            let Some(mirror) = mirror_guard.as_mut() else {
                if wrote_fallback {
                    crate::watch::ask_once_void();
                }
                return None;
            };
            let mut any_failed = false;
            for (sub_at, sub_n, slice_idx, slice_off) in retry_subs {
                if sub_at + sub_n > mirror.len() {
                    any_failed = true;
                    continue;
                }
                let gpu = &gpu_slices[slice_idx][slice_off..slice_off + sub_n];
                let old = &mut mirror[sub_at..sub_at + sub_n];
                match self.apply_rebar_relay_subrange(
                    start,
                    sub_at,
                    gpu,
                    old,
                    &mut title,
                    false,
                    &mut wrote_fallback,
                ) {
                    Some(n) => written += n,
                    None => any_failed = true,
                }
            }
            if wrote_fallback {
                crate::watch::ask_once_void();
            }
            return if any_failed { None } else { Some(written) };
        }

        if wrote_fallback {
            crate::watch::ask_once_void();
        }
        Some(written)
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_rebar_relay_subrange(
        &self,
        start: usize,
        sub_at: usize,
        gpu: &[u8],
        old: &mut [u8],
        title: &mut Vec<u8>,
        via_own: bool,
        wrote_fallback: &mut bool,
    ) -> Option<usize> {
        let sub_n = gpu.len();
        title.clear();
        title.resize(sub_n, 0);
        if via_own {
            if !crate::inplace::read_own_blocking(start + sub_at, title) {
                return None;
            }
            let old_backup = old.to_vec();
            if merge_piece(gpu, old, title) {
                crate::host::shaders_write(start + sub_at, &title[..]);
                let Some(shown) = crate::inplace::write_own_blocking(start + sub_at, &title[..])
                else {
                    old.copy_from_slice(&old_backup);
                    return None;
                };
                let _ = self.record_own_write(start + sub_at, &shown);
                return Some(sub_n);
            }
            return Some(0);
        }
        if crate::host::read_into_raw((start + sub_at) as *const u8, title) {
            let mut merged_old = old.to_vec();
            if merge_piece(gpu, &mut merged_old, title) {
                crate::host::shaders_write(start + sub_at, &title[..]);
                if self.put_back_own(start + sub_at, &title[..]) {
                    old.copy_from_slice(&merged_old);
                    return Some(sub_n);
                }
                crate::watch::writing(start + sub_at, sub_n);
                let written = crate::host::write_mapped_raw_with(
                    (start + sub_at) as *mut u8,
                    title.as_ptr(),
                    sub_n,
                    |piece_off, piece_len| {
                        old[piece_off..piece_off + piece_len]
                            .copy_from_slice(&merged_old[piece_off..piece_off + piece_len]);
                    },
                );
                *wrote_fallback = true;
                return (written == sub_n).then_some(written);
            }
            old.copy_from_slice(&merged_old);
            return Some(0);
        }
        // Title read failed: leave `old` (the mirror) untouched and report
        // failure so the caller does not mark dirty or publish completion.
        None
    }

    /// Own-view hook for lazy materialization of a ReBAR page addressed through
    /// `own_base_va` (either the primary VA or a surviving section alias).
    pub(crate) fn relay_merge_rebar_via_own_at(
        &self,
        offset: usize,
        own_base_va: usize,
        gpu: &[u8],
    ) -> Option<usize> {
        let rebar = self.rebar.as_ref()?;
        let mut mirror_guard = rebar.mirror.lock().unwrap_or_else(|e| e.into_inner());
        let mirror = mirror_guard.as_mut()?;
        let end = offset.checked_add(gpu.len())?;
        if gpu.is_empty() || end > mirror.len() {
            return Some(0);
        }
        let mut title = Vec::new();
        let mut written = 0usize;
        let mut at = offset;
        let mut unused_fallback = false;
        while at < end {
            let va_at = own_base_va.wrapping_add(at - offset);
            let n = (PAGE - va_at % PAGE).min(end - at);
            let gpu_page = &gpu[at - offset..at - offset + n];
            let old = &mut mirror[at..at + n];
            if gpu_page != old
                && let Some(first) = first_difference(gpu_page, old)
            {
                let last = last_difference(gpu_page, old).unwrap_or(first);
                let sub_start = own_base_va.wrapping_sub(offset);
                written += self.apply_rebar_relay_subrange(
                    sub_start,
                    at + first,
                    &gpu_page[first..=last],
                    &mut old[first..=last],
                    &mut title,
                    true,
                    &mut unused_fallback,
                )?;
            }
            at += n;
        }
        Some(written)
    }

    /// Synchronous card-copy merge (`Host::card_back`).
    pub(crate) fn card_merge(&self, at: usize, gpu: &[u8], title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_none() || at + gpu.len() > self.len as usize {
            return 0;
        }
        if crate::lazy_back::active() {
            crate::lazy_back::materialize_range(start + at, gpu.len());
        }
        // SAFETY: inside the CPU-side staging mapping; only submitting thread writes it.
        let old = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        let Some(first) = first_difference(gpu, old) else {
            return 0;
        };
        let last = last_difference(gpu, old).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, old) = (&gpu[first..=last], &mut old[first..=last]);
        title.clear();
        title.resize(n, 0);
        let mut written = 0;
        if crate::host::read_into_raw((start + at) as *const u8, title) {
            let mut differs = 0u8;
            for ((t, g), o) in title.iter_mut().zip(gpu).zip(old.iter_mut()) {
                let wrote = if *g != *o { 0xffu8 } else { 0 };
                let merged = (*t & !wrote) | (*g & wrote);
                differs |= merged ^ *t;
                *t = merged;
                *o = merged;
            }
            if differs != 0 {
                crate::host::shaders_write(start + at, &title[..]);
                if !self.put_back_own(start + at, &title[..]) {
                    crate::watch::writing(start + at, n);
                    crate::host::write_mapped_raw((start + at) as *mut u8, title.as_ptr(), n);
                }
                written = n;
            }
        } else {
            old.copy_from_slice(gpu);
        }
        self.card_changed(at, at + n);
        written
    }

    /// Relay card-copy merge (`Host::relay_fetch_back` / `Host::relay_card_inflight`).
    pub(crate) fn card_merge_relay(
        &self,
        at: usize,
        gpu: &[u8],
        fetched_after: u64,
        title: &mut Vec<u8>,
    ) -> usize {
        self.card_merge_relay_inner(
            at,
            (self.start as usize).wrapping_add(at),
            gpu,
            fetched_after,
            title,
            false,
        )
        .unwrap_or(0)
    }

    /// Own-view hook for lazy materialization of a card page.
    #[cfg(test)]
    pub(crate) fn card_merge_relay_via_own(
        &self,
        at: usize,
        gpu: &[u8],
        fetched_after: u64,
        title: &mut Vec<u8>,
    ) -> Option<usize> {
        self.card_merge_relay_inner(
            at,
            (self.start as usize).wrapping_add(at),
            gpu,
            fetched_after,
            title,
            true,
        )
    }

    /// Own-view hook for lazy materialization of a card page addressed through
    /// `own_base_va` (either the primary VA or a surviving section alias).
    pub(crate) fn card_merge_relay_via_own_at(
        &self,
        at: usize,
        own_base_va: usize,
        gpu: &[u8],
        fetched_after: u64,
        title: &mut Vec<u8>,
    ) -> Option<usize> {
        self.card_merge_relay_inner(at, own_base_va, gpu, fetched_after, title, true)
    }

    fn card_merge_relay_inner(
        &self,
        at: usize,
        own_base_va: usize,
        gpu: &[u8],
        fetched_after: u64,
        title: &mut Vec<u8>,
        via_own: bool,
    ) -> Option<usize> {
        let start = self.start as usize;
        let Some(card) = &self.card else {
            return Some(0);
        };
        if at + gpu.len() > self.len as usize {
            return Some(0);
        }
        if !via_own && crate::lazy_back::active() {
            crate::lazy_back::materialize_range(start + at, gpu.len());
        }
        let _side = self.before.read().unwrap_or_else(|e| e.into_inner());
        // SAFETY: inside staging mapping; `merge_under_writes` holds write lock on `before`.
        let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        let Some(first) = first_difference(gpu, held) else {
            return Some(0);
        };
        let last = last_difference(gpu, held).unwrap_or(first);
        let orig_at = at;
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, held) = (&gpu[first..=last], &mut held[first..=last]);
        let skip: Vec<(usize, usize)> = {
            let dirty = card.dirty.read().unwrap_or_else(|e| e.into_inner());
            let sent = card.sent.read().unwrap_or_else(|e| e.into_inner());
            let mut skip = Vec::new();
            for &(a, b) in dirty.iter() {
                if a < at + n && b > at {
                    skip.push((a.max(at), b.min(at + n)));
                }
            }
            sent.collect_skip(at, at + n, fetched_after, &mut skip);
            skip
        };
        title.clear();
        title.resize(n, 0);
        if via_own {
            let own_at = own_base_va.wrapping_add(at - orig_at);
            if !crate::inplace::read_own_blocking(own_at, title) {
                return None;
            }
            let held_backup = held.to_vec();
            let mut changed = false;
            for (a, b) in outside(at, at + n, skip) {
                let (a, b) = (a - at, b - at);
                changed |= merge_piece(&gpu[a..b], &mut held[a..b], &mut title[a..b]);
            }
            if !changed {
                return Some(0);
            }
            crate::host::shaders_write(own_at, &title[..]);
            let Some(shown) = crate::inplace::write_own_blocking(own_at, &title[..]) else {
                held.copy_from_slice(&held_backup);
                return None;
            };
            let _ = self.record_own_write(own_at, &shown);
            return Some(n);
        }
        let mapped = crate::host::read_into_raw((start + at) as *const u8, title);
        let mut changed = false;
        for (a, b) in outside(at, at + n, skip) {
            let (a, b) = (a - at, b - at);
            changed |= merge_piece(&gpu[a..b], &mut held[a..b], &mut title[a..b]);
        }
        if !mapped || !changed {
            return Some(0);
        }
        crate::host::shaders_write(start + at, &title[..]);
        if self.put_back_own(start + at, &title[..]) {
            return Some(n);
        }
        crate::watch::writing(start + at, n);
        crate::host::write_mapped_raw((start + at) as *mut u8, title.as_ptr(), n);
        crate::watch::ask_once_void();
        Some(n)
    }

    /// Relay PC-backed shadow merge (`relay_hand_back`).
    pub(crate) fn relay_merge(
        &self,
        at: usize,
        n: usize,
        now: &mut Vec<u8>,
        title: &mut Vec<u8>,
    ) -> usize {
        let start = self.start as usize;
        if self.card.is_some() || self.rebar.is_some() || n == 0 || at + n > self.len as usize {
            return 0;
        }
        if crate::lazy_back::active() {
            crate::lazy_back::materialize_range(start + at, n);
        }
        let mut before = self.before.write().unwrap_or_else(|e| e.into_inner());
        let Some((taken, snap)) = before.snapshot.as_mut() else {
            return 0;
        };
        if *taken > at || at + n > *taken + snap.len() {
            return 0;
        }
        now.clear();
        now.reserve(n);
        // SAFETY: inside `self.data` mapping (`at + n <= self.len`).
        unsafe {
            std::ptr::copy_nonoverlapping(self.data.add(at), now.as_mut_ptr(), n);
            now.set_len(n);
        }
        let held = &mut snap[at - *taken..at - *taken + n];
        let Some(first) = first_difference(now, held) else {
            return 0;
        };
        let last = last_difference(now, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (now, held) = (&now[first..=last], &mut held[first..=last]);
        title.clear();
        title.resize(n, 0);
        if !crate::host::read_into_raw((start + at) as *const u8, title) {
            return 0;
        }
        if !merge_piece(now, held, title) {
            return 0;
        }
        crate::host::shaders_write(start + at, &title[..]);
        if self.put_back_own(start + at, &title[..]) {
            return n;
        }
        crate::watch::writing(start + at, n);
        crate::host::write_mapped_raw((start + at) as *mut u8, title.as_ptr(), n);
        n
    }

    /// Releases declared shader-write `ranges` for one landed submission,
    /// returning the PC snapshot to the pool once no submission remains pending.
    pub(crate) fn copy_back_done(&self, ranges: &[(usize, usize)]) {
        if self.place.is_some() {
            return;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        if relay() {
            let free = self
                .flights
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .done(ranges);
            pending.retain(|r| !free.contains(r));
        } else {
            pending.retain(|r| !ranges.contains(r));
        }
        if !pending.is_empty() || self.card.is_some() || self.rebar.is_some() {
            return;
        }
        let mut before = self.before.write().unwrap_or_else(|e| e.into_inner());
        if let Some((_, bytes)) = before.snapshot.take() {
            before.unchanged.clear();
            give_back(bytes);
        }
    }

    /// Whether shaders of an unlanded submission may be writing this copy.
    pub(crate) fn writes_pending(&self) -> bool {
        !self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }

    fn busy_pending(&self, address: usize, len: usize) -> Vec<(usize, usize)> {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), address.saturating_add(len).min(end));
        if from >= to {
            return Vec::new();
        }
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect()
    }

    /// Collects the subranges of `[address, address + len)` where [`Shadow::update`]
    /// would leave title bytes behind because an in-flight shader write overlaps a
    /// page the title wrote since the copy last took it.
    ///
    /// For submissions sent to the card without a wait (`crate::host::late`):
    /// callers settle only flights whose `flight.back` overlaps these ranges.
    pub(crate) fn stale_ranges_under_writes(
        &self,
        address: usize,
        len: usize,
        out: &mut Vec<(usize, usize)>,
    ) {
        let busy = self.busy_pending(address, len);
        if busy.is_empty() {
            return;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            out.extend(busy);
            return;
        }
        let base = self.start as usize / PAGE;
        let shadow_ptr = self as *const Shadow as usize;
        // Acquire `LAZY_STATE` strictly outside `self.copied` (lock order).
        let owned = if crate::lazy_back::active() {
            let min_a = busy.iter().map(|&(a, _)| a).min().unwrap_or(0);
            let max_e = busy.iter().map(|&(_, e)| e).max().unwrap_or(0);
            crate::lazy_back::owned_blocks_in(shadow_ptr, min_a / PAGE, max_e.div_ceil(PAGE))
        } else {
            Vec::new()
        };
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        for (a, e) in busy {
            match crate::watch::asked(a, e) {
                Some((_, written)) => {
                    for page in stale_pages_in(base, &copied, a, e, &written) {
                        if owned.binary_search(&page).is_ok() {
                            continue;
                        }
                        let (lo, hi) = ((page * PAGE).max(a), ((page + 1) * PAGE).min(e));
                        if lo < hi {
                            out.push((lo, hi));
                        }
                    }
                }
                None => out.push((a, e)),
            }
        }
    }

    /// Merges CPU writes into a copy under pending shader writes (`late_exact`):
    /// writes only changed bytes into ReBAR WC VRAM + mirror, card staging, or
    /// PC copy + snapshot.
    ///
    /// Any stale subranges that could not be merged in place (unwatched memory,
    /// missing journal/snapshot, or exhausted card `runs_left`) are appended to
    /// `out` so the caller can settle only the overlapping flights.
    pub(crate) fn merge_ranges_under_writes(
        &self,
        address: usize,
        len: usize,
        runs_left: &mut usize,
        out: &mut Vec<(usize, usize)>,
    ) {
        let busy = self.busy_pending(address, len);
        if busy.is_empty() {
            return;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            out.extend(busy);
            MERGED[3].fetch_add(1, Ordering::Relaxed);
            return;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let base = start / PAGE;
        let shadow_ptr = self as *const Shadow as usize;
        let mut title = vec![0u8; PAGE];
        let push_unmerged =
            |stale: &[usize], a: usize, e: usize, out: &mut Vec<(usize, usize)>| -> bool {
                let before_len = out.len();
                for &p in stale {
                    let (lo, hi) = ((p * PAGE).max(a), ((p + 1) * PAGE).min(e));
                    if lo < hi {
                        out.push((lo, hi));
                    }
                }
                out.len() > before_len
            };
        for (a, e) in busy {
            let Some((now, written)) = crate::watch::writes(a, e, true) else {
                out.push((a, e));
                MERGED[4].fetch_add(1, Ordering::Relaxed);
                continue;
            };
            let mut stale = {
                let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                stale_pages_in(base, &copied, a, e, &written)
            };
            if !stale.is_empty() && crate::lazy_back::active() {
                stale.retain(|&page| !crate::lazy_back::owns_block(shadow_ptr, page));
            }
            for (idx, page) in stale.iter().copied().enumerate() {
                let (lo, hi) = ((page * PAGE).max(a), ((page + 1) * PAGE).min(e));
                if lo >= hi {
                    continue;
                }
                let n = hi - lo;
                let title = &mut title[..n];
                let at = lo - start;
                if let Some(rebar) = &self.rebar {
                    enum RebarMergeStep {
                        Merged,
                        MirrorMissing,
                        LazyFailed,
                        Unmapped,
                    }
                    let mut dirty_runs = Vec::new();
                    // ReBAR writes directly to mapped WC VRAM + cached RAM mirror
                    // with zero D3D12 `CopyBufferRegion` overhead, so it must not
                    // be capped by or consume `runs_left` (reserved for `card`).
                    let step = match lock_and_read(
                        || rebar.mirror.lock().unwrap_or_else(|e| e.into_inner()),
                        lo,
                        title,
                    ) {
                        LockRead::Ok(mut mirror) => {
                            MERGED[0].fetch_add(1, Ordering::Relaxed);
                            if let Some(held_all) = mirror.as_mut()
                                && at + n <= held_all.len()
                            {
                                let held = &mut held_all[at..at + n];
                                for (r, m) in changed_runs(title, held) {
                                    // SAFETY: inside ReBAR WC VRAM mapping (`at + r + m <= self.len`).
                                    unsafe {
                                        std::ptr::copy_nonoverlapping(
                                            title.as_ptr().add(r),
                                            self.data.add(at + r),
                                            m,
                                        );
                                    }
                                    held[r..r + m].copy_from_slice(&title[r..r + m]);
                                    dirty_runs.push((at + r, m));
                                    MERGED[1].fetch_add(1, Ordering::Relaxed);
                                    MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                                }
                                RebarMergeStep::Merged
                            } else {
                                RebarMergeStep::MirrorMissing
                            }
                        }
                        LockRead::LazyFailed => RebarMergeStep::LazyFailed,
                        LockRead::Unmapped => RebarMergeStep::Unmapped,
                    };
                    match step {
                        RebarMergeStep::Merged => {}
                        RebarMergeStep::MirrorMissing => {
                            if push_unmerged(&stale[idx..], a, e, out) {
                                MERGED[6].fetch_add(1, Ordering::Relaxed);
                            }
                            break;
                        }
                        RebarMergeStep::LazyFailed => {
                            out.push((lo, hi));
                            continue;
                        }
                        RebarMergeStep::Unmapped => continue,
                    }
                    for (off, len) in dirty_runs {
                        rebar.mark_dirty_range(off, len);
                    }
                } else {
                    let mut before = match lock_and_read(
                        || self.before.write().unwrap_or_else(|e| e.into_inner()),
                        lo,
                        title,
                    ) {
                        LockRead::Ok(guard) => guard,
                        LockRead::LazyFailed => {
                            out.push((lo, hi));
                            continue;
                        }
                        LockRead::Unmapped => continue,
                    };
                    MERGED[0].fetch_add(1, Ordering::Relaxed);
                    let fail_reason = if self.card.is_some() {
                        // SAFETY: inside staging mapping, write lock on `self.before` held.
                        let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                        let runs = join_runs(changed_runs(title, held), merge_gap());
                        if runs.len() > *runs_left {
                            Some(5)
                        } else {
                            *runs_left -= runs.len();
                            for &(r, m) in &runs {
                                held[r..r + m].copy_from_slice(&title[r..r + m]);
                                MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                            }
                            MERGED[1].fetch_add(runs.len() as u64, Ordering::Relaxed);
                            self.card_changed_runs(
                                runs.into_iter().map(|(r, m)| (at + r, at + r + m)),
                            );
                            None
                        }
                    } else if let Some((taken, snap)) = before.snapshot.as_mut()
                        && *taken <= at
                        && at + n <= *taken + snap.len()
                    {
                        let held = &mut snap[at - *taken..at - *taken + n];
                        for (r, m) in changed_runs(title, held) {
                            // SAFETY: inside `self.data` mapping (`at + r + m <= self.len`).
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    title.as_ptr().add(r),
                                    self.data.add(at + r),
                                    m,
                                );
                            }
                            held[r..r + m].copy_from_slice(&title[r..r + m]);
                            MERGED[1].fetch_add(1, Ordering::Relaxed);
                            MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                        }
                        None
                    } else {
                        Some(6)
                    };
                    drop(before);
                    if let Some(reason) = fail_reason {
                        if push_unmerged(&stale[idx..], a, e, out) {
                            MERGED[reason].fetch_add(1, Ordering::Relaxed);
                        }
                        break;
                    }
                }
                if (page * PAGE).max(start) >= a && ((page + 1) * PAGE).min(end) <= e {
                    let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(slot) = copied.get_mut(page - base) {
                        *slot = now;
                    }
                }
            }
        }
    }

    /// Whether [`Shadow::merge_ranges_under_writes`] leaves any unmerged ranges
    /// that require settling flights.
    #[cfg(test)]
    pub(crate) fn merge_under_writes(
        &self,
        address: usize,
        len: usize,
        runs_left: &mut usize,
    ) -> bool {
        let mut unmerged = Vec::new();
        self.merge_ranges_under_writes(address, len, runs_left, &mut unmerged);
        !unmerged.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn mark_unwatched_for_test(&self) {
        self.watched.store(NOT_WATCHED, Ordering::Relaxed);
    }

    /// ReBAR counterpart to [`Shadow::copy_back`]: merges post-submission GPU
    /// bytes (`gpu_slices`, already DMAed into cached `READBACK` RAM) for
    /// sorted disjoint buffer-offset `spans` against the cached RAM mirror,
    /// without ever reading write-combining VRAM on the CPU.
    ///
    /// Returns `None` (without clearing `pending`) if the shadow is not
    /// ReBAR-backed, its pre-submission mirror is missing, or `spans` and
    /// `gpu_slices` do not match.
    pub(crate) fn copy_back_rebar(
        &self,
        spans: &[(usize, usize)],
        gpu_slices: &[&[u8]],
    ) -> Option<usize> {
        let rebar = self.rebar.as_ref()?;
        if spans.len() != gpu_slices.len() {
            return None;
        }
        if !self.materialize_rebar_spans(spans, gpu_slices) {
            return None;
        }
        let mut mirror_guard = rebar.mirror.lock().unwrap_or_else(|e| e.into_inner());
        let mirror = mirror_guard.as_mut()?;
        for (&(from, to), &gpu_span) in spans.iter().zip(gpu_slices) {
            if from > to || to > mirror.len() || gpu_span.len() != to - from {
                return None;
            }
        }
        let mut written = 0;
        let mut vram_dirty: Vec<(usize, usize)> = Vec::new();
        let start = self.start as usize;
        let mut title = None;
        for (&(from, to), &gpu_span) in spans.iter().zip(gpu_slices) {
            if from == to {
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
                    if crate::host::read_into_raw((start + at) as *const u8, title_buf) {
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
                            crate::host::shaders_write(start + at, title_buf);
                            if !self.put_back_own(start + at, title_buf) {
                                crate::watch::writing(start + at, n);
                                crate::host::write_mapped_raw(
                                    (start + at) as *mut u8,
                                    title_buf.as_ptr(),
                                    n,
                                );
                            }
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
                    } else if crate::lazy_back::has_block_in_range(start + at, n) {
                        return None;
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
        drop(mirror_guard);
        for (at, n) in vram_dirty {
            rebar.mark_dirty_range(at, n);
        }
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        Some(written)
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        if self.place.is_some() {
            return 0;
        }
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if from < to && crate::lazy_back::active() {
            crate::lazy_back::materialize_range(start + from, to - from);
        }
        let mut before = self.before.write().unwrap_or_else(|e| e.into_inner());
        let Before {
            snapshot,
            unchanged,
        } = &mut *before;
        let mut written = 0;
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
                    && crate::host::read_into_raw((start + at) as *const u8, title)
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
                        crate::host::shaders_write(start + at, title);
                        if !self.put_back_own(start + at, title) {
                            crate::watch::writing(start + at, n);
                            crate::host::write_mapped_raw(
                                (start + at) as *mut u8,
                                title.as_ptr(),
                                n,
                            );
                        }
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
        pending.retain(|&r| r != (address, address.saturating_add(len)));
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
    #[cfg(test)]
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        self.update_checked(address, len).0
    }

    /// Like [`Shadow::update`], also returning `false` if materializing or
    /// copying a dirty lazy-owned page failed so the submission can abort
    /// instead of reading stale bytes.
    pub(crate) fn update_checked(&self, address: usize, len: usize) -> (usize, bool) {
        let plan = self.plan_update(address, len);
        if plan.refresh_failed {
            return (0, false);
        }
        if plan.runs.is_empty() && plan.busy_ranges.is_empty() {
            self.commit_copied_epochs(&plan.epochs, &[]);
            return (0, true);
        }
        let (mut copied, mut ok) = self.copy_runs_with_epochs(&plan.runs, &plan.epochs);
        for &(from, to) in &plan.busy_ranges {
            let (busy_bytes, busy_ok) = self.update_busy(from, to);
            copied += busy_bytes;
            ok &= busy_ok;
        }
        (copied, ok)
    }

    /// Computes the disjoint dirty byte runs of `[address, address + len)`
    /// outside pending shader writes and advances `self.copied` for whole
    /// stale pages, releasing `self.copied`'s lock before any memory copy.
    fn plan_update(&self, address: usize, len: usize) -> UpdatePlan {
        if self.place.is_some() {
            return UpdatePlan::default();
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), address.saturating_add(len).min(end));
        if from >= to {
            return UpdatePlan::default();
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        let mut plan = UpdatePlan::default();
        if !busy.is_empty() {
            busy.sort_unstable();
            let mut at = from;
            for (a, e) in busy {
                let (a, e) = (a.max(from), e.min(to));
                if a > at {
                    self.plan_update_free(at, a, &mut plan);
                }
                if e > a.max(at) {
                    plan.busy_ranges.push((a.max(at), e));
                }
                at = at.max(e);
            }
            if at < to {
                self.plan_update_free(at, to, &mut plan);
            }
            return plan;
        }
        self.plan_update_free(from, to, &mut plan);
        plan
    }

    /// [`Shadow::update`] of `[from, to)` where an earlier list declared
    /// shader writes, whether it is in flight or still being prepared.
    /// Copies title changes into the reference (`rebar.mirror`, card CPU
    /// staging, or PC `before` snapshot) so copy-back does not mistake them
    /// for GPU writes.
    /// Where an earlier flight's shaders may have written, copy only the
    /// changed bytes. A card-backed page slice named only by the submission
    /// being prepared can be uploaded as a single range before its lists run.
    fn update_busy(&self, from: usize, to: usize) -> (usize, bool) {
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return (0, true);
        }
        let Some((now, written)) = crate::watch::writes(from, to, true) else {
            return (0, true);
        };
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let base = start / PAGE;
        // Acquire `LAZY_STATE` strictly BEFORE `self.copied` (lock order).
        let owned = if crate::lazy_back::active() {
            crate::lazy_back::owned_blocks_with_clean_in(
                self as *const Shadow as usize,
                from / PAGE,
                to.div_ceil(PAGE),
            )
        } else {
            Vec::new()
        };
        let (stale, dirty_lazy): (Vec<usize>, Vec<(usize, usize)>) = {
            let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            let raw_stale = stale_pages_in(base, &copied, from, to, &written);
            if owned.is_empty() {
                (raw_stale, Vec::new())
            } else {
                let mut filtered = Vec::with_capacity(raw_stale.len());
                let mut dirty_lazy = Vec::new();
                for page in raw_stale {
                    if let Ok(idx) = owned.binary_search_by_key(&page, |&(b, _)| b) {
                        let at = page
                            .checked_sub(base)
                            .and_then(|i| copied.get(i).copied())
                            .unwrap_or(0);
                        let clean_lazy = at != 0 && owned[idx].1.is_some_and(|pre| pre <= at);
                        if clean_lazy {
                            // Clean lazy-owned page: keep deferred with zero DMA.
                            continue;
                        }
                        dirty_lazy.push((page * PAGE, PAGE));
                    }
                    filtered.push(page);
                }
                (filtered, dirty_lazy)
            }
        };
        if stale.is_empty() {
            return (0, true);
        }
        if !dirty_lazy.is_empty() && !crate::lazy_back::materialize_ranges(&dirty_lazy) {
            return (0, false);
        }
        // Preparation is serialized, so previously submitted ranges can
        // only disappear after this snapshot (the relay landed them).
        // Without relay bookkeeping in late mode, assume every busy range
        // might still have an earlier writer.
        let preparing = PREPARED.load(Ordering::Relaxed);
        let mut protected = if !self.carded() || !crate::host::late() {
            Vec::new()
        } else if relay() {
            self.flights
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .older_ranges(preparing, from, to)
        } else {
            vec![(from, to)]
        };
        let mut prot_idx = 0usize;
        let mut exact_runs_left = busy_exact_run_budget();
        let mut settled_once = false;
        let mut page_runs: Vec<(usize, usize)> = Vec::new();
        let mut title = vec![0u8; PAGE];
        let mut copied_bytes = 0usize;
        let mut all_ok = true;
        for page in stale {
            if crate::lazy_back::active()
                && crate::lazy_back::owns_block(self as *const Shadow as usize, page)
                && (!crate::lazy_back::materialize_range(page * PAGE, PAGE)
                    || crate::lazy_back::owns_block(self as *const Shadow as usize, page))
            {
                all_ok = false;
                continue;
            }
            let (lo, hi) = ((page * PAGE).max(from), ((page + 1) * PAGE).min(to));
            if lo >= hi {
                continue;
            }
            let n = hi - lo;
            let title = &mut title[..n];
            let at = lo - start;
            let mut page_ok = false;
            if let Some(rebar) = &self.rebar {
                let mut dirty_runs = Vec::new();
                {
                    let mut mirror = match lock_and_read(
                        || rebar.mirror.lock().unwrap_or_else(|e| e.into_inner()),
                        lo,
                        title,
                    ) {
                        LockRead::Ok(guard) => guard,
                        LockRead::LazyFailed => {
                            all_ok = false;
                            continue;
                        }
                        LockRead::Unmapped => continue,
                    };
                    if let Some(held_all) = mirror.as_mut()
                        && at + n <= held_all.len()
                    {
                        let held = &mut held_all[at..at + n];
                        for (r, m) in changed_runs(title, held) {
                            // SAFETY: inside ReBAR WC VRAM mapping (`at + r + m <= self.len`).
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    title.as_ptr().add(r),
                                    self.data.add(at + r),
                                    m,
                                );
                            }
                            held[r..r + m].copy_from_slice(&title[r..r + m]);
                            dirty_runs.push((at + r, m));
                            copied_bytes += m;
                        }
                        page_ok = true;
                    }
                }
                for (off, len) in dirty_runs {
                    rebar.mark_dirty_range(off, len);
                }
            } else {
                let mut before = match lock_and_read(
                    || self.before.write().unwrap_or_else(|e| e.into_inner()),
                    lo,
                    title,
                ) {
                    LockRead::Ok(guard) => guard,
                    LockRead::LazyFailed => {
                        all_ok = false;
                        continue;
                    }
                    LockRead::Unmapped => continue,
                };
                if self.card.is_some() {
                    while prot_idx < protected.len() && protected[prot_idx].1 <= lo {
                        prot_idx += 1;
                    }
                    // SAFETY: inside staging mapping, write lock on `self.before` held.
                    let mut held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                    page_runs.clear();
                    let exact_count = card_busy_page_runs(
                        lo,
                        title,
                        held,
                        &protected[prot_idx..],
                        merge_gap(),
                        &mut page_runs,
                    );
                    if exact_count > exact_runs_left && !settled_once && relay() && !in_wide() {
                        // Too many byte-level fragments under an older in-flight
                        // shader write: release `before`, settle overlapping
                        // older flight(s), re-read title memory (now merged with
                        // landed GPU writes), and retry so remaining pages upload
                        // as contiguous ranges.
                        drop(before);
                        settled_once = true;
                        let settled =
                            crate::host::settle_card_busy(self as *const Shadow as usize, lo, to);
                        if settled {
                            protected = self
                                .flights
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .older_ranges(preparing, lo, to);
                            prot_idx = 0;
                        }
                        before = match lock_and_read(
                            || self.before.write().unwrap_or_else(|e| e.into_inner()),
                            lo,
                            title,
                        ) {
                            LockRead::Ok(guard) => guard,
                            LockRead::LazyFailed => {
                                all_ok = false;
                                continue;
                            }
                            LockRead::Unmapped => continue,
                        };
                        // SAFETY: inside staging mapping, write lock on `self.before` held.
                        held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                        page_runs.clear();
                        let _ = card_busy_page_runs(
                            lo,
                            title,
                            held,
                            &protected[prot_idx..],
                            merge_gap(),
                            &mut page_runs,
                        );
                    } else {
                        exact_runs_left = exact_runs_left.saturating_sub(exact_count);
                    }
                    for &(r, m) in &page_runs {
                        held[r..r + m].copy_from_slice(&title[r..r + m]);
                        copied_bytes += m;
                    }
                    self.card_changed_runs(page_runs.iter().map(|&(r, m)| (at + r, at + r + m)));
                    drop(before);
                    page_ok = true;
                } else if let Some((taken, snap)) = before.snapshot.as_mut()
                    && *taken <= at
                    && at + n <= *taken + snap.len()
                {
                    let held = &mut snap[at - *taken..at - *taken + n];
                    for (r, m) in changed_runs(title, held) {
                        // SAFETY: inside `self.data` mapping (`at + r + m <= self.len`).
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                title.as_ptr().add(r),
                                self.data.add(at + r),
                                m,
                            );
                        }
                        held[r..r + m].copy_from_slice(&title[r..r + m]);
                        copied_bytes += m;
                    }
                    page_ok = true;
                }
            }
            if page_ok && (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(slot) = copied.get_mut(page - base) {
                    *slot = now;
                }
            }
        }
        (copied_bytes, all_ok)
    }

    /// [`Shadow::plan_update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn plan_update_free(&self, from: usize, to: usize, plan: &mut UpdatePlan) {
        let run_base = plan.runs.len();
        let epoch_base = plan.epochs.len();
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let mut dirty_lazy = Vec::new();
                self.collect_stale(
                    from,
                    to,
                    now,
                    &written,
                    &mut plan.runs,
                    &mut plan.epochs,
                    Some(&mut dirty_lazy),
                );
                if !dirty_lazy.is_empty() {
                    let ranges: Vec<(usize, usize)> =
                        dirty_lazy.into_iter().map(|p| (p * PAGE, PAGE)).collect();
                    if !crate::lazy_back::materialize_ranges(&ranges) {
                        plan.runs.truncate(run_base);
                        plan.epochs.truncate(epoch_base);
                        plan.refresh_failed = true;
                        return;
                    }
                    plan.runs.truncate(run_base);
                    plan.epochs.truncate(epoch_base);
                    let mut still_dirty = Vec::new();
                    self.collect_stale(
                        from,
                        to,
                        now,
                        &written,
                        &mut plan.runs,
                        &mut plan.epochs,
                        Some(&mut still_dirty),
                    );
                    if !still_dirty.is_empty() {
                        let retry_ranges: Vec<(usize, usize)> =
                            still_dirty.into_iter().map(|p| (p * PAGE, PAGE)).collect();
                        if !crate::lazy_back::materialize_ranges(&retry_ranges) {
                            plan.runs.truncate(run_base);
                            plan.epochs.truncate(epoch_base);
                            plan.refresh_failed = true;
                            return;
                        }
                        plan.runs.truncate(run_base);
                        plan.epochs.truncate(epoch_base);
                        let mut final_dirty = Vec::new();
                        self.collect_stale(
                            from,
                            to,
                            now,
                            &written,
                            &mut plan.runs,
                            &mut plan.epochs,
                            Some(&mut final_dirty),
                        );
                        if !final_dirty.is_empty() {
                            plan.runs.truncate(run_base);
                            plan.epochs.truncate(epoch_base);
                            plan.refresh_failed = true;
                            return;
                        }
                    }
                }
                Self::exclude_owned_lazy_runs(
                    self as *const Shadow as usize,
                    &mut plan.runs,
                    run_base,
                    &mut plan.epochs,
                    epoch_base,
                );
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    let copied: usize = plan.runs[run_base..].iter().map(|&(a, b)| b - a).sum();
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return;
            }
            // Only mark the shadow permanently unwatched if it has never been
            // observed as watched. A sub-range that lands entirely in an
            // unmapped hole inside an already-watched view pool must not
            // disable write-watching for the rest of the pool.
            let _ =
                self.watched
                    .compare_exchange(0, NOT_WATCHED, Ordering::Relaxed, Ordering::Relaxed);
        }
        plan.runs.push((from, to));
        Self::exclude_owned_lazy_runs(
            self as *const Shadow as usize,
            &mut plan.runs,
            run_base,
            &mut plan.epochs,
            epoch_base,
        );
    }

    /// Removes any 64 KiB page intervals in `runs[from_idx..]` (and their pending
    /// `epochs[epoch_from..]`) currently deferred by `shadow_ptr` in `lazy_back`
    /// so a shadow neither overwrites its own authoritative VRAM with stale RAM
    /// nor acknowledges CPU writes that were never copied.
    fn exclude_owned_lazy_runs(
        shadow_ptr: usize,
        runs: &mut Vec<(usize, usize)>,
        from_idx: usize,
        epochs: &mut Vec<(usize, u64)>,
        epoch_from: usize,
    ) {
        if !crate::lazy_back::active() || from_idx >= runs.len() {
            return;
        }
        let min_lo = runs[from_idx..].iter().map(|&(a, _)| a).min().unwrap_or(0);
        let max_hi = runs[from_idx..].iter().map(|&(_, b)| b).max().unwrap_or(0);
        if min_lo >= max_hi {
            return;
        }
        let owned =
            crate::lazy_back::owned_blocks_in(shadow_ptr, min_lo / PAGE, max_hi.div_ceil(PAGE));
        if owned.is_empty() {
            return;
        }
        if epoch_from < epochs.len() {
            let mut idx = epoch_from;
            while idx < epochs.len() {
                if owned.binary_search(&epochs[idx].0).is_ok() {
                    epochs.swap_remove(idx);
                } else {
                    idx += 1;
                }
            }
        }
        let tail: Vec<(usize, usize)> = runs.drain(from_idx..).collect();
        for (mut a, b) in tail {
            for &blk in &owned {
                let p0 = blk * PAGE;
                let p1 = p0 + PAGE;
                if p1 <= a {
                    continue;
                }
                if p0 >= b {
                    break;
                }
                if a < p0 {
                    runs.push((a, p0));
                }
                a = a.max(p1);
                if a >= b {
                    break;
                }
            }
            if a < b {
                runs.push((a, b));
            }
        }
    }

    /// Collects contiguous stale byte ranges `(lo, hi)` of `[from, to)`
    /// written after their last copy (`written`: last writes by page, in page
    /// order; `now`: this asking) and records pending whole-page `copied` epochs
    /// in `epochs_out` so epochs are only committed once their segments succeed.
    #[allow(clippy::too_many_arguments)]
    fn collect_stale(
        &self,
        from: usize,
        to: usize,
        now: u64,
        written: &[(usize, u64)],
        out: &mut Vec<(usize, usize)>,
        epochs_out: &mut Vec<(usize, u64)>,
        mut dirty_lazy_out: Option<&mut Vec<usize>>,
    ) {
        let base = self.start as usize / PAGE;
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let last = (to - 1) / PAGE;
        // Acquire `LAZY_STATE` before `self.copied` (lock order).
        // Clean lazy-owned pages (`at != 0 && cpu_clean_after <= at`) stay
        // deferred with zero DMA; dirty lazy-owned pages (`at == 0`,
        // `cpu_clean_after == None`, or `cpu_clean_after > at`) are reported in
        // `dirty_lazy_out` so `plan_update_free` materializes and replans them
        // before the current list runs (and leaves their epoch pending if
        // materialization fails).
        let owned = if crate::lazy_back::active() {
            crate::lazy_back::owned_blocks_with_clean_in(
                self as *const Shadow as usize,
                from / PAGE,
                last + 1,
            )
        } else {
            Vec::new()
        };
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let mut owned_it = owned.iter().peekable();
        let mut runs: Vec<(usize, usize, bool)> = Vec::new();
        let mut run: Option<(usize, bool)> = None;
        let mut again_at: Vec<(usize, u64)> = Vec::new();
        for page in from / PAGE..=last {
            while writes.peek().is_some_and(|&&(p, _)| p < page) {
                writes.next();
            }
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            while owned_it.peek().is_some_and(|&&(p, _)| p < page) {
                owned_it.next();
            }
            let own_entry = match owned_it.peek() {
                Some(&&(p, clean_after)) if p == page => {
                    owned_it.next();
                    Some(clean_after)
                }
                _ => None,
            };
            let own_lazy = own_entry.is_some();
            let at = copied[page - base];
            let stale = at == 0 || write > at;
            if let Some(cpu_clean_after) = own_entry {
                let clean_lazy = at != 0 && cpu_clean_after.is_some_and(|pre| pre <= at);
                if !clean_lazy && let Some(dirty_lazy) = dirty_lazy_out.as_deref_mut() {
                    dirty_lazy.push(page);
                }
            }
            if !own_lazy && stale {
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    epochs_out.push((page, now));
                }
                let again = at != 0;
                if again {
                    again_at.push((page, at));
                }
                if let Some((first, known)) = run
                    && known != again
                {
                    let lo = (first * PAGE).max(from);
                    let hi = (page * PAGE).min(to);
                    if lo < hi {
                        runs.push((lo, hi, known));
                    }
                    run = None;
                }
                run.get_or_insert((page, again));
            } else if let Some((first, known)) = run.take() {
                let lo = (first * PAGE).max(from);
                let hi = (page * PAGE).min(to);
                if lo < hi {
                    runs.push((lo, hi, known));
                }
            }
        }
        if let Some((first, known)) = run {
            let lo = (first * PAGE).max(from);
            let hi = ((last + 1) * PAGE).min(to);
            if lo < hi {
                runs.push((lo, hi, known));
            }
        }
        drop(copied);
        if again_at.is_empty() {
            for (a, b, _) in runs {
                match out.last_mut() {
                    Some((_, end)) if *end == a => *end = b,
                    _ => out.push((a, b)),
                }
            }
            return;
        }
        let parts = crate::watch::parts_written(&again_at);
        for (a, b, _) in by_parts(runs, &again_at, &parts) {
            match out.last_mut() {
                Some((_, end)) if *end == a => *end = b,
                _ => out.push((a, b)),
            }
        }
    }

    /// Commits planned `copied` epochs for whole pages that did NOT overlap any
    /// failed [`CopySeg`] so failed refresh segments remain dirty and eligible
    /// for refresh on the next submission.
    fn commit_copied_epochs(&self, epochs: &[(usize, u64)], failed_segs: &[CopySeg]) {
        if epochs.is_empty() {
            return;
        }
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        for &(page, now) in epochs {
            let p_lo = page * PAGE;
            let p_hi = p_lo + PAGE;
            let failed = failed_segs
                .iter()
                .any(|s| s.len > 0 && s.src < p_hi && s.src.saturating_add(s.len) > p_lo);
            if !failed && let Some(slot) = page.checked_sub(base).and_then(|i| copied.get_mut(i)) {
                *slot = (*slot).max(now);
            }
        }
    }

    /// Converts disjoint title-address `runs` (clamped to the buffer) into
    /// [`CopySeg`] entries targeting `self.data`, returning the total bytes.
    fn push_copy_segs(&self, runs: &[(usize, usize)], out: &mut Vec<CopySeg>) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let stage_card = self.card.is_some() && crate::lazy_back::can_arm();
        let mut total = 0usize;
        for &(raw_from, raw_to) in runs {
            let (from, to) = (raw_from.max(start), raw_to.min(end));
            if from >= to {
                continue;
            }
            let off = from - start;
            let len = to - from;
            out.push(CopySeg {
                src: from,
                dst: (self.data as usize) + off,
                mirror: 0,
                len,
                stage_card,
            });
            total += len;
        }
        total
    }

    /// Copies disjoint title-memory `runs`, cut to the buffer, into the copy
    /// (in parallel across worker threads when [`wide_copy`] is enabled and
    /// total bytes reach [`WIDE_COPY_MIN`]).
    #[cfg(test)]
    fn copy_runs(&self, runs: &[(usize, usize)]) -> usize {
        self.copy_runs_with_epochs(runs, &[]).0
    }

    fn copy_runs_with_epochs(
        &self,
        runs: &[(usize, usize)],
        epochs: &[(usize, u64)],
    ) -> (usize, bool) {
        if runs.is_empty() {
            self.commit_copied_epochs(epochs, &[]);
            return (0, true);
        }
        let start = self.start as usize;
        let mut segs = Vec::with_capacity(runs.len());
        let total = self.push_copy_segs(runs, &mut segs);
        if total == 0 {
            self.commit_copied_epochs(epochs, &[]);
            return (0, true);
        }
        materialize_copy_segs(&segs);
        if let Some(rebar) = &self.rebar {
            let mut mirror = rebar.mirror.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(m) = mirror.as_mut() {
                let m_ptr = m.as_mut_ptr() as usize;
                let m_len = m.len();
                if segs.iter().all(|s| (s.src - start) + s.len <= m_len) {
                    // Copy once from guarded title memory into cached RAM, then
                    // push straight to write-combining VRAM so both see identical bytes.
                    for s in &mut segs {
                        s.mirror = m_ptr + (s.src - start);
                    }
                    let res = run_copy_segs_status(&segs);
                    let mut ok = res.ok;
                    let mut touched: Vec<CopySeg> =
                        res.touched.into_iter().map(|(_, s)| s).collect();
                    drop(mirror);
                    if ok.iter().any(|&v| !v) {
                        let failed_indices: Vec<usize> = ok
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &v)| (!v).then_some(i))
                            .collect();
                        let retry_src: Vec<CopySeg> =
                            failed_indices.iter().map(|&i| segs[i]).collect();
                        materialize_copy_segs(&retry_src);
                        let mut mirror = rebar.mirror.lock().unwrap_or_else(|e| e.into_inner());
                        if let Some(m) = mirror.as_mut() {
                            let m_ptr = m.as_mut_ptr() as usize;
                            let m_len = m.len();
                            if retry_src.iter().all(|s| (s.src - start) + s.len <= m_len) {
                                let retry_segs: Vec<CopySeg> = retry_src
                                    .into_iter()
                                    .map(|mut s| {
                                        s.mirror = m_ptr + (s.src - start);
                                        s
                                    })
                                    .collect();
                                let retry_res = run_copy_segs_status(&retry_segs);
                                touched.extend(retry_res.touched.into_iter().map(|(_, s)| s));
                                for (idx, passed) in failed_indices.into_iter().zip(retry_res.ok) {
                                    ok[idx] = passed;
                                }
                            }
                        }
                    }
                    let (_succeeded, failed, bytes) = partition_copy_segs(&segs, &ok);
                    let all_ok = failed.is_empty();
                    // Invalidate ReBAR hashes for every chunk that wrote VRAM,
                    // including successful chunks of a partially failed segment.
                    rebar.mark_dirty_segs(start, &touched);
                    self.commit_copied_epochs(epochs, &failed);
                    return (bytes, all_ok);
                }
            }
        }
        let res = run_copy_segs_status(&segs);
        let mut ok = res.ok;
        let mut touched: Vec<CopySeg> = res.touched.into_iter().map(|(_, s)| s).collect();
        let mut staged: Vec<(usize, Vec<u8>)> = res.staged;
        if ok.iter().any(|&v| !v) {
            let failed_indices: Vec<usize> = ok
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| (!v).then_some(i))
                .collect();
            let retry_segs: Vec<CopySeg> = failed_indices.iter().map(|&i| segs[i]).collect();
            materialize_copy_segs(&retry_segs);
            let retry_res = run_copy_segs_status(&retry_segs);
            touched.extend(retry_res.touched.into_iter().map(|(_, s)| s));
            staged.extend(
                retry_res
                    .staged
                    .into_iter()
                    .map(|(retry_i, buf)| (failed_indices[retry_i], buf)),
            );
            for (idx, passed) in failed_indices.into_iter().zip(retry_res.ok) {
                ok[idx] = passed;
            }
        }
        let (succeeded, failed, bytes) = partition_copy_segs(&segs, &ok);
        let all_ok = failed.is_empty();
        self.publish_and_finish_copy_segs(start, &segs, &staged, &succeeded, &touched);
        self.commit_copied_epochs(epochs, &failed);
        (bytes, all_ok)
    }

    /// Publishes staged card bytes into `card.staging` and records the matching
    /// `card.dirty` ranges atomically under `self.before.write()` (matching
    /// `card_merge_relay_inner`'s `self.before.read() -> card.dirty.read()` lock
    /// order) so no concurrent card relay merge can observe published staging
    /// bytes without their corresponding `card.dirty` metadata.
    fn publish_and_finish_copy_segs(
        &self,
        start: usize,
        segs: &[CopySeg],
        staged: &[(usize, Vec<u8>)],
        succeeded: &[CopySeg],
        touched_rebar: &[CopySeg],
    ) {
        if self.card.is_some() {
            let _before = self.before.write().unwrap_or_else(|e| e.into_inner());
            publish_staged_card_segs(segs, staged);
            #[cfg(test)]
            BETWEEN_PUBLISH_AND_DIRTY_HOOK.with(|h| {
                if let Some(cb) = h.borrow_mut().take() {
                    cb();
                }
            });
            self.finish_copy_segs(start, succeeded, touched_rebar);
        } else {
            publish_staged_card_segs(segs, staged);
            self.finish_copy_segs(start, succeeded, touched_rebar);
        }
    }

    /// Post-copy metadata invalidation for non-mirrored `segs`, acquiring each
    /// per-shadow lock at most once. `touched_rebar` includes any chunk that may
    /// have written VRAM even when another chunk of the same segment failed.
    fn finish_copy_segs(&self, start: usize, succeeded: &[CopySeg], touched_rebar: &[CopySeg]) {
        if let Some(rebar) = &self.rebar {
            rebar.mark_dirty_segs(start, touched_rebar);
        }
        if !succeeded.is_empty() {
            self.card_changed_segs(start, succeeded);
        }
    }

    #[cfg(test)]
    pub(crate) fn new_card_for_test(
        side_data: *mut u8,
        title_start: u64,
        len: u64,
        dirty: Vec<(usize, usize)>,
        sent: Vec<(usize, usize, u64)>,
    ) -> Option<Arc<Shadow>> {
        let h = host()?;
        let (buffer, _) = h.shared_buffer(64)?;
        let (staging, _) = h.shared_buffer(64)?;
        let mut shadow = Shadow::new(buffer, side_data, 0, title_start, len, None);
        shadow.card = Some(Card {
            staging,
            before: Mutex::new(None),
            dirty: RwLock::new(dirty),
            sent: RwLock::new(CardSent::from_triples(sent)),
            size: 0,
            is_window: false,
        });
        Some(Arc::new(shadow))
    }

    #[cfg(test)]
    pub(crate) fn new_card_sized_for_test(
        side_data: *mut u8,
        title_start: u64,
        len: u64,
    ) -> Option<Arc<Shadow>> {
        let h = host()?;
        let alloc = len.max(64);
        let (buffer, _) = h.shared_buffer(alloc)?;
        let (staging, _) = h.shared_buffer(alloc)?;
        let mut shadow = Shadow::new(buffer, side_data, 0, title_start, len, None);
        shadow.card = Some(Card {
            staging,
            before: Mutex::new(None),
            dirty: RwLock::default(),
            sent: RwLock::default(),
            size: 0,
            is_window: false,
        });
        Some(Arc::new(shadow))
    }

    #[cfg(test)]
    pub(crate) fn new_card_registered_for_test(
        title_start: u64,
        len: u64,
        fill: u8,
    ) -> Option<Arc<Shadow>> {
        let h = host()?;
        let alloc = len.max(64);
        let (buffer, buf_ptr) = h.shared_buffer(alloc)?;
        let (staging, staging_ptr) = h.shared_buffer(alloc)?;
        // SAFETY: freshly mapped host-visible buffers of `alloc >= len` bytes.
        unsafe {
            std::ptr::write_bytes(buf_ptr, fill, len as usize);
            std::ptr::write_bytes(staging_ptr, fill, len as usize);
        }
        let mut shadow = Shadow::new(buffer, staging_ptr, 0, title_start, len, None);
        shadow.card = Some(Card {
            staging,
            before: Mutex::new(None),
            dirty: RwLock::default(),
            sent: RwLock::default(),
            size: 0,
            is_window: false,
        });
        let arc = Arc::new(shadow);
        CARDED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(Arc::downgrade(&arc));
        Some(arc)
    }

    #[cfg(test)]
    pub(crate) fn clear_card_dirty_for_test(&self) {
        if let Some(card) = &self.card {
            card.dirty
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .clear();
        }
    }

    #[cfg(test)]
    pub(crate) fn seed_card_dirty_for_test(&self, ranges: Vec<(usize, usize)>) {
        if let Some(card) = &self.card {
            card.dirty
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .extend(ranges);
        }
    }

    #[cfg(test)]
    pub(crate) fn seed_card_sent_for_test(&self, sent_as: u64, ranges: Vec<(usize, usize)>) {
        if let Some(card) = &self.card {
            card.sent
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .push_sorted(sent_as, ranges);
        }
    }

    #[cfg(test)]
    pub(crate) fn card_sent_len_for_test(&self) -> usize {
        self.card.as_ref().map_or(0, |c| {
            c.sent.read().unwrap_or_else(|e| e.into_inner()).len()
        })
    }

    #[cfg(test)]
    pub(crate) fn card_sent_has_after_for_test(&self, fetched: u64) -> bool {
        self.card.as_ref().is_some_and(|c| {
            c.sent
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .has_after(fetched)
        })
    }

    #[cfg(test)]
    pub(crate) fn new_rebar_for_test(title_start: u64, len: u64, fill: u8) -> Option<Arc<Shadow>> {
        let h = host()?;
        let alloc = len.max(64);
        let (buffer, data) = h.shared_buffer(alloc)?;
        // SAFETY: freshly mapped buffer of `alloc >= len` bytes.
        unsafe { std::ptr::write_bytes(data, fill, len as usize) };
        // SAFETY: live buffer.
        let gpu_va = unsafe { buffer.GetGPUVirtualAddress() };
        let shadow = Shadow::new(buffer, data, gpu_va, title_start, len, Some(0));
        if let Some(rebar) = &shadow.rebar {
            *rebar.mirror.lock().unwrap_or_else(|e| e.into_inner()) =
                Some(vec![fill; len as usize]);
        }
        Some(Arc::new(shadow))
    }

    #[cfg(test)]
    pub(crate) fn write_test_bytes(&self, offset: usize, bytes: &[u8]) {
        assert!(offset + bytes.len() <= self.len as usize);
        // SAFETY: test-owned mapped shadow buffer covering `self.len` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.add(offset), bytes.len())
        };
    }
}

type Job = Box<dyn FnOnce() + Send>;

struct Workers {
    jobs: Mutex<std::collections::VecDeque<Job>>,
    more: std::sync::Condvar,
}

std::thread_local! {
    /// Set while a thread is executing a job inside [`wide`] so nested callers
    /// never re-enter the shared worker pool and deadlock.
    static IN_WIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct InWideGuard(bool);

impl InWideGuard {
    fn enter() -> Self {
        Self(IN_WIDE.with(|flag| flag.replace(true)))
    }
}

impl Drop for InWideGuard {
    fn drop(&mut self) {
        IN_WIDE.with(|flag| flag.set(self.0));
    }
}

fn in_wide() -> bool {
    IN_WIDE.with(|flag| flag.get())
}

fn workers() -> &'static Workers {
    static WORKERS: OnceLock<Workers> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let workers = WORKERS.get_or_init(|| Workers {
        jobs: Mutex::new(std::collections::VecDeque::new()),
        more: std::sync::Condvar::new(),
    });
    STARTED.call_once(|| {
        let n = std::thread::available_parallelism()
            .map_or(4, |n| n.get() / 2)
            .clamp(2, 6);
        for i in 0..n {
            let _ = std::thread::Builder::new()
                .name(format!("victory-wide-{i}"))
                .spawn(move || {
                    let workers = WORKERS.get().expect("made above");
                    loop {
                        let job = {
                            let mut jobs = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
                            loop {
                                if let Some(job) = jobs.pop_front() {
                                    break job;
                                }
                                jobs = workers.more.wait(jobs).unwrap_or_else(|e| e.into_inner());
                            }
                        };
                        let _guard = InWideGuard::enter();
                        job();
                    }
                });
        }
    });
    workers
}

fn wide(jobs: Vec<Job>) {
    if jobs.len() < 2 {
        for job in jobs {
            let _guard = InWideGuard::enter();
            job();
        }
        return;
    }
    let done = Arc::new((Mutex::new(jobs.len()), std::sync::Condvar::new()));
    let workers = workers();
    {
        let mut queue = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
        for job in jobs {
            let done = done.clone();
            queue.push_back(Box::new(move || {
                struct Done(Arc<(Mutex<usize>, std::sync::Condvar)>);
                impl Drop for Done {
                    fn drop(&mut self) {
                        let mut left = self.0.0.lock().unwrap_or_else(|e| e.into_inner());
                        *left -= 1;
                        if *left == 0 {
                            self.0.1.notify_all();
                        }
                    }
                }
                let _done = Done(done);
                job();
            }));
        }
    }
    workers.more.notify_all();
    loop {
        let job = workers
            .jobs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop_front();
        match job {
            Some(job) => {
                let _guard = InWideGuard::enter();
                job();
            }
            None => break,
        }
    }
    let mut left = done.0.lock().unwrap_or_else(|e| e.into_inner());
    while *left > 0 {
        left = done.1.wait(left).unwrap_or_else(|e| e.into_inner());
    }
}

/// Runs `jobs` on the shared worker pool (or inline/scoped if called re-entrantly).
pub(crate) fn wide_run(jobs: Vec<Box<dyn FnOnce() + Send>>) {
    run_copy_jobs(jobs);
}

/// Runs `work(i)` for `i in 0..n` across the shared worker pool.
pub(crate) fn wide_each(n: usize, work: impl Fn(usize) + Send + Sync + 'static) {
    if n < 2 {
        for i in 0..n {
            work(i);
        }
        return;
    }
    let work = Arc::new(work);
    let jobs: Vec<Job> = (0..n)
        .map(|i| {
            let work = work.clone();
            Box::new(move || work(i)) as Job
        })
        .collect();
    run_copy_jobs(jobs);
}

/// Whether large shadow refreshes, parallel batch planning, and staging fills
/// parallelize their dirty copy chunks across worker threads
/// (`VICTORY_WIDE_COPY=0` disables it; `=1` forces it; on by default on 4+
/// logical CPUs).
pub(crate) fn wide_copy() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        wide_copy_enabled(
            std::env::var("VICTORY_WIDE_COPY").ok().as_deref(),
            std::thread::available_parallelism().ok().map(|n| n.get()),
        )
    })
}

fn wide_copy_enabled(setting: Option<&str>, logical_cpus: Option<usize>) -> bool {
    match setting {
        Some("1") => true,
        Some("0") => false,
        _ => logical_cpus.is_some_and(|n| n >= 4),
    }
}

/// Minimum total dirty bytes to split across worker threads in [`run_copy_segs`].
const WIDE_COPY_MIN: usize = 1 << 20;
/// Maximum target byte size per parallel copy chunk.
const WIDE_COPY_CHUNK: usize = 512 << 10;

/// One contiguous byte range to copy from title memory into a shadow copy
/// (and optionally through a cached ReBAR RAM mirror into WC VRAM).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CopySeg {
    src: usize,
    dst: usize,
    /// Non-zero for mirrored ReBAR shadows: copy `src -> mirror`, then
    /// `mirror -> dst` (WC VRAM).
    mirror: usize,
    len: usize,
    /// When true, `dst` is a card shadow's `staging` buffer (the relay
    /// comparison baseline). Bytes are staged off-line and copied into `dst`
    /// ONLY when every chunk of the segment succeeds, so a failed segment
    /// never corrupts the comparison baseline.
    stage_card: bool,
}

/// Slices `segs` into buckets of at most `chunk_bytes` each, splitting long
/// runs cleanly without losing or duplicating a single byte.
#[cfg(test)]
fn chunk_copy_segs(segs: &[CopySeg], chunk_bytes: usize) -> Vec<Vec<CopySeg>> {
    chunk_copy_segs_indexed(segs, chunk_bytes)
        .into_iter()
        .map(|b| b.into_iter().map(|(_, s)| s).collect())
        .collect()
}

fn chunk_copy_segs_indexed(segs: &[CopySeg], chunk_bytes: usize) -> Vec<Vec<(usize, CopySeg)>> {
    let chunk_bytes = chunk_bytes.max(1);
    let mut buckets: Vec<Vec<(usize, CopySeg)>> = Vec::new();
    let mut cur: Vec<(usize, CopySeg)> = Vec::new();
    let mut bytes = 0usize;
    for (idx, &seg) in segs.iter().enumerate() {
        let mut at = 0usize;
        while at < seg.len {
            let n = (chunk_bytes - bytes).min(seg.len - at);
            cur.push((
                idx,
                CopySeg {
                    src: seg.src + at,
                    dst: seg.dst + at,
                    mirror: if seg.mirror != 0 { seg.mirror + at } else { 0 },
                    len: n,
                    stage_card: seg.stage_card,
                },
            ));
            at += n;
            bytes += n;
            if bytes >= chunk_bytes {
                buckets.push(std::mem::take(&mut cur));
                bytes = 0;
            }
        }
    }
    if !cur.is_empty() {
        buckets.push(cur);
    }
    buckets
}

fn partition_copy_segs(segs: &[CopySeg], ok: &[bool]) -> (Vec<CopySeg>, Vec<CopySeg>, usize) {
    let mut succeeded = Vec::new();
    let mut failed = Vec::new();
    let mut bytes = 0usize;
    for (&seg, &passed) in segs.iter().zip(ok) {
        if passed {
            bytes += seg.len;
            succeeded.push(seg);
        } else {
            failed.push(seg);
        }
    }
    (succeeded, failed, bytes)
}

/// Materializes any deferred `lazy_back` blocks overlapping `segs` in one batched DMA.
fn materialize_copy_segs(segs: &[CopySeg]) {
    if !segs.is_empty() && crate::lazy_back::active() {
        let ranges: Vec<(usize, usize)> = segs
            .iter()
            .filter(|s| s.len > 0)
            .map(|s| (s.src, s.len))
            .collect();
        if !ranges.is_empty() {
            let _ = crate::lazy_back::materialize_ranges(&ranges);
        }
    }
}

#[inline]
fn is_diag_no_copy_mode() -> bool {
    #[cfg(feature = "diagnostics")]
    {
        crate::copystat::mode() == 2
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        false
    }
}

#[inline]
fn execute_copy_seg_checked(seg: CopySeg) -> bool {
    if seg.len == 0 || is_diag_no_copy_mode() {
        return true;
    }
    if seg.mirror != 0 {
        // Whenever lazy readback can arm (`VICTORY_LAZY_BACK=1` in production
        // or in tests), stage mirrored reads through a temporary buffer so a
        // concurrent lazy arm cannot leave a partial prefix in `seg.mirror`
        // out of sync with WC VRAM `seg.dst`. Zero allocation when disabled.
        if crate::lazy_back::can_arm() {
            let mut tmp = vec![0u8; seg.len];
            if !crate::host::copy_mapped_raw_checked(
                tmp.as_mut_ptr(),
                seg.src as *const u8,
                seg.len,
                false,
            ) {
                return false;
            }
            // SAFETY: `seg` lies inside the shadow's mirror and WC VRAM mapping
            // (`off + len <= self.len`), and chunks are strictly disjoint.
            unsafe {
                std::ptr::copy_nonoverlapping(tmp.as_ptr(), seg.mirror as *mut u8, seg.len);
                std::ptr::copy_nonoverlapping(tmp.as_ptr(), seg.dst as *mut u8, seg.len);
            }
            return true;
        }
        if !crate::host::copy_mapped_raw_checked(
            seg.mirror as *mut u8,
            seg.src as *const u8,
            seg.len,
            false,
        ) {
            return false;
        }
        // SAFETY: `seg` lies inside the shadow's mirror and WC VRAM mapping
        // (`off + len <= self.len`), and chunks are strictly disjoint.
        unsafe {
            std::ptr::copy_nonoverlapping(seg.mirror as *const u8, seg.dst as *mut u8, seg.len);
        }
        true
    } else {
        crate::host::copy_mapped_raw_checked(
            seg.dst as *mut u8,
            seg.src as *const u8,
            seg.len,
            false,
        )
    }
}

/// Executes copy jobs on the warm `workers()` pool when called from outside
/// [`wide`], or via a bounded `std::thread::scope` if called re-entrantly from
/// inside a `wide` worker so the pool can never deadlock.
fn run_copy_jobs(jobs: Vec<Job>) {
    if jobs.len() < 2 {
        for job in jobs {
            job();
        }
        return;
    }
    if !in_wide() {
        wide(jobs);
        return;
    }
    // Re-entrant call from inside a `wide` worker: never push back into the
    // pool queue; fan out with a small scoped group instead.
    let threads = jobs.len().min(4);
    let queue = Mutex::new(std::collections::VecDeque::from(jobs));
    std::thread::scope(|s| {
        for _ in 1..threads {
            let queue = &queue;
            s.spawn(move || {
                while let Some(job) = queue.lock().unwrap_or_else(|e| e.into_inner()).pop_front() {
                    job();
                }
            });
        }
        while let Some(job) = queue.lock().unwrap_or_else(|e| e.into_inner()).pop_front() {
            job();
        }
    });
}

struct CopySegsResult {
    /// Per-input-segment full-success flag (`true` iff every chunk of `segs[i]` succeeded).
    ok: Vec<bool>,
    /// `(input_seg_index, chunk)` for every chunk that wrote destination bytes
    /// (every successful chunk, plus any attempted non-mirrored non-staged
    /// chunk that failed because `copy_mapped_raw_checked` may have written a
    /// partial prefix directly into `seg.dst`).
    touched: Vec<(usize, CopySeg)>,
    /// `(input_seg_index, staged_bytes)` for `stage_card` segments whose EVERY
    /// chunk succeeded; callers publish these into `card.staging` right before
    /// marking `card.dirty`. Failed `stage_card` segments are discarded so they
    /// never pollute the relay comparison baseline.
    staged: Vec<(usize, Vec<u8>)>,
}

#[cfg(test)]
std::thread_local! {
    static BETWEEN_PUBLISH_AND_DIRTY_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

/// Copies successfully staged card refresh buffers into their target
/// `card.staging` mappings right before publishing matching `card.dirty` ranges.
fn publish_staged_card_segs(segs: &[CopySeg], staged: &[(usize, Vec<u8>)]) {
    for (idx, buf) in staged {
        let seg = segs[*idx];
        debug_assert_eq!(seg.len, buf.len());
        if seg.len > 0 {
            // SAFETY: `seg.dst` lies inside the shadow's mapped `card.staging`
            // (`off + len <= self.len`), and `buf` holds `seg.len` valid bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(buf.as_ptr(), seg.dst as *mut u8, seg.len);
            }
        }
    }
}

fn run_copy_segs_status(segs: &[CopySeg]) -> CopySegsResult {
    let mut ok = vec![true; segs.len()];
    let mut touched = Vec::new();
    let mut staged = Vec::new();
    let total: usize = segs.iter().map(|s| s.len).sum();
    if total == 0 || is_diag_no_copy_mode() {
        return CopySegsResult {
            ok,
            touched,
            staged,
        };
    }

    // For card staging segments when lazy readback can arm, redirect writes
    // into per-segment off-line buffers so failed segments never alter
    // `card.staging`.
    let has_staged = segs.iter().any(|s| s.stage_card && s.len > 0);
    let mut stage_bufs: Vec<Option<Vec<u8>>> = Vec::new();
    let exec_owned: Vec<CopySeg>;
    let exec_segs: &[CopySeg] = if has_staged {
        stage_bufs.reserve(segs.len());
        let mut redirected = Vec::with_capacity(segs.len());
        for &seg in segs {
            if seg.stage_card && seg.len > 0 {
                let mut buf = vec![0u8; seg.len];
                let ptr = buf.as_mut_ptr() as usize;
                stage_bufs.push(Some(buf));
                redirected.push(CopySeg { dst: ptr, ..seg });
            } else {
                stage_bufs.push(None);
                redirected.push(seg);
            }
        }
        exec_owned = redirected;
        &exec_owned
    } else {
        segs
    };

    let finish_staged = |ok: &[bool], stage_bufs: Vec<Option<Vec<u8>>>| -> Vec<(usize, Vec<u8>)> {
        let mut out = Vec::new();
        for (i, buf) in stage_bufs.into_iter().enumerate() {
            if ok[i]
                && let Some(bytes) = buf
            {
                out.push((i, bytes));
            }
        }
        out
    };

    if !wide_copy() || total < WIDE_COPY_MIN {
        for (i, &seg) in exec_segs.iter().enumerate() {
            if execute_copy_seg_checked(seg) {
                if seg.len > 0 && !seg.stage_card {
                    touched.push((i, segs[i]));
                }
            } else {
                ok[i] = false;
                if seg.len > 0 && seg.mirror == 0 && !seg.stage_card {
                    touched.push((i, segs[i]));
                }
            }
        }
        if has_staged {
            staged = finish_staged(&ok, stage_bufs);
        }
        return CopySegsResult {
            ok,
            touched,
            staged,
        };
    }
    let chunk_bytes = (total / 4).clamp(256 << 10, WIDE_COPY_CHUNK);
    let buckets = chunk_copy_segs_indexed(exec_segs, chunk_bytes);
    if buckets.len() < 2 {
        for (i, &seg) in exec_segs.iter().enumerate() {
            if execute_copy_seg_checked(seg) {
                if seg.len > 0 && !seg.stage_card {
                    touched.push((i, segs[i]));
                }
            } else {
                ok[i] = false;
                if seg.len > 0 && seg.mirror == 0 && !seg.stage_card {
                    touched.push((i, segs[i]));
                }
            }
        }
        if has_staged {
            staged = finish_staged(&ok, stage_bufs);
        }
        return CopySegsResult {
            ok,
            touched,
            staged,
        };
    }
    let seg_ok: Arc<Vec<std::sync::atomic::AtomicBool>> = Arc::new(
        (0..exec_segs.len())
            .map(|_| std::sync::atomic::AtomicBool::new(true))
            .collect(),
    );
    let touched_chunks: Arc<Mutex<Vec<(usize, CopySeg)>>> = Arc::new(Mutex::new(Vec::new()));
    let jobs: Vec<Job> = buckets
        .into_iter()
        .map(|bucket| {
            let seg_ok = seg_ok.clone();
            let touched_chunks = touched_chunks.clone();
            Box::new(move || {
                let mut local_touched = Vec::new();
                for (idx, seg) in bucket {
                    if execute_copy_seg_checked(seg) {
                        if seg.len > 0 && !seg.stage_card {
                            local_touched.push((idx, seg));
                        }
                    } else {
                        seg_ok[idx].store(false, Ordering::Release);
                        if seg.len > 0 && seg.mirror == 0 && !seg.stage_card {
                            local_touched.push((idx, seg));
                        }
                    }
                }
                if !local_touched.is_empty() {
                    touched_chunks
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .extend(local_touched);
                }
            }) as Job
        })
        .collect();
    run_copy_jobs(jobs);
    for (slot, flag) in ok.iter_mut().zip(seg_ok.iter()) {
        *slot = flag.load(Ordering::Acquire);
    }
    let touched = std::mem::take(&mut *touched_chunks.lock().unwrap_or_else(|e| e.into_inner()));
    if has_staged {
        staged = finish_staged(&ok, stage_bufs);
    }
    CopySegsResult {
        ok,
        touched,
        staged,
    }
}

/// Copies `len` bytes from title memory `src` into writable host memory `dst`,
/// splitting large fills across worker threads when [`wide_copy`] is enabled,
/// returning `false` if lazy materialization or retries fail.
pub(crate) fn copy_mapped_wide_checked(dst: *mut u8, src: *const u8, len: usize) -> bool {
    if len == 0 {
        return true;
    }
    if !wide_copy() || len < WIDE_COPY_MIN {
        return crate::host::copy_mapped_checked(dst, src, len, false);
    }
    if crate::lazy_back::active() && !crate::lazy_back::materialize_range(src as usize, len) {
        return false;
    }
    let seg = [CopySeg {
        src: src as usize,
        dst: dst as usize,
        mirror: 0,
        len,
        stage_card: false,
    }];
    if run_copy_segs_status(&seg).ok[0] {
        return true;
    }
    crate::host::copy_mapped_checked(dst, src, len, false)
}

/// Dirty byte ranges discovered by [`Shadow::plan_update`] before copying.
#[derive(Default)]
struct UpdatePlan {
    runs: Vec<(usize, usize)>,
    epochs: Vec<(usize, u64)>,
    busy_ranges: Vec<(usize, usize)>,
    refresh_failed: bool,
}

/// Two-phase refresh for a [`crate::host::Batch`] flush:
/// 1. Discover stale byte runs for each shadow (in parallel across shadows when
///    `parallel_plan` is enabled and there are at least 4 shadows), releasing
///    each shadow's `copied` lock immediately after planning.
/// 2. Flatten all non-mirrored dirty runs across the entire batch into balanced
///    copy chunks and execute them in one flat [`wide`] pass, then apply
///    per-shadow metadata updates (ReBAR dirty bits) and copy any mirrored
///    ReBAR shadows.
///
/// Returns `(bytes_copied, ok)` where `ok == false` if any dirty lazy-owned
/// page failed to materialize or copy so the submission can abort rather than
/// running with stale bytes.
///
/// This eliminates both straggler single-thread bottlenecks (one huge 50–174 MB
/// buffer stalling a multi-shadow flush) and nested `wide` re-entrancy.
pub(crate) fn refresh_batch(
    items: Vec<(Arc<Shadow>, usize, usize)>,
    parallel_plan: bool,
) -> (usize, bool) {
    if items.is_empty() {
        return (0, true);
    }
    if items.len() == 1 {
        let (shadow, from, to) = items.into_iter().next().unwrap();
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let (copied, ok) = shadow.update_checked(from, to - from);
        #[cfg(feature = "diagnostics")]
        {
            if let Some(bytes) = skipped(to - from, copied) {
                crate::copystat::add(crate::copystat::C::Skipped, 1);
                crate::copystat::add(crate::copystat::C::SkippedBytes, bytes);
            }
            crate::copystat::range(from, to - from, started.elapsed().as_micros() as u64, 0);
        }
        return (copied, ok);
    }

    #[cfg(feature = "diagnostics")]
    let batch_started = std::time::Instant::now();

    type PlannedItem = (Arc<Shadow>, usize, usize, UpdatePlan);

    let count = items.len();
    let planned: Vec<PlannedItem> = if parallel_plan && count >= 4 && !in_wide() {
        let slots: Arc<Vec<Mutex<Option<PlannedItem>>>> =
            Arc::new((0..count).map(|_| Mutex::new(None)).collect());
        let jobs: Vec<Job> = items
            .into_iter()
            .enumerate()
            .map(|(i, (shadow, from, to))| {
                let slots = slots.clone();
                Box::new(move || {
                    let plan = shadow.plan_update(from, to - from);
                    *slots[i].lock().unwrap_or_else(|e| e.into_inner()) =
                        Some((shadow, from, to, plan));
                }) as Job
            })
            .collect();
        wide(jobs);
        slots
            .iter()
            .filter_map(|slot| slot.lock().unwrap_or_else(|e| e.into_inner()).take())
            .collect()
    } else {
        items
            .into_iter()
            .map(|(shadow, from, to)| {
                let plan = shadow.plan_update(from, to - from);
                (shadow, from, to, plan)
            })
            .collect()
    };

    let mut all_segs: Vec<CopySeg> = Vec::new();
    let mut non_mirrored: Vec<(usize, usize, usize)> = Vec::new();
    let mut mirrored: Vec<usize> = Vec::new();
    let mut item_copied = vec![0usize; planned.len()];
    let mut total_copied = 0usize;
    let mut all_ok = true;

    for (idx, (shadow, _, _, plan)) in planned.iter().enumerate() {
        if plan.refresh_failed {
            all_ok = false;
            continue;
        }
        if plan.runs.is_empty() {
            shadow.commit_copied_epochs(&plan.epochs, &[]);
            continue;
        }
        if shadow.rebar.as_ref().is_some_and(RebarShadow::has_mirror) {
            mirrored.push(idx);
            continue;
        }
        let seg_lo = all_segs.len();
        let bytes = shadow.push_copy_segs(&plan.runs, &mut all_segs);
        let seg_hi = all_segs.len();
        if bytes > 0 {
            non_mirrored.push((idx, seg_lo, seg_hi));
        } else {
            shadow.commit_copied_epochs(&plan.epochs, &[]);
        }
    }

    if crate::lazy_back::active() && (!all_segs.is_empty() || !mirrored.is_empty()) {
        let mut ranges: Vec<(usize, usize)> = all_segs
            .iter()
            .filter(|s| s.len > 0)
            .map(|s| (s.src, s.len))
            .collect();
        for &idx in &mirrored {
            let (shadow, _, _, plan) = &planned[idx];
            let (start, end) = (shadow.start as usize, (shadow.start + shadow.len) as usize);
            for &(raw_from, raw_to) in &plan.runs {
                let (from, to) = (raw_from.max(start), raw_to.min(end));
                if from < to {
                    ranges.push((from, to - from));
                }
            }
        }
        if !ranges.is_empty() {
            let _ = crate::lazy_back::materialize_ranges(&ranges);
        }
    }

    if !all_segs.is_empty() {
        let res = run_copy_segs_status(&all_segs);
        let mut ok = res.ok;
        let mut touched = res.touched;
        let mut staged = res.staged;
        if ok.iter().any(|&v| !v) {
            let failed_indices: Vec<usize> = ok
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| (!v).then_some(i))
                .collect();
            let retry_segs: Vec<CopySeg> = failed_indices.iter().map(|&i| all_segs[i]).collect();
            materialize_copy_segs(&retry_segs);
            let retry_res = run_copy_segs_status(&retry_segs);
            touched.extend(
                retry_res
                    .touched
                    .into_iter()
                    .map(|(retry_i, seg)| (failed_indices[retry_i], seg)),
            );
            staged.extend(
                retry_res
                    .staged
                    .into_iter()
                    .map(|(retry_i, buf)| (failed_indices[retry_i], buf)),
            );
            for (idx, passed) in failed_indices.into_iter().zip(retry_res.ok) {
                ok[idx] = passed;
            }
        }
        let mut touched_by_shadow: Vec<Vec<CopySeg>> = vec![Vec::new(); non_mirrored.len()];
        for (seg_idx, seg) in touched {
            let owner = non_mirrored.partition_point(|&(_, _, hi)| hi <= seg_idx);
            if let Some(slot) = touched_by_shadow.get_mut(owner) {
                slot.push(seg);
            }
        }
        let mut staged_by_shadow: Vec<Vec<(usize, Vec<u8>)>> = vec![Vec::new(); non_mirrored.len()];
        for (seg_idx, buf) in staged {
            let owner = non_mirrored.partition_point(|&(_, _, hi)| hi <= seg_idx);
            if let Some(slot) = staged_by_shadow.get_mut(owner) {
                slot.push((seg_idx, buf));
            }
        }
        for ((&(idx, seg_lo, seg_hi), shadow_touched), shadow_staged) in non_mirrored
            .iter()
            .zip(touched_by_shadow.iter())
            .zip(staged_by_shadow.iter())
        {
            let (shadow, _, _, plan) = &planned[idx];
            let (succeeded, failed, bytes) =
                partition_copy_segs(&all_segs[seg_lo..seg_hi], &ok[seg_lo..seg_hi]);
            if !failed.is_empty() {
                all_ok = false;
            }
            shadow.publish_and_finish_copy_segs(
                shadow.start as usize,
                &all_segs,
                shadow_staged,
                &succeeded,
                shadow_touched,
            );
            shadow.commit_copied_epochs(&plan.epochs, &failed);
            item_copied[idx] = bytes;
            total_copied += bytes;
        }
    }

    for idx in mirrored {
        let (shadow, _, _, plan) = &planned[idx];
        let (bytes, ok) = shadow.copy_runs_with_epochs(&plan.runs, &plan.epochs);
        if !ok {
            all_ok = false;
        }
        item_copied[idx] = bytes;
        total_copied += bytes;
    }

    for (idx, (shadow, _, _, plan)) in planned.iter().enumerate() {
        if plan.refresh_failed {
            continue;
        }
        for &(from, to) in &plan.busy_ranges {
            let (bytes, ok) = shadow.update_busy(from, to);
            if !ok {
                all_ok = false;
            }
            item_copied[idx] += bytes;
            total_copied += bytes;
        }
    }

    #[cfg(feature = "diagnostics")]
    {
        let batch_us = batch_started.elapsed().as_micros() as u64;
        for ((_, from, to, _), copied) in planned.iter().zip(item_copied) {
            if let Some(bytes) = skipped(to.saturating_sub(*from), copied) {
                crate::copystat::add(crate::copystat::C::Skipped, 1);
                crate::copystat::add(crate::copystat::C::SkippedBytes, bytes);
            }
            crate::copystat::range(*from, to.saturating_sub(*from), batch_us, 0);
        }
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = item_copied;

    (total_copied, all_ok)
}

/// Pages of `[a, e)` that have never been copied (`copied == 0`) or were
/// written by the title after their last copy (`write > copied`). Matches
/// [`Shadow::copy_stale`], which copies both cases even when a never-copied
/// page does not appear in the write-watch delta `written`.
fn stale_pages_in(
    base: usize,
    copied: &[u64],
    a: usize,
    e: usize,
    written: &[(usize, u64)],
) -> Vec<usize> {
    if a >= e {
        return Vec::new();
    }
    let mut writes = written.iter().peekable();
    let mut out = Vec::new();
    for page in a / PAGE..=(e - 1) / PAGE {
        while writes.peek().is_some_and(|&&(p, _)| p < page) {
            writes.next();
        }
        let write = match writes.peek() {
            Some(&&(p, w)) if p == page => {
                writes.next();
                w
            }
            _ => 0,
        };
        if page >= base
            && copied
                .get(page - base)
                .is_none_or(|&at| at == 0 || write > at)
        {
            out.push(page);
        }
    }
    out
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
/// `RwLock` allows parallel recording threads to look up existing shadows
/// concurrently without contending on a global mutex.
static BUFFERS: RwLock<BTreeMap<(u64, usize), Entry>> = RwLock::new(BTreeMap::new());

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    let before = BUFFERS.write().unwrap_or_else(|e| e.into_inner()).insert(
        (start, resource),
        Entry {
            // A size from the title's description: one past the address
            // space ends at its top instead of wrapping below `start`.
            end: start.saturating_add(len),
            shadow: None,
        },
    );
    // A buffer made again at the same place and address: its old copy may
    // still be named by lists and descriptors, as one the title released.
    if let Some(shadow) = before.and_then(|e| e.shadow) {
        if crate::lazy_back::active() {
            crate::lazy_back::materialize_shadow(Arc::as_ptr(&shadow) as usize);
        }
        let born = shadow.born_seq;
        crate::host::retire_born(shadow, born);
    }
}

/// Removes any in-place (`VICTORY_INPLACE` / `VICTORY_STRETCH`) shadows
/// overlapping `[at, at + len)` when title memory is unmapped or changed.
pub(crate) fn in_place_let_go(at: u64, len: u64) -> u64 {
    let end = at.saturating_add(len);
    let mut gone: Vec<Arc<Shadow>> = Vec::new();
    {
        let mut guard = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<(u64, u64)> = guard
            .entries
            .range((at.saturating_sub(WINDOW), 0)..(end, 0))
            .filter(|(key, shadow)| shadow.place.is_some() && key.0 + key.1 > at)
            .map(|(key, _)| *key)
            .collect();
        gone.extend(keys.into_iter().filter_map(|key| {
            let s = guard.entries.remove(&key)?;
            guard.total_bytes = guard.total_bytes.saturating_sub(s.len);
            Some(s)
        }));
    }
    {
        let mut stretches = STRETCHES.write().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<u64> = stretches
            .range(..end)
            .filter(|(_, s)| s.start + s.len > at)
            .map(|(k, _)| *k)
            .collect();
        gone.extend(keys.into_iter().filter_map(|k| stretches.remove(&k)));
    }
    {
        let mut map = BUFFERS.write().unwrap_or_else(|e| e.into_inner());
        for (_, entry) in map.range_mut(..(end, 0)).rev().take(256) {
            if entry.end > at && entry.shadow.as_ref().is_some_and(|s| s.place.is_some()) {
                gone.extend(entry.shadow.take());
            }
        }
    }
    let n = gone.len() as u64;
    for shadow in gone {
        let born = shadow.born_seq;
        crate::host::retire_born(shadow, born);
    }
    n
}

/// Retires any active `VICTORY_STRETCH` window opened on `section` when the
/// runtime reports that `section`'s handle number has been reused.
pub(crate) fn stretch_section_reused(section: usize) {
    let gone: Vec<Arc<Shadow>> = {
        let mut stretches = STRETCHES.write().unwrap_or_else(|e| e.into_inner());
        if stretches.is_empty() {
            return;
        }
        let keys: Vec<u64> = stretches
            .iter()
            .filter(|(_, s)| {
                s.stretch_view
                    .as_ref()
                    .is_some_and(|v| v.section == section)
            })
            .map(|(&k, _)| k)
            .collect();
        keys.into_iter()
            .filter_map(|k| stretches.remove(&k))
            .collect()
    };
    for shadow in gone {
        let born = shadow.born_seq;
        crate::host::retire_born(shadow, born);
    }
}

/// Spans of registered buffers and active windows for `inplace` census.
pub(crate) fn spans() -> Vec<(u64, u64, bool)> {
    let mut spans: Vec<(u64, u64, bool)> = BUFFERS
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .map(|(key, entry)| (key.0, entry.end - key.0, false))
        .collect();
    let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
    spans.extend(loose.entries.keys().map(|&(start, len)| (start, len, true)));
    let stretches = STRETCHES.read().unwrap_or_else(|e| e.into_inner());
    spans.extend(stretches.values().map(|s| (s.start, s.len, true)));
    spans
}

/// The title released it; lists that still hold its copy keep it alive,
/// and any active epoch keeps it until that work completes.
pub(crate) fn unregister(resource: usize, start: u64) {
    let shadow = {
        let mut map = BUFFERS.write().unwrap_or_else(|e| e.into_inner());
        map.remove(&(start, resource)).and_then(|e| e.shadow)
    };
    if let Some(shadow) = shadow {
        if crate::lazy_back::active() {
            crate::lazy_back::materialize_shadow(Arc::as_ptr(&shadow) as usize);
        }
        let born = shadow.born_seq;
        crate::host::retire_born(shadow, born);
    }
}

/// Reserves `size` bytes against `used` up to `budget` and allocates a ReBAR
/// buffer, rolling back the reservation if D3D12 allocation fails.
fn try_alloc_rebar_in(
    used: &AtomicU64,
    size: u64,
    budget: u64,
) -> Option<(ID3D12Resource, *mut u8, u64)> {
    let h = host()?;
    if try_reserve_rebar(used, size, budget) {
        if let Some((buffer, data)) = h.try_rebar_shadow_buffer(size) {
            return Some((buffer, data, size));
        }
        used.fetch_sub(size, Ordering::Relaxed);
    }
    None
}

/// Allocates a shadow backing buffer: prefers ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`)
/// VRAM within the adapter's budget, falling back cleanly to system-RAM
/// [`Host::shared_buffer`] when ReBAR is unavailable, disabled, or full.
fn alloc_shadow_buffer(size: u64) -> Option<(ID3D12Resource, *mut u8, Option<u64>)> {
    let h = host()?;
    if let Some((buffer, data, bytes)) =
        try_alloc_rebar_in(&REBAR_SHADOW_BYTES, size, h.rebar_shadow_budget())
    {
        return Some((buffer, data, Some(bytes)));
    }
    let (buffer, data) = h.shared_buffer(size)?;
    Some((buffer, data, None))
}

/// Allocates a ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`) backing buffer for a
/// `LOOSE` window from the dedicated [`REBAR_LOOSE_BYTES`] budget so `LOOSE`
/// windows never starve registered `BUFFERS` out of [`REBAR_SHADOW_BYTES`].
fn alloc_loose_rebar(size: u64) -> Option<(ID3D12Resource, *mut u8, u64)> {
    let h = host()?;
    try_alloc_rebar_in(&REBAR_LOOSE_BYTES, size, h.rebar_loose_budget())
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    // Fast path: concurrent read lock with a single BTreeMap lookup.
    let missing_key = {
        let map = BUFFERS.read().unwrap_or_else(|e| e.into_inner());
        match map
            .range(..=(address, usize::MAX))
            .rev()
            .take(64)
            .find(|(_, e)| e.end >= end)
        {
            Some((_, entry)) if entry.shadow.is_some() => return entry.shadow.clone(),
            Some((&key, _)) => Some(key),
            None => None,
        }
    };
    let Some(key) = missing_key else {
        return loose(address, end);
    };
    // Slow path: first use of this buffer creates its host copy under write lock.
    let mut map = BUFFERS.write().unwrap_or_else(|e| e.into_inner());
    let key = if map.get(&key).is_some_and(|e| e.end >= end) {
        key
    } else {
        match map
            .range(..=(address, usize::MAX))
            .rev()
            .take(64)
            .find(|(_, e)| e.end >= end)
            .map(|(k, _)| *k)
        {
            Some(k) => k,
            None => {
                drop(map);
                return loose(address, end);
            }
        }
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let alloc_size = len.checked_next_multiple_of(1 << 16)?;
        let big = cards() && alloc_size >= card_limits().0;
        if !big && let Some(placed) = crate::inplace::place_for(key.0, len, false) {
            entry.shadow = Some(Arc::new(Shadow::placed(placed, key.0, len, false)));
            return entry.shadow.clone();
        }
        let (buffer, data, card, rebar_bytes) = match card_for(alloc_size) {
            Some(on_card) => {
                let (staging, data) = host()?.shared_buffer(alloc_size)?;
                crate::host::name(&staging, || format!("title buffer {:#x}+{len:#x}", key.0));
                crate::host::name(&on_card, || {
                    format!("title buffer {:#x}+{len:#x} on the card", key.0)
                });
                let card = Card {
                    staging,
                    before: Mutex::default(),
                    dirty: RwLock::default(),
                    sent: RwLock::default(),
                    size: alloc_size,
                    is_window: false,
                };
                (on_card, data, Some(card), None)
            }
            None => {
                let (buffer, data, rebar_bytes) = alloc_shadow_buffer(alloc_size)?;
                crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
                (buffer, data, None, rebar_bytes)
            }
        };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                card = card.is_some(),
                rebar = rebar_bytes.is_some(),
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                card = card.is_some(),
                rebar = rebar_bytes.is_some(),
                "buffer copied to the host"
            );
        }
        let mut shadow = Shadow::new(buffer, data, gpu, key.0, len, rebar_bytes);
        shadow.card = card;
        let shadow = Arc::new(shadow);
        if shadow.card.is_some() {
            CARDED
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(Arc::downgrade(&shadow));
        }
        entry.shadow = Some(shadow);
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (from their
/// own allocators). Such memory gets copies of the aligned windows it touches,
/// bounded by a RAM-scaled LRU cache.
const WINDOW: u64 = 4 << 20;
const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// Monotonic access clock for LRU eviction of loose shadow windows.
static LOOSE_TICK: AtomicU64 = AtomicU64::new(1);

fn total_physical_ram_bytes() -> Option<u64> {
    let mut status = MEMORYSTATUSEX {
        dwLength: size_of::<MEMORYSTATUSEX>() as u32,
        dwMemoryLoad: 0,
        ullTotalPhys: 0,
        ullAvailPhys: 0,
        ullTotalPageFile: 0,
        ullAvailPageFile: 0,
        ullTotalVirtual: 0,
        ullAvailVirtual: 0,
        ullAvailExtendedVirtual: 0,
    };
    // SAFETY: `status` is initialized with `dwLength` set to its byte size as
    // required by `GlobalMemoryStatusEx`.
    let ok = unsafe { GlobalMemoryStatusEx(&raw mut status) };
    (ok != 0 && status.ullTotalPhys > 0).then_some(status.ullTotalPhys)
}

fn loose_budget_for_ram(total_ram_bytes: Option<u64>) -> u64 {
    match total_ram_bytes {
        None => 256 * MIB,
        Some(ram) if ram <= 16 * GIB => 128 * MIB,
        Some(ram) if ram <= 32 * GIB => 256 * MIB,
        Some(_) => 512 * MIB,
    }
}

#[cfg(all(test, windows))]
static TEST_LOOSE_BUDGET: AtomicU64 = AtomicU64::new(0);

#[cfg(all(test, windows))]
static TEST_LOOSE_LOCK: Mutex<()> = Mutex::new(());

#[cfg(all(test, windows))]
pub(crate) struct LooseBudgetGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
}

#[cfg(all(test, windows))]
impl Drop for LooseBudgetGuard {
    fn drop(&mut self) {
        TEST_LOOSE_BUDGET.store(0, Ordering::SeqCst);
        let drained: Vec<Arc<Shadow>> = {
            let mut guard = LOOSE.write().unwrap_or_else(|e| e.into_inner());
            guard.total_bytes = 0;
            std::mem::take(&mut guard.entries).into_values().collect()
        };
        drop(drained);
    }
}

/// Test-only helper that clears the global loose shadow cache and forces an
/// explicit byte budget until the returned guard drops.
#[cfg(all(test, windows))]
pub(crate) fn isolated_loose_budget(budget: u64) -> LooseBudgetGuard {
    let lock = TEST_LOOSE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    {
        let mut guard = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        guard.total_bytes = 0;
        guard.entries.clear();
    }
    TEST_LOOSE_BUDGET.store(budget, Ordering::SeqCst);
    LooseBudgetGuard { _lock: lock }
}

fn loose_budget() -> u64 {
    #[cfg(all(test, windows))]
    {
        let override_budget = TEST_LOOSE_BUDGET.load(Ordering::SeqCst);
        if override_budget != 0 {
            return override_budget;
        }
    }
    static BUDGET: OnceLock<u64> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let ram = total_physical_ram_bytes();
        let budget = loose_budget_for_ram(ram);
        tracing::info!(
            total_ram_mib = ram.map(|b| b / MIB),
            budget_mib = budget / MIB,
            "loose shadow cache budget"
        );
        budget
    })
}

struct LooseCache {
    entries: BTreeMap<(u64, u64), Arc<Shadow>>,
    total_bytes: u64,
}

impl LooseCache {
    const fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            total_bytes: 0,
        }
    }

    /// Evicts least-recently-used idle windows while `total_bytes > budget`,
    /// never evicting `keep`, entries still held outside the cache
    /// (`Arc::strong_count > 1`), or entries with pending GPU copy-back ranges.
    fn evict_over_budget(&mut self, keep: (u64, u64), budget: u64) -> Vec<Arc<Shadow>> {
        if self.total_bytes <= budget {
            return Vec::new();
        }
        let mut candidates: Vec<(u64, (u64, u64))> = self
            .entries
            .iter()
            .filter_map(|(&key, shadow)| {
                if key == keep || Arc::strong_count(shadow) != 1 {
                    return None;
                }
                if !shadow
                    .pending
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_empty()
                {
                    return None;
                }
                Some((shadow.last_used.load(Ordering::Relaxed), key))
            })
            .collect();
        candidates.sort_unstable_by_key(|&(tick, _)| tick);
        let mut evicted = Vec::new();
        for (_, key) in candidates {
            if self.total_bytes <= budget {
                break;
            }
            if let Some(shadow) = self.entries.remove(&key) {
                self.total_bytes = self.total_bytes.saturating_sub(shadow.len);
                evicted.push(shadow);
            }
        }
        if !evicted.is_empty() {
            tracing::debug!(
                evicted = evicted.len(),
                total_bytes = self.total_bytes,
                budget,
                "evicted idle loose shadow windows"
            );
        }
        evicted
    }

    /// Collects LRU idle windows whose only extra strong references are
    /// deferred `lazy_back` entries (`strong_count == 1 + lazy_refs`), enough
    /// to bring `total_bytes` back within `budget` once materialized outside
    /// the cache lock.
    fn lazy_eviction_candidates(&self, keep: (u64, u64), budget: u64) -> Vec<Arc<Shadow>> {
        if self.total_bytes <= budget || !crate::lazy_back::active() {
            return Vec::new();
        }
        let mut candidates: Vec<(u64, (u64, u64), u64)> = self
            .entries
            .iter()
            .filter_map(|(&key, shadow)| {
                if key == keep {
                    return None;
                }
                let lazy_refs =
                    crate::lazy_back::block_count_for_shadow(Arc::as_ptr(shadow) as usize);
                if lazy_refs == 0 || Arc::strong_count(shadow) != 1 + lazy_refs {
                    return None;
                }
                if !shadow
                    .pending
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_empty()
                {
                    return None;
                }
                Some((shadow.last_used.load(Ordering::Relaxed), key, shadow.len))
            })
            .collect();
        candidates.sort_unstable_by_key(|&(tick, _, _)| tick);
        let need = self.total_bytes.saturating_sub(budget);
        let mut reclaimed = 0u64;
        let mut out = Vec::new();
        for (_, key, len) in candidates {
            if reclaimed >= need {
                break;
            }
            if let Some(shadow) = self.entries.get(&key) {
                out.push(shadow.clone());
                reclaimed = reclaimed.saturating_add(len);
            }
        }
        out
    }
}

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: RwLock<LooseCache> = RwLock::new(LooseCache::new());

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

/// Contiguous section-view stretches mapped once in place (`VICTORY_STRETCH`),
/// keyed by start address.
static STRETCHES: RwLock<BTreeMap<u64, Arc<Shadow>>> = RwLock::new(BTreeMap::new());

const STRETCH_REACH: usize = 1 << 30;
const STRETCH_LEAST: usize = 4 << 20;

/// Whether contiguous 64 KiB section stretches are opened in place
/// (`VICTORY_STRETCH=0` disables; on by default).
fn stretch_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_STRETCH").is_none_or(|v| v != "0"))
}

/// Active stretch count and total bytes covered (`VICTORY_STRETCH`).
pub(crate) fn stretch_stats() -> (usize, u64) {
    let map = STRETCHES.read().unwrap_or_else(|e| e.into_inner());
    let bytes = map.values().map(|s| s.len).sum();
    (map.len(), bytes)
}

fn stretch_find_in(
    stretches: &RwLock<BTreeMap<u64, Arc<Shadow>>>,
    address: u64,
    end: u64,
) -> Option<Arc<Shadow>> {
    let map = stretches.read().unwrap_or_else(|e| e.into_inner());
    let s = map
        .range(..=address)
        .rev()
        .take(8)
        .map(|(_, s)| s)
        .find(|s| s.start + s.len >= end)?;
    s.touch();
    Some(s.clone())
}

fn through_another_view(
    device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
    address: u64,
    end: u64,
) -> Option<(Shadow, crate::watch::Reading)> {
    use windows_sys::Win32::System::Memory::{
        FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, UnmapViewOfFile,
    };
    let (mut start, mut stop, _, _) =
        crate::inplace::stretch(address as usize, end as usize, STRETCH_REACH)?;
    if stop - start < STRETCH_LEAST {
        return None;
    }
    // Hold `title_memory_reading(start, len)` across `inplace::stretch` +
    // `MapViewOfFile` so a concurrent remap cannot close and reuse the section
    // handle before `MapViewOfFile` completes.
    let (reading, start, len, section, offset) = {
        let mut attempts = 0;
        loop {
            let reading = crate::watch::title_memory_reading(start, stop - start);
            let (s2, e2, section, offset) =
                crate::inplace::stretch(address as usize, end as usize, STRETCH_REACH)?;
            let len = e2 - s2;
            if len < STRETCH_LEAST {
                return None;
            }
            if s2 >= start && e2 <= stop {
                break (reading, s2, len, section, offset);
            }
            drop(reading);
            attempts += 1;
            if attempts >= 4 {
                return None;
            }
            start = s2;
            stop = e2;
        }
    };
    // SAFETY: section kept alive by `reading` while views stay mapped.
    let view = unsafe {
        MapViewOfFile(
            section as _,
            FILE_MAP_WRITE,
            (offset >> 32) as u32,
            offset as u32,
            len,
        )
    };
    if view.Value.is_null() {
        return None;
    }
    crate::watch::writing(start, len);
    let mut shadow = match crate::inplace::place_stretch(device, view.Value as usize, len as u64) {
        Ok(placed) => Shadow::placed(placed, start as u64, len as u64, true),
        Err(_) => {
            // SAFETY: our view, used by nothing.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
            }
            return None;
        }
    };
    shadow.stretch_view = Some(StretchView {
        addr: view.Value as usize,
        section,
    });
    Some((shadow, reading))
}

fn stretch_create_in(
    stretches: &RwLock<BTreeMap<u64, Arc<Shadow>>>,
    address: u64,
    end: u64,
    enabled: bool,
) -> Option<Arc<Shadow>> {
    if !enabled {
        return None;
    }
    if let Some(s) = stretch_find_in(stretches, address, end) {
        return Some(s);
    }
    let h = host()?;
    // Acquire `Reading` inside `through_another_view` BEFORE taking
    // `stretches.write()` so lock order (`READING -> STRETCHES`) matches
    // `REMAPPING -> title_memory_remapped -> in_place_let_go -> STRETCHES`.
    let (created, reading) = through_another_view(&h.device, address, end)?;
    let candidate = Arc::new(created);
    let (window, retired, unused) = {
        let mut map = stretches.write().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = map
            .range(..=address)
            .rev()
            .take(8)
            .map(|(_, s)| s)
            .find(|s| s.start + s.len >= end)
        {
            s.touch();
            (s.clone(), Vec::new(), Some(candidate))
        } else {
            candidate.touch();
            let (w_start, w_end) = (candidate.start, candidate.start + candidate.len);
            let overlap: Vec<u64> = map
                .range(..w_end)
                .filter(|(_, s)| s.start + s.len > w_start)
                .map(|(&k, _)| k)
                .collect();
            let retired: Vec<Arc<Shadow>> =
                overlap.into_iter().filter_map(|k| map.remove(&k)).collect();
            map.insert(w_start, candidate.clone());
            (candidate, retired, None)
        }
    };
    drop(reading);
    drop(unused);
    for old in retired {
        let born = old.born_seq;
        crate::host::retire_born(old, born);
    }
    Some(window)
}

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    if stretch_enabled() {
        if let Some(s) = stretch_find_in(&STRETCHES, address, end) {
            return Some(s);
        }
        let start = address & !(WINDOW - 1);
        let len = end.next_multiple_of(WINDOW) - start;
        if let Some(shadow) = LOOSE
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .entries
            .get(&(start, len))
        {
            shadow.touch();
            return Some(shadow.clone());
        }
        if let Some(s) = stretch_create_in(&STRETCHES, address, end, true) {
            return Some(s);
        }
    }
    loose_in(&LOOSE, address, end, loose_budget())
}

fn loose_in(
    cache: &RwLock<LooseCache>,
    address: u64,
    end: u64,
    budget: u64,
) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let key = (start, len);
    if let Some(shadow) = cache
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .entries
        .get(&key)
    {
        shadow.touch();
        return Some(shadow.clone());
    }
    let (shadow, mut evicted, lazy_candidates) = {
        let mut guard = cache.write().unwrap_or_else(|e| e.into_inner());
        if let Some(shadow) = guard.entries.get(&key) {
            shadow.touch();
            return Some(shadow.clone());
        }
        let shadow = if let Some(placed) = crate::inplace::place_for(start, len, true) {
            Arc::new(Shadow::placed(placed, start, len, true))
        } else {
            let (buffer, data, card, rebar_bytes) = match window_card_for(len) {
                Some(on_card) => {
                    let (staging, data) = host()?.shared_buffer(len)?;
                    crate::host::name(&staging, || format!("title memory {start:#x}+{len:#x}"));
                    crate::host::name(&on_card, || {
                        format!("title memory {start:#x}+{len:#x} on the card")
                    });
                    let card = Card {
                        staging,
                        before: Mutex::default(),
                        dirty: RwLock::default(),
                        sent: RwLock::default(),
                        size: len,
                        is_window: true,
                    };
                    (on_card, data, Some(card), None)
                }
                None => match alloc_loose_rebar(len) {
                    Some((buffer, data, rebar_bytes)) => {
                        crate::host::name(&buffer, || {
                            format!("title memory {start:#x}+{len:#x} (ReBAR loose)")
                        });
                        (buffer, data, None, Some(rebar_bytes))
                    }
                    None => {
                        let (buffer, data) = host()?.shared_buffer(len)?;
                        crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
                        (buffer, data, None, None)
                    }
                },
            };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            let mut shadow = Shadow::new_loose(buffer, data, gpu, start, len, rebar_bytes);
            shadow.card = card;
            let shadow = Arc::new(shadow);
            if shadow.card.is_some() {
                CARDED
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(Arc::downgrade(&shadow));
            }
            shadow
        };
        shadow.touch();
        guard.entries.insert(key, shadow.clone());
        guard.total_bytes = guard.total_bytes.saturating_add(len);
        let evicted = guard.evict_over_budget(key, budget);
        let lazy_candidates = guard.lazy_eviction_candidates(key, budget);
        (shadow, evicted, lazy_candidates)
    };
    if !lazy_candidates.is_empty() {
        // Drain lazy-only candidates strictly outside `cache.write()` so GPU
        // materialization drops their `LazyEntry` `Arc<Shadow>` references,
        // then re-acquire the cache lock to recheck refs & pending work.
        for cand in lazy_candidates {
            let _ = crate::lazy_back::materialize_shadow(Arc::as_ptr(&cand) as usize);
            drop(cand);
        }
        let mut guard = cache.write().unwrap_or_else(|e| e.into_inner());
        evicted.extend(guard.evict_over_budget(key, budget));
    }
    for old in evicted {
        if crate::lazy_back::active() {
            let _ = crate::lazy_back::materialize_shadow(Arc::as_ptr(&old) as usize);
        }
        let born = old.born_seq;
        crate::host::retire_born(old, born);
    }
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = BUFFERS
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .entries
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

/// Serializes ReBAR GPU tests that share the process-wide `Host` hash stats
/// and `REBAR_SHADOW_BYTES` / `REBAR_LOOSE_BYTES` counters.
#[cfg(all(test, windows))]
static REBAR_GPU_TEST: Mutex<()> = Mutex::new(());

#[cfg(all(test, windows))]
pub(crate) fn lock_rebar_gpu_test() -> std::sync::MutexGuard<'static, ()> {
    REBAR_GPU_TEST.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Range;

    #[test]
    fn a_buffer_sized_past_the_address_space_ends_at_its_top() {
        let start = u64::MAX - 0x0fff;
        register(usize::MAX, start, u64::MAX);
        let end = BUFFERS
            .read()
            .unwrap()
            .get(&(start, usize::MAX))
            .map(|e| e.end);
        unregister(usize::MAX, start);
        assert_eq!(end, Some(u64::MAX));
    }

    #[test]
    fn a_buffer_too_big_to_round_to_a_shadow_size_gets_no_copy() {
        // The end saturates at the top, so rounding its length up to 64 KiB
        // would overflow: no copy, before any host allocation.
        register(usize::MAX - 1, 0x1000, u64::MAX);
        let found = find(0x1000, 1);
        unregister(usize::MAX - 1, 0x1000);
        assert!(found.is_none());
    }

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
        let _rebar = lock_rebar_gpu_test();
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
        let _rebar = lock_rebar_gpu_test();
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
        let _rebar = lock_rebar_gpu_test();
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
        let _rebar = lock_rebar_gpu_test();
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

    #[cfg(windows)]
    #[test]
    fn rebar_loose_budget_is_isolated_and_releases_on_drop() {
        let _rebar = REBAR_GPU_TEST.lock().unwrap_or_else(|e| e.into_inner());
        let _retire = crate::host::isolated_retire_scope();
        let Some(h) = host() else {
            return;
        };
        let before_shadow = rebar_shadow_bytes();
        let before_loose = rebar_loose_bytes();

        // Verify drop accounting isolates `REBAR_LOOSE_BYTES` from `REBAR_SHADOW_BYTES`.
        let (buf_a, ptr_a) = h.shared_buffer(4096).expect("shared_buffer a");
        let (buf_b, ptr_b) = h.shared_buffer(4096).expect("shared_buffer b");
        // SAFETY: live D3D12 buffers.
        let gpu_a = unsafe { buf_a.GetGPUVirtualAddress() };
        // SAFETY: live D3D12 buffers.
        let gpu_b = unsafe { buf_b.GetGPUVirtualAddress() };

        assert!(try_reserve_rebar(
            &REBAR_LOOSE_BYTES,
            4096,
            before_loose + 4096
        ));
        assert_eq!(rebar_loose_bytes(), before_loose + 4096);
        assert_eq!(rebar_shadow_bytes(), before_shadow);

        let loose_shadow = Shadow::new_loose(buf_a, ptr_a, gpu_a, 0x2000_0000, 4096, Some(4096));
        assert!(loose_shadow.is_rebar());
        assert!(loose_shadow.rebar.as_ref().is_some_and(|r| r.is_loose));

        assert!(try_reserve_rebar(
            &REBAR_SHADOW_BYTES,
            8192,
            before_shadow + 8192
        ));
        let reg_shadow = Shadow::new(buf_b, ptr_b, gpu_b, 0x3000_0000, 8192, Some(8192));
        assert!(reg_shadow.is_rebar());
        assert!(reg_shadow.rebar.as_ref().is_some_and(|r| !r.is_loose));
        assert_eq!(rebar_loose_bytes(), before_loose + 4096);
        assert_eq!(rebar_shadow_bytes(), before_shadow + 8192);

        drop(loose_shadow);
        assert_eq!(rebar_loose_bytes(), before_loose);
        assert_eq!(rebar_shadow_bytes(), before_shadow + 8192);

        drop(reg_shadow);
        assert_eq!(rebar_loose_bytes(), before_loose);
        assert_eq!(rebar_shadow_bytes(), before_shadow);

        // Exercise live `loose_in` allocation and eviction under the host's ReBAR loose budget.
        // Because `try_rebar_shadow_buffer` can still fall back to shared memory under VRAM
        // pressure or D3D12 allocation failure, derive per-window counter expectations from
        // the backend each `loose_in` call actually returned.
        let check_window = |shadow: &Shadow| -> bool {
            let is_rebar = shadow.is_rebar();
            if is_rebar {
                let r = shadow.rebar.as_ref().expect("rebar state");
                assert!(r.is_loose);
                assert!(r.mirror.lock().unwrap().is_none());
                assert!(r.hashes.lock().unwrap().is_none());
                assert_eq!(rebar_loose_bytes(), before_loose + WINDOW);
            } else {
                assert!(shadow.rebar.is_none());
                assert_eq!(rebar_loose_bytes(), before_loose);
            }
            assert_eq!(rebar_shadow_bytes(), before_shadow);
            is_rebar
        };

        let cache = RwLock::new(LooseCache::new());
        let base = 0x2000_0000_0000_u64;
        let avail_loose = h.rebar_loose_budget().saturating_sub(before_loose);
        if avail_loose >= WINDOW {
            let s0 = loose_in(&cache, base, base + 64, WINDOW).expect("loose win 0");
            let s0_rebar = check_window(&s0);
            drop(s0);

            // Inserting a second window evicts the first idle window and releases its ReBAR bytes.
            // Because `loose_in` reserves ReBAR before eviction runs, `s1` cannot acquire ReBAR
            // when `s0` already consumed the only available window in the loose budget (e.g.
            // `VICTORY_REBAR_LOOSE_MB=4`); evicting `s0` returns `rebar_loose_bytes()` to
            // `before_loose`.
            let s1 =
                loose_in(&cache, base + WINDOW, base + WINDOW + 64, WINDOW).expect("loose win 1");
            if s0_rebar && avail_loose < 2 * WINDOW {
                assert!(
                    !s1.is_rebar(),
                    "single-window loose budget must reject concurrent second ReBAR reservation"
                );
            }
            let s1_rebar = check_window(&s1);
            drop(s1);
            if !s1_rebar {
                // Evicting `s0` freed any single-window ReBAR reservation, so a third window can
                // reserve ReBAR again when VRAM headroom permits.
                let s2 = loose_in(&cache, base + 2 * WINDOW, base + 2 * WINDOW + 64, WINDOW)
                    .expect("loose win 2");
                let _ = check_window(&s2);
                drop(s2);
            }
            drop(cache);
            assert_eq!(rebar_loose_bytes(), before_loose);
            assert_eq!(rebar_shadow_bytes(), before_shadow);
        } else {
            // Opt-out (`VICTORY_REBAR_LOOSE_MB=0`) or insufficient remaining loose budget:
            // `loose_in` cleanly falls back to system RAM with zero ReBAR loose accounting.
            let s0 = loose_in(&cache, base, base + 64, WINDOW).expect("loose win fallback");
            assert!(!s0.is_rebar());
            let _ = check_window(&s0);
            drop(s0);
            drop(cache);
            assert_eq!(rebar_loose_bytes(), before_loose);
            assert_eq!(rebar_shadow_bytes(), before_shadow);
        }
    }

    #[cfg(windows)]
    #[test]
    fn rebar_mark_dirty_spans_restores_failed_readback_pieces() {
        let _rebar = lock_rebar_gpu_test();
        let _retire = crate::host::isolated_retire_scope();
        let Some(h) = host() else {
            return;
        };
        let len = (3 * PAGE) as u64;
        let (buf, ptr) = h.shared_buffer(len).expect("shared buffer");
        // SAFETY: live D3D12 buffer.
        let gpu = unsafe { buf.GetGPUVirtualAddress() };
        let shadow = Shadow::new_rebar_from_parts_for_test(buf, ptr, gpu, 0x4000_0000, len);
        // Seed mirror so `prepare_rebar_hashes` initializes the hash UAV & dirty bitset.
        assert!(shadow.writing(0x4000_0000, PAGE, |_, n| Some(vec![0u8; n as usize])));
        let initial = shadow
            .prepare_rebar_hashes(|n| h.default_buffer(n))
            .expect("initial rebar hashes");
        assert_eq!(initial.pre_runs, vec![(0, 3)]);
        let clean = shadow
            .prepare_rebar_hashes(|n| h.default_buffer(n))
            .expect("clean rebar hashes");
        assert!(clean.pre_runs.is_empty());

        // Simulate readback failure restoring dirty spans across piece 0 and piece 2.
        shadow.mark_rebar_dirty_spans(&[(16, 64), (2 * PAGE + 8, 2 * PAGE + 32)]);
        let restored = shadow
            .prepare_rebar_hashes(|n| h.default_buffer(n))
            .expect("restored rebar hashes");
        assert_eq!(restored.pre_runs, vec![(0, 1), (2, 1)]);
        shadow.clear_pending_rebar();
        drop(shadow);
    }

    #[cfg(windows)]
    #[test]
    fn rebar_failed_copy_back_keeps_pending_and_preserves_gpu_bytes_on_refresh() {
        let _rebar = lock_rebar_gpu_test();
        let _retire = crate::host::isolated_retire_scope();
        let Some(h) = host() else {
            return;
        };
        let title = vec![0x11u8; 4096];
        let start = title.as_ptr() as u64;
        let len = title.len() as u64;
        let (buf, ptr) = h.shared_buffer(len).expect("shared buffer");
        // SAFETY: live D3D12 buffer.
        let gpu = unsafe { buf.GetGPUVirtualAddress() };
        let shadow = Shadow::new_rebar_from_parts_for_test(buf, ptr, gpu, start, len);
        // Simulate an unwatched buffer whose GPU bytes were written to `[64..128)`.
        shadow.watched.store(NOT_WATCHED, Ordering::Relaxed);
        // SAFETY: `ptr` is mapped for `len` bytes of shared RAM in this test.
        unsafe {
            std::ptr::write_bytes(ptr, 0x22, len as usize);
            std::ptr::write_bytes(ptr.add(64), 0xAA, 64);
        }

        // If initial mirror readback fails, `writing` returns `false`, `copy_back_rebar`
        // returns `None`, and `pending` stays intact so a later unwatched refresh does
        // not overwrite `[64..128)` with stale title bytes (`0x11`).
        assert!(!shadow.writing(start as usize + 64, 64, |_, _| None));
        assert!(shadow.has_pending());
        assert!(
            shadow
                .copy_back_rebar(&[(64, 128)], &[&[0xAA; 64]])
                .is_none()
        );
        assert!(shadow.has_pending());

        shadow.update(start as usize, len as usize);
        let bytes = shadow.read_test_bytes(start, len as usize);
        assert!(bytes[..64].iter().all(|&b| b == 0x11));
        assert!(
            bytes[64..128].iter().all(|&b| b == 0xAA),
            "pending GPU result must survive unwatched refresh when copy-back failed"
        );
        assert!(bytes[128..].iter().all(|&b| b == 0x11));
    }

    #[cfg(windows)]
    #[test]
    fn rebar_shadow_gpu_write_round_trips_and_preserves_concurrent_cpu_writes() {
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COPY_DEST;

        let _rebar = lock_rebar_gpu_test();
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

    #[cfg(windows)]
    #[test]
    fn concurrent_find_reads_share_the_same_shadow_under_rwlock() {
        let _rebar = lock_rebar_gpu_test();
        let _retire = crate::host::isolated_retire_scope();
        if host().is_none() {
            return;
        }
        let data = vec![0u8; 8192];
        let start = data.as_ptr() as u64;
        let resource = 0xdead_beef_0006_usize;
        register(resource, start, data.len() as u64);
        let first = find(start, 64).expect("initial shadow");
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let addr = start + (i * 256) as u64;
                std::thread::spawn(move || find(addr, 64).expect("concurrent shadow"))
            })
            .collect();
        for h in handles {
            let got = h.join().expect("thread");
            assert!(Arc::ptr_eq(&first, &got));
        }
        drop(first);
        unregister(resource, start);
    }

    /// Never-copied pages (`copied == 0`) inside a range must still be
    /// reported as stale even when absent from the write-watch journal,
    /// and leading entries before the range must not stall the scan.
    #[test]
    fn stale_pages_include_uncopied_pages_absent_from_write_watch() {
        let base = 0x100usize;
        // Pages 0..4: page 0 copied at t=5, page 1 never copied (0), page 2
        // copied at t=5, page 3 copied at t=10.
        let copied = [5u64, 0, 5, 10];
        let a = base * PAGE;
        let e = (base + 4) * PAGE;
        // Write-watch includes a leading out-of-range page plus pages 2 and 3.
        let written = [(base - 1, 99u64), (base + 2, 7u64), (base + 3, 8u64)];
        assert_eq!(
            stale_pages_in(base, &copied, a, e, &written),
            vec![base + 1, base + 2]
        );
        // Even with an empty write-watch delta, uncopied page 1 is still stale.
        assert_eq!(stale_pages_in(base, &copied, a, e, &[]), vec![base + 1]);
        // Once all pages are copied and clean, nothing is stale.
        assert!(stale_pages_in(base, &[10, 10, 10, 10], a, e, &written).is_empty());
    }

    #[test]
    fn loose_budget_scales_across_ram_tiers() {
        assert_eq!(loose_budget_for_ram(None), 256 * MIB);
        assert_eq!(loose_budget_for_ram(Some(8 * GIB)), 128 * MIB);
        assert_eq!(loose_budget_for_ram(Some(16 * GIB)), 128 * MIB);
        assert_eq!(loose_budget_for_ram(Some(16 * GIB + 1)), 256 * MIB);
        assert_eq!(loose_budget_for_ram(Some(32 * GIB)), 256 * MIB);
        assert_eq!(loose_budget_for_ram(Some(32 * GIB + 1)), 512 * MIB);
        assert_eq!(loose_budget_for_ram(Some(64 * GIB)), 512 * MIB);
    }

    #[cfg(windows)]
    #[test]
    fn loose_evicts_lru_idle_windows_over_budget_and_skips_in_use_or_pending() {
        let _rebar = lock_rebar_gpu_test();
        let _retire = crate::host::isolated_retire_scope();
        let Some(host) = host() else {
            return;
        };
        let cache = RwLock::new(LooseCache::new());
        let budget = 2 * WINDOW;
        let base = 0x1000_0000_0000_u64;
        let win = |i: u64| base + i * WINDOW;

        let weak0 = Arc::downgrade(&loose_in(&cache, win(0), win(0) + 64, budget).expect("win 0"));
        let weak1 = Arc::downgrade(&loose_in(&cache, win(1), win(1) + 64, budget).expect("win 1"));
        assert_eq!(weak0.strong_count(), 1);
        assert_eq!(weak1.strong_count(), 1);

        // Touch window 0 so window 1 becomes the least-recently-used entry.
        drop(loose_in(&cache, win(0), win(0) + 64, budget).expect("touch win 0"));

        // Inserting window 2 exceeds the 2-window budget and immediately drops
        // idle LRU window 1 through `retire`.
        let weak2 = Arc::downgrade(&loose_in(&cache, win(2), win(2) + 64, budget).expect("win 2"));
        assert_eq!(weak1.strong_count(), 0, "idle LRU window 1 must be evicted");
        assert_eq!(weak0.strong_count(), 1, "MRU window 0 must stay cached");
        assert_eq!(weak2.strong_count(), 1, "new window 2 must stay cached");
        {
            let guard = cache.read().unwrap();
            assert_eq!(guard.total_bytes, budget);
            assert_eq!(guard.entries.len(), 2);
        }

        // With an active epoch, evicted window 0 leaves the cache immediately
        // but stays alive in the retirement queue until the epoch completes.
        let list = host.new_list().expect("host list");
        list.mark_work();
        let weak3 = Arc::downgrade(&loose_in(&cache, win(3), win(3) + 64, budget).expect("win 3"));
        {
            let guard = cache.read().unwrap();
            assert!(!guard.entries.contains_key(&(win(0), WINDOW)));
            assert_eq!(guard.total_bytes, budget);
        }
        assert_eq!(
            weak0.strong_count(),
            1,
            "evicted loose window must wait for active epochs"
        );
        list.close();
        list.reset();
        assert_eq!(
            weak0.strong_count(),
            0,
            "evicted loose window must drop once active epochs finish"
        );

        // Pin window 2 via an external Arc (strong_count > 1) and window 3 via
        // a pending GPU UAV copy-back range; neither may be evicted.
        let held2 = loose_in(&cache, win(2), win(2) + 64, budget).expect("hold win 2");
        {
            let s3 = loose_in(&cache, win(3), win(3) + 64, budget).expect("mark win 3 pending");
            s3.pending
                .lock()
                .unwrap()
                .push((win(3) as usize, win(3) as usize + 64));
        }
        let weak4 = Arc::downgrade(&loose_in(&cache, win(4), win(4) + 64, budget).expect("win 4"));
        assert_eq!(
            weak2.strong_count(),
            2,
            "in-use window 2 must not be evicted"
        );
        assert_eq!(
            weak3.strong_count(),
            1,
            "window 3 with pending copy-back must not be evicted"
        );
        assert_eq!(weak4.strong_count(), 1);
        {
            let guard = cache.read().unwrap();
            assert_eq!(guard.entries.len(), 3);
            assert_eq!(guard.total_bytes, 3 * WINDOW);
        }

        // Once window 2 is released and window 3's pending copy-back clears,
        // the next insert drains both stale windows back down to budget.
        drop(held2);
        weak3
            .upgrade()
            .expect("win 3 still live")
            .pending
            .lock()
            .unwrap()
            .clear();
        let weak5 = Arc::downgrade(&loose_in(&cache, win(5), win(5) + 64, budget).expect("win 5"));
        assert_eq!(weak2.strong_count(), 0);
        assert_eq!(weak3.strong_count(), 0);
        assert_eq!(weak4.strong_count(), 1);
        assert_eq!(weak5.strong_count(), 1);
        {
            let guard = cache.read().unwrap();
            assert_eq!(guard.entries.len(), 2);
            assert_eq!(guard.total_bytes, budget);
        }

        // Re-accessing previously evicted window 1 cleanly recreates its shadow.
        let recreated =
            loose_in(&cache, win(1) + 128, win(1) + 256, budget).expect("recreate win 1");
        assert_eq!(recreated.start, win(1));
        assert_eq!(recreated.len, WINDOW);
        assert_ne!(recreated.gpu(win(1) + 128), 0);
        assert_eq!(weak4.strong_count(), 0);
        assert_eq!(weak5.strong_count(), 1);
        drop(recreated);
        {
            let guard = cache.read().unwrap();
            assert_eq!(guard.entries.len(), 2);
            assert_eq!(guard.total_bytes, budget);
            assert!(guard.entries.contains_key(&(win(1), WINDOW)));
        }
    }

    #[cfg(windows)]
    #[test]
    fn reregistering_a_title_buffer_retires_its_old_shadow_by_epoch() {
        let _rebar = lock_rebar_gpu_test();
        let _retire = crate::host::isolated_retire_scope();
        let Some(h) = host() else {
            return;
        };
        const LEN: u64 = 2 << 20;
        let resource = 0xdead_beef_0004usize;
        let start = 0x79f0_0000_0000u64;

        register(resource, start, LEN);
        let first = find(start, 1).expect("shadow created for registered buffer");
        assert_eq!(first.left(start + 0x10_0000), LEN - 0x10_0000);
        let weak = Arc::downgrade(&first);
        let born = first.born_seq;
        drop(first);

        let list = h.new_list().expect("host list");
        list.mark_work();
        assert!(list.has_epoch());

        // Re-register at the exact same address and resource while `list` holds an epoch.
        register(resource, start, LEN);
        assert_eq!(
            weak.strong_count(),
            1,
            "replaced Shadow (born_seq={born}) must be retired, not dropped while an active list epoch may name it"
        );

        list.close();
        list.reset();
        assert_eq!(
            weak.strong_count(),
            0,
            "retired Shadow must drop once its blocking epoch finishes"
        );

        unregister(resource, start);
    }

    #[test]
    fn wide_copy_is_default_on_multicore_and_can_be_overridden() {
        assert!(wide_copy_enabled(None, Some(4)));
        assert!(wide_copy_enabled(None, Some(16)));
        assert!(!wide_copy_enabled(None, Some(2)));
        assert!(!wide_copy_enabled(None, None));
        assert!(wide_copy_enabled(Some("1"), Some(1)));
        assert!(!wide_copy_enabled(Some("0"), Some(16)));
        assert!(wide_copy_enabled(Some("auto"), Some(4)));
    }

    #[test]
    fn chunk_copy_segs_preserves_boundaries_and_mirror_offsets() {
        let segs = [
            CopySeg {
                src: 0x1000,
                dst: 0x2000,
                mirror: 0x3000,
                len: 700_000,
                stage_card: false,
            },
            CopySeg {
                src: 0x1000 + 800_000,
                dst: 0x2000 + 800_000,
                mirror: 0x3000 + 800_000,
                len: 123,
                stage_card: false,
            },
            CopySeg {
                src: 0x1000 + 900_000,
                dst: 0x2000 + 900_000,
                mirror: 0,
                len: 1_500_000,
                stage_card: false,
            },
        ];
        let buckets = chunk_copy_segs(&segs, 256 << 10);
        assert!(buckets.len() >= 8);
        let mut flat: Vec<CopySeg> = Vec::new();
        for bucket in &buckets {
            let sum: usize = bucket.iter().map(|s| s.len).sum();
            assert!(sum > 0 && sum <= 256 << 10);
            flat.extend_from_slice(bucket);
        }
        let mut merged: Vec<CopySeg> = Vec::new();
        for seg in flat {
            match merged.last_mut() {
                Some(last)
                    if last.src + last.len == seg.src
                        && last.dst + last.len == seg.dst
                        && ((last.mirror == 0 && seg.mirror == 0)
                            || (last.mirror != 0 && last.mirror + last.len == seg.mirror)) =>
                {
                    last.len += seg.len;
                }
                _ => merged.push(seg),
            }
        }
        assert_eq!(merged, segs);
    }

    #[test]
    fn parallel_shadow_update_preserves_boundaries_unmapped_pages_and_pending_gaps() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        let Some(h) = host() else {
            return;
        };
        const MEM_TOP_DOWN: u32 = 0x10_0000;
        const LEN: usize = 4 << 20; // 4 MiB (> WIDE_COPY_MIN)
        // SAFETY: reserve a 4 MiB address range and commit parts around a hole.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                LEN,
                MEM_RESERVE | MEM_TOP_DOWN,
                PAGE_NOACCESS,
            )
            .cast::<u8>()
        };
        assert!(!base.is_null());
        let mut expected = vec![0u8; LEN];
        // Leave [1 MiB, 1 MiB + 128 KiB) unmapped (hole) and keep [2 MiB, 2 MiB + 128 KiB)
        // as a pending shader-write range that `update` must skip completely.
        let hole = (1 << 20, (1 << 20) + 2 * PAGE);
        let pending = (2 << 20, (2 << 20) + 2 * PAGE);
        for &(from, to) in &[(0usize, hole.0), (hole.1, LEN)] {
            // SAFETY: committing subranges inside our reserved allocation.
            let p = unsafe {
                VirtualAlloc(base.add(from).cast(), to - from, MEM_COMMIT, PAGE_READWRITE)
            };
            assert_eq!(p, base.wrapping_add(from).cast());
            for (off, dst_byte) in expected.iter_mut().enumerate().take(to).skip(from) {
                let v = ((off.wrapping_mul(31) ^ (off >> 12)) & 0xff) as u8;
                // SAFETY: committed above.
                unsafe { *base.add(off) = v };
                *dst_byte = v;
            }
        }

        let Some((buffer, data)) = h.shared_buffer(LEN as u64) else {
            // SAFETY: freeing our test allocation.
            unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
            return;
        };
        // Pre-fill shadow staging with a sentinel so untouched regions are verifiable.
        // SAFETY: `data` maps `LEN` bytes.
        unsafe { std::ptr::write_bytes(data, 0xA5, LEN) };
        // SAFETY: live D3D12 buffer.
        let gpu_va = unsafe { buffer.GetGPUVirtualAddress() };
        let shadow = Shadow::new(buffer, data, gpu_va, base as u64, LEN as u64, None);
        shadow
            .pending
            .lock()
            .unwrap()
            .push((base as usize + pending.0, base as usize + pending.1));

        // Refresh [37, LEN - 41): tests unaligned ends, hole zeroing, pending gap,
        // and multi-chunk parallel copy.
        let req_lo = 37usize;
        let req_hi = LEN - 41;
        let copied = shadow.update(base as usize + req_lo, req_hi - req_lo);
        assert_eq!(copied, (pending.0 - req_lo) + (req_hi - pending.1));

        // SAFETY: `data` maps `LEN` bytes and `update` has finished all workers.
        let got = unsafe { std::slice::from_raw_parts(data, LEN) };
        assert!(
            got[..req_lo].iter().all(|&b| b == 0xA5),
            "bytes before refreshed start must stay untouched"
        );
        assert!(
            got[req_hi..].iter().all(|&b| b == 0xA5),
            "bytes after refreshed end must stay untouched"
        );
        assert!(
            got[pending.0..pending.1].iter().all(|&b| b == 0xA5),
            "pending shader-write range must stay untouched"
        );
        assert_eq!(got[req_lo..hole.0], expected[req_lo..hole.0]);
        assert!(
            got[hole.0..hole.1].iter().all(|&b| b == 0),
            "unmapped hole must be zeroed"
        );
        assert_eq!(got[hole.1..pending.0], expected[hole.1..pending.0]);
        assert_eq!(got[pending.1..req_hi], expected[pending.1..req_hi]);

        // SAFETY: freeing our test allocation.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }

    #[test]
    fn parallel_shadow_update_rebar_mirror_syncs_mirror_and_vram() {
        let Some(h) = host() else {
            return;
        };
        const LEN: usize = 3 << 20; // 3 MiB (> WIDE_COPY_MIN)
        let title: Vec<u8> = (0..LEN)
            .map(|i| ((i.wrapping_mul(17) ^ (i >> 8)) & 0xff) as u8)
            .collect();
        let start = title.as_ptr() as u64;
        let Some((buffer, data)) = h.shared_buffer(LEN as u64) else {
            return;
        };
        // SAFETY: `data` maps `LEN` bytes.
        unsafe { std::ptr::write_bytes(data, 0x5A, LEN) };
        // SAFETY: live D3D12 buffer.
        let gpu_va = unsafe { buffer.GetGPUVirtualAddress() };
        let shadow = Shadow::new(buffer, data, gpu_va, start, LEN as u64, Some(0));
        {
            let rebar = shadow.rebar.as_ref().expect("rebar state");
            *rebar.mirror.lock().unwrap() = Some(vec![0x5Au8; LEN]);
        }

        // Disjoint runs totaling > 2 MiB that straddle chunk cuts.
        let runs = [
            (start as usize + 19, start as usize + (1 << 20) + 123),
            (start as usize + (1 << 20) + PAGE, start as usize + LEN - 29),
        ];
        let copied = shadow.copy_runs(&runs);
        let expected_copied: usize = runs.iter().map(|&(a, b)| b - a).sum();
        assert_eq!(copied, expected_copied);

        let mirror = shadow
            .rebar
            .as_ref()
            .unwrap()
            .mirror
            .lock()
            .unwrap()
            .clone()
            .unwrap();
        // SAFETY: `data` maps `LEN` bytes.
        let vram = unsafe { std::slice::from_raw_parts(data, LEN) };
        let r0 = (runs[0].0 - start as usize, runs[0].1 - start as usize);
        let r1 = (runs[1].0 - start as usize, runs[1].1 - start as usize);
        assert_eq!(mirror[r0.0..r0.1], title[r0.0..r0.1]);
        assert_eq!(vram[r0.0..r0.1], title[r0.0..r0.1]);
        assert_eq!(mirror[r1.0..r1.1], title[r1.0..r1.1]);
        assert_eq!(vram[r1.0..r1.1], title[r1.0..r1.1]);
        assert!(mirror[..r0.0].iter().all(|&b| b == 0x5A));
        assert!(vram[..r0.0].iter().all(|&b| b == 0x5A));
        assert!(mirror[r0.1..r1.0].iter().all(|&b| b == 0x5A));
        assert!(vram[r0.1..r1.0].iter().all(|&b| b == 0x5A));
        assert!(mirror[r1.1..].iter().all(|&b| b == 0x5A));
        assert!(vram[r1.1..].iter().all(|&b| b == 0x5A));
    }

    #[test]
    fn wide_refresh_reentrancy_and_batch_do_not_deadlock() {
        let Some(h) = host() else {
            return;
        };
        const BIG: usize = 2 << 20;
        const SMALL: usize = 64 << 10;
        let mut backing: Vec<Vec<u8>> = Vec::new();
        let mut shadows: Vec<Arc<Shadow>> = Vec::new();
        for i in 0..8usize {
            let len = if i < 4 { BIG } else { SMALL };
            let buf: Vec<u8> = (0..len)
                .map(|j| (((i + 1) * 29 + j * 7 + (j >> 9)) & 0xff) as u8)
                .collect();
            let start = buf.as_ptr() as u64;
            let Some((res, data)) = h.shared_buffer(len as u64) else {
                return;
            };
            // SAFETY: live D3D12 buffer.
            let gpu_va = unsafe { res.GetGPUVirtualAddress() };
            let shadow = Arc::new(Shadow::new(res, data, gpu_va, start, len as u64, None));
            backing.push(buf);
            shadows.push(shadow);
        }

        // 1. Direct re-entrancy stress: invoke multi-MB `Shadow::update` from
        // inside `wide` workers concurrently. Must never deadlock `workers()`.
        let jobs: Vec<Job> = shadows
            .iter()
            .map(|s| {
                let s = s.clone();
                Box::new(move || {
                    let n = s.update(s.start as usize, s.len as usize);
                    assert_eq!(n, s.len as usize);
                }) as Job
            })
            .collect();
        wide(jobs);
        for (shadow, src) in shadows.iter().zip(&backing) {
            // SAFETY: `shadow.data` maps `shadow.len` bytes.
            let got = unsafe { std::slice::from_raw_parts(shadow.data, src.len()) };
            assert_eq!(got, src.as_slice());
        }

        // 2. Two-phase `refresh_batch` mixing big + small shadows (parallel plan
        // + single flat parallel copy pass).
        for shadow in &shadows {
            // SAFETY: clear staging before batch refresh.
            unsafe { std::ptr::write_bytes(shadow.data, 0, shadow.len as usize) };
        }
        let batch_items: Vec<(Arc<Shadow>, usize, usize)> = shadows
            .iter()
            .map(|s| (s.clone(), s.start as usize, (s.start + s.len) as usize))
            .collect();
        let expected_total: usize = backing.iter().map(Vec::len).sum();
        assert_eq!(refresh_batch(batch_items, true), (expected_total, true));
        for (shadow, src) in shadows.iter().zip(&backing) {
            // SAFETY: `shadow.data` maps `shadow.len` bytes.
            let got = unsafe { std::slice::from_raw_parts(shadow.data, src.len()) };
            assert_eq!(got, src.as_slice());
        }

        // 3. `copy_mapped_wide` splits a multi-MB `Step::Fill` cleanly.
        let mut wide_dst = vec![0u8; BIG];
        assert!(copy_mapped_wide_checked(
            wide_dst.as_mut_ptr(),
            backing[0].as_ptr(),
            BIG,
        ));
        assert_eq!(wide_dst, backing[0]);
    }

    #[test]
    fn card_limits_scales_default_budget_for_high_vram_and_honors_override() {
        const MIB: u64 = 1 << 20;
        const GIB: u64 = 1 << 30;
        assert_eq!(
            card_limits_for(None, None, Some(8 * GIB)),
            (MIB, 1024 * MIB)
        );
        assert_eq!(
            card_limits_for(None, None, Some(16 * GIB)),
            (MIB, 1024 * MIB)
        );
        assert_eq!(
            card_limits_for(None, None, Some(24 * GIB)),
            (MIB, 2048 * MIB)
        );
        assert_eq!(
            card_limits_for(None, None, Some(32 * GIB)),
            (MIB, 2048 * MIB)
        );
        assert_eq!(card_limits_for(None, None, None), (MIB, 1024 * MIB));
        assert_eq!(
            card_limits_for(Some(16), Some(1536), Some(24 * GIB)),
            (16 * MIB, 1536 * MIB)
        );
        assert_eq!(card_limits_for(None, Some(0), Some(24 * GIB)), (MIB, 0));
    }

    #[test]
    fn card_piece_runs_and_inflight_slot_math_agree() {
        let joined_runs = joined_piece_runs(vec![
            CardSpan {
                first: 5,
                skew: 128,
                pieces: 2,
                from: 0,
                to: 100,
            },
            CardSpan {
                first: 7,
                skew: 128,
                pieces: 3,
                from: 100,
                to: 300,
            },
            CardSpan {
                first: 1,
                skew: 128,
                pieces: 1,
                from: 0,
                to: 10,
            },
        ]);
        assert_eq!(joined_runs.len(), 2);
        assert_eq!((joined_runs[0].first, joined_runs[0].pieces), (1, 1));
        assert_eq!((joined_runs[1].first, joined_runs[1].pieces), (5, 5));

        let start = 0x1000_0005usize;
        let len = 256 << 10;
        let span = card_span_of(start, 0x10, 0x3_0020);
        for i in 0..span.pieces {
            let (lo, hi) = card_piece_of(start, len, &span, i);
            let from = crate::host::inflight_piece_from(&span, i);
            assert!(lo >= from && hi > lo);
            assert!((hi - from) as u64 <= crate::host::INFLIGHT_SLOT);
        }

        assert_eq!(crate::host::inflight_slots(1), 1);
        assert_eq!(crate::host::inflight_slots(64 << 20), 1024);
        assert_eq!(crate::host::inflight_slots(63 << 20), 512);
        assert!(crate::host::inflight_gets_slot(u32::MAX, u32::MAX - 2, 4));
        assert!(crate::host::inflight_gets_slot(0, u32::MAX - 2, 4));
        assert!(!crate::host::inflight_gets_slot(1, u32::MAX - 2, 4));
        assert_eq!(crate::host::inflight_raise(u32::MAX, u32::MAX), 0);
        assert_eq!(crate::host::inflight_raise(5, 3), 5);
        assert_eq!(crate::host::inflight_raise(5, 5), 6);
        assert_eq!(
            outside(10, 50, vec![(20, 30), (25, 35)]),
            [(10, 20), (35, 50)]
        );
    }

    #[test]
    fn first_and_last_difference_find_exact_edges() {
        let mut a = vec![0x5au8; 257];
        let mut b = a.clone();
        assert_eq!(first_difference(&a, &b), None);
        assert_eq!(last_difference(&a, &b), None);
        b[19] = 0x11;
        b[213] = 0x22;
        assert_eq!(first_difference(&a, &b), Some(19));
        assert_eq!(last_difference(&a, &b), Some(213));
        a[0] = 0x99;
        assert_eq!(first_difference(&a, &b), Some(0));
    }

    #[test]
    fn parts_written_cuts_64k_pages_into_4k_runs() {
        const PART: usize = 4 << 10;
        let base = 0x2000_0000usize;
        let bits = (1u16 << 1) | (1u16 << 2) | (1u16 << 15);
        let mut runs = Vec::new();
        let mut mask = bits;
        let mut i = 0usize;
        while mask != 0 {
            let start = mask.trailing_zeros() as usize;
            let shifted = mask >> start;
            let len = (!shifted).trailing_zeros() as usize;
            runs.push((base + (i + start) * PART, base + (i + start + len) * PART));
            i += start + len;
            mask = if start + len >= 16 { 0 } else { shifted >> len };
        }
        assert_eq!(
            runs,
            [
                (base + PART, base + 3 * PART),
                (base + 15 * PART, base + 16 * PART)
            ]
        );
    }

    #[test]
    fn relay_merge_rebar_merges_gpu_writes_without_touching_concurrent_cpu_writes() {
        let mut title = vec![0x10u8; 4096];
        let Some(shadow) = Shadow::new_rebar_for_test(title.as_mut_ptr() as u64, 4096, 0x10) else {
            return;
        };
        // Simulate concurrent CPU write to [100..200) and GPU write to [300..400).
        title[100..200].fill(0xaa);
        let mut gpu = vec![0x10u8; 4096];
        gpu[300..400].fill(0xbb);
        let written = shadow
            .relay_merge_rebar(&[(0, 4096)], &[&gpu])
            .expect("relay_merge_rebar");
        assert_eq!(written, 100);
        assert!(title[100..200].iter().all(|&b| b == 0xaa));
        assert!(title[300..400].iter().all(|&b| b == 0xbb));
        assert!(title[..100].iter().all(|&b| b == 0x10));
    }

    #[test]
    fn relay_merge_rebar_parallel_scan_merges_large_buffer_across_chunks() {
        const N: usize = 8 << 20;
        let mut title = vec![0x22u8; N];
        let Some(shadow) = Shadow::new_rebar_for_test(title.as_mut_ptr() as u64, N as u64, 0x22)
        else {
            return;
        };
        let mut gpu = vec![0x22u8; N];
        // Dirty stripes across multiple 1 MiB chunks.
        for chunk in 0..8 {
            let at = chunk * (1 << 20) + 1234;
            gpu[at..at + 512].fill((0x30 + chunk) as u8);
        }
        title[500..600].fill(0xee);
        let written = shadow
            .relay_merge_rebar(&[(0, N)], &[&gpu])
            .expect("relay_merge_rebar parallel");
        assert_eq!(written, 8 * 512);
        assert!(title[500..600].iter().all(|&b| b == 0xee));
        for chunk in 0..8 {
            let at = chunk * (1 << 20) + 1234;
            assert!(
                title[at..at + 512]
                    .iter()
                    .all(|&b| b == (0x30 + chunk) as u8)
            );
        }
    }

    #[test]
    fn title_writes_under_a_busy_range_are_copied_for_rebar_mirror() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        const N: usize = 128 << 10;
        // SAFETY: isolated write-watched allocation freed at end of test.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                N,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
            .cast::<u8>()
        };
        assert!(!base.is_null());
        // SAFETY: committed writable region of `N` bytes.
        unsafe { std::ptr::write_bytes(base, 0x41, N) };
        let Some(shadow) = Shadow::new_rebar_for_test(base as u64, N as u64, 0x41) else {
            // SAFETY: valid allocation from above.
            unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
            return;
        };
        // Clear initial write-watch bits, mark [0..N) busy under flight writes,
        // then mutate a slice on the CPU and merge it under writes.
        let _ = crate::watch::writes(base as usize, base as usize + N, false);
        shadow.writing(base as usize, N, |_, _| None);
        // SAFETY: committed writable slice.
        unsafe { std::ptr::write_bytes(base.add(4096), 0x77, 128) };
        let mut runs_left = 4096usize;
        let need_settle = shadow.merge_under_writes(base as usize, N, &mut runs_left);
        assert!(!need_settle);
        // ReBAR mirror should now contain 0x77 at [4096..4096+128).
        {
            let guard = shadow.rebar.as_ref().unwrap().mirror.lock().unwrap();
            let mirror = guard.as_ref().unwrap();
            assert!(mirror[4096..4096 + 128].iter().all(|&b| b == 0x77));
            assert!(mirror[..4096].iter().all(|&b| b == 0x41));
        }
        shadow.copy_back_done(&[(base as usize, base as usize + N)]);
        // SAFETY: valid allocation from above.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }

    #[test]
    fn card_shadow_test_helper_creates_and_tracks_dirty_ranges() {
        let _retire = crate::host::isolated_retire_scope();
        let mut backing = vec![0x55u8; 64 << 10];
        let mut side = vec![0x55u8; 64 << 10];
        let Some(shadow) = Shadow::new_card_for_test(
            side.as_mut_ptr(),
            backing.as_mut_ptr() as u64,
            64 << 10,
            vec![(0, 4096)],
            vec![(4096, 8192, 1)],
        ) else {
            return;
        };
        assert!(shadow.carded());
        let (staging, before, size) = shadow.card_sides().expect("card sides");
        assert!(before.is_none());
        assert_eq!(size, 0);
        assert_eq!(
            shadow
                .card
                .as_ref()
                .unwrap()
                .dirty
                .read()
                .unwrap()
                .as_slice(),
            &[(0, 4096)]
        );
        let _ = staging;
    }

    #[test]
    fn refresh_batch_materializes_deferred_lazy_block_instead_of_zeroing() {
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            PAGE_READWRITE, UnmapViewOfFile,
        };

        let Some(h) = host() else {
            return;
        };
        let pages = 2usize;
        let len = pages * PAGE;
        // SAFETY: anonymous section mapped as a `2 * PAGE` view so `lazy_back`
        // has a host-only `PAGE_READWRITE` own view.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                len as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: mapping `len` bytes from offset 0.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len) };
        assert!(!view.Value.is_null());
        let title = view.Value.cast::<u8>();
        let va0 = title as usize;
        crate::inplace::title_memory_section(section as usize, len as u64, true);
        for k in 0..pages {
            assert!(crate::inplace::title_memory_view(
                va0 + k * PAGE,
                PAGE,
                section as usize,
                (k * PAGE) as u64,
            ));
        }
        // SAFETY: committed above.
        unsafe { std::ptr::write_bytes(title, 0xA5, len) };

        let cleanup = || {
            crate::watch::title_memory_remapped(va0, len);
            // SAFETY: view and section allocated above.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
                CloseHandle(section);
            }
        };

        let mut side_a = vec![0xA5u8; len];
        let mut side_b = vec![0x00u8; len];
        let mut side_c = vec![0x00u8; len];
        let Some(shadow_a) =
            Shadow::new_card_sized_for_test(side_a.as_mut_ptr(), va0 as u64, len as u64)
        else {
            cleanup();
            return;
        };
        let Some(shadow_b) =
            Shadow::new_card_sized_for_test(side_b.as_mut_ptr(), va0 as u64, len as u64)
        else {
            drop(shadow_a);
            cleanup();
            return;
        };
        let Some(shadow_c) =
            Shadow::new_card_sized_for_test(side_c.as_mut_ptr(), va0 as u64, len as u64)
        else {
            drop(shadow_b);
            drop(shadow_a);
            cleanup();
            return;
        };

        // Seed shadow_a's GPU card buffer with `0xA5` across `len` and `0x5C` in the
        // first 64 bytes so materialization merges `0x5C` into title RAM when `VICTORY_LAZY_BACK=1`.
        let (up, up_ptr) = h.upload_buffer(len as u64).expect("upload");
        // SAFETY: mapped `len`-byte upload buffer.
        unsafe {
            std::ptr::write_bytes(up_ptr, 0xA5, len);
            std::ptr::write_bytes(up_ptr, 0x5C, 64);
        }
        let list = h.new_list().expect("seed list");
        // SAFETY: live buffers on open command list.
        unsafe {
            list.list
                .CopyBufferRegion(&shadow_a.buffer, 0, &up, 0, len as u64);
        }
        list.close();
        h.execute(&[&list]);
        list.reset();
        shadow_a.writing(va0, PAGE, |_, _| None);

        let deferred = crate::lazy_back::try_defer_page(&shadow_a, 0, PAGE, 0);
        if deferred {
            assert!(crate::lazy_back::has_block(va0 / PAGE));
        }

        // `refresh_batch` with >= 2 items exercises the flat `all_segs` +
        // `run_copy_segs_status` path across page 0 (`shadow_b`) and page 1 (`shadow_c`).
        let (copied, ok) = refresh_batch(
            vec![
                (shadow_b.clone(), va0, va0 + PAGE),
                (shadow_c.clone(), va0 + PAGE, va0 + 2 * PAGE),
            ],
            false,
        );
        assert!(ok && copied >= 2 * PAGE);
        assert!(!crate::lazy_back::has_block(va0 / PAGE));
        if deferred {
            assert_eq!(&side_b[..64], &[0x5Cu8; 64]);
        } else {
            assert_eq!(&side_b[..64], &[0xA5u8; 64]);
        }
        assert_eq!(&side_c[PAGE..PAGE + 64], &[0xA5u8; 64]);

        shadow_a.copy_back_done(&[(va0, va0 + PAGE)]);
        drop(shadow_c);
        drop(shadow_b);
        drop(shadow_a);
        cleanup();
    }

    #[test]
    fn materialize_rebar_spans_ignores_inverted_span_without_underflow() {
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            PAGE_READWRITE, UnmapViewOfFile,
        };

        let Some(_h) = host() else {
            return;
        };
        // SAFETY: anonymous 1-page section so `lazy_back` can actually arm a block
        // and `materialize_rebar_spans` executes its active span-filtering path.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: map 1 page.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        assert!(!view.Value.is_null());
        let ptr = view.Value.cast::<u8>();
        let va0 = ptr as usize;
        let block = va0 / PAGE;
        crate::lazy_back::clear_hot_for_test_pub(block);
        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        assert!(crate::inplace::title_memory_view(
            va0,
            PAGE,
            section as usize,
            0
        ));
        // SAFETY: seed memory.
        unsafe { std::ptr::write_bytes(ptr, 0x11, PAGE) };

        let shadow = Shadow::new_rebar_for_test(va0 as u64, PAGE as u64, 0x11)
            .expect("rebar test shadow must be created when host is present");
        assert!(
            crate::lazy_back::try_defer_page_for_test(&shadow, 0, PAGE, 0),
            "must arm a lazy block so `lazy_back::active()` is true"
        );
        assert!(
            crate::lazy_back::active(),
            "lazy_back must be active so materialize_rebar_spans runs its filter/map path"
        );
        // Inverted span `(8, 4)` must return Some(0) without panicking in overflow-checked builds.
        assert_eq!(shadow.relay_merge_rebar(&[(8, 4)], &[&[]]), Some(0));
        assert!(
            crate::lazy_back::has_block(block),
            "inverted span must be ignored without materializing unrelated block"
        );
        assert!(crate::lazy_back::materialize_range(va0, PAGE));
        assert!(!crate::lazy_back::has_block(block));
        drop(shadow);

        crate::lazy_back::clear_hot_for_test_pub(block);
        crate::watch::title_memory_remapped(va0, PAGE);
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
            CloseHandle(section);
        }
    }

    #[cfg(windows)]
    #[test]
    fn wide_rebar_copy_partial_failure_with_failed_retry_invalidates_written_hashes() {
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            PAGE_READWRITE, UnmapViewOfFile,
        };

        let _rebar = lock_rebar_gpu_test();
        let _retire = crate::host::isolated_retire_scope();
        let Some(h) = host() else {
            return;
        };
        if !wide_copy() {
            return;
        }
        let len = WIDE_COPY_MIN;
        let pages = len / PAGE;
        assert!(pages >= 16);
        // SAFETY: anonymous section of `WIDE_COPY_MIN` bytes (16 pages -> 4 parallel chunks).
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                len as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: map all 16 pages.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len) };
        assert!(!view.Value.is_null());
        let ptr = view.Value.cast::<u8>();
        let va0 = ptr as usize;
        let last_va = va0 + (pages - 1) * PAGE;
        let last_block = last_va / PAGE;
        crate::lazy_back::clear_hot_for_test_pub(last_block);
        crate::inplace::title_memory_section(section as usize, len as u64, true);
        assert!(crate::inplace::title_memory_view(
            va0,
            len,
            section as usize,
            0
        ));
        // SAFETY: `ptr` is mapped RW for `len` bytes.
        unsafe { std::ptr::write_bytes(ptr, 0x11, len) };

        if let Some(shadow) = Shadow::new_rebar_for_test(va0 as u64, len as u64, 0x11) {
            // Initialize cached ReBAR hashes and clear dirty bits so `pre_runs` starts empty.
            let initial = shadow
                .prepare_rebar_hashes(|n| h.default_buffer(n))
                .expect("initial rebar hashes");
            assert_eq!(initial.pre_runs, vec![(0, pages as u32)]);
            let clean = shadow
                .prepare_rebar_hashes(|n| h.default_buffer(n))
                .expect("clean rebar hashes");
            assert!(clean.pre_runs.is_empty());

            // Write new bytes to chunks 0..3 (pages 0..12) while deferring the last page
            // (page 15, inside chunk 3) via `lazy_back`.
            // SAFETY: pages `0..12` are writable mapped title memory.
            unsafe { std::ptr::write_bytes(ptr, 0x22, (pages - 4) * PAGE) };
            assert!(crate::lazy_back::try_defer_page_for_test(
                &shadow,
                (pages - 1) * PAGE,
                PAGE,
                0
            ));
            assert!(crate::lazy_back::has_block(last_block));

            // Fail BOTH pre-pass-1 materialization AND pre-retry materialization so chunk 3
            // fails on both attempts while chunks 0..2 succeed and write `0x22` into VRAM.
            crate::lazy_back::set_fail_next_fetches_for_test(2);
            let epochs: Vec<(usize, u64)> = (0..pages).map(|p| (va0 / PAGE + p, 42)).collect();
            let (copied, ok) = shadow.copy_runs_with_epochs(&[(va0, va0 + len)], &epochs);
            crate::lazy_back::set_fail_next_fetches_for_test(0);

            assert!(
                !ok,
                "failed segment must report ok == false so submission can abort"
            );
            assert_eq!(
                copied, 0,
                "failed segment must not be counted as fully copied"
            );
            // VRAM bytes for chunks 0..2 were already modified to `0x22`.
            let vram_p0 = shadow.read_test_bytes(va0 as u64, 64);
            assert_eq!(vram_p0, vec![0x22u8; 64]);

            // Cached ReBAR hashes MUST be invalidated for all chunks that wrote VRAM
            // even though the overall segment failed on both initial pass and retry.
            let after = shadow
                .prepare_rebar_hashes(|n| h.default_buffer(n))
                .expect("rebar hashes after partial wide copy failure");
            assert!(
                after
                    .pre_runs
                    .iter()
                    .any(|&(first, count)| first == 0 && count >= (pages - 4) as u32),
                "written chunks (pages 0..{}) must be marked dirty in ReBAR hashes, got {:?}",
                pages - 4,
                after.pre_runs
            );

            // Epochs for the failed segment must NOT have been committed.
            let committed = shadow.copied.lock().unwrap_or_else(|e| e.into_inner());
            assert!(
                committed.iter().all(|&ep| ep == 0),
                "epochs of a failed segment must stay uncommitted"
            );
            drop(committed);

            assert!(crate::lazy_back::materialize_range(last_va, PAGE));
            assert!(!crate::lazy_back::has_block(last_block));
            drop(shadow);
        }

        crate::lazy_back::clear_hot_for_test_pub(last_block);
        crate::watch::title_memory_remapped(va0, len);
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
            CloseHandle(section);
        }
    }

    #[test]
    fn collect_stale_does_not_acknowledge_excluded_lazy_owned_cpu_write() {
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            PAGE_READWRITE, UnmapViewOfFile,
        };

        let Some(h) = host() else {
            return;
        };
        // SAFETY: section-backed page so `lazy_back` can defer it.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: map 1 page.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        assert!(!view.Value.is_null());
        let ptr = view.Value.cast::<u8>();
        let va0 = ptr as usize;
        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        assert!(crate::inplace::title_memory_view(
            va0,
            PAGE,
            section as usize,
            0
        ));
        // SAFETY: seed memory.
        unsafe { std::ptr::write_bytes(ptr, 0x11, PAGE) };

        let mut side = vec![0x11u8; PAGE];
        if let Some(shadow) =
            Shadow::new_card_sized_for_test(side.as_mut_ptr(), va0 as u64, PAGE as u64)
        {
            // Initial refresh marks page copied.
            let _ = shadow.update(va0, PAGE);
            shadow.clear_card_dirty_for_test();
            // Seed GPU buffer with a change at offset 0 (different 4 KiB part from CPU write).
            let (up, up_ptr) = h.upload_buffer(PAGE as u64).expect("upload");
            // SAFETY: mapped upload buffer.
            unsafe {
                std::ptr::write_bytes(up_ptr, 0x11, PAGE);
                std::ptr::write_bytes(up_ptr, 0xAA, 16);
            }
            let list = h.new_list().expect("list");
            // SAFETY: live buffers.
            unsafe {
                list.list
                    .CopyBufferRegion(&shadow.buffer, 0, &up, 0, PAGE as u64);
            }
            list.close();
            h.execute(&[&list]);
            list.reset();

            // 1. Clean lazy-owned page stays deferred with zero DMA on `shadow.update`.
            assert!(crate::lazy_back::try_defer_page_for_test(
                &shadow, 0, PAGE, 0
            ));
            assert_eq!(shadow.update(va0, PAGE), 0);
            assert!(
                crate::lazy_back::has_block(va0 / PAGE),
                "clean lazy-owned page must remain deferred"
            );
            assert!(crate::lazy_back::materialize_range(va0, PAGE));
            // Re-establish a clean guarded baseline after materialization, then
            // write a new GPU byte at offset 0 before the CPU writes offset 8192.
            let _ = shadow.update(va0, PAGE);
            shadow.clear_card_dirty_for_test();
            // SAFETY: mapped upload buffer.
            unsafe {
                std::ptr::write_bytes(up_ptr, 0xBB, 16);
            }
            // SAFETY: live buffers.
            unsafe {
                list.list
                    .CopyBufferRegion(&shadow.buffer, 0, &up, 0, PAGE as u64);
            }
            list.close();
            h.execute(&[&list]);
            list.reset();

            // 2. Title writes a CPU byte in a different 4 KiB part (offset 8192) before defer.
            // SAFETY: committed view memory.
            unsafe { ptr.add(8192).write(0x77) };

            crate::lazy_back::clear_hot_for_test_pub(va0 / PAGE);
            assert!(crate::lazy_back::try_defer_page_for_test(
                &shadow, 0, PAGE, 0
            ));
            // 2a. If materialization fails on the dirty lazy-owned page, `shadow.update`
            //     returns 0 and must NOT acknowledge the pending CPU write epoch.
            crate::lazy_back::set_fail_next_fetches_for_test(1);
            assert_eq!(shadow.update(va0, PAGE), 0);
            assert!(
                crate::lazy_back::has_block(va0 / PAGE),
                "failed materialization must leave dirty lazy page armed"
            );
            // 2b. Once DMA succeeds, `shadow.update` materializes and replans the
            //     dirty lazy-owned page immediately so the current list sees `0x77`
            //     while preserving GPU's `0xBB` at offset 0.
            assert!(shadow.update(va0, PAGE) > 0);
            assert!(!crate::lazy_back::has_block(va0 / PAGE));
            assert_eq!(side[8192], 0x77);
            assert_eq!(&side[..16], &[0xBB; 16]);
            drop(shadow);
        }

        crate::watch::title_memory_remapped(va0, PAGE);
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
            CloseHandle(section);
        }
    }

    #[test]
    fn evict_over_budget_drains_and_evicts_idle_deferred_windows() {
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            PAGE_READWRITE, UnmapViewOfFile,
        };

        let Some(h) = host() else {
            return;
        };
        let _guard = isolated_loose_budget(WINDOW);
        let total_len = (2 * WINDOW) as usize;
        // SAFETY: 2-window anonymous section mapped for loose windows.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                total_len as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: map 2 windows.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, total_len) };
        assert!(!view.Value.is_null());
        let base = view.Value as usize;
        // Align to `WINDOW` boundary inside the mapping if needed.
        let w0 = (base as u64).next_multiple_of(WINDOW) as usize;
        let w1 = w0 + WINDOW as usize;
        if w1 + PAGE <= base + total_len {
            crate::inplace::title_memory_section(section as usize, total_len as u64, true);
            assert!(crate::inplace::title_memory_view(
                w0,
                PAGE,
                section as usize,
                (w0 - base) as u64
            ));
            // SAFETY: seed page.
            unsafe { std::ptr::write_bytes(w0 as *mut u8, 0x11, PAGE) };

            let mut side = vec![0x11u8; WINDOW as usize];
            if let Some(s0) = Shadow::new_card_sized_for_test(side.as_mut_ptr(), w0 as u64, WINDOW)
            {
                let (up, up_ptr) = h.upload_buffer(PAGE as u64).expect("upload");
                // SAFETY: mapped upload buffer.
                unsafe { std::ptr::write_bytes(up_ptr, 0x42, PAGE) };
                let list = h.new_list().expect("list");
                // SAFETY: live buffers.
                unsafe {
                    list.list
                        .CopyBufferRegion(&s0.buffer, 0, &up, 0, PAGE as u64);
                }
                list.close();
                h.execute(&[&list]);
                list.reset();

                let cache = RwLock::new(LooseCache::new());
                {
                    let mut g = cache.write().unwrap();
                    g.entries.insert((w0 as u64, WINDOW), s0.clone());
                    g.total_bytes = WINDOW;
                }
                let deferred = crate::lazy_back::try_defer_page_for_test(&s0, 0, PAGE, 0);
                assert!(deferred, "section-backed loose window must defer");
                // Drop caller ref: `strong_count == 2` (cache + LazyEntry).
                drop(s0);
                assert!(crate::lazy_back::has_block(w0 / PAGE));
                // Requesting a second window `w1` exceeds budget `WINDOW`, so
                // `loose_in` must drain `w0`'s lazy entry outside the cache lock
                // and evict `w0`.
                let s1 = loose_in(&cache, w1 as u64, w1 as u64 + PAGE as u64, WINDOW);
                assert!(s1.is_some());
                assert!(
                    !crate::lazy_back::has_block(w0 / PAGE),
                    "idle deferred window must be materialized on budget eviction"
                );
                let g = cache.read().unwrap();
                assert!(
                    !g.entries.contains_key(&(w0 as u64, WINDOW)),
                    "idle deferred window must be evicted when over budget"
                );
                assert!(g.total_bytes <= WINDOW);
                drop(g);
                drop(s1);
                // SAFETY: materialized GPU bytes landed in `w0`.
                assert_eq!(unsafe { *(w0 as *const u8) }, 0x42);
            }
            crate::watch::title_memory_remapped(w0, PAGE);
        }
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
            CloseHandle(section);
        }
    }

    #[test]
    fn stale_and_merge_ranges_under_writes_scope_to_conflicting_subranges_only() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_TOP_DOWN: u32 = 0x10_0000;
        const MEM_WRITE_WATCH: u32 = 0x20_0000;

        let pages = 4usize;
        let len = pages * PAGE;
        // SAFETY: isolated write-watched test buffer freed at the end of the test.
        // Use `MEM_TOP_DOWN` so freed test VAs are not reused by bottom-up `watch` tests.
        let title = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_TOP_DOWN | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
            .cast::<u8>()
        };
        assert!(!title.is_null());
        // SAFETY: committed above.
        unsafe { std::ptr::write_bytes(title, 0x11, len) };
        let va0 = title as usize;

        let mut side = vec![0x11u8; len];
        let Some(shadow) = Shadow::new_card_for_test(
            side.as_mut_ptr(),
            va0 as u64,
            len as u64,
            Vec::new(),
            Vec::new(),
        ) else {
            // SAFETY: cleanup.
            unsafe { VirtualFree(title.cast(), 0, MEM_RELEASE) };
            return;
        };

        // Mark all pages copied once.
        let initial_ask = crate::watch::writes(va0, va0 + len, true)
            .expect("watched")
            .0;
        shadow.copied.lock().unwrap().fill(initial_ask);

        // Declare two disjoint pending GPU write ranges: page 0 and page 2.
        shadow.writing(va0, PAGE, |_, _| None);
        shadow.writing(va0 + 2 * PAGE, PAGE, |_, _| None);

        // Title writes only page 0 and page 3 (page 3 is not pending; page 2 is clean).
        // SAFETY: within committed buffer.
        unsafe {
            std::ptr::write_bytes(title.add(16), 0x5A, 64);
            std::ptr::write_bytes(title.add(3 * PAGE + 16), 0x7E, 64);
        }

        // `stale_ranges_under_writes` via `ask_once` must report ONLY page 0.
        crate::watch::ask_once(&[(va0, va0 + len)]);
        let mut stale = Vec::new();
        shadow.stale_ranges_under_writes(va0, len, &mut stale);
        crate::watch::ask_once_end();
        assert_eq!(stale, vec![(va0, va0 + PAGE)]);

        // With `runs_left == 0`, `merge_ranges_under_writes` fails to merge page 0
        // and reports ONLY page 0 (not page 2 or page 3).
        let mut runs_left = 0usize;
        let mut unmerged = Vec::new();
        shadow.merge_ranges_under_writes(va0, len, &mut runs_left, &mut unmerged);
        assert_eq!(unmerged, vec![(va0, va0 + PAGE)]);

        // With sufficient `runs_left`, page 0 merges cleanly and `unmerged` is empty.
        let mut runs_left = 16usize;
        unmerged.clear();
        shadow.merge_ranges_under_writes(va0, len, &mut runs_left, &mut unmerged);
        assert!(unmerged.is_empty());
        assert_eq!(&side[16..80], &[0x5Au8; 64]);

        // SAFETY: free test allocation.
        unsafe { VirtualFree(title.cast(), 0, MEM_RELEASE) };
    }

    #[test]
    fn rebar_merge_ranges_under_writes_handles_many_strided_runs_without_consuming_runs_left() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_TOP_DOWN: u32 = 0x10_0000;
        const MEM_WRITE_WATCH: u32 = 0x20_0000;

        let Some(h) = host() else {
            return;
        };
        let len = PAGE;
        // SAFETY: isolated write-watched test buffer freed at the end of the test.
        let title = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_TOP_DOWN | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
            .cast::<u8>()
        };
        assert!(!title.is_null());
        // SAFETY: committed above.
        unsafe { std::ptr::write_bytes(title, 0x11, len) };
        let va0 = title as usize;

        let Some((buffer, data)) = h.shared_buffer(len as u64) else {
            // SAFETY: cleanup.
            unsafe { VirtualFree(title.cast(), 0, MEM_RELEASE) };
            return;
        };
        // SAFETY: `data` maps `len` bytes.
        unsafe { std::ptr::write_bytes(data, 0x11, len) };
        // SAFETY: live D3D12 buffer.
        let gpu_va = unsafe { buffer.GetGPUVirtualAddress() };
        let shadow = Shadow::new(buffer, data, gpu_va, va0 as u64, len as u64, Some(0));

        // Seed copied epoch and pending shader-write range with ReBAR mirror.
        let initial_ask = crate::watch::writes(va0, va0 + len, true)
            .expect("watched")
            .0;
        shadow.copied.lock().unwrap().fill(initial_ask);
        shadow.writing(va0, len, |_, size| Some(vec![0x11u8; size as usize]));

        // Simulate 700 strided CPU writes (4 bytes every 16 bytes -> 700 separate runs,
        // exceeding old 512 cap) while a GPU shader wrote 0xCC into untouched bytes between them.
        for i in 0..700usize {
            let off = i * 16;
            // SAFETY: `700 * 16 = 11200 < PAGE (65536)`.
            unsafe {
                std::ptr::write_bytes(title.add(off), 0x5A, 4);
                std::ptr::write_bytes(data.add(off + 8), 0xCC, 4);
            }
        }

        // Even with `runs_left == 0` (reserved for `card`), ReBAR must merge all 700 runs
        // directly into WC VRAM + mirror without forcing `WHY_STALE` or touching GPU bytes.
        let mut runs_left = 0usize;
        let mut unmerged = Vec::new();
        shadow.merge_ranges_under_writes(va0, len, &mut runs_left, &mut unmerged);
        assert!(
            unmerged.is_empty(),
            "ReBAR merge must not fail on runs_left"
        );
        assert_eq!(runs_left, 0, "ReBAR merge must not consume card runs_left");

        let mirror = shadow
            .rebar
            .as_ref()
            .unwrap()
            .mirror
            .lock()
            .unwrap()
            .clone()
            .unwrap();
        // SAFETY: `data` maps `len` bytes.
        let vram = unsafe { std::slice::from_raw_parts(data, len) };
        for i in 0..700usize {
            let off = i * 16;
            assert_eq!(&vram[off..off + 4], &[0x5Au8; 4]);
            assert_eq!(&mirror[off..off + 4], &[0x5Au8; 4]);
            assert_eq!(
                &vram[off + 8..off + 12],
                &[0xCCu8; 4],
                "concurrent GPU write between CPU strides must stay intact"
            );
            assert_eq!(&mirror[off + 8..off + 12], &[0x11u8; 4]);
        }

        shadow.clear_pending_rebar();
        // SAFETY: free test allocation.
        unsafe { VirtualFree(title.cast(), 0, MEM_RELEASE) };
    }

    #[test]
    fn merge_ranges_under_writes_records_fallback_reasons() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_TOP_DOWN: u32 = 0x10_0000;
        const MEM_WRITE_WATCH: u32 = 0x20_0000;

        let Some(h) = host() else {
            return;
        };
        let len = PAGE;
        // SAFETY: isolated write-watched test buffer freed at the end of the test.
        let title = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_TOP_DOWN | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
            .cast::<u8>()
        };
        assert!(!title.is_null());
        // SAFETY: committed above.
        unsafe { std::ptr::write_bytes(title, 0x11, len) };
        let va0 = title as usize;

        // 1. `runs_cap` on card shadow when `runs_left == 0`.
        let mut side = vec![0x11u8; len];
        let Some(card_shadow) = Shadow::new_card_for_test(
            side.as_mut_ptr(),
            va0 as u64,
            len as u64,
            Vec::new(),
            Vec::new(),
        ) else {
            // SAFETY: cleanup.
            unsafe { VirtualFree(title.cast(), 0, MEM_RELEASE) };
            return;
        };
        let ask = crate::watch::writes(va0, va0 + len, true)
            .expect("watched")
            .0;
        card_shadow.copied.lock().unwrap().fill(ask);
        card_shadow.writing(va0, len, |_, _| None);
        // SAFETY: within committed buffer.
        unsafe { std::ptr::write_bytes(title.add(16), 0x77, 16) };

        let _ = merged_stats();
        let mut runs_left = 0usize;
        let mut unmerged = Vec::new();
        card_shadow.merge_ranges_under_writes(va0, len, &mut runs_left, &mut unmerged);
        assert_eq!(unmerged, vec![(va0, va0 + len)]);
        let stats = merged_stats();
        assert_eq!(stats[5], 1, "card runs overflow must count runs_cap");

        // 2. `no_ref` on ReBAR shadow when `rebar.mirror` is `None`.
        let Some((buffer, data)) = h.shared_buffer(len as u64) else {
            // SAFETY: cleanup.
            unsafe { VirtualFree(title.cast(), 0, MEM_RELEASE) };
            return;
        };
        // SAFETY: live D3D12 buffer.
        let gpu_va = unsafe { buffer.GetGPUVirtualAddress() };
        let rebar_shadow = Shadow::new(buffer, data, gpu_va, va0 as u64, len as u64, Some(0));
        rebar_shadow.copied.lock().unwrap().fill(ask);
        rebar_shadow.writing(va0, len, |_, _| None);
        unmerged.clear();
        let mut runs_left = 64usize;
        rebar_shadow.merge_ranges_under_writes(va0, len, &mut runs_left, &mut unmerged);
        assert_eq!(unmerged, vec![(va0, va0 + len)]);
        let stats = merged_stats();
        assert_eq!(stats[6], 1, "missing ReBAR mirror must count no_ref");

        // 3. `unwatched` when `self.watched == NOT_WATCHED`.
        card_shadow.mark_unwatched_for_test();
        unmerged.clear();
        card_shadow.merge_ranges_under_writes(va0, len, &mut runs_left, &mut unmerged);
        assert_eq!(unmerged, vec![(va0, va0 + len)]);
        let stats = merged_stats();
        assert_eq!(stats[3], 1, "NOT_WATCHED shadow must count unwatched");

        rebar_shadow.clear_pending_rebar();
        // SAFETY: free test allocation.
        unsafe { VirtualFree(title.cast(), 0, MEM_RELEASE) };
    }

    /// Shaders only declared during preparation have not run when the card
    /// uploads the refresh. Diffing individual bytes in this case produces
    /// thousands of needless CopyBufferRegion calls for strided constants.
    #[cfg(windows)]
    #[test]
    fn card_busy_refresh_coalesces_current_submission_constants() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_TOP_DOWN: u32 = 0x10_0000;
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let len = 16 * PAGE;
        // SAFETY: fresh write-watched allocation, released below. Use high
        // addresses so other tests' fresh allocations do not reuse the
        // address of a page still recorded in the watch journal.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_TOP_DOWN | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            ) as usize
        };
        assert_ne!(p, 0);
        let mut held = vec![0u8; len];
        let Some(copy) = Shadow::new_card_for_test(
            held.as_mut_ptr(),
            p as u64,
            len as u64,
            Vec::new(),
            Vec::new(),
        ) else {
            // SAFETY: valid allocation from above.
            unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
            return; // no host GPU here
        };
        let card = copy.card.as_ref().unwrap();
        copy.update(p, len);
        card.dirty.write().unwrap().clear();
        copy.writing(p, len, |_, _| None);

        let (lo, hi) = (19, len - 23);
        for i in (lo..hi).step_by(16) {
            // SAFETY: inside the write-watched allocation.
            unsafe { ((p + i) as *mut u8).write_volatile(0x55) };
        }
        // A write outside the refresh must not be uploaded.
        // SAFETY: inside the write-watched allocation.
        unsafe { ((p + lo - 1) as *mut u8).write_volatile(0xaa) };
        let copied = copy.update(p + lo, hi - lo);
        let uploads = joined(std::mem::take(&mut *card.dirty.write().unwrap()));
        drop(copy);
        // SAFETY: valid allocation from above.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };

        assert!(held[lo..hi].iter().step_by(16).all(|&byte| byte == 0x55));
        assert_eq!(held[lo - 1], 0);
        assert_eq!(
            uploads.len(),
            1,
            "expected one contiguous upload, got {} fragments",
            uploads.len()
        );
        assert_eq!(uploads, [(lo, hi)]);
        assert_eq!(copied, hi - lo);
    }

    #[test]
    fn card_busy_refresh_preserves_bytes_from_older_flights() {
        let mut flights = Flights::default();
        let shared = (108, 120);
        let older_only = (128, 133);
        flights.declare(shared, 1);
        flights.declare(older_only, 1);
        flights.declare((140, 150), 1);
        flights.declare(shared, 2);
        flights.declare((100, 140), 2);
        let protected = flights.older_ranges(2, 100, 140);
        assert_eq!(protected, [(108, 120), (128, 133)]);

        let held = [0u8; 40];
        let mut title = held;
        let mut device = held;
        // Simulate two GPU writes since staging last matched the device,
        // with new title bytes on either side of each write.
        for at in [9, 30] {
            device[at] = 0xaa;
            title[at - 1] = 0x55;
            title[at + 1] = 0x55;
        }
        for (at, len) in card_busy_runs(&title, &held, !protected.is_empty()) {
            device[at..at + len].copy_from_slice(&title[at..at + len]);
        }
        assert_eq!(device[8..11], [0x55, 0xaa, 0x55]);
        assert_eq!(device[29..32], [0x55, 0xaa, 0x55]);

        // Landing the older flight frees both ranges for batching, even
        // though the current submission still declares the shared range.
        flights.done(&[shared, older_only]);
        assert!(flights.older_ranges(2, 100, 140).is_empty());
    }

    /// Strided writes on a 64 KiB page coalesce into a single upload where no
    /// older flight overlaps, while a partial-page older flight keeps exact
    /// byte diffs strictly inside its subrange and preserves GPU writes there.
    #[test]
    fn card_busy_page_runs_coalesces_unprotected_and_preserves_older_gpu_writes() {
        let base = 0x10_0000usize;
        let held = vec![0u8; PAGE];
        let mut title = held.clone();
        // Strided 4-byte writes every 16 bytes across the full 64 KiB page.
        for i in (0..PAGE).step_by(16) {
            title[i..i + 4].copy_from_slice(&[0x55, 0x66, 0x77, 0x88]);
        }

        // Case 1: no older flight -> 1 single contiguous range, 0 exact runs.
        let mut runs = Vec::new();
        let exact = card_busy_page_runs(base, &title, &held, &[], 0, &mut runs);
        assert_eq!(exact, 0);
        assert_eq!(runs, [(0, PAGE)]);

        // Case 2: older flight covers only [2048, 2560) and wrote GPU bytes at
        // offsets between the strided title writes inside that subrange.
        let prot = [(base + 2048, base + 2560)];
        let mut device = held.clone();
        for at in (2056..2560).step_by(16) {
            device[at] = 0xaa;
        }
        runs.clear();
        let exact = card_busy_page_runs(base, &title, &held, &prot, 0, &mut runs);
        assert_eq!(exact, (2560 - 2048) / 16);
        // Unprotected prefix [0, 2048) + 32 exact runs in [2048, 2560) +
        // unprotected suffix [2560, PAGE) (prefix touches first run at 2048).
        assert!(
            runs.len() <= exact + 2,
            "expected unprotected regions to coalesce, got {} runs",
            runs.len()
        );
        for &(r, m) in &runs {
            device[r..r + m].copy_from_slice(&title[r..r + m]);
        }
        // Every title write landed.
        for i in (0..PAGE).step_by(16) {
            assert_eq!(&device[i..i + 4], &[0x55, 0x66, 0x77, 0x88]);
        }
        // Every older GPU write inside [2048, 2560) survived untouched.
        for at in (2056..2560).step_by(16) {
            assert_eq!(device[at], 0xaa, "older GPU write at {at} clobbered");
        }
    }

    /// `CardSent` binary-searches per-send sorted ranges, skips only sends
    /// strictly newer than `fetched_after`, tracks exact total count, and
    /// prunes landed sends cleanly.
    #[test]
    fn card_sent_index_skips_only_newer_sends_and_prunes_in_order() {
        let mut sent = CardSent::default();
        sent.push_sorted(5, vec![(10, 20), (100, 120)]);
        // 4,096 strided ranges in send 7 across 1 MiB.
        let send7: Vec<(usize, usize)> = (0..4096).map(|i| (i * 256, i * 256 + 4)).collect();
        sent.push_sorted(7, send7);
        sent.push_sorted(9, vec![(65_536, 65_600)]);

        assert_eq!(sent.len(), 2 + 4096 + 1);
        assert!(sent.has_after(8));
        assert!(!sent.has_after(9));

        // Query [0, 1024) with fetched_after = 6: must include send 7's 4
        // ranges in [0, 1024), exclude send 5 (<= 6) and send 9 (outside).
        let mut skip = Vec::new();
        sent.collect_skip(0, 1024, 6, &mut skip);
        assert_eq!(skip, [(0, 4), (256, 260), (512, 516), (768, 772)]);

        // Query with fetched_after = 7: send 7 is excluded too.
        skip.clear();
        sent.collect_skip(0, 1024, 7, &mut skip);
        assert!(skip.is_empty());

        // Query [65_500, 65_700) with fetched_after = 7: hits send 9 clamped.
        sent.collect_skip(65_500, 65_700, 7, &mut skip);
        assert_eq!(skip, [(65_536, 65_600)]);

        // Prune up to 7: drops sends 5 and 7 in O(1), keeping only send 9.
        assert!(sent.needs_prune(7));
        sent.prune(7);
        assert_eq!(sent.len(), 1);
        assert!(!sent.needs_prune(7));
        assert!(sent.has_after(8));
        sent.prune(9);
        assert_eq!(sent.len(), 0);
        assert!(!sent.has_after(0));
    }

    /// End-to-end `Shadow::card_merge_relay` against indexed `CardSent` with
    /// thousands of entries: skips only ranges with `sent_as > fetched_after`
    /// while merging shader writes cleanly.
    #[test]
    fn card_merge_relay_indexed_sent_skips_only_ranges_after_fetched() {
        const LEN: usize = 2 * PIECE;
        let base: Vec<u8> = (0..LEN)
            .map(|i| (i.wrapping_mul(17) ^ (i >> 8)) as u8)
            .collect();
        let mut title = base.clone();
        let mut side = base.clone();
        // Byte 100 sent in send 5 (<= fetched_after=6): must NOT be skipped.
        // Byte 200 sent in send 7 (> fetched_after=6): MUST be skipped.
        // Byte 300 sent in send 9 (> fetched_after=6): MUST be skipped.
        // Byte 400 written only by shader: MUST land.
        title[100] ^= 0x11;
        side[100] = title[100];
        title[200] ^= 0x22;
        side[200] = title[200];
        title[300] ^= 0x33;
        side[300] = title[300];

        let mut entries = vec![(100, 101, 5u64), (200, 201, 7u64), (300, 301, 9u64)];
        // Add 4,096 unrelated ranges in piece 1 under send 8 to exercise indexing.
        entries.extend((0..4096).map(|i| (PIECE + i * 16, PIECE + i * 16 + 4, 8u64)));

        let Some(shadow) = Shadow::new_card_for_test(
            side.as_mut_ptr(),
            title.as_ptr() as u64,
            LEN as u64,
            Vec::new(),
            entries,
        ) else {
            return; // no host GPU here
        };

        let mut gpu = base[..PIECE].to_vec();
        gpu[100] = 0x99; // shader overwrote byte 100 after send 5
        gpu[400] ^= 0x44; // shader wrote byte 400
        let written = shadow.card_merge_relay(0, &gpu, 6, &mut Vec::new());
        assert!(written > 0);
        drop(shadow);

        assert_eq!(
            title[100], 0x99,
            "send 5 (<= 6) should not block shader write"
        );
        assert_eq!(
            title[200],
            base[200] ^ 0x22,
            "send 7 (> 6) title byte must survive"
        );
        assert_eq!(
            title[300],
            base[300] ^ 0x33,
            "send 9 (> 6) title byte must survive"
        );
        assert_eq!(
            title[400],
            base[400] ^ 0x44,
            "shader write at 400 must land"
        );
    }

    /// Multi-threaded lock-order and concurrency stress test exercising:
    /// - `Shadow::before` (`RwLock`) concurrent readers (`card_merge_relay` on
    ///   disjoint pieces) vs writer (`update_busy` / `card_changed_runs`),
    /// - `Card::dirty` -> `Card::sent` (`CardSent`) concurrent insert/query/prune
    ///   (both fast-path untouched read check and slow-path write drain),
    /// - `Shadow::pending` -> `Shadow::flights` declare/`older_ranges`/`copy_back_done`,
    /// - simulated `settle_card_busy` waiting on a `LINE_LANDED`-style condvar
    ///   STRICTLY AFTER dropping `Shadow::before`, paired with a relay thread
    ///   calling `copy_back_done` STRICTLY BEFORE clearing `settling` and notifying.
    #[test]
    fn card_sent_busy_and_before_lock_order_stress() {
        use std::sync::atomic::{AtomicBool, AtomicU64};
        use std::sync::{Barrier, Condvar};

        const PIECES: usize = 8;
        const LEN: usize = PIECES * PIECE;
        const ROUNDS: u64 = 120;

        let title = vec![0x20u8; LEN];
        let mut side = vec![0x20u8; LEN];
        let title_base = title.as_ptr() as usize;
        let Some(shadow) = Shadow::new_card_for_test(
            side.as_mut_ptr(),
            title_base as u64,
            LEN as u64,
            Vec::new(),
            Vec::new(),
        ) else {
            return; // no host GPU here
        };

        let fetched = Arc::new(AtomicU64::new(0));
        let done = Arc::new(AtomicBool::new(false));
        let line = Arc::new((Mutex::new(false), Condvar::new()));
        let barrier = Arc::new(Barrier::new(4));

        // Scoped threads + unwind guard ensure all workers exit cleanly before
        // `title`, `side`, and `shadow` drop, even if an assertion fails.
        struct StopOnDrop<'a>(&'a AtomicBool, &'a (Mutex<bool>, Condvar));
        impl Drop for StopOnDrop<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
                *self.1.0.lock().unwrap_or_else(|e| e.into_inner()) = false;
                self.1.1.notify_all();
            }
        }

        std::thread::scope(|scope| {
            // Thread 2: `card_dirty` drainer (`Card::dirty` -> `Card::sent` fast + slow paths).
            let s2 = shadow.clone();
            let fetched2 = fetched.clone();
            let done2 = done.clone();
            let b2 = barrier.clone();
            let t_dirty = scope.spawn(move || {
                b2.wait();
                let card = s2.card.as_ref().unwrap();
                let mut sent_as = 1u64;
                while !done2.load(Ordering::Acquire) {
                    let f = fetched2.load(Ordering::Acquire);
                    let dirty_empty = card.dirty.read().unwrap().is_empty();
                    if dirty_empty {
                        let needs_prune = card.sent.read().unwrap().needs_prune(f);
                        if needs_prune {
                            card.sent.write().unwrap().prune(f);
                        }
                    } else {
                        let mut dirty = card.dirty.write().unwrap();
                        let taken = std::mem::take(&mut *dirty);
                        let ranges = if taken.is_empty() {
                            Vec::new()
                        } else {
                            joined(taken)
                        };
                        let mut sent = card.sent.write().unwrap();
                        sent.prune(f);
                        if !ranges.is_empty() {
                            sent.push_sorted(sent_as, ranges);
                            sent_as += 1;
                        }
                        let expected_total: usize = sent.sends.iter().map(|s| s.1.len()).sum();
                        assert_eq!(sent.len(), expected_total);
                    }
                    std::thread::yield_now();
                }
            });

            // Thread 3: relay worker A — merges pieces [4..6) (`before.read() -> dirty.read() -> sent.read()`)
            // AND performs `copy_back_done` strictly BEFORE clearing `line.settling` and notifying.
            let s3 = shadow.clone();
            let fetched3 = fetched.clone();
            let done3 = done.clone();
            let line3 = line.clone();
            let b3 = barrier.clone();
            let t_relay_a = scope.spawn(move || {
                b3.wait();
                let mut scratch = Vec::new();
                let mut gpu = vec![0x20u8; PIECE];
                let mut iter = 0u64;
                while !done3.load(Ordering::Acquire) {
                    iter += 1;
                    let piece = 4 + (iter as usize % 2);
                    gpu[(iter as usize * 31) % PIECE] = ((iter & 0x7f) as u8) | 0x80;
                    let f = fetched3.load(Ordering::Acquire);
                    let _ = s3.card_merge_relay(piece * PIECE, &gpu, f, &mut scratch);
                    if iter.is_multiple_of(3) {
                        fetched3.fetch_add(1, Ordering::AcqRel);
                    }
                    // Drain any pending flight range via `copy_back_done` BEFORE clearing `settling`.
                    let settling_now = *line3.0.lock().unwrap();
                    if settling_now {
                        let pending_ranges = s3.pending.lock().unwrap().clone();
                        if !pending_ranges.is_empty() {
                            s3.copy_back_done(&pending_ranges);
                            if !relay() {
                                // `copy_back_done` updates `flights` only when `relay()` is on;
                                // keep `flights` in sync in non-relay unit test runs as well.
                                s3.flights.lock().unwrap().done(&pending_ranges);
                            }
                        }
                        let mut settling = line3.0.lock().unwrap();
                        *settling = false;
                        line3.1.notify_all();
                    }
                }
            });

            // Thread 4: relay worker B — concurrent reader on disjoint pieces [6..8).
            let s4 = shadow.clone();
            let fetched4 = fetched.clone();
            let done4 = done.clone();
            let b4 = barrier.clone();
            let t_relay_b = scope.spawn(move || {
                b4.wait();
                let mut scratch = Vec::new();
                let mut gpu = vec![0x20u8; PIECE];
                let mut iter = 0u64;
                while !done4.load(Ordering::Acquire) {
                    iter += 1;
                    let piece = 6 + (iter as usize % 2);
                    gpu[(iter as usize * 53) % PIECE] = ((iter & 0x3f) as u8) | 0x40;
                    let f = fetched4.load(Ordering::Acquire);
                    let _ = s4.card_merge_relay(piece * PIECE, &gpu, f, &mut scratch);
                }
            });

            // Thread 1 (main test thread): submitting-side `update_busy`
            // (`before.write()` -> `card_changed_runs`) plus simulated
            // `settle_card_busy` (dropping `before` BEFORE waiting on `line`).
            {
                let _stop = StopOnDrop(&done, &line);
                barrier.wait();
                let mut runs = Vec::new();
                let mut local_title = vec![0u8; PAGE];
                for r in 1..=ROUNDS {
                    let piece = (r as usize) % (PIECES / 2); // disjoint from workers on [4..8)
                    let off = piece * PIECE;
                    let lo = title_base + off;
                    let hi = lo + PAGE;
                    {
                        let mut pending = shadow.pending.lock().unwrap();
                        shadow.flights.lock().unwrap().declare((lo, hi), r);
                        if !pending.contains(&(lo, hi)) {
                            pending.push((lo, hi));
                        }
                    }
                    {
                        let mut settling = line.0.lock().unwrap();
                        *settling = true;
                    }
                    let protected = shadow.flights.lock().unwrap().older_ranges(r + 1, lo, hi);
                    let mut before = shadow.before.write().unwrap();
                    // SAFETY: inside `side` allocation (`off + PAGE <= LEN`), exclusive under `before.write()`.
                    let held =
                        unsafe { std::slice::from_raw_parts_mut(shadow.data.add(off), PAGE) };
                    for i in (0..PAGE).step_by(64) {
                        local_title[i] = (r & 0xff) as u8;
                    }
                    runs.clear();
                    let exact =
                        card_busy_page_runs(lo, &local_title, held, &protected, 0, &mut runs);
                    if exact > 8 {
                        // Critical lock-order rule: drop `before` BEFORE waiting on `LINE_LANDED`.
                        drop(before);
                        let mut settling = line.0.lock().unwrap();
                        while *settling {
                            settling = line.1.wait(settling).unwrap();
                        }
                        drop(settling);
                        // Because `relay_land` calls `copy_back_done` BEFORE clearing
                        // `settling` and notifying `LINE_LANDED`, `older_ranges` for the
                        // landed flight must already be cleared upon wakeup.
                        let after_settle =
                            shadow.flights.lock().unwrap().older_ranges(r + 1, lo, hi);
                        assert!(
                            after_settle.is_empty(),
                            "older_ranges must already be cleared after copy_back_done -> LINE_LANDED"
                        );
                        assert!(
                            !shadow.writes_pending(),
                            "pending must already be cleared after copy_back_done -> LINE_LANDED"
                        );
                        before = shadow.before.write().unwrap();
                    }
                    // SAFETY: `before.write()` is held again.
                    let held =
                        unsafe { std::slice::from_raw_parts_mut(shadow.data.add(off), PAGE) };
                    for &(at, n) in &runs {
                        held[at..at + n].copy_from_slice(&local_title[at..at + n]);
                    }
                    shadow.card_changed_runs(runs.iter().map(|&(at, n)| (off + at, off + at + n)));
                    drop(before);
                    // Ensure relay clears the flight for the next iteration.
                    let mut settling = line.0.lock().unwrap();
                    while *settling {
                        settling = line.1.wait(settling).unwrap();
                    }
                }
            }

            t_dirty.join().expect("t_dirty thread panicked");
            t_relay_a.join().expect("t_relay_a thread panicked");
            t_relay_b.join().expect("t_relay_b thread panicked");
        });

        // Final invariants: prune all `CardSent` entries and confirm total == 0.
        let card = shadow.card.as_ref().unwrap();
        {
            let mut sent = card.sent.write().unwrap();
            sent.prune(u64::MAX);
            assert_eq!(sent.len(), 0);
        }
        assert!(shadow.flights.lock().unwrap().0.is_empty());
        drop(shadow);
        let _ = (&title, &side);
    }

    #[test]
    fn relay_merge_rebar_preserves_mirror_and_returns_none_on_materialization_or_read_failure() {
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            PAGE_NOACCESS, PAGE_READWRITE, UnmapViewOfFile, VirtualProtect,
        };

        let Some(_h) = host() else {
            return;
        };
        // SAFETY: 1-page anonymous section.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: map 1 page.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        assert!(!view.Value.is_null());
        let ptr = view.Value.cast::<u8>();
        let va0 = ptr as usize;
        let block = va0 / PAGE;
        crate::lazy_back::clear_hot_for_test_pub(block);
        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        assert!(crate::inplace::title_memory_view(
            va0,
            PAGE,
            section as usize,
            0
        ));
        // SAFETY: seed memory.
        unsafe { std::ptr::write_bytes(ptr, 0x11, PAGE) };

        if let Some(shadow) = Shadow::new_rebar_for_test(va0 as u64, PAGE as u64, 0x11) {
            let gpu_new = vec![0x22u8; PAGE];

            // 1. Pre-pass `materialize_rebar_spans` failure must return `None` without
            //    overwriting the mirror.
            assert!(crate::lazy_back::try_defer_page_for_test(
                &shadow, 0, PAGE, 0
            ));
            crate::lazy_back::set_fail_next_fetches_for_test(1);
            assert_eq!(shadow.relay_merge_rebar(&[(0, PAGE)], &[&gpu_new]), None);
            {
                let mirror = shadow.rebar.as_ref().unwrap().mirror.lock().unwrap();
                assert!(
                    mirror.as_ref().unwrap().iter().all(|&b| b == 0x11),
                    "failed materialization must leave ReBAR mirror unchanged"
                );
            }
            assert!(crate::lazy_back::materialize_range(va0, PAGE));

            // 2. Title read failure (`PAGE_NOACCESS` without lazy block) must ALSO
            //    leave the mirror unchanged and return `None`.
            let mut old_prot = 0u32;
            // SAFETY: temporarily protect our test page to force `read_into_raw` failure.
            let protected =
                unsafe { VirtualProtect(va0 as *const _, PAGE, PAGE_NOACCESS, &mut old_prot) };
            assert_ne!(protected, 0);
            assert_eq!(shadow.relay_merge_rebar(&[(0, PAGE)], &[&gpu_new]), None);
            {
                let mirror = shadow.rebar.as_ref().unwrap().mirror.lock().unwrap();
                assert!(
                    mirror.as_ref().unwrap().iter().all(|&b| b == 0x11),
                    "failed title read must leave ReBAR mirror unchanged"
                );
            }
            // SAFETY: restore RW on our test page.
            unsafe { VirtualProtect(va0 as *const _, PAGE, PAGE_READWRITE, &mut old_prot) };
            drop(shadow);
        }

        crate::lazy_back::clear_hot_for_test_pub(block);
        crate::watch::title_memory_remapped(va0, PAGE);
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
            CloseHandle(section);
        }
    }

    #[test]
    fn mirrored_copy_seg_never_writes_partial_prefix_when_lazy_can_arm() {
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            PAGE_READWRITE, UnmapViewOfFile,
        };

        let Some(_h) = host() else {
            return;
        };
        let len = 2 * PAGE;
        // SAFETY: 2-page anonymous section.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                len as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: map 2 contiguous pages.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len) };
        assert!(!view.Value.is_null());
        let ptr = view.Value.cast::<u8>();
        let va0 = ptr as usize;
        let va1 = va0 + PAGE;
        let blk1 = va1 / PAGE;
        crate::lazy_back::clear_hot_for_test_pub(blk1);
        crate::inplace::title_memory_section(section as usize, len as u64, true);
        assert!(crate::inplace::title_memory_view(
            va0,
            len,
            section as usize,
            0
        ));
        // SAFETY: page 0 holds `0x22`, page 1 holds `0x11`.
        unsafe {
            std::ptr::write_bytes(ptr, 0x22, PAGE);
            std::ptr::write_bytes(ptr.add(PAGE), 0x11, PAGE);
        }

        let shadow =
            Shadow::new_rebar_for_test(va0 as u64, len as u64, 0x11).expect("rebar test shadow");
        // Isolate this thread's view of `lazy_back::active()` to `false` across
        // destination selection (so parallel lazy tests cannot affect this test
        // AND a regression from `can_arm()` back to `active()` deterministically
        // selects direct `seg.mirror` and fails), then clear the override and arm
        // page 1 *mid-copy* right after `execute_copy_seg_checked` selects its
        // destination buffer.
        let _force_inactive = crate::lazy_back::force_active_for_test(Some(false));
        assert!(
            !crate::lazy_back::active(),
            "isolated active() must be false before destination selection"
        );
        let armed_mid_copy = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let armed_cb = armed_mid_copy.clone();
        let shadow_cb = shadow.clone();
        crate::host::set_before_copy_mapped_raw_hook(Some(Box::new(move || {
            crate::lazy_back::clear_force_active_for_test();
            let ok = crate::lazy_back::try_defer_page_for_test(&shadow_cb, PAGE, PAGE, 0);
            armed_cb.store(ok, Ordering::SeqCst);
        })));
        let mut mirror = vec![0x11u8; len];
        let mut dst = vec![0x11u8; len];
        let seg = CopySeg {
            src: va0,
            dst: dst.as_mut_ptr() as usize,
            mirror: mirror.as_mut_ptr() as usize,
            len,
            stage_card: false,
        };
        let copy_ok = execute_copy_seg_checked(seg);
        drop(_force_inactive);
        crate::host::set_before_copy_mapped_raw_hook(None);
        assert!(
            armed_mid_copy.load(Ordering::SeqCst),
            "mid-copy hook must arm page 1"
        );
        assert!(
            crate::lazy_back::has_block(blk1),
            "page 1 must be armed by the mid-copy hook"
        );
        assert!(!copy_ok, "copy reaching mid-copy-armed page 1 must fail");
        assert!(
            mirror.iter().all(|&b| b == 0x11),
            "failed mirrored copy must NOT write partial prefix into mirror"
        );
        assert!(
            dst.iter().all(|&b| b == 0x11),
            "failed mirrored copy must NOT write partial prefix into VRAM dst"
        );
        assert!(crate::lazy_back::materialize_range(va1, PAGE));
        drop(shadow);

        crate::lazy_back::clear_hot_for_test_pub(blk1);
        crate::watch::title_memory_remapped(va0, len);
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
            CloseHandle(section);
        }
    }

    #[test]
    fn failed_wide_card_refresh_preserves_staging_baseline_and_keeps_good_segments() {
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            PAGE_READWRITE, UnmapViewOfFile,
        };

        let Some(_h) = host() else {
            return;
        };
        let pages = (WIDE_COPY_MIN / PAGE) + 4;
        let len = pages * PAGE;
        // SAFETY: multi-page anonymous section large enough to trigger wide chunking.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                len as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: map `len` bytes.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len) };
        assert!(!view.Value.is_null());
        let ptr = view.Value.cast::<u8>();
        let va0 = ptr as usize;
        let last_off = (pages - 1) * PAGE;
        let last_va = va0 + last_off;
        let last_block = last_va / PAGE;
        crate::lazy_back::clear_hot_for_test_pub(last_block);
        crate::inplace::title_memory_section(section as usize, len as u64, true);
        assert!(crate::inplace::title_memory_view(
            va0,
            len,
            section as usize,
            0
        ));
        // SAFETY: segment 0 (page 0) = `0x33`, segment 1 (pages 2..pages) = `0x22`.
        unsafe {
            std::ptr::write_bytes(ptr, 0x33, PAGE);
            std::ptr::write_bytes(ptr.add(PAGE), 0x11, PAGE);
            std::ptr::write_bytes(ptr.add(2 * PAGE), 0x22, (pages - 2) * PAGE);
        }

        let mut side = vec![0x11u8; len];
        if let Some(shadow) =
            Shadow::new_card_sized_for_test(side.as_mut_ptr(), va0 as u64, len as u64)
        {
            shadow.clear_card_dirty_for_test();
            // Arm the final page of segment 1 in `lazy_back`, and force all
            // materialization attempts to fail so segment 1 fails on both tries.
            assert!(crate::lazy_back::try_defer_page_for_test(
                &shadow, last_off, PAGE, 0
            ));
            crate::lazy_back::set_fail_next_fetches_for_test(4);

            let copied = shadow.copy_runs(&[(va0, va0 + PAGE), (va0 + 2 * PAGE, va0 + len)]);
            crate::lazy_back::set_fail_next_fetches_for_test(0);

            // Independently successful segment 0 MUST land in `card.staging` and `card.dirty`.
            assert_eq!(copied, PAGE);
            assert!(
                side[..PAGE].iter().all(|&b| b == 0x33),
                "successful segment 0 must be published into card.staging"
            );
            // Failed wide segment 1 MUST leave `card.staging` untouched at `0x11`.
            assert!(
                side[2 * PAGE..].iter().all(|&b| b == 0x11),
                "failed wide card refresh must preserve old card.staging comparison baseline"
            );
            let dirty = shadow.card.as_ref().unwrap().dirty.read().unwrap().clone();
            assert_eq!(
                dirty,
                vec![(0, PAGE)],
                "card.dirty must contain only the succeeded segment"
            );

            assert!(crate::lazy_back::materialize_range(last_va, PAGE));
            drop(shadow);
        }

        crate::lazy_back::clear_hot_for_test_pub(last_block);
        crate::watch::title_memory_remapped(va0, len);
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
            CloseHandle(section);
        }
    }

    #[test]
    fn apply_rebar_relay_subrange_short_write_fails_and_preserves_unwritten_mirror() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READONLY, PAGE_READWRITE, VirtualAlloc,
            VirtualFree, VirtualProtect,
        };

        let Some(_h) = host() else {
            return;
        };
        const P4K: usize = 4 << 10;
        let len = 2 * P4K;
        // Private allocation (no `inplace` own view) so `apply_rebar_relay_subrange`
        // uses the `write_mapped_raw_with` fallback path.
        // SAFETY: fresh 2x4KiB private allocation.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
            .cast::<u8>()
        };
        assert!(!base.is_null());
        // SAFETY: seed both 4 KiB pages with `0x11`.
        unsafe { std::ptr::write_bytes(base, 0x11, len) };

        let shadow =
            Shadow::new_rebar_for_test(base as u64, len as u64, 0x11).expect("rebar test shadow");

        // Case 1: Page 0 is PAGE_READONLY and page 1 is PAGE_READWRITE -> short write
        // writes ONLY page 1 (4 KiB of 8 KiB). Must return `None`, leave mirror[0..4K]
        // at `0x11` (not advanced!), and update mirror[4K..8K] to `0xAA` to match title.
        let mut old_prot = 0u32;
        assert_ne!(
            // SAFETY: making page 0 read-only while page 1 stays writable.
            unsafe { VirtualProtect(base.cast(), P4K, PAGE_READONLY, &mut old_prot) },
            0
        );
        let gpu = vec![0xAAu8; len];
        let mut old = vec![0x11u8; len];
        let mut title = Vec::new();
        let mut wrote_fallback = false;
        let res = shadow.apply_rebar_relay_subrange(
            base as usize,
            0,
            &gpu,
            &mut old,
            &mut title,
            false,
            &mut wrote_fallback,
        );
        assert_eq!(
            res, None,
            "short write_mapped_raw must fail the ReBAR relay subrange"
        );
        assert!(wrote_fallback);
        assert!(
            old[..P4K].iter().all(|&b| b == 0x11),
            "unwritten read-only page 0 must keep pre-merge mirror bytes"
        );
        assert!(
            old[P4K..].iter().all(|&b| b == 0xAA),
            "confirmed-written page 1 must reflect merged bytes in mirror"
        );
        // SAFETY: page 0 is readable (PAGE_READONLY) and page 1 is PAGE_READWRITE.
        let (title0, title1) = unsafe {
            (
                std::slice::from_raw_parts(base, P4K),
                std::slice::from_raw_parts(base.add(P4K), P4K),
            )
        };
        assert!(title0.iter().all(|&b| b == 0x11));
        assert!(title1.iter().all(|&b| b == 0xAA));

        // Case 2: Both pages read-only -> written == 0, must return `None` and leave mirror untouched.
        assert_ne!(
            // SAFETY: making page 1 read-only too.
            unsafe { VirtualProtect(base.add(P4K).cast(), P4K, PAGE_READONLY, &mut old_prot) },
            0
        );
        let gpu2 = vec![0xBBu8; len];
        let old_before = old.clone();
        let res2 = shadow.apply_rebar_relay_subrange(
            base as usize,
            0,
            &gpu2,
            &mut old,
            &mut title,
            false,
            &mut wrote_fallback,
        );
        assert_eq!(res2, None);
        assert_eq!(
            old, old_before,
            "zero-byte write must leave mirror unchanged"
        );

        // Case 3: Restore RW -> full write succeeds and updates mirror.
        assert_ne!(
            // SAFETY: restoring RW on both pages.
            unsafe { VirtualProtect(base.cast(), len, PAGE_READWRITE, &mut old_prot) },
            0
        );
        let res3 = shadow.apply_rebar_relay_subrange(
            base as usize,
            0,
            &gpu2,
            &mut old,
            &mut title,
            false,
            &mut wrote_fallback,
        );
        assert_eq!(res3, Some(len));
        assert!(old.iter().all(|&b| b == 0xBB));

        drop(shadow);
        // SAFETY: cleanup.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }

    #[test]
    fn update_busy_reports_failure_when_page_backed_by_other_shadows_lazy_block() {
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEM_RELEASE, MEM_RESERVE,
            MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_NOACCESS, PAGE_READWRITE,
            UnmapViewOfFile, VirtualAlloc, VirtualFree,
        };

        let Some(_h) = host() else {
            return;
        };
        let len = 2 * PAGE;
        // SAFETY: 2-page anonymous section.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                len as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: map 2 contiguous pages.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len) };
        assert!(!view.Value.is_null());
        let ptr = view.Value.cast::<u8>();
        let va0 = ptr as usize;
        let blk0 = va0 / PAGE;
        crate::lazy_back::clear_hot_for_test_pub(blk0);
        crate::inplace::title_memory_section(section as usize, len as u64, true);
        assert!(crate::inplace::title_memory_view(
            va0,
            len,
            section as usize,
            0
        ));
        // SAFETY: seed page 0 = `0x11`, page 1 = `0x44`.
        unsafe {
            std::ptr::write_bytes(ptr, 0x11, PAGE);
            std::ptr::write_bytes(ptr.add(PAGE), 0x44, PAGE);
        }

        // Shadow B owns page 0's deferred lazy block; Shadow A (both card and ReBAR)
        // covers [va0, va0 + 2*PAGE) and must report `all_ok == false` when page 0's
        // materialization fails, while preserving page 1's copied bytes & epoch.
        let mut side_b = vec![0x11u8; PAGE];
        let shadow_b =
            Shadow::new_card_sized_for_test(side_b.as_mut_ptr(), va0 as u64, PAGE as u64)
                .expect("owner shadow B");
        assert!(crate::lazy_back::try_defer_page_for_test(
            &shadow_b, 0, PAGE, 0
        ));

        // 1. Card-backed Shadow A (non-ReBAR branch in `update_busy`).
        let mut side_a = vec![0x11u8; len];
        let card_a = Shadow::new_card_sized_for_test(side_a.as_mut_ptr(), va0 as u64, len as u64)
            .expect("card shadow A");
        crate::lazy_back::set_fail_next_fetches_for_test(4);
        let (copied_card, ok_card) = card_a.update_busy(va0, va0 + len);
        crate::lazy_back::set_fail_next_fetches_for_test(0);
        assert!(
            !ok_card,
            "card update_busy must report failure when a busy page is backed by another shadow's failed lazy block"
        );
        assert_eq!(copied_card, PAGE, "good page 1 bytes must still be copied");
        assert!(side_a[PAGE..].iter().all(|&b| b == 0x44));
        {
            let copied_epochs = card_a.copied.lock().unwrap();
            assert_eq!(copied_epochs[0], 0, "failed page 0 epoch must stay pending");
            assert_ne!(copied_epochs[1], 0, "good page 1 epoch must advance");
        }
        drop(card_a);

        // 2. ReBAR-backed Shadow A (ReBAR branch in `update_busy`).
        let rebar_a =
            Shadow::new_rebar_for_test(va0 as u64, len as u64, 0x11).expect("rebar shadow A");
        crate::lazy_back::set_fail_next_fetches_for_test(4);
        let (copied_rebar, ok_rebar) = rebar_a.update_busy(va0, va0 + len);
        crate::lazy_back::set_fail_next_fetches_for_test(0);
        assert!(
            !ok_rebar,
            "ReBAR update_busy must report failure when a busy page is backed by another shadow's failed lazy block"
        );
        assert_eq!(copied_rebar, PAGE, "good page 1 bytes must still be copied");
        {
            let copied_epochs = rebar_a.copied.lock().unwrap();
            assert_eq!(copied_epochs[0], 0, "failed page 0 epoch must stay pending");
            assert_ne!(copied_epochs[1], 0, "good page 1 epoch must advance");
        }
        drop(rebar_a);

        // Clean up lazy block on page 0.
        assert!(crate::lazy_back::materialize_range(va0, PAGE));
        drop(shadow_b);

        // 3. Ordinary unmapped hole with NO lazy block must remain skippable (`all_ok == true`).
        // SAFETY: reserved (uncommitted) 1-page hole.
        let hole = unsafe { VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!hole.is_null());
        let mut side_h = vec![0u8; PAGE];
        let hole_shadow =
            Shadow::new_card_sized_for_test(side_h.as_mut_ptr(), hole as u64, PAGE as u64)
                .expect("hole shadow");
        let (_, ok_hole) = hole_shadow.update_busy(hole as usize, hole as usize + PAGE);
        assert!(
            ok_hole,
            "ordinary unmapped hole without a lazy block must remain skippable"
        );
        drop(hole_shadow);
        // SAFETY: cleanup.
        unsafe {
            VirtualFree(hole, 0, MEM_RELEASE);
            crate::lazy_back::clear_hot_for_test_pub(blk0);
            crate::watch::title_memory_remapped(va0, len);
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
            CloseHandle(section);
        }
    }

    #[test]
    fn update_busy_reports_failure_when_page_armed_after_ownership_check() {
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            PAGE_READWRITE, UnmapViewOfFile,
        };

        let Some(_h) = host() else {
            return;
        };
        let len = 2 * PAGE;
        // SAFETY: 2-page anonymous section.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                len as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: map 2 contiguous pages.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len) };
        assert!(!view.Value.is_null());
        let ptr = view.Value.cast::<u8>();
        let va0 = ptr as usize;
        let blk0 = va0 / PAGE;
        crate::lazy_back::clear_hot_for_test_pub(blk0);
        crate::inplace::title_memory_section(section as usize, len as u64, true);
        assert!(crate::inplace::title_memory_view(
            va0,
            len,
            section as usize,
            0
        ));
        // SAFETY: seed page 0 = `0x22`, page 1 = `0x55`.
        unsafe {
            std::ptr::write_bytes(ptr, 0x22, PAGE);
            std::ptr::write_bytes(ptr.add(PAGE), 0x55, PAGE);
        }

        // 1. Card-backed shadow: page 0 is NOT armed at the start of `update_busy`
        //    (passing both `owned_blocks_with_clean_in` and `owns_block`), then gets
        //    armed right before `lock_and_read`'s `read_into_raw` with failing DMA.
        //    `lock_and_read` must return `LockRead::LazyFailed` -> `all_ok == false`.
        let mut side_a = vec![0x11u8; len];
        let card_a = Shadow::new_card_sized_for_test(side_a.as_mut_ptr(), va0 as u64, len as u64)
            .expect("card shadow");
        let card_for_hook = card_a.clone();
        BEFORE_LOCK_AND_READ_RAW_HOOK.with(|h| {
            *h.borrow_mut() = Some(Box::new(move || {
                assert!(crate::lazy_back::try_defer_page_for_test(
                    &card_for_hook,
                    0,
                    PAGE,
                    0
                ));
                crate::lazy_back::set_fail_next_fetches_for_test(1);
            }));
        });
        let (copied_card, ok_card) = card_a.update_busy(va0, va0 + len);
        BEFORE_LOCK_AND_READ_RAW_HOOK.with(|h| *h.borrow_mut() = None);
        crate::lazy_back::set_fail_next_fetches_for_test(0);
        assert!(
            !ok_card,
            "card update_busy must report failure when page 0 is armed after ownership check and fails materialization"
        );
        assert_eq!(copied_card, PAGE, "good page 1 bytes must still be copied");
        {
            let copied_epochs = card_a.copied.lock().unwrap();
            assert_eq!(copied_epochs[0], 0, "failed page 0 epoch must stay pending");
            assert_ne!(copied_epochs[1], 0, "good page 1 epoch must advance");
        }
        assert!(crate::lazy_back::materialize_range(va0, PAGE));
        crate::lazy_back::clear_hot_for_test_pub(blk0);
        drop(card_a);

        // 2. ReBAR-backed shadow: same post-ownership-check arming race with failing DMA.
        let rebar_a =
            Shadow::new_rebar_for_test(va0 as u64, len as u64, 0x11).expect("rebar shadow");
        let rebar_for_hook = rebar_a.clone();
        BEFORE_LOCK_AND_READ_RAW_HOOK.with(|h| {
            *h.borrow_mut() = Some(Box::new(move || {
                assert!(crate::lazy_back::try_defer_page_for_test(
                    &rebar_for_hook,
                    0,
                    PAGE,
                    0
                ));
                crate::lazy_back::set_fail_next_fetches_for_test(1);
            }));
        });
        let (copied_rebar, ok_rebar) = rebar_a.update_busy(va0, va0 + len);
        BEFORE_LOCK_AND_READ_RAW_HOOK.with(|h| *h.borrow_mut() = None);
        crate::lazy_back::set_fail_next_fetches_for_test(0);
        assert!(
            !ok_rebar,
            "ReBAR update_busy must report failure when page 0 is armed after ownership check and fails materialization"
        );
        assert_eq!(copied_rebar, PAGE, "good page 1 bytes must still be copied");
        {
            let copied_epochs = rebar_a.copied.lock().unwrap();
            assert_eq!(copied_epochs[0], 0, "failed page 0 epoch must stay pending");
            assert_ne!(copied_epochs[1], 0, "good page 1 epoch must advance");
        }
        assert!(crate::lazy_back::materialize_range(va0, PAGE));
        crate::lazy_back::clear_hot_for_test_pub(blk0);

        // 3. When materialization succeeds after post-ownership-check arming,
        //    `lock_and_read` retries cleanly and returns `Ok` (`all_ok == true`).
        let rebar_retry_hook = rebar_a.clone();
        BEFORE_LOCK_AND_READ_RAW_HOOK.with(|h| {
            *h.borrow_mut() = Some(Box::new(move || {
                assert!(crate::lazy_back::try_defer_page_for_test(
                    &rebar_retry_hook,
                    0,
                    PAGE,
                    0
                ));
            }));
        });
        let (_, ok_retry) = rebar_a.update_busy(va0, va0 + PAGE);
        BEFORE_LOCK_AND_READ_RAW_HOOK.with(|h| *h.borrow_mut() = None);
        assert!(
            ok_retry,
            "update_busy must succeed when post-ownership-check arming materializes cleanly on retry"
        );
        drop(rebar_a);

        // SAFETY: cleanup.
        unsafe {
            crate::lazy_back::clear_hot_for_test_pub(blk0);
            crate::watch::title_memory_remapped(va0, len);
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
            CloseHandle(section);
        }
    }

    #[test]
    fn card_staging_publication_and_dirty_marking_are_atomic_with_relay_merge() {
        use std::sync::Condvar;
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            PAGE_READWRITE, UnmapViewOfFile,
        };

        struct HookSync {
            hook_entered: bool,
            refresh_finished: bool,
            worker_ready: bool,
            worker_exited: bool,
        }
        struct WorkerExitGuard(Arc<(Mutex<HookSync>, Condvar)>);
        impl Drop for WorkerExitGuard {
            fn drop(&mut self) {
                let (lock, cv) = &*self.0;
                let mut s = lock.lock().unwrap_or_else(|e| e.into_inner());
                s.worker_exited = true;
                cv.notify_all();
            }
        }
        struct FinishGapGuard(Arc<(Mutex<HookSync>, Condvar)>);
        impl Drop for FinishGapGuard {
            fn drop(&mut self) {
                let (lock, cv) = &*self.0;
                let mut s = lock.lock().unwrap_or_else(|e| e.into_inner());
                s.refresh_finished = true;
                cv.notify_all();
            }
        }

        let Some(_h) = host() else {
            return;
        };
        // Exercise BOTH single-shadow (`copy_runs_with_epochs`) and batch (`refresh_batch`) paths.
        for use_batch in [false, true] {
            // SAFETY: 1-page anonymous section.
            let section: HANDLE = unsafe {
                CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    std::ptr::null(),
                    PAGE_READWRITE,
                    0,
                    PAGE as u32,
                    std::ptr::null(),
                )
            };
            assert!(!section.is_null());
            // SAFETY: map 1 page.
            let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
            assert!(!view.Value.is_null());
            let ptr = view.Value.cast::<u8>();
            let va = ptr as usize;
            crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
            assert!(crate::inplace::title_memory_view(
                va,
                PAGE,
                section as usize,
                0
            ));
            // Title now holds newer CPU refresh value `0x77`; card.staging and stale GPU hold `0x11`.
            // SAFETY: writing title view.
            unsafe { std::ptr::write_bytes(ptr, 0x77, PAGE) };

            let mut side = vec![0x11u8; PAGE];
            let shadow = Shadow::new_card_sized_for_test(side.as_mut_ptr(), va as u64, PAGE as u64)
                .expect("card shadow");
            shadow.clear_card_dirty_for_test();

            // Coordinate hook entry, worker readiness, refresh completion, and
            // timeouts under a single mutex+condvar so neither thread can ever
            // hang under any scheduling order or timeout.
            let hook_sync = Arc::new((
                Mutex::new(HookSync {
                    hook_entered: false,
                    refresh_finished: false,
                    worker_ready: false,
                    worker_exited: false,
                }),
                Condvar::new(),
            ));
            let lock_was_exclusive = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let sync_cb = hook_sync.clone();
            let exclusive_cb = lock_was_exclusive.clone();
            let shadow_cb = shadow.clone();
            BETWEEN_PUBLISH_AND_DIRTY_HOOK.with(|h| {
                *h.borrow_mut() = Some(Box::new(move || {
                    // Right after `publish_staged_card_segs` and before `finish_copy_segs`:
                    // `shadow.before` MUST be write-locked so `card_merge_relay_inner` cannot
                    // observe the new `0x77` staging bytes without their `card.dirty` range.
                    exclusive_cb.store(shadow_cb.before.try_read().is_err(), Ordering::SeqCst);
                    let (lock, cv) = &*sync_cb;
                    let mut s = lock.lock().unwrap_or_else(|e| e.into_inner());
                    s.hook_entered = true;
                    cv.notify_all();
                    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                    while !s.worker_ready && !s.worker_exited {
                        let remaining =
                            deadline.saturating_duration_since(std::time::Instant::now());
                        if remaining.is_zero() {
                            break;
                        }
                        let (guard, timeout) = cv
                            .wait_timeout(s, remaining)
                            .unwrap_or_else(|e| e.into_inner());
                        s = guard;
                        if timeout.timed_out() {
                            break;
                        }
                    }
                }));
            });

            let shadow_t2 = shadow.clone();
            let sync_t2 = hook_sync.clone();
            let merge_thread = std::thread::spawn(move || {
                let _worker_exit = WorkerExitGuard(sync_t2.clone());
                let hook_fired = {
                    let (lock, cv) = &*sync_t2;
                    let mut s = lock.lock().unwrap_or_else(|e| e.into_inner());
                    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                    while !s.hook_entered && !s.refresh_finished {
                        let remaining =
                            deadline.saturating_duration_since(std::time::Instant::now());
                        if remaining.is_zero() {
                            break;
                        }
                        let (guard, timeout) = cv
                            .wait_timeout(s, remaining)
                            .unwrap_or_else(|e| e.into_inner());
                        s = guard;
                        if timeout.timed_out() {
                            break;
                        }
                    }
                    if s.hook_entered {
                        s.worker_ready = true;
                        cv.notify_all();
                        true
                    } else {
                        s.worker_exited = true;
                        cv.notify_all();
                        false
                    }
                };
                if !hook_fired {
                    return None;
                }
                // Stale GPU bytes (`0x11`, matching old staging) arrive right as staging
                // was published to `0x77`: must block on `self.before.read()` until
                // `card.dirty` is recorded, then skip the dirty range instead of
                // overwriting title RAM `0x77` back to `0x11`.
                let stale_gpu = vec![0x11u8; PAGE];
                let mut scratch = Vec::new();
                Some(
                    shadow_t2
                        .card_merge_relay_via_own_at(0, va, &stale_gpu, 0, &mut scratch)
                        .expect("relay merge"),
                )
            });

            {
                let _finish_gap = FinishGapGuard(hook_sync.clone());
                if use_batch {
                    // Include a second empty-span shadow so `refresh_batch` takes
                    // the multi-shadow batch path (`items.len() >= 2`) rather than
                    // the single-item `update_checked` shortcut.
                    let mut dummy_side = vec![0u8; PAGE];
                    let dummy = Shadow::new_card_sized_for_test(
                        dummy_side.as_mut_ptr(),
                        va as u64,
                        PAGE as u64,
                    )
                    .expect("dummy card shadow");
                    let items = vec![(shadow.clone(), va, va + PAGE), (dummy, va, va)];
                    let (bytes, ok) = refresh_batch(items, false);
                    assert!(ok);
                    assert_eq!(bytes, PAGE);
                } else {
                    let (bytes, ok) = shadow.copy_runs_with_epochs(&[(va, va + PAGE)], &[]);
                    assert!(ok);
                    assert_eq!(bytes, PAGE);
                }
            }
            BETWEEN_PUBLISH_AND_DIRTY_HOOK.with(|h| *h.borrow_mut() = None);

            let merged = merge_thread
                .join()
                .unwrap()
                .expect("BETWEEN_PUBLISH_AND_DIRTY_HOOK must fire during card refresh");
            assert!(
                lock_was_exclusive.load(Ordering::SeqCst),
                "self.before.write() must be held across staging publication and dirty marking (batch={use_batch})"
            );
            assert_eq!(
                merged, 0,
                "concurrent relay merge must see card.dirty alongside new staging and write 0 bytes (batch={use_batch})"
            );
            // SAFETY: title page is still mapped and readable.
            let got = unsafe { std::slice::from_raw_parts(ptr, 32) };
            assert_eq!(
                got, &[0x77; 32],
                "newer CPU refresh bytes must not be overwritten by stale GPU readback (batch={use_batch})"
            );

            drop(shadow);
            crate::watch::title_memory_remapped(va, PAGE);
            // SAFETY: cleanup.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
                CloseHandle(section);
            }
        }
    }

    #[test]
    fn lock_and_read_retries_when_page_materialized_between_failed_read_and_block_lookup() {
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            PAGE_READWRITE, UnmapViewOfFile,
        };

        let Some(_h) = host() else {
            return;
        };
        // SAFETY: 1-page anonymous section.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: map 1 page.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        assert!(!view.Value.is_null());
        let ptr = view.Value.cast::<u8>();
        let va = ptr as usize;
        let blk = va / PAGE;
        crate::lazy_back::clear_hot_for_test_pub(blk);
        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        assert!(crate::inplace::title_memory_view(
            va,
            PAGE,
            section as usize,
            0
        ));
        // SAFETY: seed initial title bytes with `0x22`.
        unsafe { std::ptr::write_bytes(ptr, 0x22, PAGE) };

        let rebar_src = Shadow::new_rebar_for_test(va as u64, PAGE as u64, 0x11)
            .expect("rebar shadow for gpu bytes");
        rebar_src.write_test_bytes(0, &[0x9E; 32]);
        let shadow_for_arm = rebar_src.clone();
        BEFORE_LOCK_AND_READ_RAW_HOOK.with(|h| {
            *h.borrow_mut() = Some(Box::new(move || {
                // Page arms AFTER pre-pass, causing the first `read_into_raw` to fail.
                assert!(crate::lazy_back::try_defer_page_for_test(
                    &shadow_for_arm,
                    0,
                    PAGE,
                    0
                ));
            }));
        });
        AFTER_LOCK_AND_READ_FAIL_HOOK.with(|h| {
            *h.borrow_mut() = Some(Box::new(move || {
                // Another thread materializes the page BEFORE `has_block_in_range` runs!
                assert!(crate::lazy_back::materialize_range(va, PAGE));
                assert!(!crate::lazy_back::has_block(blk));
            }));
        });

        let mut buf = vec![0u8; PAGE];
        let outcome = lock_and_read(|| (), va, &mut buf);
        BEFORE_LOCK_AND_READ_RAW_HOOK.with(|h| *h.borrow_mut() = None);
        AFTER_LOCK_AND_READ_FAIL_HOOK.with(|h| *h.borrow_mut() = None);
        assert!(
            matches!(outcome, LockRead::Ok(())),
            "lock_and_read must retry and return Ok instead of Unmapped when page materialized after first failed read"
        );
        assert_eq!(
            &buf[..32],
            &[0x9E; 32],
            "retried read must observe materialized GPU bytes"
        );

        drop(rebar_src);
        crate::lazy_back::clear_hot_for_test_pub(blk);
        crate::watch::title_memory_remapped(va, PAGE);
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
            CloseHandle(section);
        }
    }

    #[test]
    fn merge_ranges_under_writes_reports_lazy_failed_page_without_skipping_other_pages() {
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEM_RELEASE, MEM_RESERVE,
            MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_NOACCESS, PAGE_READWRITE,
            UnmapViewOfFile, VirtualAlloc, VirtualFree,
        };

        let Some(_h) = host() else {
            return;
        };
        let len = 2 * PAGE;
        // SAFETY: 2-page anonymous section.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                len as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: map 2 contiguous pages.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len) };
        assert!(!view.Value.is_null());
        let ptr = view.Value.cast::<u8>();
        let va0 = ptr as usize;
        let va1 = va0 + PAGE;
        let blk0 = va0 / PAGE;
        crate::lazy_back::clear_hot_for_test_pub(blk0);
        crate::inplace::title_memory_section(section as usize, len as u64, true);
        assert!(crate::inplace::title_memory_view(
            va0,
            len,
            section as usize,
            0
        ));
        // SAFETY: seed page 0 = `0x22`, page 1 = `0x55`.
        unsafe {
            std::ptr::write_bytes(ptr, 0x22, PAGE);
            std::ptr::write_bytes(ptr.add(PAGE), 0x55, PAGE);
        }

        // Another shadow owns page 0 and defers it.
        let mut owner_side = vec![0x11u8; PAGE];
        let owner =
            Shadow::new_card_sized_for_test(owner_side.as_mut_ptr(), va0 as u64, PAGE as u64)
                .expect("owner card shadow");

        // 1. Card branch of `merge_ranges_under_writes`:
        //    page 0 fails lazy materialization (`LockRead::LazyFailed`) -> must be
        //    reported in `out` with its `copied` epoch unchanged, while page 1
        //    is still merged and its `copied` epoch advances.
        let mut card_side = vec![0x11u8; len];
        let card = Shadow::new_card_sized_for_test(card_side.as_mut_ptr(), va0 as u64, len as u64)
            .expect("target card shadow");
        card.pending.lock().unwrap().push((va0, va0 + len));
        assert!(crate::lazy_back::try_defer_page_for_test(
            &owner, 0, PAGE, 0
        ));
        crate::lazy_back::set_fail_next_fetches_for_test(1);
        let mut runs_left = 64usize;
        let mut out = Vec::new();
        card.merge_ranges_under_writes(va0, len, &mut runs_left, &mut out);
        crate::lazy_back::set_fail_next_fetches_for_test(0);
        assert_eq!(
            out,
            vec![(va0, va1)],
            "card merge_ranges_under_writes must report LazyFailed page 0 in out for settlement"
        );
        assert_eq!(
            &card_side[PAGE..PAGE + 32],
            &[0x55; 32],
            "card merge_ranges_under_writes must still merge healthy page 1"
        );
        {
            let epochs = card.copied.lock().unwrap();
            assert_eq!(epochs[0], 0, "failed page 0 epoch must not advance");
            assert_ne!(epochs[1], 0, "merged page 1 epoch must advance");
        }
        assert!(crate::lazy_back::materialize_range(va0, PAGE));
        crate::lazy_back::clear_hot_for_test_pub(blk0);
        drop(card);

        // 2. ReBAR branch of `merge_ranges_under_writes`: same scenario.
        // SAFETY: restore page 1 bytes in case anything touched them.
        unsafe { std::ptr::write_bytes(ptr.add(PAGE), 0x66, PAGE) };
        let rebar =
            Shadow::new_rebar_for_test(va0 as u64, len as u64, 0x11).expect("target rebar shadow");
        rebar.pending.lock().unwrap().push((va0, va0 + len));
        assert!(crate::lazy_back::try_defer_page_for_test(
            &owner, 0, PAGE, 0
        ));
        crate::lazy_back::set_fail_next_fetches_for_test(1);
        let mut rebar_runs = 64usize;
        let mut rebar_out = Vec::new();
        rebar.merge_ranges_under_writes(va0, len, &mut rebar_runs, &mut rebar_out);
        crate::lazy_back::set_fail_next_fetches_for_test(0);
        assert_eq!(
            rebar_out,
            vec![(va0, va1)],
            "rebar merge_ranges_under_writes must report LazyFailed page 0 in out for settlement"
        );
        {
            let epochs = rebar.copied.lock().unwrap();
            assert_eq!(epochs[0], 0, "rebar failed page 0 epoch must not advance");
            assert_ne!(epochs[1], 0, "rebar merged page 1 epoch must advance");
            let mirror = rebar.rebar.as_ref().unwrap().mirror.lock().unwrap();
            assert_eq!(
                &mirror.as_ref().unwrap()[PAGE..PAGE + 32],
                &[0x66; 32],
                "rebar merge_ranges_under_writes must still merge healthy page 1"
            );
        }
        assert!(crate::lazy_back::materialize_range(va0, PAGE));
        drop(rebar);
        drop(owner);

        // 3. Ordinary unmapped hole without a lazy block stays skippable (not added to `out`).
        // SAFETY: reserved uncommitted 64 KiB region.
        let hole = unsafe { VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!hole.is_null());
        let hole_va = hole as usize;
        let mut hole_side = vec![0x11u8; PAGE];
        let hole_card =
            Shadow::new_card_sized_for_test(hole_side.as_mut_ptr(), hole_va as u64, PAGE as u64)
                .expect("hole card shadow");
        hole_card
            .pending
            .lock()
            .unwrap()
            .push((hole_va, hole_va + PAGE));
        let mut hole_runs = 64usize;
        let mut hole_out = Vec::new();
        hole_card.merge_ranges_under_writes(hole_va, PAGE, &mut hole_runs, &mut hole_out);
        assert!(
            hole_out.is_empty(),
            "ordinary unmapped hole without lazy block must stay skippable"
        );
        drop(hole_card);

        // SAFETY: cleanup.
        unsafe {
            VirtualFree(hole, 0, MEM_RELEASE);
            crate::lazy_back::clear_hot_for_test_pub(blk0);
            crate::watch::title_memory_remapped(va0, len);
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
            CloseHandle(section);
        }
    }
}
