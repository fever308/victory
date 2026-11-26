//! Demand-paged / lazy GPU-to-CPU readback (`VICTORY_LAZY_BACK=1`).
//!
//! Instead of eagerly DMAing GPU-written `card` and `rebar` 64 KiB pages back
//! to title RAM on the relay landing thread, eligible 64 KiB blocks are armed
//! with `PAGE_NOACCESS` across all mapped title views. If the GPU overwrites the
//! same block across multiple flights without any CPU access, zero DMA occurs.
//!
//! On the first CPU read/write fault (via `watch.rs` VEH) or internal runtime
//! read/remap (`read_into`, `read_into_fast`, `copy_mapped`, `write_mapped`,
//! `kernel_writing`, `title_memory_remapping`, `unregister`), the block is
//! fetched once on a dedicated `D3D12_COMMAND_LIST_TYPE_COPY` queue, merged
//! into title RAM via `Shadow::card_merge_relay` / `Shadow::relay_merge_rebar`,
//! and restored to `PAGE_READWRITE`.
//!
//! # Lock ordering
//! - `LAZY_STATE` is NEVER held while acquiring `FETCHER`, `watch::GUARDS`,
//!   `inplace::VIEWS`, `shadow::BUFFERS`/`LOOSE`/`STRETCHES`, or any `Shadow` lock.
//! - `FETCHER` is a strict leaf lock used only for the `COPY`-queue DMA wait.
//! - `watch::GUARDS` may acquire `LAZY_STATE` (`has_block`, `try_arm_lazy_under_guards`),
//!   which is acyclic because `LAZY_STATE` never acquires `watch::GUARDS`.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, OnceLock};

use windows::Win32::Foundation::{CloseHandle, E_FAIL, HANDLE};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMAND_LIST_TYPE_COPY, D3D12_COMMAND_QUEUE_DESC, D3D12_FENCE_FLAG_NONE,
    ID3D12CommandAllocator, ID3D12CommandList, ID3D12CommandQueue, ID3D12Fence,
    ID3D12GraphicsCommandList, ID3D12Resource,
};
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_NOACCESS,
    PAGE_READONLY, PAGE_READWRITE, VirtualProtect, VirtualQuery,
};

use crate::shadow::Shadow;
use crate::watch::PAGE;

/// Default number of distinct frames with CPU/internal materializations before
/// a block promotes to hot.
const DEFAULT_PROMOTE_FRAMES: u32 = 2;
/// Distinct idle frames without a fault or active GPU write after which a hot
/// entry decays back to lazy.
const HOT_DECAY_FRAMES: u64 = 120;
/// Base interval (in frames) before an actively GPU-written hot block allows a
/// single reprobe deferral (doubles on each re-promotion up to 16x).
const HOT_REPROBE_FRAMES: u64 = 1800;
/// Bounded ring of recently materialized/discarded blocks to absorb multi-thread read-fault races.
const RECENT_CAP: usize = 2048;
/// Initial reusable readback buffer size for the dedicated lazy fetcher (4 MiB).
const FETCHER_INIT_BYTES: usize = 4 << 20;

/// Fast atomic count of currently deferred primary 64 KiB blocks.
static LAZY_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Count of threads currently inside `try_defer_page_inner`'s `LAZY_STATE` arming
/// section (paired with `inplace::GENERATION` via `SeqCst` so concurrent
/// `title_memory_view` / `views_mapped` cannot miss an in-progress first arm).
static ARMING_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Monotonic counter incremented whenever any entry is removed from `LAZY_STATE`,
/// allowing `host::read_sparse_checked` / `read_pieces_checked` to detect a stale
/// `PAGE_NOACCESS` snapshot after a lazy block finishes materializing.
static REMOVE_GEN: AtomicU64 = AtomicU64::new(0);
/// Set once any block has ever been deferred so `handle_fault` can skip `LAZY_STATE`
/// completely when lazy readback has never been used.
static EVER_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Whether any lazy block has ever been deferred in this process.
#[inline]
pub(crate) fn ever_active() -> bool {
    EVER_ACTIVE.load(Ordering::Acquire)
}

/// Monotonic generation bumped whenever a lazy block is removed from `LAZY_STATE`.
#[inline]
pub(crate) fn remove_generation() -> u64 {
    REMOVE_GEN.load(Ordering::SeqCst)
}

/// Whether any lazy blocks are currently armed or a thread is mid-arm inside `LAZY_STATE`.
/// Reads `ARMING_COUNT` strictly BEFORE `LAZY_COUNT` (`SeqCst`) so an armer that
/// publishes `LAZY_COUNT` and then decrements `ARMING_COUNT` cannot cause both
/// reads to observe zero after passing `inplace::generation()`.
#[inline]
pub(crate) fn active_or_arming() -> bool {
    let arming = ARMING_COUNT.load(Ordering::SeqCst);
    #[cfg(test)]
    if let Some(hook) = BETWEEN_ACTIVE_OR_ARMING_LOADS_HOOK.with(|h| h.borrow_mut().take()) {
        hook();
    }
    arming > 0 || LAZY_COUNT.load(Ordering::SeqCst) > 0
}

#[inline]
fn should_skip_fault_lookup(active_count: usize, ever_active: bool) -> bool {
    active_count == 0 && !ever_active
}

struct ArmingGuard;

impl ArmingGuard {
    fn enter() -> Self {
        ARMING_COUNT.fetch_add(1, Ordering::SeqCst);
        Self
    }
}

impl Drop for ArmingGuard {
    fn drop(&mut self) {
        ARMING_COUNT.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Whether `VICTORY_LAZY_BACK=1` is enabled for production relay readback.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAZY_BACK").is_some_and(|v| v == "1"))
}

/// Number of distinct frames a block must fault on before promoting to hot (`VICTORY_LAZY_PROMOTE_FRAMES`, default 2).
fn promote_frames() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_LAZY_PROMOTE_FRAMES")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .map(|n| n.max(1))
            .unwrap_or(DEFAULT_PROMOTE_FRAMES)
    })
}

/// Whether internal runtime materializations contribute to hot block promotion
/// (`VICTORY_LAZY_INTERNAL_PROMOTE`, default `1`).
fn internal_promote_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAZY_INTERNAL_PROMOTE").is_none_or(|v| v != "0"))
}

/// Fast lock-free check for whether any lazy blocks are currently armed.
#[inline]
pub(crate) fn active() -> bool {
    #[cfg(test)]
    if let Some(forced) = FORCE_ACTIVE_FOR_TEST.with(std::cell::Cell::get) {
        return forced;
    }
    LAZY_COUNT.load(Ordering::Acquire) > 0
}

#[cfg(test)]
pub(crate) struct ForceActiveGuard(Option<bool>);

#[cfg(test)]
impl Drop for ForceActiveGuard {
    fn drop(&mut self) {
        FORCE_ACTIVE_FOR_TEST.with(|c| c.set(self.0));
    }
}

/// Test helper: overrides `lazy_back::active()` on the current thread only so
/// tests can exercise `!active()` decision paths without racing parallel tests.
#[cfg(test)]
pub(crate) fn force_active_for_test(forced: Option<bool>) -> ForceActiveGuard {
    let prev = FORCE_ACTIVE_FOR_TEST.with(|c| c.replace(forced));
    ForceActiveGuard(prev)
}

#[cfg(test)]
pub(crate) fn clear_force_active_for_test() {
    FORCE_ACTIVE_FOR_TEST.with(|c| c.set(None));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ArmedVa {
    /// Exact start VA protected with `PAGE_NOACCESS` (`block * PAGE` for full
    /// 64 KiB views; `cur` for sub-64 KiB aliases attached on failed materialize).
    va: usize,
    /// Exact byte extent protected with `PAGE_NOACCESS` (`PAGE` for full 64 KiB
    /// views; `step <= PAGE` for sub-64 KiB aliases).
    len: usize,
}

struct LazyEntry {
    shadow: Arc<Shadow>,
    offset: usize,
    len: usize,
    fetched_after: u64,
    armed_vas: Vec<ArmedVa>,
    is_view: bool,
    cpu_clean_after: Option<u64>,
    materializing: bool,
    /// Set if restoring `PAGE_READWRITE` via `VirtualProtect` failed on this
    /// entry so fault waiters observe `MaterializeOutcome::Failed` instead of
    /// assuming the page is readable.
    restore_failed: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MaterializeReason {
    CpuFault { is_write: bool },
    InternalRead,
    Retire,
}

#[derive(Clone, Copy)]
struct HotEntry {
    faults: u32,
    last_frame: u64,
    promoted_frame: u64,
    reprobes: u32,
    reprobing: bool,
}

struct LazyTable {
    pages: BTreeMap<usize, LazyEntry>,
    alias_to_primary: BTreeMap<usize, usize>,
    by_shadow: HashMap<usize, BTreeSet<usize>>,
    hot: HashMap<usize, HotEntry>,
    recent_order: VecDeque<usize>,
    recent_set: HashSet<usize>,
    next_remap_id: u64,
    /// Active `(id, first_block, last_block)` remap ranges registered by
    /// `watch::title_memory_remapping` so no block in a remapped range can be
    /// armed between the pre-remap drain and `title_memory_remapped`'s discard.
    active_remaps: Vec<(u64, usize, usize)>,
}

impl LazyTable {
    fn new() -> Self {
        Self {
            pages: BTreeMap::new(),
            alias_to_primary: BTreeMap::new(),
            by_shadow: HashMap::new(),
            hot: HashMap::new(),
            recent_order: VecDeque::new(),
            recent_set: HashSet::new(),
            next_remap_id: 1,
            active_remaps: Vec::new(),
        }
    }

    fn overlaps_remap(&self, blocks: &[usize]) -> bool {
        if self.active_remaps.is_empty() {
            return false;
        }
        blocks.iter().any(|&b| {
            self.active_remaps
                .iter()
                .any(|&(_, first, last)| b >= first && b < last)
        })
    }

    fn note_recent(&mut self, block: usize) {
        if self.recent_set.insert(block) {
            self.recent_order.push_back(block);
            if self.recent_order.len() > RECENT_CAP
                && let Some(evicted) = self.recent_order.pop_front()
            {
                self.recent_set.remove(&evicted);
            }
        }
    }

    fn clear_recent(&mut self, block: usize) {
        if self.recent_set.remove(&block) {
            self.recent_order.retain(|&b| b != block);
        }
    }
}

static LAZY_STATE: LazyLock<Mutex<LazyTable>> = LazyLock::new(|| Mutex::new(LazyTable::new()));
static LAZY_CV: Condvar = Condvar::new();

struct LazyStats {
    deferred_new: AtomicU64,
    deferred_reused: AtomicU64,
    skipped_bytes: AtomicU64,
    materialized_bytes: AtomicU64,
    read_faults: AtomicU64,
    write_faults: AtomicU64,
    internal_materializes: AtomicU64,
    retire_materializes: AtomicU64,
    dma_batches: AtomicU64,
    hot_promoted: AtomicU64,
}

static STATS: LazyStats = LazyStats {
    deferred_new: AtomicU64::new(0),
    deferred_reused: AtomicU64::new(0),
    skipped_bytes: AtomicU64::new(0),
    materialized_bytes: AtomicU64::new(0),
    read_faults: AtomicU64::new(0),
    write_faults: AtomicU64::new(0),
    internal_materializes: AtomicU64::new(0),
    retire_materializes: AtomicU64::new(0),
    dma_batches: AtomicU64::new(0),
    hot_promoted: AtomicU64::new(0),
};

std::thread_local! {
    /// Primary blocks currently being materialized on this thread so nested
    /// `read_into` / `write_mapped` inside `card_merge_relay` / `relay_merge_rebar`
    /// never wait on `LAZY_CV` for themselves.
    static IN_MATERIALIZE: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn current_thread_materializing(block: usize) -> bool {
    IN_MATERIALIZE.with(|v| v.borrow().contains(&block))
}

struct MaterializeBatchScope {
    blocks: Vec<(usize, bool)>,
    finished: bool,
}

impl MaterializeBatchScope {
    fn enter(blocks: Vec<usize>) -> Self {
        IN_MATERIALIZE.with(|v| v.borrow_mut().extend_from_slice(&blocks));
        Self {
            blocks: blocks.into_iter().map(|b| (b, false)).collect(),
            finished: false,
        }
    }

    fn mark_merged(&mut self, block: usize) {
        if let Some(slot) = self.blocks.iter_mut().find(|(b, _)| *b == block) {
            slot.1 = true;
        }
    }
}

#[cfg(test)]
static FAIL_RESTORE_PROTECT_VAS: LazyLock<Mutex<HashMap<usize, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[cfg(test)]
static FAIL_PROTECT_NEW_ALIAS_VAS: LazyLock<Mutex<HashMap<usize, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[cfg(test)]
struct FailProtectVaGuard {
    map: &'static LazyLock<Mutex<HashMap<usize, usize>>>,
    va: usize,
}

#[cfg(test)]
impl FailProtectVaGuard {
    fn insert(map: &'static LazyLock<Mutex<HashMap<usize, usize>>>, va: usize) -> Self {
        *map.lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(va)
            .or_insert(0) += 1;
        Self { map, va }
    }
}

#[cfg(test)]
impl Drop for FailProtectVaGuard {
    fn drop(&mut self) {
        let mut guard = self.map.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(count) = guard.get_mut(&self.va) {
            *count -= 1;
            if *count == 0 {
                guard.remove(&self.va);
            }
        }
    }
}

#[cfg(test)]
fn fail_restore_protect_guard(va: usize) -> FailProtectVaGuard {
    FailProtectVaGuard::insert(&FAIL_RESTORE_PROTECT_VAS, va)
}

#[cfg(test)]
fn fail_protect_new_alias_guard(va: usize) -> FailProtectVaGuard {
    FailProtectVaGuard::insert(&FAIL_PROTECT_NEW_ALIAS_VAS, va)
}

fn restore_page_readwrite(armed: ArmedVa) -> bool {
    #[cfg(test)]
    if FAIL_RESTORE_PROTECT_VAS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .contains_key(&armed.va)
    {
        return false;
    }
    let mut old = 0u32;
    // SAFETY: `armed.va..armed.va + armed.len` was armed by us with
    // `PAGE_NOACCESS`; restoring `PAGE_READWRITE` over that exact extent.
    unsafe { VirtualProtect(armed.va as *const _, armed.len, PAGE_READWRITE, &mut old) != 0 }
}

fn protect_new_alias_noaccess(va: usize, step: usize) -> bool {
    #[cfg(test)]
    if FAIL_PROTECT_NEW_ALIAS_VAS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .contains_key(&va)
    {
        return false;
    }
    let mut old = 0u32;
    // SAFETY: `va..va+step` is the freshly mapped title view.
    unsafe { VirtualProtect(va as *const _, step, PAGE_NOACCESS, &mut old) != 0 }
}

impl Drop for MaterializeBatchScope {
    fn drop(&mut self) {
        IN_MATERIALIZE.with(|v| {
            let mut list = v.borrow_mut();
            for &(block, _) in self.blocks.iter().rev() {
                if let Some(pos) = list.iter().rposition(|&b| b == block) {
                    list.remove(pos);
                }
            }
        });
        if !self.finished {
            // Unwind safety:
            // - For blocks whose GPU bytes have NOT merged into RAM (`merged == false`),
            //   keep the entry armed at `PAGE_NOACCESS`, clear `materializing = false`,
            //   and wake `LAZY_CV` waiters so they retry and fetch the real GPU bytes
            //   instead of reading stale RAM.
            // - For blocks whose GPU bytes HAVE merged (`merged == true`), restore
            //   `PAGE_READWRITE` on every armed VA BEFORE removing the entry and waking
            //   waiters. If `VirtualProtect` fails for any VA, retain that entry with
            //   `restore_failed = true` and `materializing = false`.
            let mut unmerged_blocks = Vec::new();
            let merged_block_vas: Vec<(usize, Vec<ArmedVa>)> = {
                let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
                let mut merged = Vec::new();
                for &(block, is_merged) in &self.blocks {
                    if is_merged {
                        if let Some(entry) = table.pages.get(&block) {
                            merged.push((block, entry.armed_vas.clone()));
                        }
                    } else {
                        unmerged_blocks.push(block);
                    }
                }
                merged
            };
            let mut restored_blocks = Vec::with_capacity(merged_block_vas.len());
            let mut failed_blocks = Vec::new();
            for (block, vas) in merged_block_vas {
                let mut all_restored = true;
                for armed in vas {
                    if !restore_page_readwrite(armed) {
                        all_restored = false;
                    }
                }
                if all_restored {
                    restored_blocks.push(block);
                } else {
                    tracing::error!(
                        primary = format_args!("{:#x}", block * PAGE),
                        "lazy copy-back unwind VirtualProtect restoration failed; retaining entry"
                    );
                    failed_blocks.push(block);
                }
            }
            let mut removed = Vec::with_capacity(restored_blocks.len());
            {
                let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
                for block in unmerged_blocks {
                    if let Some(entry) = table.pages.get_mut(&block) {
                        entry.materializing = false;
                    }
                }
                for block in restored_blocks {
                    if let Some(entry) = remove_entry_locked_no_notify(&mut table, block, true) {
                        removed.push(entry);
                    }
                }
                for block in failed_blocks {
                    if let Some(entry) = table.pages.get_mut(&block) {
                        entry.materializing = false;
                        entry.restore_failed = true;
                    }
                }
                LAZY_COUNT.store(table.pages.len(), Ordering::Release);
                LAZY_CV.notify_all();
            }
            drop(removed);
        }
    }
}

type BatchFetch = (Vec<u8>, Vec<(usize, usize)>, u64);

struct LazyFetcher {
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    event: HANDLE,
    sent: u64,
    /// Set immediately before `ExecuteCommandLists` and cleared ONLY after
    /// GPU fence completion is proven so an unproven in-flight allocator or
    /// readback buffer is never reset or released.
    in_flight: bool,
    /// Set once `self.queue.Signal(&self.fence, self.sent)` succeeded for the
    /// current in-flight batch so `direct_submit_guard()` knows whether polling
    /// `self.fence.GetCompletedValue() >= self.sent` can prove GPU completion.
    signal_succeeded: bool,
    /// Source buffers referenced by `CopyBufferRegion` while `in_flight` is
    /// true. Cleared immediately once the fence confirms completion; if the
    /// fetcher is detached after an unconfirmed post-submit error, these COM
    /// references stay alive with the detached fetcher so D3D12 COPY cannot
    /// read from a released `ID3D12Resource`.
    in_flight_sources: Vec<ID3D12Resource>,
    readback: Option<(ID3D12Resource, *mut u8, usize)>,
}

// SAFETY: D3D12 COM pointers and Win32 event HANDLE are accessed exclusively under `FETCHER`.
unsafe impl Send for LazyFetcher {}

impl Drop for LazyFetcher {
    fn drop(&mut self) {
        // SAFETY: `self.event` was created by `CreateEventW` and is owned solely by `LazyFetcher`.
        unsafe {
            let _ = CloseHandle(self.event);
        }
    }
}

impl LazyFetcher {
    fn new() -> windows::core::Result<Self> {
        let host = crate::host::host().ok_or_else(|| windows::core::Error::from_hresult(E_FAIL))?;
        // SAFETY: plain D3D12 COPY queue, allocator, list, fence, and Win32 auto-reset event.
        unsafe {
            let queue: ID3D12CommandQueue =
                host.device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_COPY,
                    ..Default::default()
                })?;
            let allocator: ID3D12CommandAllocator = host
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
            let list: ID3D12GraphicsCommandList =
                host.device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
            list.Close()?;
            let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            let readback = host
                .shared_buffer(FETCHER_INIT_BYTES as u64)
                .or_else(|| host.shared_buffer(PAGE as u64))
                .map(|(res, ptr)| {
                    crate::host::name(&res, || "lazy copy-back readback".to_owned());
                    let cap = if ptr.is_null() {
                        0
                    } else {
                        res.GetDesc().Width as usize
                    };
                    (res, ptr, cap)
                });
            Ok(Self {
                queue,
                allocator,
                list,
                fence,
                event,
                sent: 0,
                in_flight: false,
                signal_succeeded: false,
                in_flight_sources: Vec::new(),
                readback,
            })
        }
    }

    fn wait_for_fence(&self, target: u64) -> windows::core::Result<()> {
        use windows::Win32::Foundation::WAIT_OBJECT_0;
        // SAFETY: `self.fence` and `self.event` are live and serialized under `FETCHER`.
        unsafe {
            let completed = self.fence.GetCompletedValue();
            if completed == u64::MAX {
                return Err(windows::core::Error::from_hresult(E_FAIL));
            }
            if completed >= target {
                return Ok(());
            }
            let event_ok = self.fence.SetEventOnCompletion(target, self.event).is_ok()
                && WaitForSingleObject(self.event, INFINITE) == WAIT_OBJECT_0;
            if event_ok {
                let after = self.fence.GetCompletedValue();
                if after == u64::MAX {
                    return Err(windows::core::Error::from_hresult(E_FAIL));
                }
                if after >= target {
                    return Ok(());
                }
            }
            // Fallback when event registration or `WaitForSingleObject` fails:
            // poll `GetCompletedValue` until completion is established or the
            // device is removed / times out.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            let mut spins = 0u32;
            loop {
                let val = self.fence.GetCompletedValue();
                if val == u64::MAX {
                    return Err(windows::core::Error::from_hresult(E_FAIL));
                }
                if val >= target {
                    return Ok(());
                }
                if std::time::Instant::now() >= deadline {
                    return Err(windows::core::Error::from_hresult(E_FAIL));
                }
                spins = spins.saturating_add(1);
                if spins < 64 {
                    std::hint::spin_loop();
                } else if spins < 256 {
                    std::thread::yield_now();
                } else {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        }
    }

    fn fetch_batch(
        &mut self,
        items: &[(&Shadow, usize, usize)],
        wait_direct: bool,
    ) -> windows::core::Result<BatchFetch> {
        if items.is_empty() {
            return Ok((Vec::new(), Vec::new(), 0));
        }
        let host = crate::host::host().ok_or_else(|| windows::core::Error::from_hresult(E_FAIL))?;
        let mut total_bytes = 0usize;
        let mut slices = Vec::with_capacity(items.len());
        for &(_, _, len) in items {
            let dst_off = total_bytes;
            total_bytes = total_bytes
                .checked_add(len)
                .ok_or_else(|| windows::core::Error::from_hresult(E_FAIL))?;
            slices.push((dst_off, len));
        }
        let needed = total_bytes.max(PAGE);
        if self
            .readback
            .as_ref()
            .is_none_or(|(_, _, cap)| *cap < needed)
        {
            let alloc_len = needed.max(FETCHER_INIT_BYTES).next_multiple_of(PAGE);
            let (res, ptr) = host
                .shared_buffer(alloc_len as u64)
                .ok_or_else(|| windows::core::Error::from_hresult(E_FAIL))?;
            crate::host::name(&res, || "lazy copy-back readback".to_owned());
            self.readback = Some((res, ptr, alloc_len));
        }
        let (rb_res, rb_ptr, _) = self
            .readback
            .as_ref()
            .ok_or_else(|| windows::core::Error::from_hresult(E_FAIL))?;

        // SAFETY: every `shadow.buffer` is live (held by `Arc<Shadow>` and
        // retained in `self.in_flight_sources` until fence completion is
        // proven) in `COMMON` state (implicit promotion to `COPY_SOURCE` and
        // decay back to `COMMON` at `ExecuteCommandLists`), `rb_res` is live
        // with capacity >= `total_bytes`, and `FETCHER` serializes this
        // queue/list/fence.
        unsafe {
            self.in_flight_sources.clear();
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            for (&(shadow, offset, _), &(dst_off, len)) in items.iter().zip(&slices) {
                if len > 0 {
                    self.in_flight_sources.push(shadow.buffer.clone());
                    self.list.CopyBufferRegion(
                        rb_res,
                        dst_off as u64,
                        &shadow.buffer,
                        offset as u64,
                        len as u64,
                    );
                }
            }
            self.list.Close()?;
            let fetched_after = host.sync_copy_queue_for(&self.queue, wait_direct)?;
            let lists = [Some(self.list.cast::<ID3D12CommandList>()?)];
            self.in_flight = true;
            self.signal_succeeded = false;
            self.queue.ExecuteCommandLists(&lists);
            self.sent += 1;
            #[cfg(test)]
            THREAD_DMA_BATCHES.with(|c| c.set(c.get() + 1));
            STATS.dma_batches.fetch_add(1, Ordering::Relaxed);
            self.queue.Signal(&self.fence, self.sent)?;
            self.signal_succeeded = true;
            self.wait_for_fence(self.sent)?;
            #[cfg(test)]
            if FAIL_AFTER_SUBMIT_FOR_TEST.with(|f| f.replace(false)) {
                return Err(windows::core::Error::from_hresult(E_FAIL));
            }
            self.in_flight = false;
            self.signal_succeeded = false;
            self.in_flight_sources.clear();
            let mut out = vec![0u8; total_bytes];
            if total_bytes > 0 {
                std::ptr::copy_nonoverlapping(*rb_ptr, out.as_mut_ptr(), total_bytes);
            }
            Ok((out, slices, fetched_after))
        }
    }
}

/// A detached in-flight COPY fetcher whose completion was not confirmed before
/// `run_fetcher_batch` returned an error. Kept alive inside `FETCHER` until
/// its fence proves GPU completion (if `signalled`) so neither its COM
/// references nor its ordering behind subsequent DIRECT submissions are lost.
struct UnconfirmedCopy {
    fetcher: LazyFetcher,
    fence_value: u64,
    signalled: bool,
}

impl UnconfirmedCopy {
    fn is_completed(&self) -> bool {
        if !self.signalled {
            return false;
        }
        // SAFETY: `self.fetcher.fence` is live for as long as `self` is retained.
        let completed = unsafe { self.fetcher.fence.GetCompletedValue() };
        completed != u64::MAX && completed >= self.fence_value
    }
}

struct FetcherState {
    active: Option<LazyFetcher>,
    unconfirmed: Vec<UnconfirmedCopy>,
}

/// Drops any detached COPY fetches whose fence now proves completion, leaving
/// unproven fetches alive and tracked.
fn retire_completed_unconfirmed(unconfirmed: &mut Vec<UnconfirmedCopy>) {
    unconfirmed.retain(|entry| !entry.is_completed());
}

/// Leaf lock protecting the dedicated `COPY`-queue fetcher AND serializing
/// `Host::queue` DIRECT submissions against active or unconfirmed `COPY`-queue
/// reads when `lazy_back` can arm. Never acquired inside `LAZY_STATE`,
/// `watch::GUARDS`, `shadow::BUFFERS`, or any `Shadow` lock.
static FETCHER: Mutex<FetcherState> = Mutex::new(FetcherState {
    active: None,
    unconfirmed: Vec::new(),
});

/// Opaque guard holding `FETCHER` across a `Host::queue` DIRECT submission
/// (`ExecuteCommandLists + Signal + note_signalled`) when `lazy_back` can arm.
pub(crate) struct DirectSubmitGuard<'a>(
    #[allow(dead_code)] Option<std::sync::MutexGuard<'a, FetcherState>>,
);

/// Whether lazy copy-back can arm blocks in this build (`enabled()` in
/// production, always `true` in tests where `try_defer_page_for_test` can arm
/// without `VICTORY_LAZY_BACK=1`).
#[inline]
pub(crate) fn can_arm() -> bool {
    #[cfg(test)]
    {
        true
    }
    #[cfg(not(test))]
    {
        enabled()
    }
}

#[inline]
fn direct_submit_guard_with_state<'a>(
    state: &'a Mutex<FetcherState>,
    serialize: bool,
) -> DirectSubmitGuard<'a> {
    if !serialize {
        return DirectSubmitGuard(None);
    }
    let mut guard = state.lock().unwrap_or_else(|e| e.into_inner());
    if !guard.unconfirmed.is_empty() {
        retire_completed_unconfirmed(&mut guard.unconfirmed);
        if !guard.unconfirmed.is_empty() {
            let remaining = guard.unconfirmed.len();
            // Drop `state` before calling `host::fail` so tests using
            // `fail_panics_for_test()` never poison the mutex.
            drop(guard);
            crate::host::fail(format_args!(
                "DIRECT submission rejected: {remaining} unconfirmed COPY-queue lazy fetch(es) still unproven"
            ));
        }
    }
    DirectSubmitGuard(Some(guard))
}

#[inline]
fn direct_submit_guard_if(serialize: bool) -> DirectSubmitGuard<'static> {
    direct_submit_guard_with_state(&FETCHER, serialize)
}

/// Serializes a DIRECT-queue submission (`ExecuteCommandLists + Signal +
/// note_signalled`) against in-flight `COPY`-queue lazy fetches so no DIRECT
/// writer can race with an unordered COPY read on the same shadow buffer,
/// including across the very first lazy activation. Fails closed if an earlier
/// detached COPY fetch has not been proven complete on its fence.
/// Lock-free no-op in production when `VICTORY_LAZY_BACK` is not enabled.
pub(crate) fn direct_submit_guard() -> DirectSubmitGuard<'static> {
    direct_submit_guard_if(can_arm())
}

#[cfg(test)]
std::thread_local! {
    static FORCE_ACTIVE_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    static THREAD_DMA_BATCHES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static FAIL_NEXT_FETCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static FAIL_AFTER_SUBMIT_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static LAST_DETACHED_SOURCE_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static BEFORE_FETCH_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
    static AFTER_MERGE_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
    static BEFORE_LAZY_CV_WAIT_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
    static AFTER_OWN_VIEW_CHECK_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
    static BETWEEN_ACTIVE_OR_ARMING_LOADS_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
    static EVER_ACTIVE_OBSERVER: std::cell::RefCell<Option<Box<dyn FnMut()>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn set_before_fetch_hook_for_test(hook: Option<Box<dyn FnOnce()>>) {
    BEFORE_FETCH_HOOK.with(|h| *h.borrow_mut() = hook);
}

#[cfg(test)]
pub(crate) fn set_after_merge_hook_for_test(hook: Option<Box<dyn FnOnce()>>) {
    AFTER_MERGE_HOOK.with(|h| *h.borrow_mut() = hook);
}

#[cfg(test)]
pub(crate) fn set_fail_next_fetches_for_test(count: usize) {
    FAIL_NEXT_FETCHES.with(|f| f.set(count));
}

#[cfg(test)]
pub(crate) fn clear_hot_for_test_pub(block: usize) {
    let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
    table.hot.remove(&block);
}

#[cfg(test)]
fn thread_dma_batches() -> u64 {
    THREAD_DMA_BATCHES.with(|c| c.get())
}

fn run_fetcher_batch(
    guard: &mut Option<LazyFetcher>,
    unconfirmed: &mut Vec<UnconfirmedCopy>,
    items: &[(&Shadow, usize, usize)],
    wait_direct: bool,
) -> windows::core::Result<BatchFetch> {
    if !unconfirmed.is_empty() {
        retire_completed_unconfirmed(unconfirmed);
    }
    if guard.is_none() {
        *guard = Some(LazyFetcher::new()?);
    }
    let result = guard
        .as_mut()
        .ok_or_else(|| windows::core::Error::from_hresult(E_FAIL))?
        .fetch_batch(items, wait_direct);
    if result.is_err() {
        if guard.as_ref().is_some_and(|f| f.in_flight) {
            // Completion could not be proven after `ExecuteCommandLists`: detach
            // the in-flight fetcher (including its retained `in_flight_sources`
            // COM references) and track it in `unconfirmed` so neither its
            // resources are released prematurely nor can a subsequent DIRECT
            // submission overlap an unproven COPY read.
            if let Some(in_flight_fetcher) = guard.take() {
                #[cfg(test)]
                LAST_DETACHED_SOURCE_COUNT
                    .with(|c| c.set(in_flight_fetcher.in_flight_sources.len()));
                let fence_value = in_flight_fetcher.sent;
                let signalled = in_flight_fetcher.signal_succeeded;
                unconfirmed.push(UnconfirmedCopy {
                    fetcher: in_flight_fetcher,
                    fence_value,
                    signalled,
                });
            }
        } else if let Some(fetcher) = guard.as_mut() {
            fetcher.in_flight_sources.clear();
        }
    }
    result
}

fn fetch_gpu_batch(items: &[(&Shadow, usize, usize)]) -> windows::core::Result<BatchFetch> {
    if items.is_empty() {
        return Ok((Vec::new(), Vec::new(), 0));
    }
    #[cfg(test)]
    if FAIL_NEXT_FETCHES.with(|f| {
        let rem = f.get();
        if rem > 0 {
            f.set(rem - 1);
            true
        } else {
            false
        }
    }) {
        return Err(windows::core::Error::from_hresult(E_FAIL));
    }
    // Acquire `FETCHER` BEFORE snapshotting `wait_direct` so DIRECT submissions
    // serialized via `direct_submit_guard()` are strictly ordered either before
    // this check (setting `wait_direct = true` + `last_signalled`) or after our
    // COPY read finishes on the GPU and releases `FETCHER`.
    let mut guard = FETCHER.lock().unwrap_or_else(|e| e.into_inner());
    let wait_direct = items
        .iter()
        .any(|&(shadow, _, _)| shadow.has_inflight_gpu_write());
    let FetcherState {
        active,
        unconfirmed,
    } = &mut *guard;
    run_fetcher_batch(active, unconfirmed, items, wait_direct)
}

#[cfg(test)]
fn fetch_gpu_bytes(shadow: &Shadow, offset: usize, len: usize) -> windows::core::Result<Vec<u8>> {
    let mut guard = FETCHER.lock().unwrap_or_else(|e| e.into_inner());
    let FetcherState {
        active,
        unconfirmed,
    } = &mut *guard;
    let (buf, _, _) = run_fetcher_batch(active, unconfirmed, &[(shadow, offset, len)], true)?;
    Ok(buf)
}

fn remove_entry_locked_no_notify(
    table: &mut LazyTable,
    primary_block: usize,
    record_recent: bool,
) -> Option<LazyEntry> {
    let entry = table.pages.remove(&primary_block)?;
    if entry.is_view {
        crate::watch::clear_lazy_view_armed(entry.armed_vas.iter().map(|a| a.va));
    }
    for &armed in &entry.armed_vas {
        let b = armed.va / PAGE;
        table.alias_to_primary.remove(&b);
        if record_recent {
            table.note_recent(b);
        }
    }
    let shadow_ptr = Arc::as_ptr(&entry.shadow) as usize;
    if let Some(blocks) = table.by_shadow.get_mut(&shadow_ptr) {
        blocks.remove(&primary_block);
        if blocks.is_empty() {
            table.by_shadow.remove(&shadow_ptr);
        }
    }
    REMOVE_GEN.fetch_add(1, Ordering::SeqCst);
    Some(entry)
}

fn is_readable_now(at: usize) -> bool {
    // SAFETY: plain `VirtualQuery` into a zeroed `MEMORY_BASIC_INFORMATION`.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid out pointer.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0
        && info.State == MEM_COMMIT
        && matches!(
            info.Protect,
            PAGE_READONLY | PAGE_READWRITE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE
        )
}

/// Attempts to defer copy-back of `[offset, offset + len)` in `shadow` by arming
/// `PAGE_NOACCESS` across its title virtual address and any mapped section aliases.
/// Returns `true` if the block was deferred (new or reused), `false` if the caller
/// must copy it back eagerly.
pub(crate) fn try_defer_page(
    shadow: &Arc<Shadow>,
    offset: usize,
    len: usize,
    fetched_after: u64,
) -> bool {
    if !enabled() {
        return false;
    }
    try_defer_page_inner(shadow, offset, len, fetched_after)
}

#[cfg(test)]
pub(crate) fn try_defer_page_for_test(
    shadow: &Arc<Shadow>,
    offset: usize,
    len: usize,
    fetched_after: u64,
) -> bool {
    try_defer_page_inner(shadow, offset, len, fetched_after)
}

#[cfg(test)]
std::thread_local! {
    static BEFORE_PROTECT_HOOK: std::cell::RefCell<Option<Box<dyn FnMut()>>> =
        const { std::cell::RefCell::new(None) };
}

fn try_defer_page_inner(
    shadow: &Arc<Shadow>,
    offset: usize,
    len: usize,
    fetched_after: u64,
) -> bool {
    if len != PAGE
        || shadow.is_placed()
        || (!shadow.carded() && !shadow.is_rebar())
        || offset.saturating_add(len) > shadow.len() as usize
    {
        return false;
    }
    let primary_va = (shadow.start() as usize).wrapping_add(offset);
    if !primary_va.is_multiple_of(PAGE) || crate::host::host().is_none() {
        return false;
    }
    let primary_block = primary_va / PAGE;
    let frame = crate::frame::current();

    {
        let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(hot) = table.hot.get_mut(&primary_block) {
            let threshold = promote_frames();
            if frame > hot.last_frame.saturating_add(HOT_DECAY_FRAMES) {
                table.hot.remove(&primary_block);
            } else if hot.faults >= threshold {
                let reprobe_after = HOT_REPROBE_FRAMES << hot.reprobes.min(4);
                if frame > hot.promoted_frame.saturating_add(reprobe_after) {
                    hot.faults = threshold.saturating_sub(1);
                    hot.last_frame = frame;
                    hot.promoted_frame = frame;
                    hot.reprobes = hot.reprobes.saturating_add(1);
                    hot.reprobing = true;
                } else {
                    hot.last_frame = frame;
                    return false;
                }
            }
        }
        if let Some(existing) = table.pages.get_mut(&primary_block) {
            if existing.materializing
                || existing.restore_failed
                || !Arc::ptr_eq(&existing.shadow, shadow)
            {
                return false;
            }
            existing.fetched_after = existing.fetched_after.max(fetched_after);
            if existing.is_view {
                crate::watch::note_lazy_view_reused(existing.armed_vas.iter().map(|a| a.va));
            }
            STATS.deferred_reused.fetch_add(1, Ordering::Relaxed);
            STATS.skipped_bytes.fetch_add(len as u64, Ordering::Relaxed);
            return true;
        }
        if table.alias_to_primary.contains_key(&primary_block) {
            return false;
        }
    }

    // Capture `inplace::generation()` (`SeqCst`) strictly BEFORE
    // `has_own_view_for`, `has_view_at`, and `pages_of` so any remap completing
    // after this snapshot (including between `has_own_view_for` and `pages_of`)
    // either causes revalidation to reject arming below or observes
    // `ARMING_COUNT`/`LAZY_COUNT` in `views_mapped`.
    let view_gen = crate::inplace::generation();

    // Only defer pages that have a host-only `PAGE_READWRITE` section own view
    // so materialization can ALWAYS merge GPU bytes into physical RAM before
    // lifting `PAGE_NOACCESS` on title views (preventing concurrent title
    // threads from ever reading stale RAM). Private non-section allocations
    // without an own-view merge path stay on eager readback.
    if !crate::inplace::has_own_view_for(primary_va, PAGE) {
        return false;
    }
    #[cfg(test)]
    if let Some(hook) = AFTER_OWN_VIEW_CHECK_HOOK.with(|h| h.borrow_mut().take()) {
        hook();
    }

    let is_view =
        crate::inplace::has_view_at(primary_va) || crate::watch::is_known_view_block(primary_block);
    let mut vas = crate::inplace::pages_of(primary_va);
    if !vas.contains(&primary_va) {
        vas.push(primary_va);
    }
    if let Some(pos) = vas.iter().position(|&va| va == primary_va)
        && pos != 0
    {
        // Arm `primary_va` first so a protection failure on it never leaves aliases modified.
        vas.swap(0, pos);
    }
    if vas.len() > crate::watch::MANY_VIEWS || vas.iter().any(|&va| !va.is_multiple_of(PAGE)) {
        return false;
    }
    let blocks: Vec<usize> = vas.iter().map(|&va| va / PAGE).collect();

    if !crate::watch::ensure_handler() {
        return false;
    }
    crate::watch::try_arm_lazy_under_guards(&blocks, |cpu_clean_after| {
        let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
        let _arming = ArmingGuard::enter();
        if table.overlaps_remap(&blocks) || crate::inplace::generation() != view_gen {
            return false;
        }
        if let Some(existing) = table.pages.get_mut(&primary_block) {
            if existing.materializing
                || existing.restore_failed
                || !Arc::ptr_eq(&existing.shadow, shadow)
            {
                return false;
            }
            existing.fetched_after = existing.fetched_after.max(fetched_after);
            if existing.is_view {
                crate::watch::note_lazy_view_reused(existing.armed_vas.iter().map(|a| a.va));
            }
            STATS.deferred_reused.fetch_add(1, Ordering::Relaxed);
            STATS.skipped_bytes.fetch_add(len as u64, Ordering::Relaxed);
            return true;
        }
        if table.alias_to_primary.contains_key(&primary_block)
            || vas
                .iter()
                .any(|&va| table.alias_to_primary.contains_key(&(va / PAGE)))
        {
            return false;
        }

        // Publish `EVER_ACTIVE` BEFORE arming `PAGE_NOACCESS` so any thread faulting on
        // the very first deferred page observes `EVER_ACTIVE == true` in `handle_fault`.
        EVER_ACTIVE.store(true, Ordering::Release);
        #[cfg(test)]
        EVER_ACTIVE_OBSERVER.with(|obs| {
            if let Some(cb) = obs.borrow_mut().as_mut() {
                cb();
            }
        });
        #[cfg(test)]
        BEFORE_PROTECT_HOOK.with(|hook| {
            if let Some(cb) = hook.borrow_mut().as_mut() {
                cb();
            }
        });

        let mut protected: Vec<(usize, u32)> = Vec::with_capacity(vas.len());
        for &va in &vas {
            let mut old = 0u32;
            // SAFETY: `va` is a 64 KiB-aligned page in title memory; any protection
            // failure (primary or alias) immediately rolls back all previously
            // modified protections and rejects deferral so eager readback runs.
            let ok = unsafe { VirtualProtect(va as *const _, PAGE, PAGE_NOACCESS, &mut old) != 0 };
            if ok {
                protected.push((va, old));
            } else {
                for (prev_va, prev_prot) in protected {
                    let mut dummy = 0u32;
                    // SAFETY: rolling back protection changed moments ago.
                    unsafe { VirtualProtect(prev_va as *const _, PAGE, prev_prot, &mut dummy) };
                }
                return false;
            }
        }

        let armed_vas: Vec<ArmedVa> = protected
            .into_iter()
            .map(|(va, _)| ArmedVa { va, len: PAGE })
            .collect();
        if is_view {
            crate::watch::note_lazy_view_armed(armed_vas.iter().map(|a| a.va));
        }
        for &armed in &armed_vas {
            let b = armed.va / PAGE;
            table.clear_recent(b);
            table.alias_to_primary.insert(b, primary_block);
        }
        table
            .by_shadow
            .entry(Arc::as_ptr(shadow) as usize)
            .or_default()
            .insert(primary_block);
        table.pages.insert(
            primary_block,
            LazyEntry {
                shadow: shadow.clone(),
                offset,
                len,
                fetched_after,
                armed_vas,
                is_view,
                cpu_clean_after: if is_view { cpu_clean_after } else { None },
                materializing: false,
                restore_failed: false,
            },
        );
        LAZY_COUNT.store(table.pages.len(), Ordering::SeqCst);
        STATS.deferred_new.fetch_add(1, Ordering::Relaxed);
        STATS.skipped_bytes.fetch_add(len as u64, Ordering::Relaxed);
        true
    })
}

/// Number of currently deferred primary blocks owned by `shadow_ptr`.
pub(crate) fn block_count_for_shadow(shadow_ptr: usize) -> usize {
    if !active() {
        return 0;
    }
    let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
    table.by_shadow.get(&shadow_ptr).map_or(0, BTreeSet::len)
}

#[inline]
fn begin_remap_if(at: usize, len: usize, allow: bool) -> u64 {
    if len == 0 || !allow {
        return 0;
    }
    let first = at / PAGE;
    let last = at.saturating_add(len).div_ceil(PAGE);
    if first >= last {
        return 0;
    }
    let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
    let id = table.next_remap_id;
    table.next_remap_id = table.next_remap_id.wrapping_add(1).max(1);
    table.active_remaps.push((id, first, last));
    id
}

/// Registers `[at, at + len)` as actively remapping so no lazy block in that
/// range can be armed between the pre-remap drain and `discard_range`.
/// Lock-free no-op in production when `VICTORY_LAZY_BACK` is not enabled.
pub(crate) fn begin_remap(at: usize, len: usize) -> u64 {
    begin_remap_if(at, len, can_arm())
}

/// Unregisters an active remap range previously registered by [`begin_remap`].
pub(crate) fn end_remap(id: u64) {
    if id == 0 {
        return;
    }
    let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(pos) = table
        .active_remaps
        .iter()
        .position(|&(rid, _, _)| rid == id)
    {
        table.active_remaps.swap_remove(pos);
    }
}

/// Returns `true` if `block` is currently registered as a lazy block (or alias of one).
pub(crate) fn has_block(block: usize) -> bool {
    if !active() {
        return false;
    }
    let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
    table.alias_to_primary.contains_key(&block)
}

/// Returns `true` if any active lazy block (or alias of one) overlaps `[at, at + len)`.
/// Acquires `LAZY_STATE` at most once (O(log N) range probe).
pub(crate) fn has_block_in_range(at: usize, len: usize) -> bool {
    if len == 0 || !active() {
        return false;
    }
    let first = at / PAGE;
    let last = at.saturating_add(len).div_ceil(PAGE);
    if first >= last {
        return false;
    }
    let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
    table.alias_to_primary.range(first..last).next().is_some()
}

/// Returns `true` if `shadow_ptr` currently owns `page` as a deferred primary block.
pub(crate) fn owns_block(shadow_ptr: usize, page: usize) -> bool {
    if !active() {
        return false;
    }
    let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
    table
        .pages
        .get(&page)
        .is_some_and(|e| Arc::as_ptr(&e.shadow) as usize == shadow_ptr)
}

/// Returns sorted `(block, cpu_clean_after)` pairs for all primary 64 KiB block
/// indices in `[first_page, last_page)` owned by `shadow_ptr`.
///
/// `cpu_clean_after == Some(pre_epoch)` proves the block was guard-clean at
/// arm time with no CPU write after `pre_epoch` (so `pre_epoch <= copied` means
/// the owning shadow can keep it deferred with zero DMA). `None` or
/// `pre_epoch > copied` means a newer CPU write may exist and must be
/// materialized before the current list runs.
pub(crate) fn owned_blocks_with_clean_in(
    shadow_ptr: usize,
    first_page: usize,
    last_page: usize,
) -> Vec<(usize, Option<u64>)> {
    if !active() || first_page >= last_page {
        return Vec::new();
    }
    let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(blocks) = table.by_shadow.get(&shadow_ptr) else {
        return Vec::new();
    };
    blocks
        .range(first_page..last_page)
        .filter_map(|&b| {
            table.pages.get(&b).map(|entry| {
                (
                    b,
                    (!entry.materializing && !entry.restore_failed)
                        .then_some(entry.cpu_clean_after)
                        .flatten(),
                )
            })
        })
        .collect()
}

/// Returns all primary 64 KiB block indices in `[first_page, last_page)` owned by `shadow_ptr`.
pub(crate) fn owned_blocks_in(
    shadow_ptr: usize,
    first_page: usize,
    last_page: usize,
) -> Vec<usize> {
    if !active() || first_page >= last_page {
        return Vec::new();
    }
    let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(blocks) = table.by_shadow.get(&shadow_ptr) else {
        return Vec::new();
    };
    blocks.range(first_page..last_page).copied().collect()
}

/// Returns sorted `(block, pre_epoch)` pairs for non-`materializing` view
/// blocks in `[first_page, last_page)` owned by `shadow_ptr` that were
/// guard-clean before being armed.
#[cfg(test)]
pub(crate) fn owned_cpu_clean_blocks_in(
    shadow_ptr: usize,
    first_page: usize,
    last_page: usize,
) -> Vec<(usize, u64)> {
    if !active() || first_page >= last_page {
        return Vec::new();
    }
    let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(blocks) = table.by_shadow.get(&shadow_ptr) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for &b in blocks.range(first_page..last_page) {
        if let Some(entry) = table.pages.get(&b)
            && !entry.materializing
            && !entry.restore_failed
            && let Some(pre_epoch) = entry.cpu_clean_after
        {
            out.push((b, pre_epoch));
        }
    }
    out
}

fn note_fault_hot_locked(table: &mut LazyTable, primary_block: usize) {
    let frame = crate::frame::current();
    let threshold = promote_frames();
    let entry = table.hot.entry(primary_block).or_insert(HotEntry {
        faults: 0,
        last_frame: frame,
        promoted_frame: frame,
        reprobes: 0,
        reprobing: false,
    });
    if entry.faults == 0
        || (!entry.reprobing && frame > entry.last_frame.saturating_add(HOT_DECAY_FRAMES))
    {
        entry.faults = 1;
        entry.last_frame = frame;
        entry.reprobing = false;
        if threshold == 1 {
            entry.promoted_frame = frame;
            STATS.hot_promoted.fetch_add(1, Ordering::Relaxed);
        }
    } else if entry.reprobing || frame != entry.last_frame {
        let was_below = entry.faults < threshold;
        entry.faults = entry.faults.saturating_add(1);
        entry.last_frame = frame;
        entry.reprobing = false;
        if was_below && entry.faults >= threshold {
            entry.promoted_frame = frame;
            STATS.hot_promoted.fetch_add(1, Ordering::Relaxed);
        }
    }
}

struct ClaimedBlock {
    primary_block: usize,
    shadow: Arc<Shadow>,
    offset: usize,
    len: usize,
    fetched_after: u64,
    /// VA used to resolve the host-only `PAGE_READWRITE` section own view
    /// (normally `primary_block * PAGE`, or a surviving alias during `discard_range`).
    own_va: usize,
    /// Title VAs and exact protected byte lengths whose `PAGE_NOACCESS`
    /// protection should be restored to `PAGE_READWRITE` once GPU bytes have
    /// landed through the own view.
    restore_vas: Vec<ArmedVa>,
}

/// Materializes `claimed` blocks via the dedicated COPY queue and merges GPU
/// bytes through each block's host-only own view strictly BEFORE restoring
/// `PAGE_READWRITE` on title views.
///
/// Returns `true` if every claimed block materialized cleanly. On DMA or merge
/// failure, failed blocks stay at `PAGE_NOACCESS` in `LAZY_STATE` with
/// `materializing = false` (so stale RAM is never exposed and retries can
/// recover if the error was transient) and this returns `false`.
fn materialize_claimed_batch(claimed: Vec<ClaimedBlock>, reason: MaterializeReason) -> bool {
    if claimed.is_empty() {
        return true;
    }
    let scope_blocks: Vec<usize> = claimed.iter().map(|c| c.primary_block).collect();
    let mut scope = MaterializeBatchScope::enter(scope_blocks);
    #[cfg(test)]
    if let Some(hook) = BEFORE_FETCH_HOOK.with(|h| h.borrow_mut().take()) {
        hook();
    }
    let fetch_items: Vec<(&Shadow, usize, usize)> = claimed
        .iter()
        .map(|c| (&*c.shadow, c.offset, c.len))
        .collect();
    let fetched = fetch_gpu_batch(&fetch_items);

    let restore_prot = |vas: &[ArmedVa]| -> bool {
        let mut all_ok = true;
        for &armed in vas {
            if !restore_page_readwrite(armed) {
                all_ok = false;
            }
        }
        all_ok
    };

    let mut succeeded: Vec<usize> = Vec::with_capacity(claimed.len());
    let mut failed: Vec<usize> = Vec::new();
    let mut restore_failed: Vec<usize> = Vec::new();

    match fetched {
        Ok((buf, slices, live_fetched_after)) => {
            for (item, (buf_off, buf_len)) in claimed.iter().zip(slices) {
                let gpu_bytes = &buf[buf_off..buf_off + buf_len];
                let merge_after = item.fetched_after.max(live_fetched_after);
                // Every deferred block has a verified section own view: merge
                // GPU->title bytes via our host-only `PAGE_READWRITE` view
                // strictly BEFORE restoring `PAGE_READWRITE` on title views so
                // concurrent title threads never observe stale RAM.
                let merged_via_own = if item.shadow.carded() {
                    let mut scratch = Vec::new();
                    item.shadow
                        .card_merge_relay_via_own_at(
                            item.offset,
                            item.own_va,
                            gpu_bytes,
                            merge_after,
                            &mut scratch,
                        )
                        .is_some()
                } else if item.shadow.is_rebar() {
                    if item
                        .shadow
                        .relay_merge_rebar_via_own_at(item.offset, item.own_va, gpu_bytes)
                        .is_some()
                    {
                        item.shadow.mark_rebar_dirty_range(item.offset, item.len);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };

                if merged_via_own {
                    scope.mark_merged(item.primary_block);
                    #[cfg(test)]
                    if let Some(hook) = AFTER_MERGE_HOOK.with(|h| h.borrow_mut().take()) {
                        hook();
                    }
                    if restore_prot(&item.restore_vas) {
                        if reason == (MaterializeReason::CpuFault { is_write: true }) {
                            for &armed in &item.restore_vas {
                                crate::watch::wrote(armed.va, armed.len);
                            }
                        }
                        STATS
                            .materialized_bytes
                            .fetch_add(item.len as u64, Ordering::Relaxed);
                        succeeded.push(item.primary_block);
                    } else {
                        tracing::error!(
                            primary = format_args!("{:#x}", item.primary_block * PAGE),
                            "lazy copy-back VirtualProtect restoration failed; retaining entry"
                        );
                        restore_failed.push(item.primary_block);
                        failed.push(item.primary_block);
                    }
                } else if !crate::inplace::has_view_at(item.own_va) {
                    // View was already unmapped; drop tracking without reopening stale RAM.
                    succeeded.push(item.primary_block);
                } else {
                    tracing::error!(
                        primary = format_args!("{:#x}", item.primary_block * PAGE),
                        own_va = format_args!("{:#x}", item.own_va),
                        "lazy copy-back own-view merge failed; keeping page armed at PAGE_NOACCESS"
                    );
                    failed.push(item.primary_block);
                }
            }
        }
        Err(e) => {
            tracing::error!(
                error = %e,
                count = claimed.len(),
                "lazy copy-back batched DMA fetch failed; retaining armed entries at PAGE_NOACCESS"
            );
            for item in &claimed {
                failed.push(item.primary_block);
            }
        }
    }

    let mut removed = Vec::with_capacity(succeeded.len());
    {
        let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
        for &block in &succeeded {
            if let Some(entry) = remove_entry_locked_no_notify(&mut table, block, true) {
                removed.push(entry);
            }
        }
        for &block in &failed {
            if let Some(entry) = table.pages.get_mut(&block) {
                entry.materializing = false;
            }
        }
        for &block in &restore_failed {
            if let Some(entry) = table.pages.get_mut(&block) {
                entry.restore_failed = true;
            }
        }
        LAZY_COUNT.store(table.pages.len(), Ordering::Release);
        LAZY_CV.notify_all();
    }
    scope.finished = true;
    drop(scope);
    drop(removed);
    drop(claimed);
    failed.is_empty()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MaterializeOutcome {
    Handled,
    NotPresent,
    Failed,
}

/// Materializes `primary_block` outside `LAZY_STATE`.
fn materialize_primary(primary_block: usize, reason: MaterializeReason) -> MaterializeOutcome {
    let claimed = {
        let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
        let mut waited = false;
        loop {
            match table.pages.get_mut(&primary_block) {
                Some(entry) if entry.materializing => {
                    waited = true;
                    #[cfg(test)]
                    if let Some(hook) = BEFORE_LAZY_CV_WAIT_HOOK.with(|h| h.borrow_mut().take()) {
                        hook();
                    }
                    table = LAZY_CV.wait(table).unwrap_or_else(|e| e.into_inner());
                }
                Some(entry) if entry.restore_failed => {
                    return MaterializeOutcome::Failed;
                }
                Some(entry) => {
                    entry.materializing = true;
                    let claimed = ClaimedBlock {
                        primary_block,
                        shadow: entry.shadow.clone(),
                        offset: entry.offset,
                        len: entry.len,
                        fetched_after: entry.fetched_after,
                        own_va: entry
                            .armed_vas
                            .first()
                            .map_or(primary_block * PAGE, |a| a.va),
                        restore_vas: entry.armed_vas.clone(),
                    };
                    match reason {
                        MaterializeReason::CpuFault { is_write } => {
                            note_fault_hot_locked(&mut table, primary_block);
                            if is_write {
                                STATS.write_faults.fetch_add(1, Ordering::Relaxed);
                            } else {
                                STATS.read_faults.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        MaterializeReason::InternalRead => {
                            if internal_promote_enabled() {
                                note_fault_hot_locked(&mut table, primary_block);
                            }
                            STATS.internal_materializes.fetch_add(1, Ordering::Relaxed);
                        }
                        MaterializeReason::Retire => {
                            STATS.retire_materializes.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    break claimed;
                }
                None => {
                    return if waited {
                        MaterializeOutcome::Handled
                    } else {
                        MaterializeOutcome::NotPresent
                    };
                }
            }
        }
    };

    if materialize_claimed_batch(vec![claimed], reason) {
        MaterializeOutcome::Handled
    } else {
        MaterializeOutcome::Failed
    }
}

/// Handles a CPU `EXCEPTION_ACCESS_VIOLATION` (`is_write == true` for write, `false` for read)
/// at `at`. Called by `watch.rs`'s VEH handler before checking write guards.
pub(crate) fn handle_fault(at: usize, is_write: bool) -> bool {
    if should_skip_fault_lookup(
        LAZY_COUNT.load(Ordering::Acquire),
        EVER_ACTIVE.load(Ordering::Acquire),
    ) {
        return false;
    }
    let va_block = at / PAGE;
    let primary = {
        let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(&p) = table.alias_to_primary.get(&va_block) {
            Some(p)
        } else if !is_write && table.recent_set.contains(&va_block) && is_readable_now(at) {
            // Another thread just materialized this block to readable between our CPU fault and now.
            return true;
        } else {
            None
        }
    };
    let Some(primary_block) = primary else {
        return false;
    };
    match materialize_primary(primary_block, MaterializeReason::CpuFault { is_write }) {
        MaterializeOutcome::Handled => return true,
        MaterializeOutcome::Failed => return false,
        MaterializeOutcome::NotPresent => {}
    }
    if !is_write {
        let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
        if table.recent_set.contains(&va_block) && is_readable_now(at) {
            return true;
        }
    }
    false
}

/// Materializes any deferred lazy blocks overlapping any `[at, at + len)` in `ranges`
/// in a single batched `COPY`-queue DMA before internal runtime reads, kernel writes,
/// copies, or memory remaps. Returns `false` if any block failed to materialize.
pub(crate) fn materialize_ranges(ranges: &[(usize, usize)]) -> bool {
    materialize_ranges_inner(ranges, true)
}

/// Like [`materialize_ranges`], but always acquires `LAZY_STATE` even if
/// `LAZY_COUNT` was transiently `0` (paired with [`active_or_arming`] in
/// `watch::views_mapped`).
pub(crate) fn materialize_ranges_unconditional(ranges: &[(usize, usize)]) -> bool {
    materialize_ranges_inner(ranges, false)
}

fn materialize_ranges_inner(ranges: &[(usize, usize)], check_active: bool) -> bool {
    if ranges.is_empty() || (check_active && !active()) {
        return true;
    }
    loop {
        let (claimed, wait_only) = {
            let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
            let mut to_claim = BTreeSet::new();
            let mut wait_only = false;
            for &(at, len) in ranges {
                if len == 0 {
                    continue;
                }
                let end = at.saturating_add(len);
                let first = at / PAGE;
                let last = end.div_ceil(PAGE);
                for (_, &primary_block) in table.alias_to_primary.range(first..last) {
                    if current_thread_materializing(primary_block) {
                        continue;
                    }
                    if let Some(entry) = table.pages.get(&primary_block) {
                        if entry.restore_failed {
                            return false;
                        }
                        if entry.materializing {
                            wait_only = true;
                        } else {
                            to_claim.insert(primary_block);
                        }
                    }
                }
            }
            if to_claim.is_empty() {
                if wait_only {
                    drop(LAZY_CV.wait(table).unwrap_or_else(|e| e.into_inner()));
                    continue;
                }
                return true;
            }
            let do_promote = internal_promote_enabled();
            let mut claimed = Vec::with_capacity(to_claim.len());
            for primary_block in to_claim {
                if let Some(entry) = table.pages.get_mut(&primary_block)
                    && !entry.materializing
                {
                    entry.materializing = true;
                    claimed.push(ClaimedBlock {
                        primary_block,
                        shadow: entry.shadow.clone(),
                        offset: entry.offset,
                        len: entry.len,
                        fetched_after: entry.fetched_after,
                        own_va: entry
                            .armed_vas
                            .first()
                            .map_or(primary_block * PAGE, |a| a.va),
                        restore_vas: entry.armed_vas.clone(),
                    });
                    if do_promote {
                        note_fault_hot_locked(&mut table, primary_block);
                    }
                    STATS.internal_materializes.fetch_add(1, Ordering::Relaxed);
                }
            }
            (claimed, wait_only)
        };
        if !claimed.is_empty()
            && !materialize_claimed_batch(claimed, MaterializeReason::InternalRead)
        {
            return false;
        }
        if !wait_only {
            return true;
        }
    }
}

/// Materializes any deferred lazy blocks overlapping `[at, at + len)` before an internal
/// runtime read, kernel write, copy, or memory remap.
pub(crate) fn materialize_range(at: usize, len: usize) -> bool {
    if len > 0 && active() {
        materialize_ranges(&[(at, len)])
    } else {
        true
    }
}

/// Materializes all deferred lazy blocks owned by `shadow_ptr` before retirement or unregister
/// in a single batched `COPY`-queue DMA without hot promotion.
pub(crate) fn materialize_shadow(shadow_ptr: usize) -> bool {
    if !active() {
        return true;
    }
    loop {
        let (claimed, wait_only) = {
            let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
            let mut to_claim = Vec::new();
            let mut wait_only = false;
            if let Some(blocks) = table.by_shadow.get(&shadow_ptr) {
                for &b in blocks {
                    if current_thread_materializing(b) {
                        continue;
                    }
                    if let Some(entry) = table.pages.get(&b) {
                        if entry.restore_failed {
                            return false;
                        }
                        if entry.materializing {
                            wait_only = true;
                        } else {
                            to_claim.push(b);
                        }
                    }
                }
            }
            if to_claim.is_empty() {
                if wait_only {
                    drop(LAZY_CV.wait(table).unwrap_or_else(|e| e.into_inner()));
                    continue;
                }
                return true;
            }
            let mut claimed = Vec::with_capacity(to_claim.len());
            for primary_block in to_claim {
                if let Some(entry) = table.pages.get_mut(&primary_block)
                    && !entry.materializing
                {
                    entry.materializing = true;
                    STATS.retire_materializes.fetch_add(1, Ordering::Relaxed);
                    claimed.push(ClaimedBlock {
                        primary_block,
                        shadow: entry.shadow.clone(),
                        offset: entry.offset,
                        len: entry.len,
                        fetched_after: entry.fetched_after,
                        own_va: entry
                            .armed_vas
                            .first()
                            .map_or(primary_block * PAGE, |a| a.va),
                        restore_vas: entry.armed_vas.clone(),
                    });
                }
            }
            (claimed, wait_only)
        };
        if !claimed.is_empty() && !materialize_claimed_batch(claimed, MaterializeReason::Retire) {
            return false;
        }
        if !wait_only {
            return true;
        }
    }
}

/// Detaches all armed VAs in `[first, last)` from `primary`'s entry in `table`.
///
/// - If every armed VA lies inside `[first, last)`, removes the entry and
///   returns `(Some(removed_entry), None)`.
/// - If surviving VAs remain outside `[first, last)`, detaches the removed VAs
///   and promotes `surviving[0] / PAGE` as the entry's primary/merge block
///   (preserving `restore_failed` and all other metadata), returning
///   `(None, Some(new_primary))`.
fn detach_remapped_vas_locked(
    table: &mut LazyTable,
    primary: usize,
    first: usize,
    last: usize,
) -> (Option<LazyEntry>, Option<usize>) {
    let Some(mut entry) = table.pages.remove(&primary) else {
        return (None, None);
    };
    let old_primary = entry.armed_vas.first().copied();
    let old_sub_off = old_primary.map_or(0, |a| a.va % PAGE);
    let base_offset = entry.offset.saturating_sub(old_sub_off);
    let shadow_ptr = Arc::as_ptr(&entry.shadow) as usize;
    if let Some(blocks) = table.by_shadow.get_mut(&shadow_ptr) {
        blocks.remove(&primary);
        if blocks.is_empty() {
            table.by_shadow.remove(&shadow_ptr);
        }
    }
    for &armed in &entry.armed_vas {
        table.alias_to_primary.remove(&(armed.va / PAGE));
    }
    let mut surviving = Vec::with_capacity(entry.armed_vas.len());
    for &armed in &entry.armed_vas {
        let b = armed.va / PAGE;
        if b >= first && b < last {
            if entry.is_view {
                crate::watch::clear_lazy_view_armed_block(b);
            }
            table.note_recent(b);
        } else {
            surviving.push(armed);
        }
    }
    REMOVE_GEN.fetch_add(1, Ordering::SeqCst);
    if surviving.is_empty() {
        return (Some(entry), None);
    }
    // Prefer a surviving alias covering the full 64 KiB page; otherwise pick the
    // longest surviving alias and recompute `entry.offset` / `entry.len` from
    // the block's page-aligned base offset so repeated promotions are idempotent
    // and never double-shift `entry.offset` or shrink `entry.len` toward zero.
    let best_idx = surviving
        .iter()
        .enumerate()
        .find(|(_, a)| a.va.is_multiple_of(PAGE) && a.len >= PAGE)
        .or_else(|| surviving.iter().enumerate().max_by_key(|(_, a)| a.len))
        .map(|(i, _)| i)
        .unwrap_or(0);
    if best_idx != 0 {
        surviving.swap(0, best_idx);
    }
    let promoted = surviving[0];
    let new_primary = promoted.va / PAGE;
    if old_primary != Some(promoted) {
        let sub_off = promoted.va % PAGE;
        entry.offset = base_offset.saturating_add(sub_off);
        entry.len = (PAGE - sub_off).min(promoted.len);
    }
    entry.armed_vas = surviving;
    for &armed in &entry.armed_vas {
        table.alias_to_primary.insert(armed.va / PAGE, new_primary);
    }
    table
        .by_shadow
        .entry(shadow_ptr)
        .or_default()
        .insert(new_primary);
    table.pages.insert(new_primary, entry);
    (None, Some(new_primary))
}

/// Discards deferred entries (or unmapped alias VAs) in `[at, at + len)` after
/// `title_memory_remapped`. If any primary entry inside `[at, at + len)` still
/// has surviving mapped aliases outside the remapped range (e.g. if a caller
/// invoked `title_memory_remapped` without holding `Remapping`), immediately
/// detaches the removed addresses, promotes a surviving alias as the entry's
/// primary/merge address (preserving `restore_failed` if already set), and
/// materializes its GPU bytes through that surviving alias.
pub(crate) fn discard_range(at: usize, len: usize) {
    if len == 0 || !active() {
        return;
    }
    let end = at.saturating_add(len);
    let first = at / PAGE;
    let last = end.div_ceil(PAGE);
    let mut dropped: Vec<LazyEntry> = Vec::new();
    let mut surviving_claimed: Vec<ClaimedBlock> = Vec::new();
    {
        let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            let waiting = table
                .alias_to_primary
                .range(first..last)
                .any(|(_, &p)| table.pages.get(&p).is_some_and(|e| e.materializing));
            if waiting {
                table = LAZY_CV.wait(table).unwrap_or_else(|e| e.into_inner());
            } else {
                break;
            }
        }
        let primaries: BTreeSet<usize> = table
            .alias_to_primary
            .range(first..last)
            .map(|(_, &primary)| primary)
            .collect();
        let mut changed = false;
        for primary in primaries {
            let primary_removed = primary >= first && primary < last;
            let (removed, promoted) = detach_remapped_vas_locked(&mut table, primary, first, last);
            if let Some(entry) = removed {
                dropped.push(entry);
                changed = true;
            } else if let Some(new_primary) = promoted {
                changed = true;
                if primary_removed
                    && let Some(entry) = table.pages.get_mut(&new_primary)
                    && !entry.materializing
                    && !entry.restore_failed
                {
                    entry.materializing = true;
                    STATS.retire_materializes.fetch_add(1, Ordering::Relaxed);
                    surviving_claimed.push(ClaimedBlock {
                        primary_block: new_primary,
                        shadow: entry.shadow.clone(),
                        offset: entry.offset,
                        len: entry.len,
                        fetched_after: entry.fetched_after,
                        own_va: entry.armed_vas.first().map_or(new_primary * PAGE, |a| a.va),
                        restore_vas: entry.armed_vas.clone(),
                    });
                }
            }
        }
        if changed {
            LAZY_COUNT.store(table.pages.len(), Ordering::Release);
            LAZY_CV.notify_all();
        }
        if !table.hot.is_empty() {
            if last.saturating_sub(first) <= table.hot.len() {
                for b in first..last {
                    table.hot.remove(&b);
                }
            } else {
                table.hot.retain(|&b, _| b < first || b >= last);
            }
        }
    }
    if !surviving_claimed.is_empty() {
        let _ = materialize_claimed_batch(surviving_claimed, MaterializeReason::Retire);
    }
    drop(dropped);
}

/// Protects newly mapped alias `[at, at + len)` with `PAGE_NOACCESS` and
/// attaches each overlapping 64 KiB page to any still-armed `LazyEntry` on the
/// same physical section bytes after `materialize_ranges_unconditional` failed.
/// Ensures a newly mapped alias can never silently read stale RAM when the
/// pre-expose drain failed.
///
/// Returns `false` if `VirtualProtect(..., PAGE_NOACCESS)` failed on any
/// overlapping armed page so the caller can fail closed rather than exposing
/// an unprotected readable alias to stale RAM.
pub(crate) fn attach_new_alias_on_failed_materialize(
    at: usize,
    len: usize,
    section: usize,
    offset: u64,
) -> bool {
    if len == 0 || !active_or_arming() {
        return true;
    }
    let mut all_protected = true;
    let end = at.saturating_add(len);
    let mut cur = at;
    while cur < end {
        let step = (PAGE - cur % PAGE).min(end - cur);
        let page_off = offset.saturating_add((cur - at) as u64);
        let peer_ranges = crate::inplace::views_of_section_all(section, page_off, step);
        if peer_ranges.is_empty() {
            cur += step;
            continue;
        }
        let cur_block = cur / PAGE;
        enum AttachStep {
            Wait,
            Done(bool),
        }
        loop {
            // Acquire `cur_block`'s guard shard BEFORE `LAZY_STATE` (strictly
            // matching `GUARD_SHARDS_TABLE -> LAZY_STATE` lock order) so
            // `guard_run_shard` / `protect_here` can never overwrite our
            // `PAGE_NOACCESS` with `PAGE_READONLY`. Never wait on `LAZY_CV`
            // while holding the guard shard.
            let step_res = crate::watch::under_guard_shard_for_lazy_attach(cur_block, || {
                let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
                let mut found_primary = None;
                let mut waiting = false;
                for &(peer_va, peer_len) in &peer_ranges {
                    if peer_len == 0 {
                        continue;
                    }
                    let p_first = peer_va / PAGE;
                    let p_last = peer_va.saturating_add(peer_len).div_ceil(PAGE);
                    for (_, &primary) in table.alias_to_primary.range(p_first..p_last) {
                        if let Some(entry) = table.pages.get(&primary) {
                            if entry.materializing {
                                waiting = true;
                            } else {
                                found_primary = Some(primary);
                                break;
                            }
                        }
                    }
                    if found_primary.is_some() || waiting {
                        break;
                    }
                }
                if waiting {
                    return AttachStep::Wait;
                }
                let mut protected_ok = true;
                if let Some(primary) = found_primary {
                    let mut attached = None;
                    if let Some(entry) = table.pages.get_mut(&primary) {
                        let armed = ArmedVa { va: cur, len: step };
                        let already_covered = entry
                            .armed_vas
                            .iter()
                            .any(|a| a.va <= cur && a.va.saturating_add(a.len) >= cur + step);
                        if !already_covered {
                            let ok = protect_new_alias_noaccess(cur, step);
                            if ok {
                                crate::watch::clear_guard_bits_for_lazy_attach(cur_block);
                                entry.armed_vas.push(armed);
                                attached = Some((cur_block, armed.va, entry.is_view));
                            } else {
                                tracing::error!(
                                    va = format_args!("{cur:#x}"),
                                    primary = format_args!("{:#x}", primary * PAGE),
                                    "failed to protect newly mapped alias after lazy materialization failure"
                                );
                                entry.restore_failed = true;
                                protected_ok = false;
                            }
                        }
                    }
                    if let Some((cur_block, attached_va, is_view)) = attached {
                        table.clear_recent(cur_block);
                        table.alias_to_primary.insert(cur_block, primary);
                        if is_view {
                            crate::watch::note_lazy_view_armed([attached_va]);
                        }
                    }
                }
                AttachStep::Done(protected_ok)
            });
            match step_res {
                AttachStep::Wait => {
                    let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
                    let still_waiting = peer_ranges.iter().any(|&(peer_va, peer_len)| {
                        if peer_len == 0 {
                            return false;
                        }
                        let p_first = peer_va / PAGE;
                        let p_last = peer_va.saturating_add(peer_len).div_ceil(PAGE);
                        table
                            .alias_to_primary
                            .range(p_first..p_last)
                            .any(|(_, &p)| table.pages.get(&p).is_some_and(|e| e.materializing))
                    });
                    if still_waiting {
                        table = LAZY_CV.wait(table).unwrap_or_else(|e| e.into_inner());
                    }
                    drop(table);
                }
                AttachStep::Done(ok) => {
                    if !ok {
                        all_protected = false;
                    }
                    break;
                }
            }
        }
        cur += step;
    }
    all_protected
}

/// Drops any lazy entries tracked for `shadow_ptr` on `Shadow::drop`.
pub(crate) fn discard_shadow(shadow_ptr: usize) {
    if !active() {
        return;
    }
    let mut dropped: Vec<LazyEntry> = Vec::new();
    {
        let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
        let blocks: Vec<usize> = table
            .by_shadow
            .get(&shadow_ptr)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default();
        for b in blocks {
            if let Some(entry) = remove_entry_locked_no_notify(&mut table, b, true) {
                dropped.push(entry);
            }
        }
        if !dropped.is_empty() {
            LAZY_COUNT.store(table.pages.len(), Ordering::Release);
            LAZY_CV.notify_all();
        }
    }
    drop(dropped);
}

/// Logs the periodic 5s telemetry summary when `VICTORY_LAZY_BACK=1` is enabled.
pub(crate) fn log_stats(frames: f64) {
    if !enabled() || frames <= 0.0 {
        return;
    }
    let deferred_new = STATS.deferred_new.swap(0, Ordering::Relaxed) as f64 / frames;
    let deferred_reused = STATS.deferred_reused.swap(0, Ordering::Relaxed) as f64 / frames;
    let skipped_mb =
        STATS.skipped_bytes.swap(0, Ordering::Relaxed) as f64 / (1024.0 * 1024.0) / frames;
    let materialized_mb =
        STATS.materialized_bytes.swap(0, Ordering::Relaxed) as f64 / (1024.0 * 1024.0) / frames;
    let read_faults = STATS.read_faults.swap(0, Ordering::Relaxed) as f64 / frames;
    let write_faults = STATS.write_faults.swap(0, Ordering::Relaxed) as f64 / frames;
    let internal = STATS.internal_materializes.swap(0, Ordering::Relaxed) as f64 / frames;
    let retire = STATS.retire_materializes.swap(0, Ordering::Relaxed) as f64 / frames;
    let batches = STATS.dma_batches.swap(0, Ordering::Relaxed) as f64 / frames;
    let hot_promoted = STATS.hot_promoted.swap(0, Ordering::Relaxed) as f64;
    tracing::info!(
        "lazy copy-back (VICTORY_LAZY_BACK=1): deferred {deferred_new:.1} new / {deferred_reused:.1} reused ({skipped_mb:.1} MB/frame skipped); materialized {materialized_mb:.1} MB/frame (read faults {read_faults:.1}, write faults {write_faults:.1}, internal {internal:.1}, retire {retire:.1}, dma batches {batches:.1}); hot promoted {hot_promoted:.0}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_WRITE, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE,
        MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, UnmapViewOfFile, VirtualAlloc, VirtualFree,
    };

    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    static TEST_SECTIONS: LazyLock<Mutex<HashMap<usize, (usize, usize)>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    fn alloc_watch_pages(pages: usize) -> *mut u8 {
        let len = pages * PAGE;
        // SAFETY: fresh anonymous section mapped as a `pages`-page view so
        // `inplace` has a host-only `PAGE_READWRITE` own view (matching
        // production title section memory).
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null(),
                PAGE_READWRITE,
                (len as u64 >> 32) as u32,
                len as u32,
                ptr::null(),
            )
        };
        assert!(!section.is_null(), "CreateFileMappingW failed");
        // SAFETY: mapping `len` bytes from offset 0 of `section`.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len) };
        assert!(!view.Value.is_null(), "MapViewOfFile failed");
        let base = view.Value as usize;
        crate::inplace::title_memory_section(section as usize, len as u64, true);
        for k in 0..pages {
            assert!(crate::inplace::title_memory_view(
                base + k * PAGE,
                PAGE,
                section as usize,
                (k * PAGE) as u64,
            ));
        }
        TEST_SECTIONS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(base, (section as usize, len));
        base as *mut u8
    }

    unsafe fn free_watch_pages(p: *mut u8) {
        let base = p as usize;
        let entry = TEST_SECTIONS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&base);
        if let Some((section, len)) = entry {
            crate::watch::title_memory_remapped(base, len);
            // SAFETY: view and section allocated by `alloc_watch_pages`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: p.cast() });
                CloseHandle(section as HANDLE);
            }
        } else {
            // SAFETY: fallback for raw `VirtualAlloc` allocations.
            unsafe {
                VirtualFree(p.cast(), 0, MEM_RELEASE);
            }
        }
    }

    fn make_card_shadow_at(
        title_start: u64,
        pages: usize,
        init_byte: u8,
    ) -> Option<(Arc<Shadow>, Vec<u8>)> {
        let len = pages * PAGE;
        let mut side = vec![init_byte; len];
        let shadow = Shadow::new_card_sized_for_test(side.as_mut_ptr(), title_start, len as u64)?;
        // Seed `shadow.buffer` (card side) with `init_byte` so non-written bytes match `side`.
        let h = crate::host::host()?;
        let (up, up_ptr) = h.upload_buffer(len as u64)?;
        // SAFETY: `up_ptr` is mapped for `len` bytes.
        unsafe { ptr::write_bytes(up_ptr, init_byte, len) };
        let list = h.new_list().ok()?;
        // SAFETY: live D3D12 buffers in COMMON / GENERIC_READ states.
        unsafe {
            list.list
                .CopyBufferRegion(&shadow.buffer, 0, &up, 0, len as u64);
        }
        list.close();
        h.execute(&[&list]);
        list.reset();
        Some((shadow, side))
    }

    fn gpu_write_shadow(shadow: &Shadow, offset: usize, bytes: &[u8]) {
        let h = crate::host::host().expect("host GPU");
        let (up, up_ptr) = h.upload_buffer(bytes.len() as u64).expect("upload buffer");
        // SAFETY: `up_ptr` is mapped for `bytes.len()` bytes.
        unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), up_ptr, bytes.len()) };
        let list = h.new_list().expect("host list");
        // SAFETY: `shadow.buffer` is live in COMMON state.
        unsafe {
            list.list
                .CopyBufferRegion(&shadow.buffer, offset as u64, &up, 0, bytes.len() as u64);
        }
        list.close();
        h.execute(&[&list]);
        list.reset();
        let _ = D3D12_RESOURCE_STATES::default();
    }

    fn advance_frames(n: usize) {
        let mut token = 0u64;
        for _ in 0..n {
            let _ = crate::frame::wait(ptr::null(), 0, 0, ptr::null(), 2, &mut token);
        }
    }

    fn clear_hot_for_test(block: usize) {
        let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
        table.hot.remove(&block);
    }

    #[test]
    fn lazy_read_fault_materializes_gpu_bytes() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(2);
        // SAFETY: `ptr` owns `2 * PAGE` committed bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, 2 * PAGE) };
        let va = ptr as usize;
        clear_hot_for_test(va / PAGE);
        clear_hot_for_test(va / PAGE + 1);

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 2, 0x11) else {
            // SAFETY: freeing test allocation.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        gpu_write_shadow(&shadow, 64, &[0xAA; 64]);

        assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
        assert!(has_block(va / PAGE));
        assert!(!is_readable_now(va));

        // Real CPU read fault via VEH.
        // SAFETY: reading inside the deferred allocation triggers VEH materialization.
        let got = unsafe { std::slice::from_raw_parts(ptr.add(64), 64).to_vec() };
        assert_eq!(got, vec![0xAA; 64]);
        // Untouched byte in the same page remains 0x11.
        // SAFETY: inside `ptr`.
        assert_eq!(unsafe { ptr.read() }, 0x11);
        assert!(!has_block(va / PAGE));
        assert!(is_readable_now(va));
        // Page 1 was never deferred or touched.
        assert!(is_readable_now(va + PAGE));

        clear_hot_for_test(va / PAGE);
        drop(shadow);
        // SAFETY: test allocation cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_write_fault_preserves_both_gpu_and_cpu_bytes() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        clear_hot_for_test(va / PAGE);
        // Reset initial guard state from the seed memset.
        let (baseline_epoch, _) = crate::watch::writes(va, va + PAGE, true).expect("watched");

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        gpu_write_shadow(&shadow, 32, &[0xBB; 8]);

        assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
        // Real CPU write fault via VEH at offset 128.
        // SAFETY: writing inside deferred page triggers VEH write materialization.
        unsafe { ptr.add(128).write(0x77) };

        assert!(!has_block(va / PAGE));
        // SAFETY: page is now readable/writable.
        let gpu_slice = unsafe { std::slice::from_raw_parts(ptr.add(32), 8) };
        assert_eq!(
            gpu_slice, &[0xBB; 8],
            "GPU bytes must land before CPU write"
        );
        // SAFETY: byte 128 was written by the CPU instruction after VEH resumed.
        let cpu_byte = unsafe { ptr.add(128).read() };
        assert_eq!(cpu_byte, 0x77, "CPU write must survive");
        // Write watch must report the page dirty after the baseline so next upload picks up `0x77`.
        let (_, written) = crate::watch::writes(va, va + PAGE, true).expect("watched");
        assert!(
            written
                .iter()
                .any(|&(p, epoch)| p == va / PAGE && epoch > baseline_epoch),
            "write fault must leave page dirty in write-watch"
        );

        clear_hot_for_test(va / PAGE);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_multi_flight_reuse_skips_rearming() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        clear_hot_for_test(va / PAGE);

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        gpu_write_shadow(&shadow, 0, &[0x22; 16]);
        assert!(try_defer_page_inner(&shadow, 0, PAGE, 1));

        let reused_before = STATS.deferred_reused.load(Ordering::Relaxed);
        gpu_write_shadow(&shadow, 16, &[0x33; 16]);
        assert!(try_defer_page_inner(&shadow, 0, PAGE, 2));
        assert!(
            STATS.deferred_reused.load(Ordering::Relaxed) > reused_before,
            "second flight on same block must reuse armed entry"
        );

        // SAFETY: CPU read materializes both flights' GPU writes in one DMA.
        let first = unsafe { std::slice::from_raw_parts(ptr, 16).to_vec() };
        // SAFETY: reading bytes 16..32.
        let second = unsafe { std::slice::from_raw_parts(ptr.add(16), 16).to_vec() };
        assert_eq!(first, vec![0x22; 16]);
        assert_eq!(second, vec![0x33; 16]);

        clear_hot_for_test(va / PAGE);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_hot_promotion_after_three_frames_and_decay() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        let block = va / PAGE;
        clear_hot_for_test(block);

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };

        let threshold = promote_frames();
        for f in 0..threshold {
            advance_frames(1);
            gpu_write_shadow(&shadow, 0, &[0x40 + f as u8; 8]);
            assert!(
                try_defer_page_inner(&shadow, 0, PAGE, (f + 1) as u64),
                "frame {f} before promotion should defer"
            );
            // SAFETY: CPU read fault increments hot frame counter.
            let b = unsafe { ptr.read() };
            assert_eq!(b, 0x40 + f as u8);
        }

        // Now promoted: next flight must NOT defer.
        advance_frames(1);
        assert!(
            !try_defer_page_inner(&shadow, 0, PAGE, 10),
            "hot block must skip lazy deferral"
        );

        // Decay after > 120 unused frames.
        advance_frames((HOT_DECAY_FRAMES + 2) as usize);
        assert!(
            try_defer_page_inner(&shadow, 0, PAGE, 11),
            "unused hot block must decay back to lazy after >120 frames"
        );
        materialize_range(va, PAGE);

        clear_hot_for_test(block);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_internal_materialization_on_read_into_kernel_writing_and_unregister() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        // 1 MiB write-watched allocation so `shadow::register` + `find` gives a real card/ReBAR shadow.
        let pages = 16;
        let ptr = alloc_watch_pages(pages);
        // SAFETY: `ptr` owns `16 * PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, pages * PAGE) };
        let va0 = ptr as usize;
        let va1 = va0 + PAGE;
        let va2 = va0 + 2 * PAGE;
        clear_hot_for_test(va0 / PAGE);
        clear_hot_for_test(va1 / PAGE);
        clear_hot_for_test(va2 / PAGE);

        let Some((card_shadow, _side)) = make_card_shadow_at(va0 as u64, 3, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        gpu_write_shadow(&card_shadow, 0, &[0xA1; 16]);
        gpu_write_shadow(&card_shadow, PAGE, &[0xB2; 16]);

        // 1. `host::read_into` materializes page 0.
        assert!(try_defer_page_inner(&card_shadow, 0, PAGE, 0));
        let mut dst = [0u8; 16];
        assert!(crate::host::read_into(va0 as *const u8, &mut dst));
        assert_eq!(dst, [0xA1; 16]);
        assert!(!has_block(va0 / PAGE));

        // 2. `watch::kernel_writing` materializes page 1.
        assert!(try_defer_page_inner(&card_shadow, PAGE, PAGE, 0));
        let kw_guard = crate::watch::kernel_writing(va1, PAGE).expect("kernel_writing page 1");
        assert!(!has_block(va1 / PAGE));
        // SAFETY: page 1 is now readable with GPU bytes.
        let page1_bytes = unsafe { std::slice::from_raw_parts(va1 as *const u8, 16) };
        assert_eq!(page1_bytes, &[0xB2; 16]);
        drop(kw_guard);
        // Clear kernel pin via `title_memory_remapped` cleanup at end or unpin by remapped.
        crate::watch::title_memory_remapped(va1, PAGE);

        // 3. `shadow::unregister` materializes registered shadow on page 2.
        drop(card_shadow);
        let reg_len = (pages * PAGE) as u64;
        crate::shadow::register(va0, va0 as u64, reg_len);
        let reg_shadow = crate::shadow::find(va0 as u64, reg_len).expect("registered shadow");
        if reg_shadow.carded() || reg_shadow.is_rebar() {
            if reg_shadow.is_rebar() {
                let list = h.new_list().expect("host list");
                reg_shadow.refresh(&list, va0 as u64, reg_len);
                list.close();
                h.execute(&[&list]);
                list.reset();
                reg_shadow.writing(va2, PAGE, |_, size| Some(vec![0x11u8; size as usize]));
            }
            gpu_write_shadow(&reg_shadow, 2 * PAGE, &[0xC3; 16]);
            assert!(try_defer_page_inner(&reg_shadow, 2 * PAGE, PAGE, 0));
            assert!(has_block(va2 / PAGE));
            drop(reg_shadow);
            crate::shadow::unregister(va0, va0 as u64);
            assert!(!has_block(va2 / PAGE));
            // SAFETY: unregister materialized GPU bytes to title RAM.
            let page2_bytes = unsafe { std::slice::from_raw_parts(va2 as *const u8, 16) };
            assert_eq!(page2_bytes, &[0xC3; 16]);
        } else {
            drop(reg_shadow);
            crate::shadow::unregister(va0, va0 as u64);
        }

        clear_hot_for_test(va0 / PAGE);
        clear_hot_for_test(va1 / PAGE);
        clear_hot_for_test(va2 / PAGE);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_aliased_section_views_protect_and_fault_across_aliases() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        // SAFETY: creating an anonymous page-file-backed section for two aliased views.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                ptr::null(),
            )
        };
        assert!(!section.is_null(), "CreateFileMappingW failed");
        // SAFETY: mapping two distinct views of offset 0 of `section`.
        let view_a = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        // SAFETY: second aliased view.
        let view_b = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        assert!(!view_a.Value.is_null() && !view_b.Value.is_null());
        let addr_a = view_a.Value as usize;
        let addr_b = view_b.Value as usize;
        assert_ne!(addr_a, addr_b);
        clear_hot_for_test(addr_a / PAGE);
        clear_hot_for_test(addr_b / PAGE);

        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        assert!(crate::inplace::title_memory_view(
            addr_a,
            PAGE,
            section as usize,
            0
        ));
        assert!(crate::inplace::title_memory_view(
            addr_b,
            PAGE,
            section as usize,
            0
        ));

        // SAFETY: `addr_a` is mapped for `PAGE` bytes.
        unsafe { ptr::write_bytes(addr_a as *mut u8, 0x11, PAGE) };
        let Some((shadow, _side)) = make_card_shadow_at(addr_a as u64, 1, 0x11) else {
            crate::watch::title_memory_remapped(addr_a, PAGE);
            crate::watch::title_memory_remapped(addr_b, PAGE);
            // SAFETY: cleanup views and section handle.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view_a.Value,
                });
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view_b.Value,
                });
                CloseHandle(section);
            }
            return;
        };
        gpu_write_shadow(&shadow, 64, &[0xDD; 16]);

        assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
        assert!(has_block(addr_a / PAGE));
        assert!(has_block(addr_b / PAGE));
        assert!(!is_readable_now(addr_a));
        assert!(!is_readable_now(addr_b));

        // Fault on the ALIAS view (`addr_b`), not `addr_a`.
        // SAFETY: reading from `addr_b` triggers VEH fault, resolving alias -> primary block.
        let got_b = unsafe { std::slice::from_raw_parts((addr_b + 64) as *const u8, 16).to_vec() };
        assert_eq!(got_b, vec![0xDD; 16]);
        // Both views are now restored and show the materialized GPU bytes.
        assert!(is_readable_now(addr_a));
        assert!(is_readable_now(addr_b));
        // SAFETY: reading from `addr_a`.
        let got_a = unsafe { std::slice::from_raw_parts((addr_a + 64) as *const u8, 16).to_vec() };
        assert_eq!(got_a, vec![0xDD; 16]);
        assert!(!has_block(addr_a / PAGE));
        assert!(!has_block(addr_b / PAGE));

        clear_hot_for_test(addr_a / PAGE);
        clear_hot_for_test(addr_b / PAGE);
        drop(shadow);
        crate::watch::title_memory_remapped(addr_a, PAGE);
        crate::watch::title_memory_remapped(addr_b, PAGE);
        // SAFETY: cleanup views and section handle.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view_a.Value,
            });
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view_b.Value,
            });
            CloseHandle(section);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_merge_under_writes_update_busy_and_copy_runs_never_deadlock() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let pages = 16;
        let ptr = alloc_watch_pages(pages);
        // SAFETY: `ptr` owns `pages * PAGE` committed write-watched bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, pages * PAGE) };
        let va0 = ptr as usize;
        clear_hot_for_test(va0 / PAGE);

        let Some((shadow_a, _side_a)) = make_card_shadow_at(va0 as u64, 2, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        let Some((shadow_b, side_b)) = make_card_shadow_at(va0 as u64, 2, 0x11) else {
            drop(shadow_a);
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        let Some((shadow_c, side_c)) = make_card_shadow_at(va0 as u64, 2, 0x11) else {
            drop(shadow_b);
            drop(shadow_a);
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };

        // Declare pending ranges on `shadow_a` and `shadow_b` so `merge_under_writes`
        // and `update` -> `update_busy` exercise the busy-page paths.
        shadow_a.writing(va0, PAGE, |_, size| Some(vec![0x11u8; size as usize]));
        shadow_b.writing(va0, PAGE, |_, size| Some(vec![0x11u8; size as usize]));

        // 1. Defer page 0 on `shadow_a`: `shadow_a.merge_under_writes` must skip its own
        //    deferred block without locking + materializing itself, keeping page 0 deferred.
        gpu_write_shadow(&shadow_a, 0, &[0xA1; 32]);
        assert!(try_defer_page_inner(&shadow_a, 0, PAGE, 0));
        assert!(has_block(va0 / PAGE));
        let mut runs_left = 1024usize;
        assert!(!shadow_a.merge_under_writes(va0, PAGE, &mut runs_left));
        assert!(
            has_block(va0 / PAGE),
            "merge_under_writes must skip own deferred block"
        );

        // 2. Overlapping `shadow_b.merge_under_writes` must materialize `shadow_a`'s
        //    deferred block outside `shadow_b.before.write()` and see `[0xA1; 32]`.
        let mut runs_left = 1024usize;
        assert!(!shadow_b.merge_under_writes(va0, PAGE, &mut runs_left));
        assert!(!has_block(va0 / PAGE));
        assert_eq!(&side_b[..32], &[0xA1; 32]);

        // 3. `update` -> `update_busy`: once `shadow_a` has a clean guarded baseline,
        //    `shadow_a.update` keeps its own clean deferred block deferred, while
        //    overlapping `shadow_b.update` materializes outside `before` lock and copies `[0xB2; 32]`.
        let _ = shadow_a.update(va0, PAGE);
        shadow_a.clear_card_dirty_for_test();
        gpu_write_shadow(&shadow_a, 0, &[0xB2; 32]);
        assert!(try_defer_page_inner(&shadow_a, 0, PAGE, 0));
        assert_eq!(shadow_a.update(va0, PAGE), 0);
        assert!(has_block(va0 / PAGE));
        assert!(shadow_b.update(va0, PAGE) >= 32);
        assert!(!has_block(va0 / PAGE));
        assert_eq!(&side_b[..32], &[0xB2; 32]);

        // 4. `update` -> `copy_runs`: `shadow_c` (no pending ranges) refreshes across
        //    `shadow_a`'s deferred block without holding any `Shadow` lock during materialization.
        gpu_write_shadow(&shadow_a, 0, &[0xC3; 32]);
        assert!(try_defer_page_inner(&shadow_a, 0, PAGE, 0));
        assert!(has_block(va0 / PAGE));
        assert!(shadow_c.update(va0, PAGE) > 0);
        assert!(!has_block(va0 / PAGE));
        assert_eq!(&side_c[..32], &[0xC3; 32]);

        // 5. `card_merge_relay` directly against an otherwise-deferred block and
        //    `relay_merge_rebar` / `merge_under_writes` when registered shadow uses ReBAR.
        let Some((shadow_d, _side_d)) = make_card_shadow_at(va0 as u64, 2, 0x11) else {
            drop(side_c);
            drop(shadow_c);
            drop(side_b);
            drop(shadow_b);
            drop(_side_a);
            drop(shadow_a);
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        assert!(try_defer_page_inner(&shadow_a, 0, PAGE, 0));
        let mut scratch = Vec::new();
        let mut gpu_card = vec![0x11u8; 64];
        gpu_card[32..64].copy_from_slice(&[0xD4; 32]);
        assert!(shadow_d.card_merge_relay(0, &gpu_card, 0, &mut scratch) > 0);
        assert!(!has_block(va0 / PAGE));
        // SAFETY: merged bytes 32..64 landed in title RAM.
        let merged = unsafe { std::slice::from_raw_parts(ptr.add(32), 32) };
        assert_eq!(merged, &[0xD4; 32]);

        drop(_side_d);
        drop(shadow_d);
        drop(side_c);
        drop(shadow_c);
        drop(side_b);
        drop(shadow_b);
        drop(_side_a);
        drop(shadow_a);

        // SAFETY: reset test memory before registering a fresh shadow over `pages * PAGE`.
        unsafe { ptr::write_bytes(ptr, 0x11, pages * PAGE) };
        let reg_len = (pages * PAGE) as u64;
        crate::shadow::register(va0, va0 as u64, reg_len);
        let reg_shadow = crate::shadow::find(va0 as u64, reg_len).expect("registered shadow");
        if reg_shadow.is_rebar() {
            let list = h.new_list().expect("host list");
            reg_shadow.refresh(&list, va0 as u64, reg_len);
            list.close();
            h.execute(&[&list]);
            list.reset();
            reg_shadow.writing(va0, PAGE, |_, size| Some(vec![0x11u8; size as usize]));
            gpu_write_shadow(&reg_shadow, 0, &[0xE5; 32]);
            assert!(try_defer_page_inner(&reg_shadow, 0, PAGE, 0));
            let mut runs_left = 1024usize;
            assert!(!reg_shadow.merge_under_writes(va0, PAGE, &mut runs_left));
            assert!(has_block(va0 / PAGE));
            let mut gpu_rebar = vec![0x11u8; 64];
            gpu_rebar[..32].copy_from_slice(&[0xE5; 32]);
            let _ = reg_shadow.relay_merge_rebar(&[(0, 64)], &[&gpu_rebar]);
            assert!(!has_block(va0 / PAGE));
            // SAFETY: materialized ReBAR GPU bytes landed in title RAM.
            assert_eq!(unsafe { std::slice::from_raw_parts(ptr, 32) }, &[0xE5; 32]);
        }
        drop(reg_shadow);
        crate::shadow::unregister(va0, va0 as u64);

        clear_hot_for_test(va0 / PAGE);
        // SAFETY: test allocation cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_first_page_sets_ever_active_before_virtual_protect() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` committed bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        clear_hot_for_test(va / PAGE);

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        gpu_write_shadow(&shadow, 0, &[0x5A; 16]);

        // Test the fast-path skip predicate on isolated values without mutating
        // the process-wide `EVER_ACTIVE` atomic (which concurrent tests share).
        assert!(should_skip_fault_lookup(0, false));
        assert!(!should_skip_fault_lookup(0, true));
        assert!(!should_skip_fault_lookup(1, false));
        assert!(!should_skip_fault_lookup(1, true));

        let step = std::rc::Rc::new(std::cell::Cell::new(0u32));
        let ever_step = std::rc::Rc::new(std::cell::Cell::new(0u32));
        let protect_step = std::rc::Rc::new(std::cell::Cell::new(0u32));
        let saw_ever_active_before_protect = std::rc::Rc::new(std::cell::Cell::new(false));
        let step_e = step.clone();
        let ever_e = ever_step.clone();
        EVER_ACTIVE_OBSERVER.with(|obs| {
            *obs.borrow_mut() = Some(Box::new(move || {
                let s = step_e.get() + 1;
                step_e.set(s);
                ever_e.set(s);
            }));
        });
        let step_p = step.clone();
        let protect_p = protect_step.clone();
        let saw_clone = saw_ever_active_before_protect.clone();
        BEFORE_PROTECT_HOOK.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move || {
                let s = step_p.get() + 1;
                step_p.set(s);
                protect_p.set(s);
                saw_clone.set(EVER_ACTIVE.load(Ordering::Acquire));
            }));
        });
        let deferred = try_defer_page_inner(&shadow, 0, PAGE, 0);
        EVER_ACTIVE_OBSERVER.with(|obs| {
            *obs.borrow_mut() = None;
        });
        BEFORE_PROTECT_HOOK.with(|hook| {
            *hook.borrow_mut() = None;
        });
        assert!(deferred);
        assert!(
            saw_ever_active_before_protect.get()
                && ever_step.get() > 0
                && ever_step.get() < protect_step.get(),
            "EVER_ACTIVE must be published strictly BEFORE VirtualProtect(PAGE_NOACCESS)"
        );
        // SAFETY: real read fault materializes the page.
        assert_eq!(unsafe { std::slice::from_raw_parts(ptr, 16) }, &[0x5A; 16]);
        assert!(EVER_ACTIVE.load(Ordering::Acquire));

        clear_hot_for_test(va / PAGE);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_try_arm_under_guards_skips_pinned_and_clears_cold_state() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        let block = va / PAGE;
        clear_hot_for_test(block);

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        gpu_write_shadow(&shadow, 0, &[0x6B; 16]);

        // Pin via `kernel_writing`: `try_arm_lazy_under_guards` must reject deferral.
        let kw_guard = crate::watch::kernel_writing(va, PAGE).expect("kernel_writing");
        assert!(
            !try_defer_page_inner(&shadow, 0, PAGE, 0),
            "pinned block must not be deferred"
        );
        assert!(!has_block(block));
        assert!(is_readable_now(va));
        drop(kw_guard);
        crate::watch::title_memory_remapped(va, PAGE);

        // Set `ColdRo`-like `PAGE_READONLY` guard via section view + `writes(..., true)`.
        // SAFETY: anonymous section mapped to test guard clearing under `try_arm_lazy_under_guards`.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: mapping view.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        assert!(!view.Value.is_null());
        let sva = view.Value as usize;
        let sblock = sva / PAGE;
        clear_hot_for_test(sblock);
        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        assert!(crate::inplace::title_memory_view(
            sva,
            PAGE,
            section as usize,
            0
        ));
        // SAFETY: seed view.
        unsafe { ptr::write_bytes(sva as *mut u8, 0x11, PAGE) };
        assert!(crate::watch::writes(sva, sva + PAGE, true).is_some());
        assert!(
            crate::watch::clean(sblock),
            "view page must be guarded clean"
        );

        let Some((view_shadow, _vside)) = make_card_shadow_at(sva as u64, 1, 0x11) else {
            crate::watch::title_memory_remapped(sva, PAGE);
            // SAFETY: cleanup.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
                CloseHandle(section);
                free_watch_pages(ptr);
            }
            return;
        };
        gpu_write_shadow(&view_shadow, 0, &[0x7C; 16]);

        assert!(
            try_defer_page_inner(&view_shadow, 0, PAGE, 0),
            "non-pinned clean guarded block must defer atomically"
        );
        assert!(
            !crate::watch::clean(sblock),
            "try_arm_lazy_under_guards must clear GuardClean/ColdRo before arming PAGE_NOACCESS"
        );
        // SAFETY: CPU read materializes back to PAGE_READWRITE (still not marked clean).
        let view_bytes = unsafe { std::slice::from_raw_parts(sva as *const u8, 16) };
        assert_eq!(view_bytes, &[0x7C; 16]);
        assert!(!crate::watch::clean(sblock));

        clear_hot_for_test(sblock);
        clear_hot_for_test(block);
        drop(view_shadow);
        drop(shadow);
        crate::watch::title_memory_remapped(sva, PAGE);
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
            CloseHandle(section);
            free_watch_pages(ptr);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_fast_copy_rip_routes_to_fast_fill_without_materializing() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        assert!(!crate::host::is_fast_copy_rip(0));

        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        clear_hot_for_test(va / PAGE);

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        gpu_write_shadow(&shadow, 0, &[0x8D; 16]);
        assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
        assert!(has_block(va / PAGE));

        // `copy_mapped_raw_checked` with `VICTORY_FAST_COPY=1` runs `victory_copy_try`,
        // which faults on `PAGE_NOACCESS`; `watch::on_write` must recognize `is_fast_copy_rip`
        // and pass to `fast_fill::on_fault` without materializing `shadow`'s deferred block.
        let mut dst = vec![0xEEu8; PAGE];
        let ok = crate::host::copy_mapped_raw_checked(dst.as_mut_ptr(), ptr, PAGE, false);
        assert!(
            !ok,
            "copy_mapped_raw_checked must return false on deferred PAGE_NOACCESS"
        );
        assert!(
            has_block(va / PAGE),
            "victory_copy_try fault must route to fast_fill without materializing lazy block"
        );

        materialize_range(va, PAGE);
        clear_hot_for_test(va / PAGE);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_fetcher_cross_queue_sync_reads_latest_direct_queue_writes() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        clear_hot_for_test(va / PAGE);

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        // Write on `Host::queue` (DIRECT) and fetch via `LazyFetcher` (COPY) using `sync_copy_queue`.
        gpu_write_shadow(&shadow, 128, &[0x9E; 64]);
        let fetched = fetch_gpu_bytes(&shadow, 0, PAGE).expect("LazyFetcher::fetch");
        assert_eq!(&fetched[128..192], &[0x9E; 64]);
        assert_eq!(&fetched[..128], &[0x11; 128]);

        clear_hot_for_test(va / PAGE);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_section_view_merges_via_own_view_before_restoring_prot() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        // SAFETY: anonymous section with two aliased title views.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: mapping two views of the same section page.
        let view_a = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        // SAFETY: second aliased view.
        let view_b = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        assert!(!view_a.Value.is_null() && !view_b.Value.is_null());
        let addr_a = view_a.Value as usize;
        let addr_b = view_b.Value as usize;
        clear_hot_for_test(addr_a / PAGE);
        clear_hot_for_test(addr_b / PAGE);

        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        assert!(crate::inplace::title_memory_view(
            addr_a,
            PAGE,
            section as usize,
            0
        ));
        assert!(crate::inplace::title_memory_view(
            addr_b,
            PAGE,
            section as usize,
            0
        ));

        // SAFETY: seed physical memory via `addr_a`.
        unsafe { ptr::write_bytes(addr_a as *mut u8, 0x11, PAGE) };
        let Some((shadow, side)) = make_card_shadow_at(addr_a as u64, 1, 0x11) else {
            crate::watch::title_memory_remapped(addr_a, PAGE);
            crate::watch::title_memory_remapped(addr_b, PAGE);
            // SAFETY: cleanup.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view_a.Value,
                });
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view_b.Value,
                });
                CloseHandle(section);
            }
            return;
        };
        gpu_write_shadow(&shadow, 64, &[0xE1; 32]);
        assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
        assert!(!is_readable_now(addr_a));
        assert!(!is_readable_now(addr_b));

        // While both title views remain `PAGE_NOACCESS`, `read_own` + `card_merge_relay_via_own`
        // must merge GPU bytes into both `shadow` CPU side and physical section RAM.
        let mut own_buf = [0u8; 32];
        assert!(crate::inplace::read_own(addr_a + 64, &mut own_buf));
        assert_eq!(own_buf, [0x11; 32]);

        let gpu_bytes = fetch_gpu_bytes(&shadow, 0, PAGE).expect("fetch");
        let mut scratch = Vec::new();
        assert_eq!(
            shadow.card_merge_relay_via_own(0, &gpu_bytes, 0, &mut scratch),
            Some(32)
        );
        assert!(
            !is_readable_now(addr_a) && !is_readable_now(addr_b),
            "title views must stay PAGE_NOACCESS during own-view merge"
        );
        assert!(crate::inplace::read_own(addr_a + 64, &mut own_buf));
        assert_eq!(own_buf, [0xE1; 32]);
        assert_eq!(&side[64..96], &[0xE1; 32]);

        // Full materialize path also succeeds cleanly.
        materialize_range(addr_a, PAGE);
        assert!(is_readable_now(addr_a) && is_readable_now(addr_b));
        // SAFETY: reading restored views.
        let view_b_bytes = unsafe { std::slice::from_raw_parts((addr_b + 64) as *const u8, 32) };
        assert_eq!(view_b_bytes, &[0xE1; 32]);

        clear_hot_for_test(addr_a / PAGE);
        clear_hot_for_test(addr_b / PAGE);
        drop(shadow);
        crate::watch::title_memory_remapped(addr_a, PAGE);
        crate::watch::title_memory_remapped(addr_b, PAGE);
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view_a.Value,
            });
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view_b.Value,
            });
            CloseHandle(section);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_view_owner_skips_self_refresh_while_other_shadow_materializes_and_promotes_hot() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let pages = 2;
        // SAFETY: anonymous section mapped for a 2-page view.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null(),
                PAGE_READWRITE,
                0,
                (pages * PAGE) as u32,
                ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: mapping 2-page view.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, pages * PAGE) };
        assert!(!view.Value.is_null());
        let va0 = view.Value as usize;
        let va1 = va0 + PAGE;
        clear_hot_for_test(va0 / PAGE);
        clear_hot_for_test(va1 / PAGE);

        crate::inplace::title_memory_section(section as usize, (pages * PAGE) as u64, true);
        assert!(crate::inplace::title_memory_view(
            va0,
            pages * PAGE,
            section as usize,
            0
        ));

        // SAFETY: `va0` is mapped for `2 * PAGE` bytes.
        unsafe { ptr::write_bytes(va0 as *mut u8, 0x11, pages * PAGE) };
        let Some((shadow_a, side_a)) = make_card_shadow_at(va0 as u64, pages, 0x11) else {
            crate::watch::title_memory_remapped(va0, pages * PAGE);
            // SAFETY: cleanup.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
                CloseHandle(section);
            }
            return;
        };
        let Some((shadow_b, side_b)) = make_card_shadow_at(va0 as u64, pages, 0x11) else {
            drop(side_a);
            drop(shadow_a);
            crate::watch::title_memory_remapped(va0, pages * PAGE);
            // SAFETY: cleanup.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
                CloseHandle(section);
            }
            return;
        };

        // 1. View owner skips self-refresh when page 0 was guard-clean before deferring.
        let _ = shadow_a.update(va0, pages * PAGE);
        shadow_a.clear_card_dirty_for_test();
        let _ = shadow_b.update(va0, pages * PAGE);
        assert!(crate::watch::clean(va0 / PAGE));
        gpu_write_shadow(&shadow_a, 0, &[0xA1; 32]);
        assert!(try_defer_page_inner(&shadow_a, 0, PAGE, 0));
        assert!(has_block(va0 / PAGE));
        assert!(
            !owned_cpu_clean_blocks_in(Arc::as_ptr(&shadow_a) as usize, va0 / PAGE, va0 / PAGE + 1)
                .is_empty()
        );
        assert_eq!(shadow_a.update(va0, PAGE), 0);
        assert!(has_block(va0 / PAGE), "view owner must skip self-refresh");
        assert!(!is_readable_now(va0));

        // 2. Pre-defer CPU write safety on page 1:
        // (a) Un-guarded CPU write before deferring leaves `cpu_clean_after == None`.
        // SAFETY: `va1` is committed RW title view memory.
        unsafe { ((va1 + 16) as *mut u8).write(0x22) };
        assert!(!crate::watch::clean(va1 / PAGE));
        gpu_write_shadow(&shadow_a, PAGE, &[0xB1; 16]);
        assert!(try_defer_page_inner(&shadow_a, PAGE, PAGE, 0));
        assert!(
            owned_cpu_clean_blocks_in(Arc::as_ptr(&shadow_a) as usize, va1 / PAGE, va1 / PAGE + 1)
                .is_empty(),
            "dirty view block before defer must not record cpu_clean_after"
        );
        materialize_range(va1, PAGE);
        assert!(shadow_a.update(va1, PAGE) > 0);
        assert_eq!(side_a[PAGE + 16], 0x22);

        // (b) CPU write after `shadow_a` refreshed, guarded by `shadow_b` before `shadow_a` defers:
        // `cpu_clean_after > shadow_a.copied[1]` so `shadow_a.update` immediately materializes
        // and uploads `0x23` before the dependent list runs (while clean pages stay deferred).
        // SAFETY: `va1` is committed RW title view memory.
        unsafe { ((va1 + 20) as *mut u8).write(0x23) };
        assert!(shadow_b.update(va1, PAGE) > 0);
        assert!(crate::watch::clean(va1 / PAGE));
        gpu_write_shadow(&shadow_a, PAGE, &[0xB2; 16]);
        assert!(try_defer_page_inner(&shadow_a, PAGE, PAGE, 0));
        assert!(shadow_a.update(va1, PAGE) > 0);
        assert!(!has_block(va1 / PAGE));
        assert_eq!(side_a[PAGE + 20], 0x23);

        // 3. Overlapping `shadow_b.update` materializes `shadow_a`'s deferred block via
        //    `InternalRead` and promotes it to hot across `promote_frames()` distinct frames.
        materialize_range(va0, PAGE);
        clear_hot_for_test(va0 / PAGE);
        let threshold = promote_frames();
        for f in 0..threshold {
            advance_frames(1);
            let _ = shadow_a.update(va0, PAGE);
            shadow_a.clear_card_dirty_for_test();
            gpu_write_shadow(&shadow_a, 0, &[0xD0 + f as u8; 16]);
            assert!(try_defer_page_inner(&shadow_a, 0, PAGE, (f + 1) as u64));
            assert!(
                shadow_b.update(va0, PAGE) >= 16,
                "overlapping shadow_b must materialize and observe GPU bytes on frame {f}"
            );
            assert!(!has_block(va0 / PAGE));
            assert_eq!(&side_b[..16], &[0xD0 + f as u8; 16]);
        }
        advance_frames(1);
        assert!(
            !try_defer_page_inner(&shadow_a, 0, PAGE, 99),
            "InternalRead must promote overlapping block to hot"
        );

        // 4. `materialize_shadow` (`Retire`) must NOT promote to hot.
        clear_hot_for_test(va1 / PAGE);
        for f in 0..threshold {
            advance_frames(1);
            gpu_write_shadow(&shadow_a, PAGE, &[0xE0 + f as u8; 16]);
            assert!(try_defer_page_inner(&shadow_a, PAGE, PAGE, (f + 10) as u64));
            materialize_shadow(Arc::as_ptr(&shadow_a) as usize);
            assert!(!has_block(va1 / PAGE));
        }
        advance_frames(1);
        assert!(
            try_defer_page_inner(&shadow_a, PAGE, PAGE, 99),
            "Retire (materialize_shadow) must not promote block to hot"
        );
        materialize_shadow(Arc::as_ptr(&shadow_a) as usize);

        clear_hot_for_test(va0 / PAGE);
        clear_hot_for_test(va1 / PAGE);
        drop(side_b);
        drop(shadow_b);
        drop(side_a);
        drop(shadow_a);
        crate::watch::title_memory_remapped(va0, pages * PAGE);
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value });
            CloseHandle(section);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_materialize_ranges_batches_multi_block_and_multi_shadow_into_single_dma() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr1 = alloc_watch_pages(2);
        let ptr2 = alloc_watch_pages(2);
        let ptr3 = alloc_watch_pages(3);
        // SAFETY: fresh allocations own their respective page counts.
        unsafe {
            ptr::write_bytes(ptr1, 0x11, 2 * PAGE);
            ptr::write_bytes(ptr2, 0x11, 2 * PAGE);
            ptr::write_bytes(ptr3, 0x11, 3 * PAGE);
        }
        let va1 = ptr1 as usize;
        let va2 = ptr2 as usize;
        let va3 = ptr3 as usize;
        for b in [
            va1 / PAGE,
            va1 / PAGE + 1,
            va2 / PAGE,
            va2 / PAGE + 1,
            va3 / PAGE,
            va3 / PAGE + 1,
            va3 / PAGE + 2,
        ] {
            clear_hot_for_test(b);
        }

        let Some((shadow1, _side1)) = make_card_shadow_at(va1 as u64, 2, 0x11) else {
            // SAFETY: cleanup.
            unsafe {
                free_watch_pages(ptr1);
                free_watch_pages(ptr2);
                free_watch_pages(ptr3);
            }
            return;
        };
        let Some((shadow2, _side2)) = make_card_shadow_at(va2 as u64, 2, 0x11) else {
            drop(shadow1);
            // SAFETY: cleanup.
            unsafe {
                free_watch_pages(ptr1);
                free_watch_pages(ptr2);
                free_watch_pages(ptr3);
            }
            return;
        };
        let Some((shadow3, _side3)) = make_card_shadow_at(va3 as u64, 3, 0x11) else {
            drop(shadow2);
            drop(shadow1);
            // SAFETY: cleanup.
            unsafe {
                free_watch_pages(ptr1);
                free_watch_pages(ptr2);
                free_watch_pages(ptr3);
            }
            return;
        };

        gpu_write_shadow(&shadow1, 0, &[0xA1; 16]);
        gpu_write_shadow(&shadow1, PAGE, &[0xA2; 16]);
        gpu_write_shadow(&shadow2, 0, &[0xB1; 16]);
        gpu_write_shadow(&shadow2, PAGE, &[0xB2; 16]);

        assert!(try_defer_page_inner(&shadow1, 0, PAGE, 0));
        assert!(try_defer_page_inner(&shadow1, PAGE, PAGE, 0));
        assert!(try_defer_page_inner(&shadow2, 0, PAGE, 0));
        assert!(try_defer_page_inner(&shadow2, PAGE, PAGE, 0));

        let sent_before = thread_dma_batches();
        assert!(materialize_ranges(&[(va1, 2 * PAGE), (va2, 2 * PAGE)]));
        assert_eq!(
            thread_dma_batches(),
            sent_before + 1,
            "materialize_ranges must batch all 4 blocks across 2 shadows into 1 DMA"
        );
        assert!(!has_block(va1 / PAGE) && !has_block(va1 / PAGE + 1));
        assert!(!has_block(va2 / PAGE) && !has_block(va2 / PAGE + 1));
        // SAFETY: all 4 pages are materialized to readable RAM.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(ptr1, 16), &[0xA1; 16]);
            assert_eq!(std::slice::from_raw_parts(ptr1.add(PAGE), 16), &[0xA2; 16]);
            assert_eq!(std::slice::from_raw_parts(ptr2, 16), &[0xB1; 16]);
            assert_eq!(std::slice::from_raw_parts(ptr2.add(PAGE), 16), &[0xB2; 16]);
        }

        // Batched `materialize_shadow` across 3 deferred blocks of `shadow3`.
        gpu_write_shadow(&shadow3, 0, &[0xC1; 16]);
        gpu_write_shadow(&shadow3, PAGE, &[0xC2; 16]);
        gpu_write_shadow(&shadow3, 2 * PAGE, &[0xC3; 16]);
        assert!(try_defer_page_inner(&shadow3, 0, PAGE, 0));
        assert!(try_defer_page_inner(&shadow3, PAGE, PAGE, 0));
        assert!(try_defer_page_inner(&shadow3, 2 * PAGE, PAGE, 0));

        let sent_before_retire = thread_dma_batches();
        assert!(materialize_shadow(Arc::as_ptr(&shadow3) as usize));
        assert_eq!(
            thread_dma_batches(),
            sent_before_retire + 1,
            "materialize_shadow must batch all 3 blocks into 1 DMA"
        );
        assert!(!has_block(va3 / PAGE) && !has_block(va3 / PAGE + 1) && !has_block(va3 / PAGE + 2));
        // SAFETY: all 3 pages are materialized to readable RAM.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(ptr3, 16), &[0xC1; 16]);
            assert_eq!(std::slice::from_raw_parts(ptr3.add(PAGE), 16), &[0xC2; 16]);
            assert_eq!(
                std::slice::from_raw_parts(ptr3.add(2 * PAGE), 16),
                &[0xC3; 16]
            );
        }

        for b in [
            va1 / PAGE,
            va1 / PAGE + 1,
            va2 / PAGE,
            va2 / PAGE + 1,
            va3 / PAGE,
            va3 / PAGE + 1,
            va3 / PAGE + 2,
        ] {
            clear_hot_for_test(b);
        }
        drop(shadow3);
        drop(shadow2);
        drop(shadow1);
        // SAFETY: cleanup.
        unsafe {
            free_watch_pages(ptr1);
            free_watch_pages(ptr2);
            free_watch_pages(ptr3);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_deferred_card_page_does_not_pin_card_sent_and_materializes_live() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let len = PAGE;
        let ptr = alloc_watch_pages(1);
        assert!(!ptr.is_null());
        let va = ptr as usize;
        clear_hot_for_test(va / PAGE);
        // SAFETY: `ptr` is valid writable `PAGE` bytes.
        unsafe {
            std::ptr::write_bytes(ptr, 0x11, len);
        }
        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };

        // Seed a nonempty `Card::sent` history on this isolated fixture (without
        // touching the process-wide `RELAY_FETCHED` watermark or `CARDED`) and
        // defer page 0 at an ancient `fetched_after = 1`.
        shadow.seed_card_sent_for_test(10, vec![(0, 64)]);
        shadow.seed_card_sent_for_test(20, vec![(64, 128)]);
        assert_eq!(shadow.card_sent_len_for_test(), 2);

        gpu_write_shadow(&shadow, 0, &[0x5A; 32]);
        assert!(try_defer_page_inner(&shadow, 0, len, 1));
        assert!(has_block(va / PAGE));

        // 1. Production watermark computation when every changed card page is
        //    deferred (`has_card == false`) advances past `fetched_after = 1`
        //    rather than pinning the watermark to the deferred page, while NEVER
        //    covering a later incomplete DIRECT upload (`sent_as = 20`) whose
        //    fence has not completed yet (`completed_order = 15`).
        let wm15 = crate::host::relay_no_card_fetch_watermark(true, Some(15), 15)
            .expect("inflight watermark");
        assert_eq!(wm15, 15);
        let wm15_direct = crate::host::relay_no_card_fetch_watermark(false, Some(15), 15)
            .expect("non-inflight watermark");
        assert_eq!(wm15_direct, 15);
        assert!(
            shadow.card_sent_has_after_for_test(wm15_direct),
            "later incomplete DIRECT upload (sent_as=20) must remain in-flight above completed_order=15"
        );
        let wm25 = crate::host::relay_no_card_fetch_watermark(true, Some(25), 25)
            .expect("inflight watermark 25");
        assert_eq!(wm25, 25);
        assert!(
            !shadow.card_sent_has_after_for_test(wm25),
            "once completed_order reaches 25, upload at 20 is no longer in-flight"
        );

        // 2. Production `card_dirty` empty-dirty fast path (`card_dirty_with`)
        //    prunes `sent_as <= 15` on this isolated fixture while page 0 stays
        //    deferred at `fetched_after = 1`.
        let mut isolated_carded = vec![Arc::downgrade(&shadow)];
        let drained_fast = crate::shadow::card_dirty_with(&mut isolated_carded, 30, wm15, true);
        assert!(drained_fast.is_empty());
        assert_eq!(shadow.card_sent_len_for_test(), 1);

        // 3. Production `card_dirty` non-empty-dirty slow path prunes `sent_as <= 25`
        //    and records the newly drained dirty range under `sent_as = 30`.
        shadow.seed_card_dirty_for_test(vec![(128, 192)]);
        let drained_slow = crate::shadow::card_dirty_with(&mut isolated_carded, 30, wm25, true);
        assert_eq!(drained_slow.len(), 1);
        assert_eq!(drained_slow[0].1, vec![(128, 192)]);
        assert_eq!(shadow.card_sent_len_for_test(), 1);

        // Materializing the cold page fetches live GPU bytes cleanly.
        assert!(materialize_range(va, len));
        assert!(!has_block(va / PAGE));
        // SAFETY: page is materialized and readable.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(ptr, 32), &[0x5A; 32]);
        }

        clear_hot_for_test(va / PAGE);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_active_hot_block_does_not_decay_after_120_frames_and_reprobes_cleanly() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        let block = va / PAGE;
        clear_hot_for_test(block);

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };

        // Promote the block across `promote_frames()` distinct frames.
        let threshold = promote_frames();
        for f in 0..threshold {
            advance_frames(1);
            gpu_write_shadow(&shadow, 0, &[0x50 + f as u8; 8]);
            assert!(try_defer_page_inner(&shadow, 0, PAGE, (f + 1) as u64));
            // SAFETY: CPU read fault increments hot counter.
            assert_eq!(unsafe { ptr.read() }, 0x50 + f as u8);
        }

        // Stay actively GPU-written for > HOT_DECAY_FRAMES (e.g. 200 frames):
        // must remain hot without thrashing every 120 frames.
        for f in 0..(HOT_DECAY_FRAMES + 80) {
            advance_frames(1);
            assert!(
                !try_defer_page_inner(&shadow, 0, PAGE, 100 + f),
                "actively GPU-written hot block must not decay at active frame {f}"
            );
        }

        // Simulate reaching the reprobe horizon while still active (stepping in
        // chunks <= HOT_DECAY_FRAMES so inactivity decay does not trigger).
        let step = HOT_DECAY_FRAMES / 2;
        let mut advanced = HOT_DECAY_FRAMES + 80;
        while advanced <= HOT_REPROBE_FRAMES {
            advance_frames(step as usize);
            advanced += step;
            let deferred = try_defer_page_inner(&shadow, 0, PAGE, 1000 + advanced);
            if advanced <= HOT_REPROBE_FRAMES {
                assert!(!deferred, "must stay hot before reprobe horizon");
            } else {
                assert!(
                    deferred,
                    "must allow one reprobe deferral past reprobe horizon"
                );
            }
        }

        // One fault on the reprobed block must immediately re-promote it on the
        // very same frame so the next flight skips deferral again.
        // SAFETY: volatile read triggers VEH materialization and re-promotion.
        let got = unsafe { ptr.read_volatile() };
        assert_eq!(got, 0x50 + (threshold - 1) as u8);
        assert!(
            !has_block(block),
            "reprobe read must materialize deferred block"
        );
        advance_frames(1);
        assert!(
            !try_defer_page_inner(&shadow, 0, PAGE, 9999),
            "single fault during reprobe must immediately re-promote hot block"
        );

        clear_hot_for_test(block);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn fast_copy_read_into_raw_and_unmapped_hole_respect_lazy_back_and_holes() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        // Reserve 3 pages, commit pages 0 and 2, leave page 1 uncommitted (hole).
        // SAFETY: reserving a fresh 3-page region for test.
        let base =
            unsafe { VirtualAlloc(ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_READWRITE) } as *mut u8;
        assert!(!base.is_null(), "VirtualAlloc reserve failed");
        // SAFETY: committing page 0 inside `base`.
        let p0 = unsafe { VirtualAlloc(base.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) } as *mut u8;
        // SAFETY: committing page 2 inside `base`.
        let p2_expected = unsafe { base.add(2 * PAGE) };
        // SAFETY: `p2_expected` lies within our 3-page reservation `base`.
        let p2 = unsafe { VirtualAlloc(p2_expected.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) }
            as *mut u8;
        assert_eq!(p0, base);
        assert_eq!(p2, p2_expected);
        // SAFETY: pages 0 and 2 are committed RW.
        unsafe {
            ptr::write_bytes(p0, 0x3A, PAGE);
            ptr::write_bytes(p2, 0x7C, PAGE);
        }

        // 1. `copy_mapped_raw_checked` across committed + hole + committed zeroes
        //    only the hole and preserves both committed pages.
        let mut dst = vec![0xEEu8; 3 * PAGE];
        assert!(crate::host::copy_mapped_raw_checked(
            dst.as_mut_ptr(),
            base,
            3 * PAGE,
            false
        ));
        assert!(dst[..PAGE].iter().all(|&b| b == 0x3A));
        assert!(dst[PAGE..2 * PAGE].iter().all(|&b| b == 0x00));
        assert!(dst[2 * PAGE..].iter().all(|&b| b == 0x7C));

        // 2. `read_into_raw` on mapped page succeeds; on uncommitted hole returns false.
        let mut buf = vec![0u8; 64];
        assert!(crate::host::read_into_raw(p0, &mut buf));
        assert!(buf.iter().all(|&b| b == 0x3A));
        // SAFETY: `base.add(PAGE)` points to the uncommitted middle page inside our 3-page reservation.
        let hole_ptr = unsafe { base.add(PAGE) };
        assert!(!crate::host::read_into_raw(hole_ptr, &mut buf));

        // 3. `read_into_raw` on a `lazy_back` `PAGE_NOACCESS` section page returns
        //    false WITHOUT triggering `lazy_back::handle_fault` materialization.
        let sec_ptr = alloc_watch_pages(1);
        // SAFETY: `sec_ptr` owns `PAGE` committed bytes.
        unsafe { ptr::write_bytes(sec_ptr, 0x3A, PAGE) };
        let va0 = sec_ptr as usize;
        clear_hot_for_test(va0 / PAGE);
        if let Some((shadow, _side)) = make_card_shadow_at(va0 as u64, 1, 0x3A) {
            gpu_write_shadow(&shadow, 0, &[0x91; 16]);
            assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
            assert!(has_block(va0 / PAGE));
            assert!(
                !crate::host::read_into_raw(sec_ptr, &mut buf),
                "read_into_raw must return false on deferred PAGE_NOACCESS"
            );
            assert!(
                has_block(va0 / PAGE),
                "read_into_raw must NOT materialize lazy_back block under caller locks"
            );
            // `read_into_fast` materializes and returns the GPU bytes.
            assert!(crate::host::read_into_fast(sec_ptr, &mut buf));
            assert_eq!(&buf[..16], &[0x91; 16]);
            assert!(!has_block(va0 / PAGE));
            clear_hot_for_test(va0 / PAGE);
            drop(shadow);
        }
        // SAFETY: freeing section test allocation.
        unsafe { free_watch_pages(sec_ptr) };

        // SAFETY: releasing test reservation.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn fill_ahead_wrote_under_rebar_mirror_lock_does_not_deadlock_on_lazy_sibling_page() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(2);
        // SAFETY: `ptr` owns `2 * PAGE` committed RW bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, 2 * PAGE) };
        let va0 = ptr as usize;
        let va1 = va0 + PAGE;
        clear_hot_for_test(va0 / PAGE);
        clear_hot_for_test(va1 / PAGE);

        let Some(shadow) = Shadow::new_rebar_for_test(va0 as u64, (2 * PAGE) as u64, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };

        // Record a multi-page fill spanning both page 0 and page 1, then defer
        // page 1 in `lazy_back` (`PAGE_NOACCESS`) on the SAME ReBAR shadow.
        crate::host::test_fill_ahead_read(va0, 2 * PAGE);
        gpu_write_shadow(&shadow, PAGE, &[0xCC; 16]);
        assert!(try_defer_page_inner(&shadow, PAGE, PAGE, 0));
        assert!(has_block(va1 / PAGE));

        // Merging GPU writes into page 0 holds `shadow.rebar.mirror` and calls
        // `fill_ahead::wrote`, which checks the `[va0, va0 + 2*PAGE)` fill.
        // Without raw non-materializing reads in `fill_ahead::wrote`, sampling
        // page 1 would invoke `lazy_back::materialize_range` ->
        // `shadow.relay_merge_rebar` -> `rebar.mirror.lock()` on the same
        // thread and permanently self-deadlock.
        let mut gpu_page0 = vec![0x11u8; PAGE];
        gpu_page0[0] = 0xAA;
        let written = shadow
            .relay_merge_rebar(&[(0, PAGE)], &[&gpu_page0])
            .expect("relay_merge_rebar on page 0");
        assert!(written > 0);
        // SAFETY: page 0 is readable and now contains the merged GPU byte.
        assert_eq!(unsafe { ptr.read() }, 0xAA);
        assert!(
            has_block(va1 / PAGE),
            "fill_ahead::wrote must not materialize sibling lazy_back blocks under Shadow locks"
        );

        // Accessing page 1 afterward still materializes its deferred GPU write cleanly.
        // SAFETY: volatile read triggers VEH materialization of page 1.
        assert_eq!(unsafe { ptr.add(PAGE).read_volatile() }, 0xCC);
        assert!(!has_block(va1 / PAGE));

        clear_hot_for_test(va0 / PAGE);
        clear_hot_for_test(va1 / PAGE);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_discard_range_saturated_len_completes_instantly_and_prunes_hot() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(2);
        // SAFETY: `ptr` owns `2 * PAGE` committed RW bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, 2 * PAGE) };
        let va0 = ptr as usize;
        let va1 = va0 + PAGE;
        let b0 = va0 / PAGE;
        let b1 = va1 / PAGE;
        clear_hot_for_test(b0);
        clear_hot_for_test(b1);

        let Some((shadow, _side)) = make_card_shadow_at(va0 as u64, 2, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };

        // Defer page 1 and seed hot entries for both page 0 (< first) and page 1 (>= first).
        gpu_write_shadow(&shadow, PAGE, &[0xAB; 16]);
        assert!(try_defer_page_inner(&shadow, PAGE, PAGE, 0));
        assert!(has_block(b1));
        assert!(!has_block_in_range(va0, PAGE));
        assert!(has_block_in_range(va1, PAGE));
        assert!(has_block_in_range(va0, usize::MAX));

        {
            let mut table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
            note_fault_hot_locked(&mut table, b0);
            note_fault_hot_locked(&mut table, b1);
            assert!(table.hot.contains_key(&b0));
            assert!(table.hot.contains_key(&b1));
        }

        // Saturated length (`usize::MAX`) with an active lazy block and a hot entry below `va1`
        // must complete in O(table.hot.len()) instead of iterating `first..last`.
        discard_range(va1, usize::MAX);
        assert!(!has_block(b1));
        assert!(!has_block_in_range(va1, usize::MAX));
        {
            let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
            assert!(
                table.hot.contains_key(&b0),
                "hot entry below discard range must stay"
            );
            assert!(
                !table.hot.contains_key(&b1),
                "hot entry inside saturated discard range must be pruned"
            );
        }

        // Restore RW on page 1 before freeing (in production the runtime remaps/frees it).
        let mut old = 0u32;
        // SAFETY: `va1` is our committed test page.
        unsafe { VirtualProtect(va1 as *const _, PAGE, PAGE_READWRITE, &mut old) };

        clear_hot_for_test(b0);
        clear_hot_for_test(b1);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_rejects_private_allocation_without_own_view() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        // SAFETY: private write-watched `VirtualAlloc` without section own-view path.
        let priv_ptr = unsafe {
            VirtualAlloc(
                ptr::null(),
                PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as *mut u8;
        assert!(!priv_ptr.is_null());
        // SAFETY: committed above.
        unsafe { ptr::write_bytes(priv_ptr, 0x11, PAGE) };
        let va = priv_ptr as usize;
        clear_hot_for_test(va / PAGE);

        if let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) {
            gpu_write_shadow(&shadow, 0, &[0xEE; 16]);
            assert!(
                !try_defer_page_inner(&shadow, 0, PAGE, 0),
                "private allocation without own-view merge path must stay on eager readback"
            );
            assert!(!has_block(va / PAGE));
            assert!(is_readable_now(va));
            drop(shadow);
        }
        // SAFETY: cleanup.
        unsafe { VirtualFree(priv_ptr.cast(), 0, MEM_RELEASE) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_rejects_deferral_and_rolls_back_primary_when_alias_protect_fails() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        // SAFETY: anonymous 64 KiB section mapped as a full 64 KiB primary view and a 4 KiB alias.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: full 64 KiB primary view.
        let primary_view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        // SAFETY: 4 KiB alias view (64 KiB `VirtualProtect` on this alias fails).
        let short_alias = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 4096) };
        assert!(!primary_view.Value.is_null() && !short_alias.Value.is_null());
        let va_p = primary_view.Value as usize;
        let va_a = short_alias.Value as usize;
        clear_hot_for_test(va_p / PAGE);
        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        assert!(crate::inplace::title_memory_view(
            va_p,
            PAGE,
            section as usize,
            0
        ));
        assert!(crate::inplace::title_memory_view(
            va_a,
            4096,
            section as usize,
            0
        ));

        // SAFETY: seed primary view.
        unsafe { ptr::write_bytes(va_p as *mut u8, 0x11, PAGE) };
        if let Some((shadow, _side)) = make_card_shadow_at(va_p as u64, 1, 0x11) {
            gpu_write_shadow(&shadow, 0, &[0xAB; 16]);
            assert!(
                !try_defer_page_inner(&shadow, 0, PAGE, 0),
                "deferral must be rejected when protecting a 4 KiB alias fails"
            );
            assert!(!has_block(va_p / PAGE));
            assert!(
                is_readable_now(va_p),
                "primary view protection must be rolled back on alias failure"
            );
            drop(shadow);
        }

        crate::watch::title_memory_remapped(va_p, PAGE);
        crate::watch::title_memory_remapped(va_a, 4096);
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: primary_view.Value,
            });
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: short_alias.Value,
            });
            CloseHandle(section);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_failed_dma_keeps_entry_armed_and_recovers_on_retry() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        clear_hot_for_test(va / PAGE);

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        gpu_write_shadow(&shadow, 0, &[0x7E; 32]);
        assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
        assert!(has_block(va / PAGE));
        assert!(!is_readable_now(va));

        // Inject a DMA failure: `handle_fault` must return `false`, keep the
        // entry tracked, and leave the page at `PAGE_NOACCESS` (never exposing stale RAM).
        set_fail_next_fetches_for_test(1);
        assert!(
            !handle_fault(va, false),
            "failed materialization must NOT report fault as handled"
        );
        assert!(
            has_block(va / PAGE),
            "entry must remain tracked after failed materialization"
        );
        assert!(
            !is_readable_now(va),
            "page must stay PAGE_NOACCESS while GPU bytes have not landed"
        );

        // Once the transient failure clears, a retry succeeds and delivers GPU bytes.
        assert!(handle_fault(va, false));
        assert!(!has_block(va / PAGE));
        assert!(is_readable_now(va));
        // SAFETY: page is now materialized and readable.
        assert_eq!(unsafe { std::slice::from_raw_parts(ptr, 32) }, &[0x7E; 32]);

        clear_hot_for_test(va / PAGE);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_remapping_blocks_deferral_and_discard_range_materializes_surviving_alias() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        // SAFETY: anonymous section with two aliased views.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: two distinct views of offset 0.
        let view_a = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        // SAFETY: second aliased view.
        let view_b = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        assert!(!view_a.Value.is_null() && !view_b.Value.is_null());
        let addr_a = view_a.Value as usize;
        let addr_b = view_b.Value as usize;
        clear_hot_for_test(addr_a / PAGE);
        clear_hot_for_test(addr_b / PAGE);
        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        assert!(crate::inplace::title_memory_view(
            addr_a,
            PAGE,
            section as usize,
            0
        ));
        assert!(crate::inplace::title_memory_view(
            addr_b,
            PAGE,
            section as usize,
            0
        ));
        // SAFETY: seed physical page.
        unsafe { ptr::write_bytes(addr_a as *mut u8, 0x11, PAGE) };

        if let Some((shadow, _side)) = make_card_shadow_at(addr_a as u64, 1, 0x11) {
            gpu_write_shadow(&shadow, 0, &[0xC9; 32]);
            // 1. Holding `title_memory_remapping` blocks deferral in `[addr_a, addr_a + PAGE)`.
            let remap = crate::watch::title_memory_remapping(addr_a, PAGE);
            assert!(
                !try_defer_page_inner(&shadow, 0, PAGE, 0),
                "deferral must be blocked while title_memory_remapping is active"
            );
            drop(remap);

            // 2. Defense-in-depth: if `discard_range(addr_a, PAGE)` is invoked while
            //    `addr_a` (primary) and `addr_b` (surviving alias) are armed, `discard_range`
            //    materializes GPU bytes via `addr_b` and restores `addr_b` before dropping.
            assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
            assert!(has_block(addr_a / PAGE) && has_block(addr_b / PAGE));
            discard_range(addr_a, PAGE);
            assert!(!has_block(addr_a / PAGE) && !has_block(addr_b / PAGE));
            assert!(
                is_readable_now(addr_b),
                "surviving alias must be readable after primary discard_range"
            );
            // SAFETY: surviving alias `addr_b` is readable with merged GPU bytes.
            let got_b = unsafe { std::slice::from_raw_parts(addr_b as *const u8, 32) };
            assert_eq!(got_b, &[0xC9; 32]);
            // Restore `addr_a` for test cleanup (in production `addr_a` was unmapped).
            let mut old = 0u32;
            // SAFETY: test cleanup of `addr_a`.
            unsafe { VirtualProtect(addr_a as *const _, PAGE, PAGE_READWRITE, &mut old) };
            drop(shadow);
        }

        clear_hot_for_test(addr_a / PAGE);
        clear_hot_for_test(addr_b / PAGE);
        crate::watch::title_memory_remapped(addr_a, PAGE);
        crate::watch::title_memory_remapped(addr_b, PAGE);
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view_a.Value,
            });
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view_b.Value,
            });
            CloseHandle(section);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_copy_mapped_title_dst_materializes_deferred_source() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let src_ptr = alloc_watch_pages(1);
        let dst_ptr = alloc_watch_pages(1);
        // SAFETY: committed pages.
        unsafe {
            ptr::write_bytes(src_ptr, 0x11, PAGE);
            ptr::write_bytes(dst_ptr, 0x22, PAGE);
        }
        let src_va = src_ptr as usize;
        clear_hot_for_test(src_va / PAGE);

        if let Some((shadow, _side)) = make_card_shadow_at(src_va as u64, 1, 0x11) {
            gpu_write_shadow(&shadow, 0, &[0xD7; 64]);
            assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
            assert!(has_block(src_va / PAGE));

            // Title-to-title copy (`title_dst = true`) with deferred `src` and
            // committed `dst` must materialize `src` instead of zero-filling `dst`.
            crate::host::copy_mapped(dst_ptr, src_ptr, PAGE, true);
            assert!(!has_block(src_va / PAGE));
            // SAFETY: `dst_ptr` holds the materialized bytes copied from `src_ptr`.
            let copied = unsafe { std::slice::from_raw_parts(dst_ptr, 64) };
            assert_eq!(copied, &[0xD7; 64]);
            drop(shadow);
        }

        clear_hot_for_test(src_va / PAGE);
        // SAFETY: cleanup.
        unsafe {
            free_watch_pages(dst_ptr);
            free_watch_pages(src_ptr);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_unwind_failed_virtual_protect_makes_failure_visible_to_waiters() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        let block = va / PAGE;
        clear_hot_for_test(block);

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        gpu_write_shadow(&shadow, 0, &[0x6B; 32]);
        assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
        assert!(has_block(block));
        assert!(!is_readable_now(va));

        let _reset_fail = fail_restore_protect_guard(va);

        let entered = Arc::new((Mutex::new(false), Condvar::new()));
        let release_panic = Arc::new((Mutex::new(false), Condvar::new()));
        let entered_t1 = entered.clone();
        let release_t1 = release_panic.clone();

        // Thread 1 claims the block (`materializing = true`), merges GPU bytes
        // into RAM (`merged = true`), waits for Thread 2 to block on `LAZY_CV`,
        // and then panics before `restore_prot` so `MaterializeBatchScope::drop`
        // attempts `VirtualProtect` restoration (which fails for `va`).
        let t1 = std::thread::spawn(move || {
            AFTER_MERGE_HOOK.with(|hook| {
                *hook.borrow_mut() = Some(Box::new(move || {
                    {
                        let mut g = entered_t1.0.lock().unwrap();
                        *g = true;
                        entered_t1.1.notify_all();
                    }
                    let mut r = release_t1.0.lock().unwrap();
                    while !*r {
                        r = release_t1.1.wait(r).unwrap();
                    }
                    panic!("simulated post-merge unwind during materialization");
                }));
            });
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = handle_fault(va, false);
            }));
        });

        {
            let mut g = entered.0.lock().unwrap();
            while !*g {
                g = entered.1.wait(g).unwrap();
            }
        }

        // Thread 2 enters `handle_fault` while Thread 1 holds `materializing = true`,
        // signals deterministically right at the `LAZY_CV.wait(table)` entry point
        // while still holding `LAZY_STATE`, and then parks on `LAZY_CV`. Because
        // Thread 1's `MaterializeBatchScope::drop` must acquire `LAZY_STATE` before
        // calling `LAZY_CV.notify_all()`, Thread 1 cannot notify until Thread 2 has
        // atomically entered `LAZY_CV.wait(table)` and released `LAZY_STATE`.
        let waiter_at_cv = Arc::new((Mutex::new(false), Condvar::new()));
        let waiter_at_cv_t2 = waiter_at_cv.clone();
        let t2 = std::thread::spawn(move || {
            BEFORE_LAZY_CV_WAIT_HOOK.with(|hook| {
                *hook.borrow_mut() = Some(Box::new(move || {
                    let mut g = waiter_at_cv_t2.0.lock().unwrap();
                    *g = true;
                    waiter_at_cv_t2.1.notify_all();
                }));
            });
            handle_fault(va, false)
        });
        {
            let mut g = waiter_at_cv.0.lock().unwrap();
            while !*g {
                g = waiter_at_cv.1.wait(g).unwrap();
            }
        }

        {
            let mut r = release_panic.0.lock().unwrap();
            *r = true;
            release_panic.1.notify_all();
        }
        t1.join().expect("t1 joined");
        let waiter_handled = t2.join().expect("t2 joined");

        assert!(
            !waiter_handled,
            "waiter must observe failure when unwind VirtualProtect restoration fails"
        );
        assert!(
            has_block(block),
            "entry must remain tracked when unwind VirtualProtect restoration fails"
        );
        assert!(
            !is_readable_now(va),
            "page remains PAGE_NOACCESS after failed VirtualProtect restoration"
        );
        assert!(
            !handle_fault(va, false),
            "subsequent fault on restore-failed entry must continue reporting failure"
        );
        assert!(
            !materialize_range(va, PAGE),
            "materialize_range on restore-failed entry must report failure"
        );

        // Clean up the intentionally broken entry and restore RW before freeing.
        drop(_reset_fail);
        discard_range(va, PAGE);
        assert!(!has_block(block));
        let mut old = 0u32;
        // SAFETY: restoring RW on our test allocation for cleanup.
        unsafe { VirtualProtect(va as *const _, PAGE, PAGE_READWRITE, &mut old) };
        clear_hot_for_test(block);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_fetch_gpu_bytes_and_batch_detach_inflight_fetcher_on_error() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        clear_hot_for_test(va / PAGE);

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };
        gpu_write_shadow(&shadow, 0, &[0x4D; 32]);

        // 1. Seed a test-owned fetcher slot so parallel tests sharing global
        //    `FETCHER` cannot race on `slot.is_none()` or `f.sent`.
        let mut slot: Option<LazyFetcher> = None;
        let mut unconfirmed: Vec<UnconfirmedCopy> = Vec::new();
        let (initial, _, _) =
            run_fetcher_batch(&mut slot, &mut unconfirmed, &[(&shadow, 0, PAGE)], true)
                .expect("initial batch");
        assert_eq!(&initial[..32], &[0x4D; 32]);
        let (old_fence, fence_before) = {
            let f = slot.as_ref().expect("fetcher initialized");
            assert_eq!(f.sent, 1);
            assert!(
                f.in_flight_sources.is_empty(),
                "confirmed fetch must release retained COPY source resources"
            );
            (f.fence.clone(), f.fence.as_raw() as usize)
        };

        // 2. Simulate an unconfirmed post-submit error in `run_fetcher_batch`:
        //    the in-flight fetcher must be detached (`None`), never reused, and
        //    must retain the COPY source resource in `unconfirmed`.
        LAST_DETACHED_SOURCE_COUNT.with(|c| c.set(0));
        FAIL_AFTER_SUBMIT_FOR_TEST.with(|f| f.set(true));
        let err = run_fetcher_batch(&mut slot, &mut unconfirmed, &[(&shadow, 0, PAGE)], true);
        FAIL_AFTER_SUBMIT_FOR_TEST.with(|f| f.set(false));
        assert!(err.is_err(), "injected post-submit error must fail");
        assert!(
            slot.is_none(),
            "run_fetcher_batch must detach in-flight fetcher on post-submit error"
        );
        assert_eq!(
            LAST_DETACHED_SOURCE_COUNT.with(|c| c.get()),
            1,
            "detached unconfirmed fetcher must retain COPY source resources"
        );
        assert_eq!(unconfirmed.len(), 1);

        // Next batch on the slot retires the completed unconfirmed entry,
        // creates a fresh fetcher, and succeeds cleanly.
        let (recovered, _, _) =
            run_fetcher_batch(&mut slot, &mut unconfirmed, &[(&shadow, 0, PAGE)], true)
                .expect("recovered batch");
        assert!(unconfirmed.is_empty());
        assert_eq!(&recovered[..32], &[0x4D; 32]);
        let fence_after = {
            let f = slot.as_ref().expect("fresh fetcher created");
            assert_eq!(f.sent, 1, "fresh fetcher must start at sent = 1");
            assert!(f.in_flight_sources.is_empty());
            f.fence.as_raw() as usize
        };
        assert_ne!(
            fence_after, fence_before,
            "fresh fetcher must replace the detached in-flight fetcher"
        );
        drop(old_fence);

        // 3. Also verify multi-item batch detach + recovery on the test-owned slot
        //    and through `fetch_gpu_bytes` / `fetch_gpu_batch` wrappers.
        LAST_DETACHED_SOURCE_COUNT.with(|c| c.set(0));
        FAIL_AFTER_SUBMIT_FOR_TEST.with(|f| f.set(true));
        let multi_err = run_fetcher_batch(
            &mut slot,
            &mut unconfirmed,
            &[(&shadow, 0, PAGE / 2), (&shadow, PAGE / 2, PAGE / 2)],
            true,
        );
        FAIL_AFTER_SUBMIT_FOR_TEST.with(|f| f.set(false));
        assert!(multi_err.is_err());
        assert!(slot.is_none());
        assert_eq!(LAST_DETACHED_SOURCE_COUNT.with(|c| c.get()), 2);
        assert_eq!(unconfirmed.len(), 1);
        retire_completed_unconfirmed(&mut unconfirmed);
        assert!(unconfirmed.is_empty());

        let (batch_buf, _, _) =
            fetch_gpu_batch(&[(&shadow, 0, PAGE)]).expect("recovered fetch_gpu_batch");
        assert_eq!(&batch_buf[..32], &[0x4D; 32]);
        let single_buf = fetch_gpu_bytes(&shadow, 0, PAGE).expect("recovered fetch_gpu_bytes");
        assert_eq!(&single_buf[..32], &[0x4D; 32]);

        clear_hot_for_test(va / PAGE);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_direct_submit_guard_serializes_across_first_activation_transition() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();

        // When serialization is disabled (production with VICTORY_LAZY_BACK unset),
        // `direct_submit_guard_if(false)` stays a lock-free no-op.
        assert!(direct_submit_guard_if(false).0.is_none());

        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        let block = va / PAGE;
        clear_hot_for_test(block);

        let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) else {
            // SAFETY: cleanup.
            unsafe { free_watch_pages(ptr) };
            return;
        };

        // Reset `EVER_ACTIVE` to simulate pre-first-activation state, restoring it on drop.
        struct RestoreEverActive(bool);
        impl Drop for RestoreEverActive {
            fn drop(&mut self) {
                EVER_ACTIVE.store(self.0, Ordering::SeqCst);
            }
        }
        let _restore_ever = RestoreEverActive(EVER_ACTIVE.swap(false, Ordering::SeqCst));
        assert!(!ever_active());

        // Prepare a DIRECT-queue command list that writes `[0xFA; 64]` to `shadow`
        // and declare the pending GPU write on `shadow` just like production submissions.
        shadow.writing(va, 64, |_, _| None);
        let (up, up_ptr) = h.upload_buffer(64).expect("upload buffer");
        // SAFETY: `up_ptr` is mapped for 64 bytes.
        unsafe { ptr::write_bytes(up_ptr, 0xFA, 64) };
        let list = h.new_list().expect("list");
        // SAFETY: live upload and shadow buffers.
        unsafe {
            list.list.CopyBufferRegion(&shadow.buffer, 0, &up, 0, 64);
        }
        list.close();

        let entered_direct = Arc::new((Mutex::new(false), Condvar::new()));
        let release_direct = Arc::new((Mutex::new(false), Condvar::new()));
        let entered_t1 = entered_direct.clone();
        let release_t1 = release_direct.clone();

        // Thread 1 starts a DIRECT submission while `ever_active() == false` and
        // pauses strictly AFTER acquiring `direct_submit_guard()` and BEFORE
        // `ExecuteCommandLists` / `Signal`.
        let t1 = std::thread::spawn(move || {
            crate::host::set_after_direct_submit_guard_hook(Some(Box::new(move || {
                {
                    let mut e = entered_t1.0.lock().unwrap();
                    *e = true;
                    entered_t1.1.notify_all();
                }
                let mut r = release_t1.0.lock().unwrap();
                while !*r {
                    r = release_t1.1.wait(r).unwrap();
                }
            })));
            h.execute(&[&list]);
            list.reset();
        });

        {
            let mut e = entered_direct.0.lock().unwrap();
            while !*e {
                e = entered_direct.1.wait(e).unwrap();
            }
        }

        // While Thread 1 is paused mid-DIRECT-submission, arm the VERY FIRST lazy
        // page (transitioning `EVER_ACTIVE`: false -> true) and kick off a COPY
        // materialization on Thread 2.
        assert!(!ever_active());
        assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
        assert!(ever_active(), "first lazy arm must flip EVER_ACTIVE");
        assert!(has_block(block));

        let copy_started = Arc::new(AtomicBool::new(false));
        let copy_finished = Arc::new(AtomicBool::new(false));
        let started_t2 = copy_started.clone();
        let finished_t2 = copy_finished.clone();
        let t2 = std::thread::spawn(move || {
            started_t2.store(true, Ordering::SeqCst);
            let ok = materialize_range(va, PAGE);
            finished_t2.store(true, Ordering::SeqCst);
            ok
        });
        while !copy_started.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert!(
            !copy_finished.load(Ordering::SeqCst),
            "COPY fetch must block behind DIRECT submission even across first-activation transition"
        );

        // Release Thread 1 so its DIRECT write executes, signals, and drops its guard.
        {
            let mut r = release_direct.0.lock().unwrap();
            *r = true;
            release_direct.1.notify_all();
        }
        t1.join().expect("t1 joined");
        shadow.copy_back_done(&[(va, va + 64)]);
        assert!(t2.join().expect("t2 joined"));
        assert!(!has_block(block));

        // Thread 2's COPY read must have executed after Thread 1's DIRECT write.
        // SAFETY: page is materialized and readable.
        let got = unsafe { std::slice::from_raw_parts(ptr, 64) };
        assert_eq!(
            got, &[0xFA; 64],
            "COPY fetch serialized after first-activation DIRECT write must read `[0xFA; 64]`"
        );

        drop(_restore_ever);
        clear_hot_for_test(block);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_active_or_arming_reads_arming_before_lazy_across_completion_interleaving() {
        assert_eq!(LAZY_COUNT.load(Ordering::SeqCst), 0);
        assert_eq!(ARMING_COUNT.load(Ordering::SeqCst), 0);
        assert!(!active_or_arming());

        struct RestoreLazyCount;
        impl Drop for RestoreLazyCount {
            fn drop(&mut self) {
                LAZY_COUNT.store(0, Ordering::SeqCst);
                BETWEEN_ACTIVE_OR_ARMING_LOADS_HOOK.with(|h| *h.borrow_mut() = None);
            }
        }
        let _restore = RestoreLazyCount;

        // Simulate an armer mid-arm (`ARMING_COUNT == 1`, `LAZY_COUNT == 0`).
        // Between the two atomic loads inside `active_or_arming()`, complete the arm
        // (`LAZY_COUNT: 0 -> 1`, then `ARMING_COUNT: 1 -> 0`).
        // Reading `ARMING_COUNT` first observes `1` and returns `true`; reading
        // `LAZY_COUNT` first would observe `0`, run the hook, then observe `ARMING_COUNT == 0`
        // and falsely return `false`.
        let arming = ArmingGuard::enter();
        let mut arming_opt = Some(arming);
        BETWEEN_ACTIVE_OR_ARMING_LOADS_HOOK.with(|h| {
            *h.borrow_mut() = Some(Box::new(move || {
                LAZY_COUNT.store(1, Ordering::SeqCst);
                drop(arming_opt.take());
            }));
        });
        assert!(
            active_or_arming(),
            "active_or_arming must observe true across (ARMING=1,LAZY=0)->(ARMING=0,LAZY=1)"
        );
        // After completion (`ARMING_COUNT == 0`, `LAZY_COUNT == 1`), second check also returns true.
        assert_eq!(ARMING_COUNT.load(Ordering::SeqCst), 0);
        assert!(active_or_arming());
        drop(_restore);
        assert!(!active_or_arming());
    }

    #[test]
    fn lazy_mapping_65th_section_view_materializes_armed_block() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let cap = crate::watch::MANY_VIEWS;
        // SAFETY: anonymous 1-page section mapped across `MANY_VIEWS + 1` (65) views.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                ptr::null(),
            )
        };
        assert!(!section.is_null());
        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);

        let mut views = Vec::with_capacity(cap + 1);
        for _ in 0..cap {
            // SAFETY: mapping one of `MANY_VIEWS` views of offset 0.
            let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
            assert!(!view.Value.is_null());
            let va = view.Value as usize;
            clear_hot_for_test(va / PAGE);
            assert!(crate::inplace::title_memory_view(
                va,
                PAGE,
                section as usize,
                0
            ));
            views.push(view);
        }

        let va0 = views[0].Value as usize;
        let block0 = va0 / PAGE;
        // SAFETY: seed physical section page via view 0.
        unsafe { ptr::write_bytes(va0 as *mut u8, 0x11, PAGE) };

        if let Some((shadow, _side)) = make_card_shadow_at(va0 as u64, 1, 0x11) {
            gpu_write_shadow(&shadow, 0, &[0x65; 32]);
            assert!(
                try_defer_page_inner(&shadow, 0, PAGE, 0),
                "arming a page with exactly MANY_VIEWS ({cap}) views must succeed"
            );
            assert!(has_block(block0));
            for v in &views {
                assert!(
                    !is_readable_now(v.Value as usize),
                    "every armed view must be PAGE_NOACCESS before the 65th view maps"
                );
            }

            // Map the 65th (`MANY_VIEWS + 1`) view: capped `views_of_section`
            // omits it, so `watch::views_mapped` must use the uncapped lookup
            // (`views_of_section_all`) to materialize the armed block.
            // SAFETY: mapping the 65th view of the same section page.
            let view65 = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
            assert!(!view65.Value.is_null());
            let va65 = view65.Value as usize;
            clear_hot_for_test(va65 / PAGE);
            views.push(view65);
            assert!(crate::inplace::title_memory_view(
                va65,
                PAGE,
                section as usize,
                0
            ));

            assert!(
                crate::inplace::views_of_section(section as usize, 0, PAGE).is_empty(),
                "capped views_of_section must still omit pages with > MANY_VIEWS views"
            );
            assert!(
                !crate::inplace::views_of_section_all(section as usize, 0, PAGE).is_empty(),
                "uncapped views_of_section_all must include pages with > MANY_VIEWS views"
            );
            assert!(
                !has_block(block0),
                "mapping the 65th view must materialize the deferred block"
            );
            for v in &views {
                assert!(
                    is_readable_now(v.Value as usize),
                    "all 65 views must be readable after materialization"
                );
            }
            // SAFETY: 65th view is readable and holds merged GPU bytes.
            let got65 = unsafe { std::slice::from_raw_parts(va65 as *const u8, 32) };
            assert_eq!(got65, &[0x65; 32]);
            drop(shadow);
        }

        for v in views {
            let va = v.Value as usize;
            clear_hot_for_test(va / PAGE);
            crate::watch::title_memory_remapped(va, PAGE);
            // SAFETY: unmapping test view.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: v.Value });
            }
        }
        // SAFETY: closing anonymous section handle.
        unsafe { CloseHandle(section) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_write_mapped_reports_materialization_failure_without_writing_or_counting() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` committed bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        let block = va / PAGE;
        clear_hot_for_test(block);

        if let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) {
            gpu_write_shadow(&shadow, 0, &[0xAA; 32]);
            assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
            assert!(has_block(block));

            let src = vec![0x55u8; PAGE];
            // Inject DMA materialization failure: `write_mapped` must return `None`
            // without writing or unarming the destination.
            set_fail_next_fetches_for_test(1);
            assert_eq!(
                crate::host::write_mapped(ptr, src.as_ptr(), PAGE),
                None,
                "write_mapped must propagate failed lazy materialization"
            );
            assert!(
                has_block(block),
                "failed write_mapped must leave destination armed at PAGE_NOACCESS"
            );
            assert!(!is_readable_now(va));

            // Once DMA recovers, `write_mapped` materializes, writes, and reports `Some(PAGE)`.
            assert_eq!(
                crate::host::write_mapped(ptr, src.as_ptr(), PAGE),
                Some(PAGE)
            );
            assert!(!has_block(block));
            // SAFETY: destination is readable and now contains `src`.
            let got = unsafe { std::slice::from_raw_parts(ptr, 32) };
            assert_eq!(got, &[0x55; 32]);
            drop(shadow);
        }

        clear_hot_for_test(block);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_note_signalled_uses_submitted_queue_order_without_locking_queue_order() {
        // Exercise the exact production `QueueOrderTimeline` lock-order,
        // candidate, submission, and fence-watermark bookkeeping against a 100%
        // isolated test-owned timeline (zero race and zero lock-order inversion
        // against `Host::card_before` under parallel tests).
        let timeline = crate::host::QueueOrderTimeline::new();
        let (prev_order, candidate) = timeline.next_candidate();
        assert_eq!((prev_order, candidate), (0, 1));

        // While holding the non-reentrant `timeline.queue_order` mutex (before
        // `candidate` is submitted), calling `note_signalled(10)` (the exact
        // path `Host::note_signalled` calls under `direct_submit_guard()`) MUST
        // NOT lock `queue_order` and must record submitted order `0 < candidate`.
        let mut order_guard = timeline.lock_queue_order();
        timeline.note_signalled(10);
        assert_eq!(timeline.last_signalled(), (10, 0));

        // COPY-queue sync plan at fence 10 must yield watermark `0 < candidate`,
        // excluding the unsubmitted candidate plan.
        let (wait_before, wm_before) = timeline.sync_copy_plan(10, true, 0);
        assert_eq!(wait_before, None);
        assert!(
            wm_before < candidate,
            "watermark before submission ({wm_before}) must exclude unsubmitted candidate ({candidate})"
        );

        // Once the candidate's upload command list is actually submitted on the
        // DIRECT queue and signalled with a later fence (`fence = 11`), the
        // committed submitted order advances to `candidate` and the later fence
        // covers it.
        let committed = timeline.commit_submitted_locked(&mut order_guard, candidate, true);
        assert_eq!(committed, candidate);
        timeline.note_signalled_with_order(11, committed);
        drop(order_guard);

        assert_eq!(timeline.last_signalled(), (11, candidate));
        let (wait_inflight, wm_after_wait) = timeline.sync_copy_plan(10, true, 0);
        assert_eq!(wait_inflight, Some(11));
        assert!(
            wm_after_wait >= candidate,
            "later fence watermark ({wm_after_wait}) must cover submitted candidate ({candidate})"
        );
        let (wait_done, wm_done) = timeline.sync_copy_plan(11, true, 0);
        assert_eq!(wait_done, None);
        assert_eq!(wm_done, candidate);
    }

    #[test]
    fn lazy_card_before_signals_fence_while_not_ever_active_for_first_copy_fetch() {
        let Some(h) = crate::host::host() else {
            return;
        };
        if !crate::shadow::cards() || !h.card_ready() {
            return;
        }
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        let block = va / PAGE;
        clear_hot_for_test(block);

        let shadow =
            crate::shadow::Shadow::new_card_registered_for_test(va as u64, PAGE as u64, 0x11)
                .expect("registered card shadow");
        // Write new CPU-uploaded bytes into `card.staging` and mark page 0 dirty so
        // `Host::card_before` records & submits a DIRECT-queue CopyBufferRegion upload.
        shadow.write_test_bytes(0, &[0x5C; 64]);
        shadow.seed_card_dirty_for_test(vec![(0, PAGE)]);

        struct RestoreEverActive(bool);
        impl Drop for RestoreEverActive {
            fn drop(&mut self) {
                EVER_ACTIVE.store(self.0, Ordering::SeqCst);
            }
        }
        let _restore_ever = RestoreEverActive(EVER_ACTIVE.swap(false, Ordering::SeqCst));
        assert!(
            !ever_active() && can_arm(),
            "test must run card_before while !ever_active() && can_arm()"
        );

        let (sig_before, order_before) = h.last_signalled_for_test();
        let (plan, submitted_after) = h
            .card_before_for_test()
            .expect("card_before must submit dirty card upload");
        let (sig_after, order_after) = h.last_signalled_for_test();
        assert!(
            sig_after > sig_before && sig_after >= submitted_after,
            "card_before must signal self.fence when can_arm() && !ever_active() (before={sig_before}, after={sig_after}, submitted={submitted_after})"
        );
        if crate::host::relay() {
            assert!(
                order_after > order_before,
                "card_before must record submitted queue order watermark (before={order_before}, after={order_after})"
            );
        }

        // Now perform a COPY-queue fetch (`sync_copy_queue_for(&self.queue, true)`)
        // BEFORE the CPU waits on `submitted_after`: it must wait on `sig_after`
        // recorded by `card_before`, read the freshly uploaded `0x5C` bytes from
        // `shadow.buffer`, and observe `watermark >= order_after`.
        {
            let mut slot: Option<LazyFetcher> = None;
            let mut unconfirmed: Vec<UnconfirmedCopy> = Vec::new();
            let (fetched_bytes, _, watermark) =
                run_fetcher_batch(&mut slot, &mut unconfirmed, &[(&shadow, 0, PAGE)], true)
                    .expect("COPY-queue fetch after card_before");
            assert_eq!(
                &fetched_bytes[..64],
                &[0x5C; 64],
                "first COPY fetch must wait on card_before's fence and read uploaded 0x5C bytes"
            );
            if crate::host::relay() {
                assert!(
                    watermark >= order_after,
                    "COPY fetch watermark ({watermark}) must cover card_before order ({order_after})"
                );
            }
        }

        // Also arm the very first lazy block (transitioning `ever_active()` to true)
        // and materialize it cleanly without D3D12 errors.
        assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
        assert!(ever_active());
        assert!(materialize_range(va, PAGE));

        h.card_done_and_wait_for_test(plan, submitted_after);
        clear_hot_for_test(block);
        drop(shadow);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_try_defer_rejects_remap_between_own_view_check_and_pages_of() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        let block = va / PAGE;
        clear_hot_for_test(block);

        if let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) {
            // Simulate a remap completing right after `has_own_view_for` and before
            // `pages_of`: because `inplace::generation()` was captured BEFORE
            // `has_own_view_for`, revalidation under `LAZY_STATE` must reject arming.
            AFTER_OWN_VIEW_CHECK_HOOK.with(|hook| {
                *hook.borrow_mut() = Some(Box::new(move || {
                    crate::inplace::views_gone(va, PAGE);
                }));
            });
            assert!(
                !try_defer_page_inner(&shadow, 0, PAGE, 0),
                "remap between has_own_view_for and pages_of must reject lazy deferral"
            );
            assert!(!has_block(block));
            assert!(is_readable_now(va));
            drop(shadow);
        }

        clear_hot_for_test(block);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_begin_remap_is_lock_free_noop_when_disabled() {
        // Hold `LAZY_STATE` on this thread: `begin_remap_if(..., false)` must
        // return `0` without attempting to acquire `LAZY_STATE` (which would
        // self-deadlock on the non-reentrant `Mutex`).
        let _guard = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(begin_remap_if(0x1_0000, PAGE, false), 0);
        end_remap(0);
    }

    #[test]
    fn lazy_discard_range_promotes_surviving_alias_on_failed_drain_and_preserves_restore_failed() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        // SAFETY: 1-page anonymous section mapped at two aliases A and B.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: mapping view A.
        let view_a = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        // SAFETY: mapping view B.
        let view_b = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        assert!(!view_a.Value.is_null() && !view_b.Value.is_null());
        let va_a = view_a.Value as usize;
        let va_b = view_b.Value as usize;
        let blk_a = va_a / PAGE;
        let blk_b = va_b / PAGE;
        clear_hot_for_test(blk_a);
        clear_hot_for_test(blk_b);
        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        assert!(crate::inplace::title_memory_view(
            va_a,
            PAGE,
            section as usize,
            0
        ));
        assert!(crate::inplace::title_memory_view(
            va_b,
            PAGE,
            section as usize,
            0
        ));
        // SAFETY: seed physical page.
        unsafe { ptr::write_bytes(va_a as *mut u8, 0x11, PAGE) };

        if let Some((shadow, _side)) = make_card_shadow_at(va_a as u64, 1, 0x11) {
            // Case 1: Primary A is unmapped while COPY drain through surviving B fails.
            gpu_write_shadow(&shadow, 0, &[0xAA; 32]);
            assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
            assert!(has_block(blk_a) && has_block(blk_b));

            crate::inplace::views_gone(va_a, PAGE);
            set_fail_next_fetches_for_test(1);
            discard_range(va_a, PAGE);

            assert!(
                !has_block(blk_a),
                "unmapped primary A must be detached even when drain through B fails"
            );
            assert!(
                has_block(blk_b),
                "surviving alias B must stay tracked as the promoted primary after failed drain"
            );
            assert!(!is_readable_now(va_b));

            // Subsequent fault on B must merge through B's own view (not gone A) and succeed.
            assert!(
                handle_fault(va_b, false),
                "retry fault on promoted surviving alias B must succeed"
            );
            assert!(!has_block(blk_b));
            assert!(is_readable_now(va_b));
            // SAFETY: B is readable and holds merged GPU bytes.
            let got_b = unsafe { std::slice::from_raw_parts(va_b as *const u8, 32) };
            assert_eq!(got_b, &[0xAA; 32]);

            // Case 2: Re-register A, arm `[A, B]`, trigger `restore_failed` on B, then
            //         unmap A via `discard_range(A)`: `restore_failed` must be preserved for B.
            clear_hot_for_test(blk_a);
            clear_hot_for_test(blk_b);
            let mut old = 0u32;
            // SAFETY: restoring RW on A before re-arming.
            unsafe { VirtualProtect(va_a as *const _, PAGE, PAGE_READWRITE, &mut old) };
            assert!(crate::inplace::title_memory_view(
                va_a,
                PAGE,
                section as usize,
                0
            ));
            gpu_write_shadow(&shadow, 0, &[0xBB; 32]);
            assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));

            let _reset = fail_restore_protect_guard(va_b);
            assert!(!materialize_range(va_a, PAGE));
            drop(_reset);

            crate::inplace::views_gone(va_a, PAGE);
            discard_range(va_a, PAGE);
            assert!(!has_block(blk_a), "A must be detached");
            assert!(
                has_block(blk_b),
                "surviving alias B must retain entry when restore_failed was already set"
            );
            assert!(
                !handle_fault(va_b, false),
                "promoted surviving alias B must preserve restore_failed on faults"
            );
            assert!(
                !materialize_range(va_b, PAGE),
                "promoted surviving alias B must preserve restore_failed on materialize_range"
            );

            // Clean up B.
            discard_range(va_b, PAGE);
            assert!(!has_block(blk_b));
            // SAFETY: restoring RW on test views before unmap.
            unsafe {
                VirtualProtect(va_a as *const _, PAGE, PAGE_READWRITE, &mut old);
                VirtualProtect(va_b as *const _, PAGE, PAGE_READWRITE, &mut old);
            }
            drop(shadow);
        }

        clear_hot_for_test(blk_a);
        clear_hot_for_test(blk_b);
        crate::watch::title_memory_remapped(va_a, PAGE);
        crate::watch::title_memory_remapped(va_b, PAGE);
        // SAFETY: unmapping test views and closing section.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view_a.Value,
            });
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view_b.Value,
            });
            CloseHandle(section);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_kernel_writing_survives_pin_expiry_during_drain_and_propagates_failure() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        let block = va / PAGE;
        clear_hot_for_test(block);

        if let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) {
            gpu_write_shadow(&shadow, 0, &[0xCC; 16]);
            assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));

            // 1. Simulate a slow blocking drain exceeding `PIN` (2s): expire all timed
            //    pins mid-drain and verify `try_arm_lazy_under_guards` STILL rejects
            //    because `kernel_writing` holds a non-expiring announcement across the drain.
            let rearm_blocked = Arc::new(AtomicBool::new(false));
            let rearm_blocked_cb = rearm_blocked.clone();
            set_before_fetch_hook_for_test(Some(Box::new(move || {
                crate::watch::expire_pins_in_range_for_test(va, PAGE);
                let armed = crate::watch::try_arm_lazy_under_guards(&[block], |_| true);
                rearm_blocked_cb.store(!armed, Ordering::SeqCst);
            })));
            let kw_guard = crate::watch::kernel_writing(va, PAGE)
                .expect("kernel_writing must return Some(guard) when drain succeeds");
            assert!(
                rearm_blocked.load(Ordering::SeqCst),
                "non-expiring announcement must block lazy arming even after timed pins expire during drain"
            );
            assert!(!has_block(block));

            // 2. Simulate a post-drain delay exceeding `PIN` (2s) BEFORE dispatch
            //    completes: expire only this allocation's timed pins while `kw_guard`
            //    is still held and verify the destination STILL rejects lazy arming
            //    until `kw_guard` drops.
            crate::watch::expire_pins_in_range_for_test(va, PAGE);
            assert!(
                !crate::watch::try_arm_lazy_under_guards(&[block], |_| true),
                "non-expiring KernelWriteGuard announcement must block lazy arming after post-drain pin expiry"
            );
            assert!(
                !try_defer_page_inner(&shadow, 0, PAGE, 0),
                "destination must stay unarmable across post-drain delay until KernelWriteGuard drops"
            );

            // Once dispatch completes and `kw_guard` drops (and pins expire), deferral works again.
            drop(kw_guard);
            crate::watch::expire_pins_in_range_for_test(va, PAGE);
            clear_hot_for_test(block);
            assert!(
                try_defer_page_inner(&shadow, 0, PAGE, 0),
                "dropping KernelWriteGuard must release the non-expiring announcement"
            );

            // 3. Verify failure propagation: when materialization fails, `kernel_writing`
            //    returns `None` and immediately releases its announcement (no leak).
            set_fail_next_fetches_for_test(1);
            assert!(
                crate::watch::kernel_writing(va, PAGE).is_none(),
                "kernel_writing must return None when lazy materialization fails"
            );
            assert!(has_block(block));
            assert!(materialize_range(va, PAGE));
            crate::watch::expire_pins_in_range_for_test(va, PAGE);
            clear_hot_for_test(block);
            assert!(
                try_defer_page_inner(&shadow, 0, PAGE, 0),
                "failed kernel_writing must not leak its non-expiring announcement"
            );
            assert!(materialize_range(va, PAGE));
            drop(shadow);
        }

        clear_hot_for_test(block);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_kernel_will_write_and_write_title_memory_propagate_failure_and_unannounce() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        let block = va / PAGE;
        clear_hot_for_test(block);

        if let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) {
            gpu_write_shadow(&shadow, 0, &[0xDD; 16]);
            assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));

            // 1. Direct `kernel_will_write` returns `false` on materialization failure.
            set_fail_next_fetches_for_test(1);
            assert!(!crate::watch::kernel_will_write(va, 32));
            crate::watch::kernel_wrote(va, 32);
            assert!(has_block(block));

            // 2. `host::write_title_memory` checks `kernel_will_write`, aborts before
            //    writing, calls `kernel_wrote` to clear the announcement, and returns `false`.
            set_fail_next_fetches_for_test(1);
            assert!(!crate::host::write_title_memory(va, &[0x99; 32]));
            assert!(has_block(block));

            // Materialize normally, then verify no announcement leaked by re-arming.
            assert!(materialize_range(va, PAGE));
            // SAFETY: destination holds GPU bytes `0xDD`, NOT aborted `0x99`.
            let got = unsafe { std::slice::from_raw_parts(ptr, 16) };
            assert_eq!(got, &[0xDD; 16]);
            clear_hot_for_test(block);
            assert!(
                try_defer_page_inner(&shadow, 0, PAGE, 0),
                "early-return kernel_wrote must clear announcement so future deferral works"
            );
            assert!(materialize_range(va, PAGE));
            drop(shadow);
        }

        clear_hot_for_test(block);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_views_mapped_protects_new_alias_when_materialization_fails() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        // SAFETY: 1-page anonymous section.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: map view A first.
        let view_a = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        assert!(!view_a.Value.is_null());
        let va_a = view_a.Value as usize;
        let blk_a = va_a / PAGE;
        clear_hot_for_test(blk_a);
        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        assert!(crate::inplace::title_memory_view(
            va_a,
            PAGE,
            section as usize,
            0
        ));
        // SAFETY: seed physical page with stale value `0x11`.
        unsafe { ptr::write_bytes(va_a as *mut u8, 0x11, PAGE) };

        if let Some((shadow, _side)) = make_card_shadow_at(va_a as u64, 1, 0x11) {
            gpu_write_shadow(&shadow, 0, &[0xEE; 32]);
            assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
            assert!(has_block(blk_a));

            // Map new alias B over the same section bytes while materialization fails:
            // `title_memory_view` must return `false` AND arm B with `PAGE_NOACCESS`
            // so B cannot silently read stale `0x11` RAM without faulting.
            // SAFETY: mapping view B of the same section page.
            let view_b = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
            assert!(!view_b.Value.is_null());
            let va_b = view_b.Value as usize;
            let blk_b = va_b / PAGE;
            clear_hot_for_test(blk_b);

            set_fail_next_fetches_for_test(1);
            assert!(
                !crate::inplace::title_memory_view(va_b, PAGE, section as usize, 0),
                "title_memory_view must report failed materialization"
            );
            assert!(
                has_block(blk_b),
                "newly mapped alias B must be attached to the armed entry on drain failure"
            );
            assert!(
                !is_readable_now(va_b),
                "newly mapped alias B must be PAGE_NOACCESS when drain fails, never readable stale RAM"
            );

            // Once DMA recovers, faulting on B materializes GPU bytes `0xEE` and reopens both A and B.
            // SAFETY: volatile read triggers VEH fault on B.
            let got = unsafe { (va_b as *const u8).read_volatile() };
            assert_eq!(got, 0xEE);
            assert!(!has_block(blk_a) && !has_block(blk_b));
            assert!(is_readable_now(va_a) && is_readable_now(va_b));

            clear_hot_for_test(blk_b);
            crate::watch::title_memory_remapped(va_b, PAGE);
            // SAFETY: unmapping view B.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view_b.Value,
                });
            }
            drop(shadow);
        }

        clear_hot_for_test(blk_a);
        crate::watch::title_memory_remapped(va_a, PAGE);
        // SAFETY: unmapping view A and closing section.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view_a.Value,
            });
            CloseHandle(section);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_views_mapped_fails_closed_when_drain_and_alias_protect_both_fail() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        // SAFETY: 1-page anonymous section.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: map view A first.
        let view_a = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        assert!(!view_a.Value.is_null());
        let va_a = view_a.Value as usize;
        let blk_a = va_a / PAGE;
        clear_hot_for_test(blk_a);
        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        assert!(crate::inplace::title_memory_view(
            va_a,
            PAGE,
            section as usize,
            0
        ));
        // SAFETY: seed physical page with stale value `0x11`.
        unsafe { ptr::write_bytes(va_a as *mut u8, 0x11, PAGE) };

        let (shadow, _side) =
            make_card_shadow_at(va_a as u64, 1, 0x11).expect("card shadow for va_a");
        gpu_write_shadow(&shadow, 0, &[0xEF; 32]);
        assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
        assert!(has_block(blk_a));

        // SAFETY: map new alias B of the same physical section page.
        let view_b = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        assert!(!view_b.Value.is_null());
        let va_b = view_b.Value as usize;
        let blk_b = va_b / PAGE;
        clear_hot_for_test(blk_b);

        let _reset_new_alias = fail_protect_new_alias_guard(va_b);
        set_fail_next_fetches_for_test(1);

        // When BOTH lazy materialization AND `VirtualProtect(..., PAGE_NOACCESS)`
        // on the newly mapped alias fail, `title_memory_view` must roll back `va_b`
        // from `inplace` views and fail closed via `host::fail` rather than
        // returning an unprotected readable alias pointing at stale RAM.
        let _fail_panics = crate::host::fail_panics_for_test();
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = crate::inplace::title_memory_view(va_b, PAGE, section as usize, 0);
        }));
        assert!(
            res.is_err(),
            "title_memory_view must fail closed via host::fail when both drain and alias VirtualProtect fail"
        );
        drop(_fail_panics);
        drop(_reset_new_alias);

        assert!(
            !crate::inplace::views_of_section_all(section as usize, 0, PAGE)
                .iter()
                .any(|&(a, _)| a == va_b),
            "failed title_memory_view must roll back view registration for the unprotected alias"
        );
        assert!(
            !handle_fault(va_a, false),
            "existing armed entry must remain marked restore_failed after failed alias protection"
        );

        // Clean up armed entry A and unmap both views.
        discard_range(va_a, PAGE);
        assert!(!has_block(blk_a));
        let mut old = 0u32;
        // SAFETY: restore RW on test view A before unmap.
        unsafe { VirtualProtect(va_a as *const _, PAGE, PAGE_READWRITE, &mut old) };
        drop(shadow);

        clear_hot_for_test(blk_a);
        clear_hot_for_test(blk_b);
        crate::watch::title_memory_remapped(va_a, PAGE);
        crate::watch::title_memory_remapped(va_b, PAGE);
        // SAFETY: unmapping test views and closing section.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view_b.Value,
            });
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view_a.Value,
            });
            CloseHandle(section);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_write_mapped_retries_concurrent_arm_and_fails_closed_when_exhausted() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(2);
        // SAFETY: `ptr` owns `2 * PAGE` committed bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, 2 * PAGE) };
        let va0 = ptr as usize;
        let va1 = va0 + PAGE;
        let blk1 = va1 / PAGE;
        clear_hot_for_test(blk1);

        if let Some((shadow, _side)) = make_card_shadow_at(va0 as u64, 2, 0x11) {
            let src = vec![0x66u8; 2 * PAGE];

            // 1. Concurrent lazy arm lands right before `write_mapped_raw` TWICE
            //    (more than the old single retry), then stops: `write_mapped` must
            //    retry materialization and complete with `Some(2 * PAGE)`.
            let arms_left = Arc::new(AtomicUsize::new(2));
            let arms_cb = arms_left.clone();
            let shadow_cb = shadow.clone();
            crate::host::set_before_write_mapped_raw_hook(Some(Box::new(move || {
                if arms_cb.load(Ordering::SeqCst) > 0 {
                    arms_cb.fetch_sub(1, Ordering::SeqCst);
                    clear_hot_for_test(blk1);
                    assert!(try_defer_page_inner(&shadow_cb, PAGE, PAGE, 0));
                }
            })));
            assert_eq!(
                crate::host::write_mapped(ptr, src.as_ptr(), 2 * PAGE),
                Some(2 * PAGE),
                "write_mapped must retry multiple concurrent lazy arms until all bytes land"
            );
            crate::host::set_before_write_mapped_raw_hook(None);
            assert_eq!(arms_left.load(Ordering::SeqCst), 0);
            assert!(!has_block(blk1));
            // SAFETY: destination is readable and holds `src`.
            let got1 = unsafe { std::slice::from_raw_parts(ptr.add(PAGE), 32) };
            assert_eq!(got1, &[0x66; 32]);

            // 2. Concurrent lazy arm lands on EVERY attempt, exhausting bounded
            //    retries: `write_mapped` must fail closed (`None`) instead of
            //    reporting the short count `Some(PAGE)` as complete.
            let shadow_cb2 = shadow.clone();
            crate::host::set_before_write_mapped_raw_hook(Some(Box::new(move || {
                clear_hot_for_test(blk1);
                let _ = try_defer_page_inner(&shadow_cb2, PAGE, PAGE, 0);
            })));
            assert_eq!(
                crate::host::write_mapped(ptr, src.as_ptr(), 2 * PAGE),
                None,
                "write_mapped must fail closed when lazy-backed bytes remain unwritten after bounded retries"
            );
            crate::host::set_before_write_mapped_raw_hook(None);
            assert!(has_block(blk1));
            assert!(materialize_range(va1, PAGE));
            drop(shadow);
        }

        clear_hot_for_test(blk1);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_direct_submit_guard_fails_closed_for_unconfirmed_copy_and_retires_completed_fence() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let fetcher = LazyFetcher::new().expect("create isolated LazyFetcher");
        let fence = fetcher.fence.clone();

        // Isolated test-owned `FetcherState` so parallel tests using global
        // `FETCHER` are never affected.
        let state = Mutex::new(FetcherState {
            active: None,
            unconfirmed: vec![UnconfirmedCopy {
                fetcher,
                fence_value: 1,
                signalled: true,
            }],
        });

        // 1. While fence completed value (0) < target (1), `direct_submit_guard_with_state`
        //    must fail closed via `host::fail` and retain the unconfirmed entry.
        {
            let _fail_panics = crate::host::fail_panics_for_test();
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _g = direct_submit_guard_with_state(&state, true);
            }));
            assert!(
                res.is_err(),
                "unproven in-flight COPY must cause direct_submit_guard to fail closed"
            );
        }
        assert_eq!(state.lock().unwrap().unconfirmed.len(), 1);

        // 2. Once the fence completes (`fence.Signal(1)`), `direct_submit_guard_with_state`
        //    retires the completed COPY cleanly and lets the DIRECT submission proceed.
        // SAFETY: CPU-side signal on our isolated test fence.
        unsafe { fence.Signal(1) }.expect("signal test fence");
        {
            let guard = direct_submit_guard_with_state(&state, true);
            assert!(guard.0.as_ref().unwrap().unconfirmed.is_empty());
        }

        // 3. If `queue.Signal` itself failed (`signalled == false`), completion
        //    cannot be proven even if the fence has a non-zero value; guard must
        //    stay failed-closed and keep the entry retained.
        let unsignalled_fetcher = LazyFetcher::new().expect("create second isolated LazyFetcher");
        let unsig_fence = unsignalled_fetcher.fence.clone();
        // SAFETY: advancing test fence value to 5 still cannot prove an unsignalled submission.
        unsafe { unsig_fence.Signal(5) }.expect("signal second test fence");
        state.lock().unwrap().unconfirmed.push(UnconfirmedCopy {
            fetcher: unsignalled_fetcher,
            fence_value: 1,
            signalled: false,
        });
        {
            let _fail_panics = crate::host::fail_panics_for_test();
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _g = direct_submit_guard_with_state(&state, true);
            }));
            assert!(
                res.is_err(),
                "unsignalled unconfirmed COPY must remain unproven and fail closed"
            );
        }
        assert_eq!(state.lock().unwrap().unconfirmed.len(), 1);
        state.lock().unwrap().unconfirmed.clear();

        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_restore_failed_excluded_from_clean_owner_proof_and_shadow_refresh() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        let block = va / PAGE;
        clear_hot_for_test(block);

        if let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) {
            // Establish a clean guarded view epoch on `shadow` first.
            let (_, ok_init) = shadow.update_checked(va, PAGE);
            assert!(ok_init);
            assert!(crate::watch::writes(va, va + PAGE, true).is_some());

            // Arm the clean view block so `cpu_clean_after` is `Some(epoch)`.
            gpu_write_shadow(&shadow, 0, &[0x9A; 16]);
            assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
            let shadow_ptr = Arc::as_ptr(&shadow) as usize;
            let before_fail = owned_blocks_with_clean_in(shadow_ptr, block, block + 1);
            assert_eq!(before_fail.len(), 1);
            assert!(
                before_fail[0].1.is_some(),
                "healthy clean-armed view must report a clean epoch"
            );

            // Force `VirtualProtect` restoration failure so `entry.restore_failed = true`.
            let _fail_restore = fail_restore_protect_guard(va);
            assert!(!materialize_range(va, PAGE));
            drop(_fail_restore);

            // `owned_blocks_with_clean_in` MUST NOT report a clean epoch for `restore_failed`.
            let after_fail = owned_blocks_with_clean_in(shadow_ptr, block, block + 1);
            assert_eq!(
                after_fail,
                vec![(block, None)],
                "restore_failed entry must be excluded from clean-owner proof"
            );
            assert!(owned_cpu_clean_blocks_in(shadow_ptr, block, block + 1).is_empty());

            // `shadow.update_checked` must take the dirty path and fail (not skip as clean).
            let (_, ok_after) = shadow.update_checked(va, PAGE);
            assert!(
                !ok_after,
                "shadow refresh must not keep a restore_failed block deferred as clean"
            );

            // Cleanup.
            discard_range(va, PAGE);
            let mut old = 0u32;
            // SAFETY: restore RW on test view before unmapping.
            unsafe { VirtualProtect(va as *const _, PAGE, PAGE_READWRITE, &mut old) };
            drop(shadow);
        }

        clear_hot_for_test(block);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_sub_64k_attached_alias_restores_exact_extent_on_materialize_and_unwind() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        const SUB_LEN: usize = 4 << 10;
        // SAFETY: 64 KiB anonymous section.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: full 64 KiB primary view A.
        let view_a = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        // SAFETY: sub-64 KiB (4 KiB) alias view B; bytes `4096..65536` at `va_b` are NOT committed.
        let view_b = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SUB_LEN) };
        assert!(!view_a.Value.is_null() && !view_b.Value.is_null());
        let va_a = view_a.Value as usize;
        let va_b = view_b.Value as usize;
        let blk_a = va_a / PAGE;
        let blk_b = va_b / PAGE;
        clear_hot_for_test(blk_a);
        clear_hot_for_test(blk_b);
        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        assert!(crate::inplace::title_memory_view(
            va_a,
            PAGE,
            section as usize,
            0
        ));
        // SAFETY: seed physical page with `0x11`.
        unsafe { ptr::write_bytes(va_a as *mut u8, 0x11, PAGE) };

        if let Some((shadow, _side)) = make_card_shadow_at(va_a as u64, 1, 0x11) {
            // 1. Normal materialize path with sub-64 KiB attached alias B.
            gpu_write_shadow(&shadow, 0, &[0xCD; 32]);
            assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
            set_fail_next_fetches_for_test(1);
            assert!(
                !crate::inplace::title_memory_view(va_b, SUB_LEN, section as usize, 0),
                "mapping alias B during failed drain must attach and protect B"
            );
            assert!(has_block(blk_a) && has_block(blk_b));
            assert!(!is_readable_now(va_b));

            // Materializing must restore exactly `SUB_LEN` (4 KiB) on B instead of
            // calling `VirtualProtect(va_b, 64 KiB, ...)` past uncommitted memory.
            assert!(
                materialize_range(va_a, PAGE),
                "sub-64 KiB alias restore must succeed without restore_failed"
            );
            assert!(!has_block(blk_a) && !has_block(blk_b));
            assert!(is_readable_now(va_a) && is_readable_now(va_b));
            // SAFETY: B's 4 KiB view is now PAGE_READWRITE and holds merged GPU bytes.
            let got_b = unsafe { std::slice::from_raw_parts(va_b as *const u8, 32) };
            assert_eq!(got_b, &[0xCD; 32]);

            // 2. Unwind BEFORE GPU merge (`BEFORE_FETCH_HOOK`): must keep the
            //    entry armed at `PAGE_NOACCESS` (never exposing stale `0xCD` RAM),
            //    clear `materializing`, and allow a retry to merge real GPU `0xEF`.
            clear_hot_for_test(blk_a);
            clear_hot_for_test(blk_b);
            crate::inplace::views_gone(va_b, SUB_LEN);
            gpu_write_shadow(&shadow, 0, &[0xEF; 32]);
            assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
            set_fail_next_fetches_for_test(1);
            assert!(!crate::inplace::title_memory_view(
                va_b,
                SUB_LEN,
                section as usize,
                0
            ));
            assert!(!is_readable_now(va_b));
            set_before_fetch_hook_for_test(Some(Box::new(|| {
                panic!("simulated pre-merge unwind with sub-64 KiB alias");
            })));
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = materialize_range(va_a, PAGE);
            }));
            assert!(res.is_err());
            assert!(
                has_block(blk_a) && has_block(blk_b),
                "pre-merge unwind must keep entry armed so stale RAM is never exposed"
            );
            assert!(
                !is_readable_now(va_a) && !is_readable_now(va_b),
                "pre-merge unwind must keep both views PAGE_NOACCESS until GPU bytes land"
            );
            assert!(
                materialize_range(va_a, PAGE),
                "retry after pre-merge unwind must succeed cleanly"
            );
            assert!(!has_block(blk_a) && !has_block(blk_b));
            assert!(is_readable_now(va_a) && is_readable_now(va_b));
            // SAFETY: both views are PAGE_READWRITE and hold merged GPU `0xEF`.
            unsafe {
                assert_eq!(
                    std::slice::from_raw_parts(va_a as *const u8, 32),
                    &[0xEF; 32]
                );
                assert_eq!(
                    std::slice::from_raw_parts(va_b as *const u8, 32),
                    &[0xEF; 32]
                );
            }

            // 3. Unwind AFTER GPU merge (`AFTER_MERGE_HOOK`): GPU bytes (`0x5A`)
            //    have landed in RAM, so `MaterializeBatchScope::drop` restores exact
            //    `SUB_LEN` on B and `PAGE` on A and removes the entry cleanly.
            clear_hot_for_test(blk_a);
            clear_hot_for_test(blk_b);
            crate::inplace::views_gone(va_b, SUB_LEN);
            gpu_write_shadow(&shadow, 0, &[0x5A; 32]);
            assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
            set_fail_next_fetches_for_test(1);
            assert!(!crate::inplace::title_memory_view(
                va_b,
                SUB_LEN,
                section as usize,
                0
            ));
            assert!(!is_readable_now(va_b));
            set_after_merge_hook_for_test(Some(Box::new(|| {
                panic!("simulated post-merge unwind with sub-64 KiB alias");
            })));
            let res_post = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = materialize_range(va_a, PAGE);
            }));
            assert!(res_post.is_err());
            assert!(
                !has_block(blk_a) && !has_block(blk_b),
                "post-merge unwind restore of sub-64 KiB alias must succeed and remove entry cleanly"
            );
            assert!(is_readable_now(va_a) && is_readable_now(va_b));
            // SAFETY: both views are PAGE_READWRITE and hold merged GPU `0x5A`.
            unsafe {
                assert_eq!(
                    std::slice::from_raw_parts(va_a as *const u8, 32),
                    &[0x5A; 32]
                );
                assert_eq!(
                    std::slice::from_raw_parts(va_b as *const u8, 32),
                    &[0x5A; 32]
                );
            }
            drop(shadow);
        }

        clear_hot_for_test(blk_a);
        clear_hot_for_test(blk_b);
        crate::watch::title_memory_remapped(va_a, PAGE);
        crate::watch::title_memory_remapped(va_b, SUB_LEN);
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view_b.Value,
            });
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view_a.Value,
            });
            CloseHandle(section);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_try_arm_under_guards_transitions_readonly_directly_to_noaccess_without_rw_window() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let ptr = alloc_watch_pages(1);
        // SAFETY: `ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(ptr, 0x11, PAGE) };
        let va = ptr as usize;
        let block = va / PAGE;
        clear_hot_for_test(block);

        // Guard the view so it is `PAGE_READONLY` with `GuardProtected + GuardClean`.
        assert!(crate::watch::writes(va, va + PAGE, true).is_some());
        let query_protect = |addr: usize| -> u32 {
            // SAFETY: zero-initialized `MEMORY_BASIC_INFORMATION`.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: querying page protection with valid out pointer.
            let n = unsafe { VirtualQuery(addr as *const _, &mut info, size_of_val(&info)) };
            assert_ne!(n, 0);
            info.Protect
        };
        assert_eq!(query_protect(va), PAGE_READONLY);

        // 1. Acceptance path: right before `VirtualProtect(..., PAGE_NOACCESS)`
        //    runs inside `try_defer_page_inner`, protection MUST still be
        //    `PAGE_READONLY` (never `PAGE_READWRITE`).
        if let Some((shadow, _side)) = make_card_shadow_at(va as u64, 1, 0x11) {
            let observed_prot = Arc::new(AtomicUsize::new(0));
            let observed_cb = observed_prot.clone();
            BEFORE_PROTECT_HOOK.with(|hook| {
                *hook.borrow_mut() = Some(Box::new(move || {
                    observed_cb.store(query_protect(va) as usize, Ordering::SeqCst);
                }));
            });
            assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
            BEFORE_PROTECT_HOOK.with(|hook| *hook.borrow_mut() = None);
            assert_eq!(
                observed_prot.load(Ordering::SeqCst) as u32,
                PAGE_READONLY,
                "view must stay PAGE_READONLY across arm closure until PAGE_NOACCESS is applied"
            );
            assert_eq!(query_protect(va), PAGE_NOACCESS);
            assert!(materialize_range(va, PAGE));
            assert_eq!(query_protect(va), PAGE_READWRITE);
            drop(shadow);
        }

        // 2. Rejection path: re-guard the view to `PAGE_READONLY`, reject inside
        //    `try_arm_lazy_under_guards`: protection must be `PAGE_READONLY`
        //    inside the closure and cleanly restored to `PAGE_READWRITE` on rollback.
        assert!(crate::watch::writes(va, va + PAGE, true).is_some());
        assert_eq!(query_protect(va), PAGE_READONLY);
        let mut prot_in_closure = 0u32;
        let armed = crate::watch::try_arm_lazy_under_guards(&[block], |clean_after| {
            assert!(clean_after.is_some());
            prot_in_closure = query_protect(va);
            false
        });
        assert!(!armed);
        assert_eq!(prot_in_closure, PAGE_READONLY);
        assert_eq!(
            query_protect(va),
            PAGE_READWRITE,
            "rejection rollback must restore PAGE_READWRITE and clear guard metadata"
        );
        assert!(!crate::watch::is_guard_protected_for_test(block));

        clear_hot_for_test(block);
        // SAFETY: cleanup.
        unsafe { free_watch_pages(ptr) };
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_arm_invalidates_clean_proof_when_alias_is_unguarded_or_view_guard_disabled() {
        use windows_sys::Win32::System::Memory::{
            MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, VirtualQuery,
        };

        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let query_protect = |addr: usize| -> u32 {
            // SAFETY: zero-initialized `MEMORY_BASIC_INFORMATION`.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: querying page protection with valid out pointer.
            let n = unsafe { VirtualQuery(addr as *const _, &mut info, size_of_val(&info)) };
            assert_ne!(n, 0);
            info.Protect
        };

        // Map a 1-page anonymous section at two section aliases A (primary) and B (alias).
        // SAFETY: creating a 1-page anonymous section.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: mapping view A.
        let view_a = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        // SAFETY: mapping view B.
        let view_b = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        assert!(!view_a.Value.is_null() && !view_b.Value.is_null());
        let va_a = view_a.Value as usize;
        let va_b = view_b.Value as usize;
        let blk_a = va_a / PAGE;
        let blk_b = va_b / PAGE;
        clear_hot_for_test(blk_a);
        clear_hot_for_test(blk_b);
        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        assert!(crate::inplace::title_memory_view(
            va_a,
            PAGE,
            section as usize,
            0
        ));
        assert!(crate::inplace::title_memory_view(
            va_b,
            PAGE,
            section as usize,
            0
        ));
        // SAFETY: seed section bytes.
        unsafe { ptr::write_bytes(view_a.Value.cast::<u8>(), 0x22, PAGE) };

        // Also allocate an unaliased single-view page to confirm single views keep
        // their zero-DMA clean proof even with `VICTORY_VIEW_GUARD=0`.
        let single_ptr = alloc_watch_pages(1);
        // SAFETY: `single_ptr` owns `PAGE` bytes.
        unsafe { ptr::write_bytes(single_ptr, 0x33, PAGE) };
        let single_va = single_ptr as usize;
        let single_blk = single_va / PAGE;
        clear_hot_for_test(single_blk);

        if let (Some((shadow, _side)), Some((single_shadow, _single_side))) = (
            make_card_shadow_at(va_a as u64, 1, 0x22),
            make_card_shadow_at(single_va as u64, 1, 0x33),
        ) {
            let shadow_ptr = Arc::as_ptr(&shadow) as usize;
            let single_ptr_id = Arc::as_ptr(&single_shadow) as usize;
            let _iso_gen = crate::inplace::isolate_generation_for_test();

            {
                let _no_vg = crate::watch::disable_view_guard_for_test();

                // 1. With `VICTORY_VIEW_GUARD=0`, guarding primary A leaves alias B writable
                //    (`PAGE_READWRITE`). Arming A MUST invalidate `cpu_clean_after` (`None`).
                assert!(crate::watch::writes(va_a, va_a + PAGE, true).is_some());
                assert_eq!(query_protect(va_a), PAGE_READONLY);
                assert_eq!(query_protect(va_b), PAGE_READWRITE);
                assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
                assert_eq!(query_protect(va_a), PAGE_NOACCESS);
                assert_eq!(query_protect(va_b), PAGE_NOACCESS);
                assert_eq!(
                    owned_blocks_with_clean_in(shadow_ptr, blk_a, blk_a + 1),
                    vec![(blk_a, None)],
                    "clean proof must be invalidated when alias B was writable under VICTORY_VIEW_GUARD=0"
                );
                assert!(materialize_range(va_a, PAGE));
                clear_hot_for_test(blk_a);

                // 2. Even if both A and B were independently guarded while `VICTORY_VIEW_GUARD=0`,
                //    cross-alias write tracking is off, so `cpu_clean_after` MUST still be `None`.
                assert!(crate::watch::writes(va_a, va_a + PAGE, true).is_some());
                assert!(crate::watch::writes(va_b, va_b + PAGE, true).is_some());
                assert_eq!(query_protect(va_a), PAGE_READONLY);
                assert_eq!(query_protect(va_b), PAGE_READONLY);
                assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
                assert_eq!(
                    owned_blocks_with_clean_in(shadow_ptr, blk_a, blk_a + 1),
                    vec![(blk_a, None)],
                    "clean proof must stay invalidated for aliased pages under VICTORY_VIEW_GUARD=0"
                );
                assert!(materialize_range(va_a, PAGE));
                clear_hot_for_test(blk_a);

                // 3. A single guarded view (no aliases) STILL retains its zero-DMA clean proof
                //    even when `VICTORY_VIEW_GUARD=0`.
                assert!(crate::watch::writes(single_va, single_va + PAGE, true).is_some());
                assert_eq!(query_protect(single_va), PAGE_READONLY);
                assert!(try_defer_page_inner(&single_shadow, 0, PAGE, 0));
                let single_clean =
                    owned_blocks_with_clean_in(single_ptr_id, single_blk, single_blk + 1);
                assert_eq!(single_clean.len(), 1);
                assert_eq!(single_clean[0].0, single_blk);
                assert!(
                    single_clean[0].1.is_some(),
                    "single guarded view must retain its clean proof"
                );
                assert!(materialize_range(single_va, PAGE));
                clear_hot_for_test(single_blk);
            }

            // 4. With view-guarding enabled, guarding A guards BOTH A and B read-only, so
            //    arming A preserves the zero-DMA clean proof (`Some(_)`).
            if crate::watch::view_guard_enabled_for_test() {
                assert!(crate::watch::writes(va_a, va_a + PAGE, true).is_some());
                assert_eq!(query_protect(va_a), PAGE_READONLY);
                assert_eq!(query_protect(va_b), PAGE_READONLY);
                assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
                let aliased_clean = owned_blocks_with_clean_in(shadow_ptr, blk_a, blk_a + 1);
                assert_eq!(aliased_clean.len(), 1);
                assert_eq!(aliased_clean[0].0, blk_a);
                assert!(
                    aliased_clean[0].1.is_some(),
                    "aliased page with all views guarded read-only must preserve its clean proof"
                );
                assert!(materialize_range(va_a, PAGE));
                clear_hot_for_test(blk_a);
            }

            drop(single_shadow);
            drop(shadow);
        }

        clear_hot_for_test(blk_a);
        clear_hot_for_test(blk_b);
        clear_hot_for_test(single_blk);
        crate::watch::title_memory_remapped(va_a, PAGE);
        crate::watch::title_memory_remapped(va_b, PAGE);
        // SAFETY: cleanup views, section, and single-view page.
        unsafe {
            free_watch_pages(single_ptr);
            UnmapViewOfFile(view_a);
            UnmapViewOfFile(view_b);
            CloseHandle(section);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_attach_new_alias_is_mutually_exclusive_with_guarded_refresh_protect_here() {
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        // SAFETY: 1-page anonymous section mapped at views A and B.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: mapping view A.
        let view_a = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        // SAFETY: mapping view B.
        let view_b = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        assert!(!view_a.Value.is_null() && !view_b.Value.is_null());
        let va_a = view_a.Value as usize;
        let va_b = view_b.Value as usize;
        let blk_a = va_a / PAGE;
        let blk_b = va_b / PAGE;
        clear_hot_for_test(blk_a);
        clear_hot_for_test(blk_b);
        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        assert!(crate::inplace::title_memory_view(
            va_a,
            PAGE,
            section as usize,
            0
        ));
        // SAFETY: seed physical page.
        unsafe { ptr::write_bytes(va_a as *mut u8, 0x11, PAGE) };

        let query_protect = |addr: usize| -> u32 {
            // SAFETY: zero-initialized `MEMORY_BASIC_INFORMATION`.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: querying page protection with valid out pointer.
            let n = unsafe { VirtualQuery(addr as *const _, &mut info, size_of_val(&info)) };
            assert_ne!(n, 0);
            info.Protect
        };

        if let Some((shadow, _side)) = make_card_shadow_at(va_a as u64, 1, 0x11) {
            gpu_write_shadow(&shadow, 0, &[0x7E; 32]);
            assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));

            // Hold `blk_b`'s guard shard (as `guard_run_shard` does across its
            // `has_block(B)` check and `protect_here` call), then spawn
            // `attach_new_alias_on_failed_materialize` on another thread.
            let guard_b = crate::watch::lock_guard_block_for_test(blk_b);
            let attach_done = Arc::new(AtomicBool::new(false));
            let attach_done_t = attach_done.clone();
            let section_id = section as usize;
            let t = std::thread::spawn(move || {
                let ok = attach_new_alias_on_failed_materialize(va_b, PAGE, section_id, 0);
                attach_done_t.store(ok, Ordering::SeqCst);
                ok
            });
            // While `guard_b` is held, `attach_new_alias_on_failed_materialize`
            // cannot run until `guard_b` drops, so `protect_here` cannot race
            // ahead and overwrite `PAGE_NOACCESS` with `PAGE_READONLY`.
            drop(guard_b);
            assert!(t.join().unwrap());
            assert!(attach_done.load(Ordering::SeqCst));
            assert_eq!(query_protect(va_b), PAGE_NOACCESS);
            assert!(!crate::watch::is_guard_protected_for_test(blk_b));

            // Also verify the reverse ordering: if `protect_here` ran FIRST
            // (marking B `PAGE_READONLY` + `GuardProtected`), attaching B must
            // upgrade it to `PAGE_NOACCESS` and clear `GuardProtected`, and a
            // subsequent `watch::writes` must NOT downgrade it to `PAGE_READONLY`.
            assert!(materialize_range(va_a, PAGE));
            clear_hot_for_test(blk_a);
            clear_hot_for_test(blk_b);
            gpu_write_shadow(&shadow, 0, &[0x8F; 32]);
            assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
            crate::watch::protect_here_block_for_test(blk_b);
            assert_eq!(query_protect(va_b), PAGE_READONLY);
            assert!(crate::watch::is_guard_protected_for_test(blk_b));

            assert!(attach_new_alias_on_failed_materialize(
                va_b,
                PAGE,
                section as usize,
                0
            ));
            assert_eq!(query_protect(va_b), PAGE_NOACCESS);
            assert!(!crate::watch::is_guard_protected_for_test(blk_b));

            // Subsequent guarded refresh on B must see B is lazy-armed and keep `PAGE_NOACCESS`.
            let _ = crate::watch::writes(va_b, va_b + PAGE, true);
            assert_eq!(
                query_protect(va_b),
                PAGE_NOACCESS,
                "guarded refresh must not overwrite attached alias PAGE_NOACCESS with PAGE_READONLY"
            );

            assert!(materialize_range(va_a, PAGE));
            assert_eq!(query_protect(va_b), PAGE_READWRITE);
            drop(shadow);
        }

        clear_hot_for_test(blk_a);
        clear_hot_for_test(blk_b);
        crate::watch::title_memory_remapped(va_a, PAGE);
        crate::watch::title_memory_remapped(va_b, PAGE);
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view_b.Value,
            });
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view_a.Value,
            });
            CloseHandle(section);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_discard_primary_with_shorter_surviving_alias_clamps_and_merges() {
        const SUB_LEN: usize = 4096;
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        // SAFETY: 1-page (64 KiB) anonymous section mapped at primary A (64 KiB)
        // and surviving short alias B (4 KiB).
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: mapping 64 KiB primary A.
        let view_a = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        // SAFETY: mapping 4 KiB alias B.
        let view_b = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SUB_LEN) };
        assert!(!view_a.Value.is_null() && !view_b.Value.is_null());
        let va_a = view_a.Value as usize;
        let va_b = view_b.Value as usize;
        let blk_a = va_a / PAGE;
        let blk_b = va_b / PAGE;
        clear_hot_for_test(blk_a);
        clear_hot_for_test(blk_b);
        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        assert!(crate::inplace::title_memory_view(
            va_a,
            PAGE,
            section as usize,
            0
        ));
        // SAFETY: seed physical page with `0x11`.
        unsafe { ptr::write_bytes(va_a as *mut u8, 0x11, PAGE) };

        if let Some((shadow, _side)) = make_card_shadow_at(va_a as u64, 1, 0x11) {
            gpu_write_shadow(&shadow, 0, &[0xAB; 64]);
            assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
            // Attach 4 KiB alias B while the 64 KiB entry is armed.
            set_fail_next_fetches_for_test(1);
            assert!(!crate::inplace::title_memory_view(
                va_b,
                SUB_LEN,
                section as usize,
                0
            ));
            assert!(has_block(blk_a) && has_block(blk_b));
            assert!(!is_readable_now(va_b));

            // Discard primary A while only 4 KiB alias B survives: `detach_remapped_vas_locked`
            // must clamp `entry.len` to `SUB_LEN` so `own_view_ptr` succeeds through B
            // and merges GPU bytes into RAM instead of wedging at `PAGE_NOACCESS`.
            crate::inplace::views_gone(va_a, PAGE);
            discard_range(va_a, PAGE);
            assert!(!has_block(blk_a));
            assert!(
                !has_block(blk_b),
                "discarding primary A must promote and materialize through surviving 4 KiB alias B"
            );
            assert!(is_readable_now(va_b));
            // SAFETY: B is now PAGE_READWRITE and holds merged GPU bytes.
            let got_b = unsafe { std::slice::from_raw_parts(va_b as *const u8, 64) };
            assert_eq!(got_b, &[0xAB; 64]);

            // Restore RW on A before cleanup.
            let mut old = 0u32;
            // SAFETY: A is still mapped in this test until UnmapViewOfFile below.
            unsafe { VirtualProtect(va_a as *const _, PAGE, PAGE_READWRITE, &mut old) };
            drop(shadow);
        }

        clear_hot_for_test(blk_a);
        clear_hot_for_test(blk_b);
        crate::watch::title_memory_remapped(va_a, PAGE);
        crate::watch::title_memory_remapped(va_b, SUB_LEN);
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view_b.Value,
            });
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view_a.Value,
            });
            CloseHandle(section);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }

    #[test]
    fn lazy_unaligned_alias_promotion_is_idempotent_across_repeated_discards() {
        const SUB_OFF: usize = 4096;
        const SUB_LEN: usize = 4096;
        let Some(h) = crate::host::host() else {
            return;
        };
        let errors_before = h.d3d12_errors();
        let _iso_gen = crate::inplace::isolate_generation_for_test();
        // SAFETY: 1-page (64 KiB) anonymous section mapped at primary A and aliases B, C.
        let section: HANDLE = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                ptr::null(),
                PAGE_READWRITE,
                0,
                PAGE as u32,
                ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: mapping 64 KiB view A.
        let view_a = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        // SAFETY: mapping 64 KiB view B.
        let view_b = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        // SAFETY: mapping 64 KiB view C.
        let view_c = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        assert!(!view_a.Value.is_null() && !view_b.Value.is_null() && !view_c.Value.is_null());
        let va_a = view_a.Value as usize;
        let va_b = view_b.Value as usize;
        let va_c = view_c.Value as usize;
        let blk_a = va_a / PAGE;
        let blk_b = va_b / PAGE;
        let blk_c = va_c / PAGE;
        clear_hot_for_test(blk_a);
        clear_hot_for_test(blk_b);
        clear_hot_for_test(blk_c);
        crate::inplace::title_memory_section(section as usize, PAGE as u64, true);
        assert!(crate::inplace::title_memory_view(
            va_a,
            PAGE,
            section as usize,
            0
        ));
        // SAFETY: seed physical page with `0x11`.
        unsafe { ptr::write_bytes(va_a as *mut u8, 0x11, PAGE) };

        if let Some((shadow, _side)) = make_card_shadow_at(va_a as u64, 1, 0x11) {
            // Write distinct GPU bytes into each 4 KiB slice so any offset shift
            // or zero-length truncation is immediately observable on materialize.
            gpu_write_shadow(&shadow, 0, &[0xAA; 64]);
            gpu_write_shadow(&shadow, SUB_OFF, &[0xBB; 64]);
            gpu_write_shadow(&shadow, 2 * SUB_OFF, &[0xCC; 64]);
            gpu_write_shadow(&shadow, 3 * SUB_OFF, &[0xDD; 64]);

            // Part 1: Promote unaligned short alias B (with failed materialization),
            // then discard a second alias C. `entry.offset` and `entry.len` must NOT
            // be double-shifted, and subsequent materialization through B must succeed.
            assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
            set_fail_next_fetches_for_test(1);
            assert!(!crate::inplace::title_memory_view(
                va_b + SUB_OFF,
                2 * SUB_LEN,
                section as usize,
                SUB_OFF as u64,
            ));
            set_fail_next_fetches_for_test(1);
            assert!(!crate::inplace::title_memory_view(
                va_c + 3 * SUB_OFF,
                SUB_LEN,
                section as usize,
                (3 * SUB_OFF) as u64,
            ));
            assert!(has_block(blk_a) && has_block(blk_b) && has_block(blk_c));

            // Discard A while retire-materialize fails: promotes longer unaligned B at `SUB_OFF`.
            set_fail_next_fetches_for_test(1);
            discard_range(va_a, PAGE);
            {
                let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
                let entry = table.pages.get(&blk_b).expect("B promoted as primary");
                assert_eq!(
                    entry.armed_vas[0],
                    ArmedVa {
                        va: va_b + SUB_OFF,
                        len: 2 * SUB_LEN,
                    }
                );
                assert_eq!(entry.offset, SUB_OFF);
                assert_eq!(entry.len, 2 * SUB_LEN);
            }

            // Discard C while unaligned B is already `armed_vas[0]`: must be idempotent.
            discard_range(va_c, PAGE);
            {
                let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
                let entry = table.pages.get(&blk_b).expect("B still primary");
                assert_eq!(
                    entry.armed_vas,
                    vec![ArmedVa {
                        va: va_b + SUB_OFF,
                        len: 2 * SUB_LEN,
                    }]
                );
                assert_eq!(
                    entry.offset, SUB_OFF,
                    "discarding a non-primary alias must not double-shift entry.offset"
                );
                assert_eq!(
                    entry.len,
                    2 * SUB_LEN,
                    "discarding a non-primary alias must not shrink entry.len"
                );
            }

            assert!(materialize_range(va_b + SUB_OFF, 2 * SUB_LEN));
            assert!(!has_block(blk_b));
            assert!(is_readable_now(va_b + SUB_OFF));
            // SAFETY: B slice is restored to PAGE_READWRITE and holds merged GPU bytes.
            let got_b0 = unsafe { std::slice::from_raw_parts((va_b + SUB_OFF) as *const u8, 64) };
            // SAFETY: second 4 KiB of B slice is also restored to PAGE_READWRITE.
            let got_b1 =
                unsafe { std::slice::from_raw_parts((va_b + 2 * SUB_OFF) as *const u8, 64) };
            assert_eq!(got_b0, &[0xBB; 64]);
            assert_eq!(got_b1, &[0xCC; 64]);

            // Part 2: Promoting from unaligned alias B (`SUB_OFF`, len `2 * SUB_LEN`) to
            // a DIFFERENT unaligned alias C (`3 * SUB_OFF`, len `SUB_LEN`) must recompute
            // `entry.offset` from the page base (`3 * SUB_OFF`, not `4 * SUB_OFF`).
            let mut old = 0u32;
            // SAFETY: restoring RW on views still mapped in this test.
            unsafe {
                VirtualProtect(va_a as *const _, PAGE, PAGE_READWRITE, &mut old);
                VirtualProtect(va_b as *const _, PAGE, PAGE_READWRITE, &mut old);
                VirtualProtect(va_c as *const _, PAGE, PAGE_READWRITE, &mut old);
                ptr::write_bytes(va_a as *mut u8, 0x11, PAGE);
            }
            crate::inplace::views_gone(va_b, PAGE);
            crate::inplace::views_gone(va_c, PAGE);
            clear_hot_for_test(blk_a);
            clear_hot_for_test(blk_b);
            clear_hot_for_test(blk_c);

            assert!(try_defer_page_inner(&shadow, 0, PAGE, 0));
            set_fail_next_fetches_for_test(1);
            assert!(!crate::inplace::title_memory_view(
                va_b + SUB_OFF,
                2 * SUB_LEN,
                section as usize,
                SUB_OFF as u64,
            ));
            set_fail_next_fetches_for_test(1);
            assert!(!crate::inplace::title_memory_view(
                va_c + 3 * SUB_OFF,
                SUB_LEN,
                section as usize,
                (3 * SUB_OFF) as u64,
            ));

            // Discard A (promotes longer unaligned B, fail materialize).
            set_fail_next_fetches_for_test(1);
            discard_range(va_a, PAGE);
            {
                let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
                let entry = table.pages.get(&blk_b).expect("B promoted as primary");
                assert_eq!(entry.offset, SUB_OFF);
                assert_eq!(entry.len, 2 * SUB_LEN);
            }

            // Discard B (promotes unaligned C, fail materialize).
            set_fail_next_fetches_for_test(1);
            discard_range(va_b, PAGE);
            {
                let table = LAZY_STATE.lock().unwrap_or_else(|e| e.into_inner());
                let entry = table.pages.get(&blk_c).expect("C promoted as primary");
                assert_eq!(
                    entry.offset,
                    3 * SUB_OFF,
                    "re-promoting from unaligned B to unaligned C must not accumulate offsets"
                );
                assert_eq!(entry.len, SUB_LEN);
            }

            assert!(materialize_range(va_c + 3 * SUB_OFF, SUB_LEN));
            assert!(!has_block(blk_c));
            assert!(is_readable_now(va_c + 3 * SUB_OFF));
            // SAFETY: C slice is restored to PAGE_READWRITE and holds merged GPU bytes.
            let got_c =
                unsafe { std::slice::from_raw_parts((va_c + 3 * SUB_OFF) as *const u8, 64) };
            assert_eq!(got_c, &[0xDD; 64]);

            // SAFETY: restore RW on views still mapped before cleanup.
            unsafe {
                VirtualProtect(va_a as *const _, PAGE, PAGE_READWRITE, &mut old);
                VirtualProtect(va_b as *const _, PAGE, PAGE_READWRITE, &mut old);
                VirtualProtect(va_c as *const _, PAGE, PAGE_READWRITE, &mut old);
            }
            drop(shadow);
        }

        clear_hot_for_test(blk_a);
        clear_hot_for_test(blk_b);
        clear_hot_for_test(blk_c);
        crate::watch::title_memory_remapped(va_a, PAGE);
        crate::watch::title_memory_remapped(va_b, PAGE);
        crate::watch::title_memory_remapped(va_c, PAGE);
        // SAFETY: cleanup.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view_c.Value,
            });
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view_b.Value,
            });
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view_a.Value,
            });
            CloseHandle(section);
        }
        h.log_messages();
        assert_eq!(h.d3d12_errors(), errors_before);
    }
}
