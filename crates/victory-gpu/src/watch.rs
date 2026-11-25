//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection.
//!
//! shortcut: writes into physical pages through another view (xmem's
//! `write_physical`, DirectStorage's page destinations) are not seen at
//! the guarded addresses; they go to the texture pool, whose textures are
//! hashed, not guarded. A system call writing into a guarded block fails
//! (none is known to).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Page ranges `[first, last)` asked ahead for the current submission
    /// ([`ask_once`]): `writes` answers any range inside one from the journal
    /// without another kernel call.
    ahead: Vec<(usize, usize)>,
    /// Whether `ahead` stands until [`ask_once_end`].
    kept: bool,
}

impl Journal {
    /// Refreshes overlapping an announced file read must ask the kernel
    /// again, even when a previous query covered their pages.
    fn forget_ahead(&mut self, (first, last): (usize, usize)) {
        const _: () = assert!(BLOCK == PAGE);
        self.ahead.retain(|&(a, b)| b <= first || last <= a);
        self.kept = !self.ahead.is_empty();
    }

    fn snapshot(&self, first: usize, last: usize) -> (u64, Vec<(usize, u64)>) {
        (
            self.asked,
            self.written
                .range(first..last)
                .map(|(&p, &w)| (p, w))
                .collect(),
        )
    }
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
    ahead: Vec::new(),
    kept: false,
});

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another `GetWriteWatch`
/// reset call.
const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// Maximum span of a single coalesced `GetWriteWatch` asking (1 GiB).
const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// Inserts 4 KiB kernel-reported addresses into `written` at 64 KiB `PAGE`
/// granularity, skipping consecutive addresses that fold into the same page.
fn insert_found(found: &[usize], written: &mut BTreeMap<usize, u64>, now: u64) {
    let mut prev = usize::MAX;
    for &at in found {
        let page = at / PAGE;
        if page != prev {
            written.insert(page, now);
            prev = page;
        }
    }
}

/// Asks the kernel once for the written pages of `[first, last)` and notes
/// them in the journal under `now`; returns `false` (leaving the journal
/// unchanged) if the range is not write-watched memory of one reservation.
fn ask_kernel(j: &mut Journal, first: usize, last: usize, now: u64) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    if failed {
        return false;
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
    }
    insert_found(&j.found[..count], &mut j.written, now);
    true
}

/// Tries one kernel asking for `pieces` (sorted page ranges); bisects if the
/// combined span crosses reservations or non-watched memory.
fn ask_run(j: &mut Journal, pieces: &[(usize, usize)], now: u64) {
    if pieces.len() < 2 {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    if ask_kernel(j, first, last, now) {
        j.ahead.push((first, last));
        return;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    ask_run(j, a, now);
    ask_run(j, b, now);
}

/// Coalesces write-watch queries for all `(start, end)` ranges of a submission
/// into a small number of `GetWriteWatch` calls so repeated flushes within the
/// submission answer from the journal until [`ask_once_end`]. Ranges that are
/// not single-allocation write-watched memory are left to [`writes`].
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    let mut pieces: Vec<(usize, usize)> = ranges
        .iter()
        .filter(|&&(start, end)| start < end)
        .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .collect();
    if pieces.is_empty() {
        return;
    }
    pieces.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(pieces.len());
    for (first, last) in pieces {
        match merged.last_mut() {
            Some(prev) if first <= prev.1 => prev.1 = prev.1.max(last),
            _ => merged.push((first, last)),
        }
    }
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // Snapshot before the kernel queries. A read already in progress may
    // complete while they run. A newly announced read takes this journal
    // lock and invalidates its range before the caller starts writing.
    let pending: Vec<_> = guards().announced.keys().copied().collect();
    j.ahead.clear();
    j.kept = false;
    j.asked += 1;
    let now = j.asked;
    let mut i = 0;
    while i < merged.len() {
        let first = merged[i].0;
        let mut last = merged[i].1;
        let mut k = i + 1;
        while k < merged.len()
            && merged[k].0 <= last + AHEAD_GAP
            && last.max(merged[k].1) - first <= AHEAD_MAX
        {
            last = last.max(merged[k].1);
            k += 1;
        }
        ask_run(j, &merged[i..k], now);
        i = k;
    }
    // Single isolated ranges (or halves that failed coalescing) are asked once
    // here so repeated flushes in the same submission do not re-query them.
    for &(first, last) in &merged {
        if !j.ahead.iter().any(|&(a, b)| a <= first && last <= b) && ask_kernel(j, first, last, now)
        {
            j.ahead.push((first, last));
        }
    }
    j.kept = !j.ahead.is_empty();
    for range in pending {
        j.forget_ahead(range);
    }
}

/// Ends the active per-submission coalesced asking so later [`writes`] calls
/// query the kernel again.
pub(crate) fn ask_once_end() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if j.kept {
        j.kept = false;
        j.ahead.clear();
    }
}

/// Notes that our submission wrote title memory `[at, at + len)` while a
/// coalesced asking stands, so refreshes recorded after the copy see those
/// pages as written without ending coalescing for the rest of the submission.
pub(crate) fn wrote(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    for page in at / PAGE..at.saturating_add(len).div_ceil(PAGE) {
        j.written.insert(page, now);
    }
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    if j.kept && j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
        return Some(j.snapshot(first, last));
    }
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // Clean mapped-view blocks are still protected. A write, kernel-write
    // announcement or remap invalidates the corresponding entries.
    if guarded && first < last && guards().clean.range(first..last).copied().eq(first..last) {
        j.asked += 1;
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::WatchUs, started.elapsed().as_micros() as u64);
        }
        return Some(j.snapshot(first, last));
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    let guarded = if failed {
        count = 0;
        match by_allocation(first * PAGE, last * PAGE, &mut j.found) {
            Some(pages) => pages,
            None if !guarded => return None,
            None => guard(first * PAGE, last * PAGE)?,
        }
    } else {
        Vec::new()
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
    }
    j.asked += 1;
    let now = j.asked;
    insert_found(&j.found[..count], &mut j.written, now);
    for page in guarded {
        j.written.insert(page, now);
    }
    Some(j.snapshot(first, last))
}

/// The pages of `[start, end)` written, asked allocation by allocation:
/// the kernel refuses a range over several (a title buffer spans as many
/// allocations as the runtime made for it). Parts never committed cannot
/// have been written, so a range none of which is committed (a reserved
/// allocation the title has not committed yet) has no writes. `None` if a
/// committed part is not write-watched (or the range is one committed
/// allocation, already refused).
fn by_allocation(start: usize, end: usize, found: &mut Vec<usize>) -> Option<Vec<usize>> {
    use windows_sys::Win32::System::Memory::{
        GetWriteWatch, MEM_COMMIT, MEM_FREE, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let query = |at: usize| {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: a query about an address; nothing is dereferenced.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        (n != 0).then_some(info)
    };
    let (mut pages, mut at, mut allocations, mut any) = (Vec::new(), start, 0, false);
    while at < end {
        let info = query(at)?;
        // A committed section view cannot belong to a MEM_WRITE_WATCH
        // reservation. Its other protection regions need no further queries.
        if info.State == MEM_COMMIT && info.Type == MEM_MAPPED {
            return None;
        }
        let base = info.AllocationBase as usize;
        let mut committed = info.State == MEM_COMMIT;
        let mut to = (info.BaseAddress as usize + info.RegionSize).min(end);
        // The rest of the allocation's regions in the range.
        while info.State != MEM_FREE && to < end {
            let next = query(to)?;
            if next.State == MEM_FREE || next.AllocationBase as usize != base {
                break;
            }
            committed |= next.State == MEM_COMMIT;
            to = (next.BaseAddress as usize + next.RegionSize).min(end);
        }
        allocations += 1;
        any |= committed;
        if committed {
            let mut count = (to - at) / KERNEL_PAGE;
            found.resize(count.max(found.len()), 0);
            let mut granularity = 0;
            // SAFETY: the answer buffer holds `count` entries.
            let failed = unsafe {
                GetWriteWatch(
                    WRITE_WATCH_FLAG_RESET,
                    at as *const _,
                    to - at,
                    found.as_mut_ptr().cast(),
                    &mut count,
                    &mut granularity,
                )
            } != 0;
            if failed {
                return None;
            }
            pages.extend(found[..count].iter().map(|&a| a / PAGE));
        }
        at = to;
    }
    (allocations > 1 || !any).then_some(pages)
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
    /// Announced kernel writes still to land, as block ranges `[first, end)`
    /// with how many announcements of that range are open: their blocks are
    /// left open until they have, however many refreshes come first. Kept
    /// as ranges, not per block, so a title-supplied length (saturated by
    /// `fread`'s size times count) costs nothing however large; counted, so
    /// a range announced again and again (overlapped reads into the same
    /// buffer, which never say they landed) stays one entry.
    announced: BTreeMap<(usize, usize), u32>,
}

impl Guards {
    /// Whether an announced kernel write still to land covers `block`.
    fn announced(&self, block: usize) -> bool {
        self.announced
            .keys()
            .any(|&(first, end)| (first..end).contains(&block))
    }
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    clean: BTreeSet::new(),
    announced: BTreeMap::new(),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the range contains no mapped view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // Every block in `clean` is also in `protected` (previously verified as
    // mapped view memory and still read-only). When the entire range is clean,
    // skip both the kernel query and per-block lookups.
    if first < last && guards().clean.range(first..last).count() == last - first {
        return Some(Vec::new());
    }
    if !has_mapped(start, end) {
        return None;
    }
    install();
    let mut g = guards();
    let clean: Vec<usize> = g.clean.range(first..last).copied().collect();
    if clean.len() == last - first {
        return Some(Vec::new());
    }
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    let mut next_clean = 0;
    for block in first..=last {
        while next_clean < clean.len() && clean[next_clean] < block {
            next_clean += 1;
        }
        let is_clean = next_clean < clean.len() && clean[next_clean] == block;
        let open = block < last && !is_clean;
        if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if g.protected.contains(&block) {
                // Written through the runtime, still read-only.
                g.clean.insert(block);
            } else if !g.announced(block) {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block);
        }
    }
    Some(written)
}

/// Whether `[start, end)` contains a mapped view and no committed non-view
/// memory. Checks `start` first via [`mapped`] (fast path); when `start` is
/// not a view (e.g. an unmapped placeholder/hole inside a view pool), walks
/// `VirtualQuery` across `[start, end)` and rejects the range if any
/// committed non-view region is present so private/image memory is never
/// read-protected.
fn has_mapped(start: usize, end: usize) -> bool {
    if mapped(start) {
        return true;
    }
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    let mut at = start;
    let mut found_mapped = false;
    while at < end {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: queries virtual memory metadata without dereferencing `at`.
        if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0
            || info.RegionSize == 0
        {
            break;
        }
        if info.Type == MEM_MAPPED {
            found_mapped = true;
        } else if info.State == MEM_COMMIT {
            return false;
        }
        let Some(next) = (info.BaseAddress as usize).checked_add(info.RegionSize) else {
            break;
        };
        if next <= at {
            break;
        }
        at = next;
    }
    found_mapped
}

/// Whether the current allocation at `at` is a mapped view. Asking only for
/// the allocation type avoids enumerating pages with identical protection,
/// which can span an entire large view after its blocks were guarded.
fn mapped(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_MAPPED, MEMORY_BASIC_INFORMATION, MemoryRegionInfo, QueryVirtualMemoryInformation,
        VirtualQuery, WIN32_MEMORY_REGION_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut region = WIN32_MEMORY_REGION_INFORMATION::default();
    // SAFETY: both queries describe an address without dereferencing it. The
    // output buffers match their information classes.
    let queried = unsafe {
        QueryVirtualMemoryInformation(
            GetCurrentProcess(),
            at as *const _,
            MemoryRegionInfo,
            (&raw mut region).cast(),
            size_of_val(&region),
            std::ptr::null_mut(),
        )
    };
    if queried != 0 {
        // These are the Windows SDK's named bitfields in declaration order.
        // windows-sys exposes only the union's raw bits.
        const PRIVATE: u32 = 1 << 0;
        const MAPPED_DATA_FILE: u32 = 1 << 1;
        const MAPPED_IMAGE: u32 = 1 << 2;
        const MAPPED_PAGE_FILE: u32 = 1 << 3;
        const MAPPED_PHYSICAL: u32 = 1 << 4;
        // SAFETY: this union was filled by the successful query above.
        match unsafe { region.Anonymous.Flags } {
            MAPPED_DATA_FILE | MAPPED_PAGE_FILE | MAPPED_PHYSICAL => return true,
            PRIVATE | MAPPED_IMAGE => return false,
            _ => {}
        }
    }
    // Unknown flag combinations and failed allocation queries keep the
    // VirtualQuery type check, including for a free address.
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: the output buffer has the required layout and size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.Type == MEM_MAPPED
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail (unmapped).
fn protect(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
        }
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
        let mut old = 0;
        // SAFETY: a block we made read-only.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
    }
    !blocks.is_empty()
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// For the title's kernel reads, which other threads' refreshes can
/// overtake, the runtime calls [`kernel_will_write`] instead.
pub(crate) fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    open(&mut guards(), at, at.saturating_add(len));
}

/// Before the title's kernel read into its memory `[at, at + len)`: opens
/// the guarded blocks there and keeps them open until [`kernel_wrote`]. A
/// refresh on another thread in between would otherwise guard them again,
/// and the read would fail.
pub fn kernel_will_write(at: usize, len: usize) {
    let end = at.saturating_add(len);
    let range = (at / BLOCK, end.div_ceil(BLOCK));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut g = guards();
    open(&mut g, at, end);
    *g.announced.entry(range).or_default() += 1;
    j.forget_ahead(range);
}

/// The kernel write [`kernel_will_write`] announced for `[at, at + len)`
/// has landed (or failed); its blocks can be guarded again.
pub fn kernel_wrote(at: usize, len: usize) {
    let range = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    let mut g = guards();
    if let Some(n) = g.announced.get_mut(&range) {
        *n -= 1;
        if *n == 0 {
            g.announced.remove(&range);
        }
    }
}

/// A storage worker finished writing these mapped guest pages through another
/// view of the same section. Their address protections are still in effect.
pub fn mapped_pages_written(addresses: impl Iterator<Item = usize>) {
    let mut addresses = addresses.peekable();
    if addresses.peek().is_none() {
        return;
    }
    let mut g = guards();
    for at in addresses {
        g.clean.remove(&(at / PAGE));
    }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded, and kernel
/// writes announced wholly inside it are moot (an overlapped read that
/// never said it landed no longer keeps those blocks open).
pub fn title_memory_remapped(at: usize, len: usize) {
    let (first, end) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    let mut g = guards();
    let blocks: Vec<usize> = g.protected.range(first..end).copied().collect();
    for block in blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
    }
    g.announced
        .retain(|&(from, to), _| !(first <= from && to <= end));
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn allocation_query_recognizes_file_views_and_rejects_nonviews() {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_READ, MapViewOfFile, PAGE_READONLY, UnmapViewOfFile,
        };
        let file = tempfile::tempfile().unwrap();
        file.set_len(PAGE as u64).unwrap();
        // SAFETY: a valid, open file and a read-only mapping of its contents.
        let section = unsafe {
            CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                PAGE_READONLY,
                0,
                0,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: an open section containing at least one page.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_READ, 0, 0, PAGE) };
        assert!(!view.Value.is_null());
        assert!(mapped(view.Value as usize + 13));
        // SAFETY: our successful mapping and its section handle.
        unsafe {
            UnmapViewOfFile(view);
            CloseHandle(section);
        }
        assert!(!mapped((&raw const file) as usize)); // private stack allocation
        assert!(!mapped(mapped as *const () as usize)); // executable image
        assert!(!mapped(0)); // no allocation to query
    }

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        // A previous test may have used these virtual addresses.
        let mut journal = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
        for page in start / PAGE..end / PAGE {
            journal.written.remove(&page);
        }
        drop(journal);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // Clear journal entries before releasing `p` so a parallel test that
        // receives the same virtual address does not inherit stale writes.
        let mut journal = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
        for page in start / PAGE..end / PAGE {
            journal.written.remove(&page);
        }
        drop(journal);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// Two write-watched allocations side by side (the runtime reserves
    /// title memory allocation by allocation), asked about as one range, as
    /// a buffer spanning them is: the pages written in each are reported.
    #[test]
    fn a_range_over_several_watched_allocations_is_watched() {
        let (a, b) = loop {
            // SAFETY: a probe reservation, released at once to find room for two.
            let span =
                unsafe { VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE, PAGE_READWRITE) };
            assert!(!span.is_null());
            // SAFETY: the probe above.
            unsafe { VirtualFree(span, 0, MEM_RELEASE) };
            let at = |p: usize| {
                // SAFETY: a fixed address in the room just released.
                let got = unsafe {
                    VirtualAlloc(
                        p as *const _,
                        2 * PAGE,
                        MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                        PAGE_READWRITE,
                    )
                };
                got as usize
            };
            let a = at(span as usize);
            let b = at(span as usize + 2 * PAGE);
            if a != 0 && b != 0 {
                break (a, b);
            }
            for p in [a, b].into_iter().filter(|&p| p != 0) {
                // SAFETY: ours.
                unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
            }
        };
        // A previous test may have used these virtual addresses.
        {
            let mut journal = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
            for page in a / PAGE..(b + 2 * PAGE) / PAGE {
                journal.written.remove(&page);
            }
        }
        let _ = writes(a, b + 2 * PAGE, false);
        // SAFETY: committed pages of ours.
        unsafe {
            ((a + 8) as *mut u8).write(1);
            ((b + PAGE + 8) as *mut u8).write(1);
        }
        let (now, found) = writes(a, b + 2 * PAGE, false).expect("watched");
        assert_eq!(found, [(a / PAGE, now), (b / PAGE + 1, now)]);
        // Not freed: the journal keeps these pages, and a later test given
        // the same addresses would find them written.
    }

    /// A range in an allocation reserved but not committed yet (as the
    /// title's are before it uses them) has no writes, and is asked again
    /// once committed.
    #[test]
    fn a_range_never_committed_has_no_writes() {
        // SAFETY: a fresh write-watched reservation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                2 * PAGE,
                MEM_RESERVE | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (_, none) = writes(p, p + 2 * PAGE, false).expect("nothing can be written there");
        assert!(none.is_empty());
        // SAFETY: committing and writing a page of our reservation.
        unsafe {
            assert!(!VirtualAlloc(p as *const _, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            ((p + 8) as *mut u8).write(1);
        }
        let (now, found) = writes(p, p + 2 * PAGE, false).expect("watched");
        assert_eq!(found, [(p / PAGE, now)]);
        // Not freed: the journal keeps these pages, and a later test given
        // the same addresses would find them written.
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        let end = view + 2 * BLOCK;
        let (first, initial) = writes(view, end, true).expect("new mapped view");
        assert_eq!(initial, [(view / PAGE, first), (view / PAGE + 1, first)]);
        let (next, clean) = writes(view + 8, end - 1, true).expect("same protected pages");
        assert!(next > first);
        assert_eq!(clean, initial);
        mapped_pages_written(std::iter::once(view + BLOCK));
        let (alias, updated) = writes(view, end, true).expect("alias write");
        assert_eq!(updated, [(view / PAGE, first), (view / PAGE + 1, alias)]);
        assert_eq!(writes(view, end, true).unwrap().1, updated);
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        assert_eq!(
            // SAFETY: the bytes just read.
            unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) },
            b"from the disk"
        );
        let (changed, pages) = writes(view, end, true).expect("kernel write visible");
        assert_eq!(pages, [(view / PAGE, changed), (view / PAGE + 1, alias)]);
        assert_eq!(writes(view, end, true).unwrap().1, pages);
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    /// A range that begins on an unmapped reserved hole before a mapped view
    /// is still guarded, whereas a range starting in committed private memory
    /// before a mapped view is rejected so private pages are never
    /// read-protected.
    #[test]
    fn guard_handles_unmapped_prefix_and_rejects_committed_private_prefix() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFileEx,
            PAGE_NOACCESS, UnmapViewOfFile,
        };
        // SAFETY: create a 1-block pagefile section.
        let section = unsafe {
            CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                BLOCK as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());

        // Deterministically place an unmapped reserved block immediately before
        // a mapped view: reserve 2 blocks, release, then re-reserve block 0 and
        // map the view at block 1.
        let mut pair = None;
        for _ in 0..16 {
            // SAFETY: probe for a free 2-block-aligned address range.
            let probe =
                unsafe { VirtualAlloc(std::ptr::null(), 2 * BLOCK, MEM_RESERVE, PAGE_NOACCESS) }
                    as usize;
            assert_ne!(probe, 0);
            // SAFETY: freeing the probe we just reserved.
            assert_ne!(unsafe { VirtualFree(probe as *mut _, 0, MEM_RELEASE) }, 0);
            // SAFETY: reserve block 0 as an unmapped placeholder.
            let prefix =
                unsafe { VirtualAlloc(probe as *const _, BLOCK, MEM_RESERVE, PAGE_NOACCESS) }
                    as usize;
            if prefix != probe {
                if prefix != 0 {
                    // SAFETY: freeing our reservation.
                    unsafe { VirtualFree(prefix as *mut _, 0, MEM_RELEASE) };
                }
                continue;
            }
            // SAFETY: map the 1-block section immediately after the reserved prefix.
            let view = unsafe {
                MapViewOfFileEx(
                    section,
                    FILE_MAP_WRITE,
                    0,
                    0,
                    BLOCK,
                    (probe + BLOCK) as *const _,
                )
                .Value as usize
            };
            if view == probe + BLOCK {
                pair = Some((prefix, view));
                break;
            }
            if view != 0 {
                // SAFETY: unmapping our view.
                unsafe {
                    UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                        Value: view as *mut _,
                    });
                }
            }
            // SAFETY: freeing our prefix reservation.
            unsafe { VirtualFree(prefix as *mut _, 0, MEM_RELEASE) };
        }
        let (prefix, view) = pair.expect("reserve prefix block immediately before mapped view");

        // 1. Unmapped reserved prefix + mapped view: must stay guarded.
        assert!(!mapped(prefix));
        assert!(
            guard(prefix, view + BLOCK).is_some(),
            "range spanning an unmapped reserved prefix into a mapped view must stay guarded"
        );
        // SAFETY: writing inside our mapped read-write view.
        unsafe { (view as *mut u8).write(42) };
        let written = guard(prefix, view + BLOCK).expect("still guarded");
        assert!(written.contains(&(view / PAGE)));

        // 2. Committed private prefix + mapped view: must NOT read-protect private memory.
        // SAFETY: committing our reserved prefix block as private read-write memory.
        let committed =
            unsafe { VirtualAlloc(prefix as *const _, BLOCK, MEM_COMMIT, PAGE_READWRITE) };
        assert!(!committed.is_null());
        assert!(
            guard(prefix, view + BLOCK).is_none(),
            "range containing committed private memory must not be read-protected by guard"
        );
        // SAFETY: writing to the committed private block must still succeed without faulting.
        unsafe { (prefix as *mut u8).write(99) };

        // SAFETY: clean up our view and prefix reservation.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            });
            VirtualFree(prefix as *mut _, 0, MEM_RELEASE);
        }
        title_memory_remapped(view, BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }

    /// `fread`'s size times count saturates to `usize::MAX`: announcing and
    /// landing that kernel read is one range, not a step per block of the
    /// whole address space, and landing it leaves nothing announced.
    #[test]
    fn a_saturated_kernel_read_is_announced_in_one_step() {
        // Kernel half of the address space: no user allocation of another
        // test running at the same time can fall in the announced range.
        let at = 0xFFFF_8000_0000_0000;
        let range = (at / BLOCK, usize::MAX.div_ceil(BLOCK));
        kernel_will_write(at, usize::MAX);
        assert!(guards().announced(at / BLOCK + 12345));
        kernel_wrote(at, usize::MAX);
        assert!(!guards().announced.contains_key(&range));
    }

    /// A range announced again and again without landing (overlapped reads
    /// into one buffer) is one counted entry, and a remap over it drops it.
    #[test]
    fn repeated_announcements_are_counted_and_a_remap_drops_them() {
        let at = 0xFFFF_9000_0000_0000;
        let range = (at / BLOCK, (at + BLOCK).div_ceil(BLOCK));
        for _ in 0..1000 {
            kernel_will_write(at, BLOCK);
        }
        assert_eq!(guards().announced.get(&range), Some(&1000));
        kernel_wrote(at, BLOCK);
        assert_eq!(guards().announced.get(&range), Some(&999));
        title_memory_remapped(at, BLOCK);
        assert!(!guards().announced.contains_key(&range));
    }

    /// A kernel read announced with `kernel_will_write` survives a refresh
    /// on another thread before it lands; once `kernel_wrote` says it has,
    /// the next refresh guards its block again.
    #[test]
    fn an_announced_kernel_read_survives_a_refresh_before_it() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        let end = view + 2 * BLOCK;
        let (first, initial) = writes(view, end, true).expect("new mapped view");
        assert_eq!(initial, [(view / PAGE, first), (view / PAGE + 1, first)]);
        let (next, clean) = writes(view + 8, end - 1, true).expect("same protected pages");
        assert!(next > first);
        assert_eq!(clean, initial);
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        kernel_will_write(view + 100, 13); // the title's thread, in `ReadFile`
        let _ = guard(view, view + 2 * BLOCK); // a refresh on another thread
        assert!(
            read(&mut file),
            "the refresh guarded the block again under the read"
        );
        kernel_wrote(view + 100, 13);
        let (changed, pages) = writes(view, end, true).expect("kernel write visible");
        assert_eq!(pages, [(view / PAGE, changed), (view / PAGE + 1, first)]);
        assert_eq!(writes(view, end, true).unwrap().1, pages);
        assert!(!read(&mut file), "guarded again once the read is done");
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    /// Serializes tests that drive the process-wide `ask_once` journal state
    /// (`JOURNAL.ahead` / `j.kept` / `j.asked`).
    static ASK_ONCE_TEST: Mutex<()> = Mutex::new(());

    #[test]
    fn ask_once_coalesces_ranges_and_surfaces_intervening_wrote() {
        let _serial = ASK_ONCE_TEST.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: a fresh write-watched allocation, freed after cleaning up.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                6 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        // Clear any journal entries from previous allocations at this VA.
        {
            let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
            for page in p / PAGE..(p + 6 * PAGE) / PAGE {
                j.written.remove(&page);
            }
        }
        // Baseline clear.
        assert!(writes(p, p + 6 * PAGE, false).is_some());

        // Write page 1 and page 4, then coalesce two disjoint ranges inside the allocation.
        // SAFETY: committed write-watched pages of ours.
        unsafe {
            ((p + PAGE + 4) as *mut u8).write(7);
            ((p + 4 * PAGE + 4) as *mut u8).write(9);
        }
        ask_once(&[(p, p + 2 * PAGE), (p + 4 * PAGE, p + 5 * PAGE)]);
        let (now1, found1) = writes(p, p + 2 * PAGE, false).expect("covered by ask_once");
        let (now2, found2) =
            writes(p + 4 * PAGE, p + 5 * PAGE, false).expect("covered by ask_once");
        assert_eq!(found1.len(), 1);
        assert_eq!(found2.len(), 1);
        assert_eq!(found1[0].0, p / PAGE + 1);
        assert_eq!(found2[0].0, p / PAGE + 4);
        let coalesced_gen = found1[0].1;
        assert_eq!(
            found2[0].1, coalesced_gen,
            "coalesced asks share one asking number"
        );
        assert!(coalesced_gen <= now1 && coalesced_gen <= now2);

        // An intervening submission copy into page 0 updates the journal and advances `now`.
        wrote(p + 16, 64);
        let (now3, found3) = writes(p, p + 2 * PAGE, false).expect("still covered");
        assert!(now3 > coalesced_gen);
        assert_eq!(found3.len(), 2);
        assert_eq!(found3[0].0, p / PAGE);
        assert!(found3[0].1 > coalesced_gen && found3[0].1 <= now3);
        assert_eq!(found3[1], (p / PAGE + 1, coalesced_gen));

        ask_once_end();
        // After `ask_once_end`, a fresh CPU write is seen by the kernel again.
        // SAFETY: committed write-watched page of ours.
        unsafe { ((p + 5 * PAGE + 8) as *mut u8).write(11) };
        let (now4, found4) = writes(p + 5 * PAGE, p + 6 * PAGE, false).expect("watched");
        assert!(now4 > now3);
        assert_eq!(found4, [(p / PAGE + 5, now4)]);

        {
            let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
            for page in p / PAGE..(p + 6 * PAGE) / PAGE {
                j.written.remove(&page);
            }
        }
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn a_file_read_after_the_submission_query_reaches_every_reader() {
        use std::io::{Read, Seek, Write};
        let _serial = ASK_ONCE_TEST.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: allocate a writable page and release it after all reads.
        let at = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(at, 0);
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"texel").unwrap();
        for already_announced in [false, true] {
            let (before, _) = writes(at, at + PAGE, false).unwrap();
            if already_announced {
                kernel_will_write(at, 5);
            }
            ask_once(&[(at, at + PAGE)]);
            if !already_announced {
                kernel_will_write(at, 5);
            }
            file.rewind().unwrap();
            // SAFETY: these five bytes belong to the committed allocation.
            let dest = unsafe { std::slice::from_raw_parts_mut(at as *mut u8, 5) };
            file.read_exact(dest).unwrap();
            assert_eq!(dest, b"texel");
            // An overlapped read can still have an open announcement when a
            // refresh runs. Another reader can ask after completion.
            for finished in [false, true] {
                if finished {
                    kernel_wrote(at, 5);
                }
                let (_, pages) = writes(at, at + PAGE, false).unwrap();
                assert!(
                    pages.iter().any(|&(p, w)| p == at / PAGE && w > before),
                    "already_announced={already_announced}, finished={finished}"
                );
            }
            ask_once_end();
        }
        {
            let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
            j.written.remove(&(at / PAGE));
        }
        // SAFETY: the reservation is ours.
        unsafe { VirtualFree(at as _, 0, MEM_RELEASE) };
    }

    #[test]
    fn ask_once_handles_mixed_watched_and_unwatched_ranges() {
        let _serial = ASK_ONCE_TEST.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: a fresh write-watched allocation.
        let watched = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                2 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(watched, 0);
        // A previous test may have used these virtual addresses.
        {
            let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
            for page in watched / PAGE..(watched + 2 * PAGE) / PAGE {
                j.written.remove(&page);
            }
        }
        let mut plain = vec![0u8; PAGE];
        let plain_start = plain.as_mut_ptr() as usize;

        // SAFETY: committed write-watched page of ours.
        unsafe { ((watched + 4) as *mut u8).write(3) };
        ask_once(&[
            (watched, watched + 2 * PAGE),
            (plain_start, plain_start + PAGE),
        ]);
        let (now, found) = writes(watched, watched + 2 * PAGE, false).expect("watched range");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, watched / PAGE);
        assert!(found[0].1 > 0 && found[0].1 <= now);
        assert!(writes(plain_start, plain_start + PAGE, false).is_none());
        ask_once_end();

        {
            let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
            for page in watched / PAGE..(watched + 2 * PAGE) / PAGE {
                j.written.remove(&page);
            }
        }
        // SAFETY: our allocation.
        unsafe { VirtualFree(watched as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn guard_clean_fast_path_returns_empty_and_still_detects_faults() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            UnmapViewOfFile,
        };
        // SAFETY: a 4-block pagefile section mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (4 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 4 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        let initial = guard(view, view + 4 * BLOCK).expect("guarded");
        assert_eq!(initial.len(), 4);
        // All blocks clean -> fast path returns empty without re-protecting.
        assert_eq!(guard(view, view + 4 * BLOCK), Some(Vec::new()));
        // Write block 2 via fault handler -> only block 2 reported dirty.
        // SAFETY: mapped view memory; our VEH opens the block on write.
        unsafe { ((view + 2 * BLOCK + 16) as *mut u8).write(0x55) };
        assert_eq!(guard(view, view + 4 * BLOCK), Some(vec![view / PAGE + 2]));
        // Clean again immediately after re-guarding.
        assert_eq!(guard(view, view + 4 * BLOCK), Some(Vec::new()));
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            });
        }
        title_memory_remapped(view, 4 * BLOCK);
    }
}
