//! Diagnostics: a sampling profiler of the whole process (title threads,
//! our workers, the driver), switched on by `VICTORY_PROFILE=<ms>` (the
//! sampling period of busy threads, 4 by default; idle threads are looked at
//! every 25 periods). Every 5 s it logs, per thread, the CPU time spent and
//! the most frequent call stacks (module!function+offset, innermost first),
//! so a frame's time can be told apart: title code, write watching,
//! translation, driver, or waiting (and where).
//!
//! A sampled thread is suspended for a few microseconds; nothing is
//! allocated or locked while it is. The title's own code shows as
//! `module+rva` (no symbols), ours and the system's by name.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use windows_sys::Win32::Foundation::{CloseHandle, FILETIME, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Diagnostics::Debug::{
    CONTEXT, GetThreadContext, RtlLookupFunctionEntry, RtlVirtualUnwind, SYMBOL_INFO, SymFromAddr,
    SymInitialize, SymSetOptions,
};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows_sys::Win32::System::LibraryLoader::GetModuleFileNameW;
use windows_sys::Win32::System::Threading::{
    GetCurrentProcess, GetCurrentProcessId, GetCurrentThreadId, GetThreadDescription,
    GetThreadTimes, OpenThread, ResumeThread, SuspendThread,
};

const CONTEXT_FULL: u32 = 0x10_0003;
const THREAD_SUSPEND_RESUME: u32 = 0x2;
const THREAD_GET_CONTEXT: u32 = 0x8;
const THREAD_QUERY_INFORMATION: u32 = 0x40;
const DEPTH: usize = 8;
const REPORT: Duration = Duration::from_secs(5);
/// A thread using at least this much CPU per second is "busy".
const BUSY_MS: u64 = 30;

type Stack = [u64; DEPTH];

struct Tracked {
    handle: HANDLE,
    name: String,
    last_cpu_100ns: u64,
    cpu_ms_second: u64,
    cpu_ms_report: u64,
    samples: u64,
    stacks: HashMap<Stack, u32>,
}

/// Starts the profiler thread when `VICTORY_PROFILE` is set; once.
pub(crate) fn start() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    let Some(period) = std::env::var("VICTORY_PROFILE")
        .ok()
        .map(|v| v.parse::<u64>().unwrap_or(4).max(1))
    else {
        return;
    };
    ONCE.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("victory-prof".into())
            .spawn(move || run(Duration::from_millis(period)));
    });
}

fn filetime(t: &FILETIME) -> u64 {
    u64::from(t.dwHighDateTime) << 32 | u64::from(t.dwLowDateTime)
}

fn thread_ids() -> Vec<u32> {
    let mut ids = Vec::new();
    // SAFETY: a snapshot of our own process's threads, walked and closed.
    unsafe {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snap == INVALID_HANDLE_VALUE {
            return ids;
        }
        let mut e: THREADENTRY32 = std::mem::zeroed();
        e.dwSize = size_of::<THREADENTRY32>() as u32;
        let pid = GetCurrentProcessId();
        let mut more = Thread32First(snap, &mut e) != 0;
        while more {
            if e.th32OwnerProcessID == pid {
                ids.push(e.th32ThreadID);
            }
            more = Thread32Next(snap, &mut e) != 0;
        }
        CloseHandle(snap);
    }
    ids
}

fn thread_name(handle: HANDLE, tid: u32) -> String {
    use windows_sys::Win32::Foundation::LocalFree;
    let mut text: *mut u16 = std::ptr::null_mut();
    // SAFETY: an open thread handle and an out pointer; the text is
    // allocated by the call and freed below.
    unsafe {
        if GetThreadDescription(handle, &mut text) >= 0 && !text.is_null() {
            let mut n = 0;
            while *text.add(n) != 0 {
                n += 1;
            }
            let name = String::from_utf16_lossy(std::slice::from_raw_parts(text, n));
            LocalFree(text.cast());
            if !name.is_empty() {
                return format!("{name}#{tid}");
            }
        }
    }
    format!("thread#{tid}")
}

fn cpu_100ns(handle: HANDLE) -> u64 {
    let mut t: [FILETIME; 4] = [FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    }; 4];
    // SAFETY: an open thread handle and four time structures.
    unsafe {
        GetThreadTimes(handle, &mut t[0], &mut t[1], &mut t[2], &mut t[3]);
    }
    filetime(&t[2]) + filetime(&t[3])
}

/// `CONTEXT` with the 16-byte alignment the x64 entry points write it with
/// (`GetThreadContext`, `RtlVirtualUnwind`); windows-sys declares `CONTEXT`
/// 8-aligned, so a plain local can sit 8 bytes off and every capture then
/// fails silently.
#[repr(align(16))]
struct Aligned16(CONTEXT);

/// Whether `[at, at + n)` lies inside one committed page run that is
/// readable, so a read of it cannot fault.
fn readable(at: usize, n: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    if at.checked_add(n).is_none() {
        return false;
    }
    // SAFETY: plain queries with our buffers; `zeroed` is all-bits-zero valid
    // for this structure.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a plain query with our buffer.
    let got = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    if got == 0 {
        return false;
    }
    let base = info.BaseAddress as usize;
    let protect = info.Protect as u32;
    info.State == MEM_COMMIT
        && protect & (PAGE_GUARD | PAGE_NOACCESS) == 0
        && at >= base
        && at + n <= base.saturating_add(info.RegionSize)
}

/// The call stack of a thread, innermost first, zero-padded. The thread is
/// suspended meanwhile; nothing here allocates or locks.
fn capture(handle: HANDLE) -> Option<Stack> {
    let mut stack = [0u64; DEPTH];
    // SAFETY: an open thread handle with the rights; the context is ours;
    // the unwinding reads the thread's stack, which is stopped.
    unsafe {
        if SuspendThread(handle) == u32::MAX {
            return None;
        }
        let mut aligned = Aligned16(std::mem::zeroed());
        let ctx = &mut aligned.0;
        ctx.ContextFlags = CONTEXT_FULL;
        if GetThreadContext(handle, ctx) != 0 {
            for slot in &mut stack {
                if ctx.Rip == 0 {
                    break;
                }
                *slot = ctx.Rip;
                let mut base = 0u64;
                let entry = RtlLookupFunctionEntry(ctx.Rip, &mut base, std::ptr::null_mut());
                if entry.is_null() {
                    // A leaf without unwind data: the return address is on top.
                    // After a bad unwind `Rsp` can point anywhere, so the read
                    // only happens inside a committed, guard-free page.
                    let rsp = ctx.Rsp as usize;
                    if !readable(rsp, 8) {
                        break;
                    }
                    ctx.Rip = *(rsp as *const u64);
                    ctx.Rsp += 8;
                } else {
                    let mut data = std::ptr::null_mut();
                    let mut frame = 0u64;
                    RtlVirtualUnwind(
                        0,
                        base,
                        ctx.Rip,
                        entry,
                        ctx,
                        &mut data,
                        &mut frame,
                        std::ptr::null_mut(),
                    );
                }
            }
        }
        ResumeThread(handle);
    }
    (stack[0] != 0).then_some(stack)
}

/// `module!function+offset`, or `module+rva`.
fn describe(addr: u64, cache: &mut HashMap<u64, String>) -> String {
    if let Some(s) = cache.get(&addr) {
        return s.clone();
    }
    let text = symbolize(addr);
    cache.insert(addr, text.clone());
    text
}

fn symbolize(addr: u64) -> String {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain queries with our buffers.
    unsafe {
        let mut base = 0u64;
        RtlLookupFunctionEntry(addr, &mut base, std::ptr::null_mut());
        let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        VirtualQuery(addr as *const _, &mut info, size_of_val(&info));
        let region = if base != 0 {
            base
        } else {
            info.AllocationBase as u64
        };
        let mut path = [0u16; 260];
        let n = GetModuleFileNameW(region as *mut _, path.as_mut_ptr(), 260) as usize;
        let module = String::from_utf16_lossy(&path[..n]);
        let module = module.rsplit('\\').next().unwrap_or("").to_string();
        let module = if module.is_empty() {
            format!("{region:#x}")
        } else {
            module
        };
        let mut buf = vec![0u8; size_of::<SYMBOL_INFO>() + 256];
        let sym = buf.as_mut_ptr().cast::<SYMBOL_INFO>();
        (*sym).SizeOfStruct = size_of::<SYMBOL_INFO>() as u32;
        (*sym).MaxNameLen = 255;
        let mut displacement = 0u64;
        if SymFromAddr(GetCurrentProcess(), addr, &mut displacement, sym) != 0 {
            let name = std::slice::from_raw_parts(
                (*sym).Name.as_ptr().cast::<u8>(),
                (*sym).NameLen as usize,
            );
            let name = String::from_utf8_lossy(name);
            let short: String = name.chars().take(90).collect();
            return format!("{module}!{short}+{displacement:#x}");
        }
        format!("{module}+{:#x}", addr - region)
    }
}

fn run(period: Duration) {
    // SAFETY: symbol handling for our own process, used from this thread only.
    // No invade: loading every module's symbols at startup is what made this
    // unsafe to run beside anything else using dbghelp.
    unsafe {
        SymSetOptions(0x2 | 0x4); // SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS
        SymInitialize(GetCurrentProcess(), std::ptr::null(), 0);
    }
    // SAFETY: no arguments and no failure: the calling thread's own id.
    let me = unsafe { GetCurrentThreadId() };
    let mut threads: HashMap<u32, Tracked> = HashMap::new();
    let mut cache = HashMap::new();
    let (mut refreshed, mut second, mut reported) = (
        Instant::now() - Duration::from_secs(9),
        Instant::now(),
        Instant::now(),
    );
    let mut tick = 0u64;
    tracing::info!(period_ms = period.as_millis() as u64, "profiler started");
    loop {
        std::thread::sleep(period);
        tick += 1;
        if refreshed.elapsed() >= Duration::from_secs(1) {
            refreshed = Instant::now();
            let ids = thread_ids();
            threads.retain(|tid, t| {
                let keep = ids.contains(tid);
                if !keep {
                    // SAFETY: a handle we opened.
                    unsafe { CloseHandle(t.handle) };
                }
                keep
            });
            for tid in ids {
                if tid == me || threads.contains_key(&tid) {
                    continue;
                }
                // SAFETY: opening a thread of our process.
                let handle = unsafe {
                    OpenThread(
                        THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                        0,
                        tid,
                    )
                };
                if handle.is_null() {
                    continue;
                }
                threads.insert(
                    tid,
                    Tracked {
                        handle,
                        name: thread_name(handle, tid),
                        last_cpu_100ns: cpu_100ns(handle),
                        cpu_ms_second: BUSY_MS,
                        cpu_ms_report: 0,
                        samples: 0,
                        stacks: HashMap::new(),
                    },
                );
            }
        }
        if second.elapsed() >= Duration::from_secs(1) {
            second = Instant::now();
            for t in threads.values_mut() {
                let now = cpu_100ns(t.handle);
                t.cpu_ms_second = (now - t.last_cpu_100ns.min(now)) / 10_000;
                t.cpu_ms_report += t.cpu_ms_second;
                t.last_cpu_100ns = now;
            }
        }
        for t in threads.values_mut() {
            if t.cpu_ms_second < BUSY_MS && !tick.is_multiple_of(25) {
                continue;
            }
            if let Some(stack) = capture(t.handle) {
                t.samples += 1;
                *t.stacks.entry(stack).or_insert(0) += 1;
            }
        }
        if reported.elapsed() >= REPORT {
            let secs = reported.elapsed().as_secs_f64();
            reported = Instant::now();
            let mut list: Vec<&mut Tracked> =
                threads.values_mut().filter(|t| t.samples > 0).collect();
            list.sort_by_key(|t| std::cmp::Reverse((t.cpu_ms_report, t.samples)));
            for t in list.into_iter().take(10) {
                tracing::info!(
                    "prof {}: cpu {:.0}% ({} ms), {} samples",
                    t.name,
                    t.cpu_ms_report as f64 / 10.0 / secs,
                    t.cpu_ms_report,
                    t.samples
                );
                let mut stacks: Vec<_> = std::mem::take(&mut t.stacks).into_iter().collect();
                stacks.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
                for (stack, n) in stacks.into_iter().take(4) {
                    let frames: Vec<String> = stack
                        .iter()
                        .take_while(|&&a| a != 0)
                        .map(|&a| describe(a, &mut cache))
                        .collect();
                    tracing::info!("prof   {:>4}x {}", n, frames.join(" < "));
                }
                (t.cpu_ms_report, t.samples) = (0, 0);
            }
            for t in threads.values_mut() {
                (t.cpu_ms_report, t.samples) = (0, 0);
                t.stacks.clear();
            }
        }
    }
}
