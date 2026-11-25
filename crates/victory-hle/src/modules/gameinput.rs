//! `gameinput.dll`: input devices from the host. Gamepads come from XInput
//! (host slots 0-3); while none is connected, the keyboard drives one
//! stand-in gamepad (`FALLBACK_BUTTONS`). The keyboard is a device of its own
//! too. A poll thread turns host state into readings (a new one per change)
//! and runs device callbacks on connect and disconnect.
#![allow(non_snake_case)]
// Title code calls these through the objects, passing pointers the real API
// would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, LazyLock, Mutex, MutexGuard, PoisonError};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use super::release_ref;
use super::xasync::{E_INVALIDARG, E_NOTIMPL};
use super::xgameruntime::{E_NOINTERFACE, E_POINTER, Guid, HRESULT, IID_IUNKNOWN, S_OK};
use crate::{HleExport, HleModule};

/// The documented errors.
pub const DEVICE_DISCONNECTED: HRESULT = 0x838A_0001_u32 as i32;
pub const DEVICE_NOT_FOUND: HRESULT = 0x838A_0002_u32 as i32;
pub const READING_NOT_FOUND: HRESULT = 0x838A_0003_u32 as i32;
pub const REFERENCE_TOO_OLD: HRESULT = 0x838A_0004_u32 as i32;

/// Public GameInput v0 interface ids (GDK `GameInput.h`).
const IID_IGAMEINPUT: Guid = Guid::new(
    0x11BE_2A7E,
    0x4254,
    0x445A,
    [0x9C, 0x09, 0xFF, 0xC4, 0x0F, 0x00, 0x69, 0x18],
);
const IID_IGAMEINPUT_DEVICE: Guid = Guid::new(
    0x31DD_86FB,
    0x4C1B,
    0x408A,
    [0x86, 0x8F, 0x43, 0x9B, 0x3C, 0xD4, 0x71, 0x25],
);
const IID_IGAMEINPUT_READING: Guid = Guid::new(
    0x2156_947A,
    0xE1FA,
    0x4DE0,
    [0xA3, 0x0B, 0xD8, 0x12, 0x93, 0x1D, 0xBD, 0x8D],
);

/// Input kinds (bits) our devices have.
const KEYBOARD_KIND: u32 = 0x10;
const GAMEPAD_KIND: u32 = 0x4_0000;
/// Device status of a connected device: connected, input enabled.
const STATUS_CONNECTED: u32 = 0x3;
const ASYNC_ENUMERATION: u32 = 1;

/// Gamepad buttons; the title maps each bit to one of its pad buttons.
const MENU: u32 = 0x1;
const VIEW: u32 = 0x2;
const A: u32 = 0x4;
const B: u32 = 0x8;
const X: u32 = 0x10;
const Y: u32 = 0x20;
const UP: u32 = 0x40;
const DOWN: u32 = 0x80;
const LEFT: u32 = 0x100;
const RIGHT: u32 = 0x200;
const LEFT_SHOULDER: u32 = 0x400;
const RIGHT_SHOULDER: u32 = 0x800;
const LEFT_STICK: u32 = 0x1000;
const RIGHT_STICK: u32 = 0x2000;

/// Devices: host pad slots, the keyboard's stand-in pad, the keyboard.
const PADS: usize = 4;
const FALLBACK: usize = PADS;
const KEYBOARD: usize = PADS + 1;
const DEVICES: usize = PADS + 2;

/// Readings kept per device for next/previous lookups.
const HISTORY: usize = 32;
const POLL: Duration = Duration::from_millis(8);

/// The keyboard's stand-in gamepad (virtual key, button); README lists it.
/// Sticks: W/A/S/D left, I/J/K/L right; triggers Z and C.
const FALLBACK_BUTTONS: [(u8, u32); 15] = [
    (0x1B, MENU), // Esc
    (0x09, VIEW), // Tab
    (0x0D, A),    // Enter
    (0x08, B),    // Backspace
    (b'O', B),    // Circle, for PlayStation players
    (b'X', X),
    (b'Y', Y),
    (0x26, UP), // arrows
    (0x28, DOWN),
    (0x25, LEFT),
    (0x27, RIGHT),
    (b'Q', LEFT_SHOULDER),
    (b'E', RIGHT_SHOULDER),
    (b'F', LEFT_STICK),
    (b'H', RIGHT_STICK),
];

/// A gamepad reading: triggers 0 to 1, sticks -1 to 1 (up positive).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PadState {
    buttons: u32,
    left_trigger: f32,
    right_trigger: f32,
    left_x: f32,
    left_y: f32,
    right_x: f32,
    right_y: f32,
}

/// Motor speeds for one pad, each 0 (off) to `u16::MAX` (full), in the
/// order of the title's four rumble levels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Motors {
    /// Low frequency, in the left grip.
    low: u16,
    /// High frequency, in the right grip.
    high: u16,
    left_trigger: u16,
    right_trigger: u16,
}

/// One key held down: scan code (0xE0xx when extended), character, key.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct KeyState {
    scan_code: u32,
    code_point: u32,
    virtual_key: u8,
    dead_key: bool,
}

const _: () = assert!(size_of::<KeyState>() == 12 && size_of::<PadState>() == 28);

/// Host input, or a fake in tests.
trait Source: Send {
    /// The pad in host slot `slot`, `None` when the slot is empty.
    fn pad(&mut self, slot: usize) -> Option<PadState>;
    /// Keys held down; none while the title's window lacks focus.
    fn keys(&mut self) -> Vec<KeyState>;
    /// Virtual keys `VICTORY_KEYS` holds down now (diagnostics builds only);
    /// `keys` reports them too.
    fn scripted(&mut self) -> Vec<u8> {
        Vec::new()
    }
    /// Motor speeds for the pad in host slot `slot`; a host without
    /// trigger motors drops those two.
    fn rumble(&mut self, slot: usize, motors: Motors);
}

fn fallback(keys: &[KeyState]) -> PadState {
    let on = |vk: u8| {
        if keys.iter().any(|k| k.virtual_key == vk) {
            1.0
        } else {
            0.0
        }
    };
    PadState {
        buttons: FALLBACK_BUTTONS
            .iter()
            .filter(|(vk, _)| on(*vk) != 0.0)
            .fold(0, |all, (_, bit)| all | bit),
        left_trigger: on(b'Z'),
        right_trigger: on(b'C'),
        left_x: on(b'D') - on(b'A'),
        left_y: on(b'W') - on(b'S'),
        right_x: on(b'L') - on(b'J'),
        right_y: on(b'I') - on(b'K'),
    }
}

/// `pad` with the stand-in pad's reading of `keys` pressed on top of it.
fn press(pad: PadState, keys: &[KeyState]) -> PadState {
    let extra = fallback(keys);
    let stick = |a: f32, b: f32| (a + b).clamp(-1.0, 1.0);
    PadState {
        buttons: pad.buttons | extra.buttons,
        left_trigger: pad.left_trigger.max(extra.left_trigger),
        right_trigger: pad.right_trigger.max(extra.right_trigger),
        left_x: stick(pad.left_x, extra.left_x),
        left_y: stick(pad.left_y, extra.left_y),
        right_x: stick(pad.right_x, extra.right_x),
        right_y: stick(pad.right_y, extra.right_y),
    }
}

#[repr(C)]
struct Table<const N: usize>([*const (); N]);
// SAFETY: function addresses only.
unsafe impl<const N: usize> Sync for Table<N> {}

/// The input object; one per hub.
#[repr(C)]
pub struct Object {
    vtable: &'static Table<22>,
    hub: *const Hub,
}

#[repr(C)]
pub struct Device {
    vtable: &'static Table<21>,
    hub: *const Hub,
    index: usize,
    info: DeviceInfo,
}

/// What a device says about itself. The title reads `id` (+0x20); the rest
/// follows the public layout and is zero where we have nothing to say.
#[repr(C)]
#[derive(Default)]
#[allow(dead_code)] // read by title code
struct DeviceInfo {
    size: u32,
    vendor: u16,
    product: u16,
    revision: u16,
    interface: u8,
    collection: u8,
    usage: [u16; 2],
    versions: [u16; 8],
    id: [u8; 32],
    root_id: [u8; 32],
    family: i32,
    capabilities: u32,
    supported_input: u32,
    rumble_motors: u32,
    /// Counts and optional detail pointers.
    rest: [u64; 26],
}

const _: () = assert!(std::mem::offset_of!(DeviceInfo, id) == 0x20);

fn kind(index: usize) -> u32 {
    if index == KEYBOARD {
        KEYBOARD_KIND
    } else {
        GAMEPAD_KIND
    }
}

impl Device {
    fn new(index: usize) -> Device {
        let mut id = [0; 32];
        id[..7].copy_from_slice(b"victory");
        id[31] = index as u8 + 1;
        let info = DeviceInfo {
            size: size_of::<DeviceInfo>() as u32,
            id,
            root_id: id,
            supported_input: kind(index),
            // Low and high frequency.
            rumble_motors: if index < PADS { 0x3 } else { 0 },
            ..Default::default()
        };
        Device {
            vtable: &DEVICE,
            hub: std::ptr::null(),
            index,
            info,
        }
    }

    fn hub(&self) -> &'static Hub {
        // SAFETY: set by `Hub::new`; hubs are never freed.
        unsafe { &*self.hub }
    }
}

#[derive(PartialEq)]
enum Value {
    Pad(PadState),
    Keys(Vec<KeyState>),
}

#[repr(C)]
pub struct Reading {
    vtable: &'static Table<25>,
    refs: AtomicU32,
    device: *const Device,
    sequence: u64,
    timestamp: u64,
    value: Value,
}

impl Reading {
    fn kind(&self) -> u32 {
        match self.value {
            Value::Pad(_) => GAMEPAD_KIND,
            Value::Keys(_) => KEYBOARD_KIND,
        }
    }
}

/// A counted reference to a reading.
struct Ref(NonNull<Reading>);

impl Ref {
    fn new(reading: Reading) -> Ref {
        Ref(NonNull::from(Box::leak(Box::new(reading))))
    }

    /// Another reference, for the title.
    fn share(&self) -> *const Reading {
        reading_add_ref(self.0.as_ptr());
        self.0.as_ptr()
    }
}

impl Deref for Ref {
    type Target = Reading;
    fn deref(&self) -> &Reading {
        // SAFETY: we hold a reference, so it is alive.
        unsafe { self.0.as_ref() }
    }
}

impl Drop for Ref {
    fn drop(&mut self) {
        reading_release(self.0.as_ptr());
    }
}

type DeviceCallback = extern "system" fn(u64, *mut c_void, *const Device, u64, u32, u32);

#[derive(Clone, Copy)]
struct Registration {
    token: u64,
    /// One device's address, or 0 for any.
    device: usize,
    kind: u32,
    status: u32,
    context: usize,
    callback: DeviceCallback,
}

impl Registration {
    fn matches(&self, devices: &[Device; DEVICES], index: usize) -> bool {
        kind(index) & self.kind != 0
            && (self.device == 0 || self.device == &devices[index] as *const Device as usize)
    }

    fn event(&self, device: usize, timestamp: u64, current: u32, previous: u32) -> Event {
        Event {
            registration: *self,
            device,
            timestamp,
            current,
            previous,
        }
    }
}

struct Event {
    registration: Registration,
    device: usize,
    timestamp: u64,
    current: u32,
    previous: u32,
}

#[derive(Default)]
struct Slot {
    connected: bool,
    sequence: u64,
    history: VecDeque<Ref>,
}

struct State {
    slots: [Slot; DEVICES],
    callbacks: Vec<Registration>,
    /// Asynchronous enumerations, delivered by the next poll.
    pending: Vec<Event>,
    /// Device callbacks running now: token and thread.
    running: Vec<(u64, ThreadId)>,
    /// Events waiting for the single delivery loop.
    queue: VecDeque<Event>,
    /// The one thread draining `queue`; other producers only append.
    deliverer: Option<ThreadId>,
    /// The scripted keys of the last poll, to log each new press once.
    scripted: Vec<u8>,
}

pub struct Hub {
    object: Object,
    devices: [Device; DEVICES],
    /// Apart from the state: XInput can be slow, and the title reads
    /// readings every frame.
    source: Mutex<Box<dyn Source>>,
    state: Mutex<State>,
    /// A device callback returned.
    returned: Condvar,
}

fn lock<T: ?Sized>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

// SAFETY: the back pointers are set before the hub is shared and never
// change; what changes sits behind the mutex or is atomic.
unsafe impl Send for Hub {}
// SAFETY: as above.
unsafe impl Sync for Hub {}

static START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Microseconds, the clock of every timestamp we hand out.
fn now() -> u64 {
    START.elapsed().as_micros() as u64
}

impl Hub {
    fn new(source: Box<dyn Source>) -> &'static Hub {
        let hub = Box::into_raw(Box::new(Hub {
            object: Object {
                vtable: &INPUT,
                hub: std::ptr::null(),
            },
            devices: std::array::from_fn(Device::new),
            source: Mutex::new(source),
            state: Mutex::new(State {
                slots: Default::default(),
                callbacks: Vec::new(),
                pending: Vec::new(),
                running: Vec::new(),
                queue: VecDeque::new(),
                deliverer: None,
                scripted: Vec::new(),
            }),
            returned: Condvar::new(),
        }));
        // SAFETY: just allocated and never freed; nothing else sees it yet.
        unsafe {
            (*hub).object.hub = hub;
            for device in &mut (*hub).devices {
                device.hub = hub;
            }
            &*hub
        }
    }

    /// A hub over the host's input, polled from now on.
    fn start(source: Box<dyn Source>) -> &'static Hub {
        let hub = Hub::new(source);
        hub.poll();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(POLL);
                hub.poll();
            }
        });
        hub
    }

    fn state(&self) -> MutexGuard<'_, State> {
        lock(&self.state)
    }

    /// Reads the host once; device callbacks run on the calling thread.
    fn poll(&self) {
        let (pads, keys, scripted) = {
            let mut source = lock(&self.source);
            let pads: [_; PADS] = std::array::from_fn(|slot| source.pad(slot));
            (pads, source.keys(), source.scripted())
        };
        let events = self.state().poll(&self.devices, pads, keys, scripted);
        self.deliver(events);
    }

    /// Runs callbacks one at a time, in the order they were queued, never
    /// under our lock: they call back into us. A callback delivered from a
    /// callback joins the same queue instead of nesting. One unregistered
    /// since the event was queued does not run, and `UnregisterCallback`
    /// waits for one that runs.
    fn deliver(&self, events: Vec<Event>) {
        let me = std::thread::current().id();
        {
            let mut state = self.state();
            state.queue.extend(events);
            if state.deliverer.is_some() {
                return;
            }
            state.deliverer = Some(me);
        }
        loop {
            let e = {
                let mut state = self.state();
                match state.queue.pop_front() {
                    Some(e) => e,
                    None => {
                        state.deliverer = None;
                        return;
                    }
                }
            };
            self.run(e, me);
        }
    }

    /// A blocking enumeration: its callbacks have all run on the calling
    /// thread before this returns. The title registers holding a lock its
    /// callback takes again, so they must not be handed to (and waited for
    /// on) another thread that is delivering: that thread would block on the
    /// lock this one holds. They run here, alongside that thread's current
    /// callback. Called from a callback on the delivering thread, they queue
    /// behind it as `deliver` does.
    fn deliver_blocking(&self, events: Vec<Event>) {
        let me = std::thread::current().id();
        let elsewhere = self.state().deliverer.is_some_and(|t| t != me);
        if !elsewhere {
            return self.deliver(events);
        }
        for e in events {
            self.run(e, me);
        }
    }

    /// Runs one event's callback on this thread, never under our lock,
    /// unless its registration was dropped since the event was made.
    fn run(&self, e: Event, me: ThreadId) {
        let r = e.registration;
        {
            let mut state = self.state();
            if !state.callbacks.iter().any(|c| c.token == r.token) {
                return;
            }
            state.running.push((r.token, me));
        }
        (r.callback)(
            r.token,
            r.context as *mut c_void,
            &self.devices[e.device],
            e.timestamp,
            e.current,
            e.previous,
        );
        let mut state = self.state();
        if let Some(i) = state.running.iter().position(|&x| x == (r.token, me)) {
            state.running.swap_remove(i);
        }
        self.returned.notify_all();
    }

    fn index(&self, device: *const Device) -> Option<usize> {
        self.devices.iter().position(|d| std::ptr::eq(d, device))
    }
}

impl State {
    fn poll(
        &mut self,
        devices: &[Device; DEVICES],
        mut pads: [Option<PadState>; PADS],
        keys: Vec<KeyState>,
        scripted: Vec<u8>,
    ) -> Vec<Event> {
        let now = now();
        let mut events = std::mem::take(&mut self.pending);
        let any_pad = pads.iter().any(Option::is_some);
        // Scripted keys press the pad the title reads. With no host pad that
        // is the stand-in, which has them from `keys`. A host pad hides the
        // stand-in (a receiver answers for its switched-off pad too), so
        // they press the first one.
        let target = pads.iter().position(Option::is_some).unwrap_or(FALLBACK);
        if !scripted.is_empty()
            && let Some(pad) = pads.get_mut(target).and_then(Option::as_mut)
        {
            let keys: Vec<KeyState> = scripted
                .iter()
                .map(|&virtual_key| KeyState {
                    virtual_key,
                    ..Default::default()
                })
                .collect();
            *pad = press(*pad, &keys);
        }
        if scripted.iter().any(|vk| !self.scripted.contains(vk)) {
            tracing::info!(
                virtual_keys = format_args!("{scripted:02x?}"),
                device = target,
                "scripted keys pressed"
            );
        }
        self.scripted = scripted;
        for (slot, pad) in pads.into_iter().enumerate() {
            self.update(devices, slot, pad.map(Value::Pad), now, &mut events);
        }
        let fallback = (!any_pad).then(|| Value::Pad(fallback(&keys)));
        self.update(devices, FALLBACK, fallback, now, &mut events);
        self.update(devices, KEYBOARD, Some(Value::Keys(keys)), now, &mut events);
        events
    }

    /// Records a device's state: a reading when it changed, callbacks when
    /// the device came or went.
    fn update(
        &mut self,
        devices: &[Device; DEVICES],
        index: usize,
        value: Option<Value>,
        now: u64,
        events: &mut Vec<Event>,
    ) {
        let slot = &mut self.slots[index];
        let was = slot.connected;
        slot.connected = value.is_some();
        if let Some(value) = value
            && (!was || slot.history.back().is_none_or(|r| r.value != value))
        {
            slot.sequence += 1;
            if slot.history.len() == HISTORY {
                slot.history.pop_front();
            }
            slot.history.push_back(Ref::new(Reading {
                vtable: &READING,
                refs: AtomicU32::new(1),
                device: &devices[index],
                sequence: slot.sequence,
                timestamp: now,
                value,
            }));
        }
        if was != slot.connected {
            tracing::info!(device = index, connected = !was, "input device");
            let (current, previous) = if was {
                (0, STATUS_CONNECTED)
            } else {
                (STATUS_CONNECTED, 0)
            };
            events.extend(
                self.callbacks
                    .iter()
                    .filter(|r| r.matches(devices, index) && r.status & STATUS_CONNECTED != 0)
                    .map(|r| r.event(index, now, current, previous)),
            );
        }
    }
}

#[cfg(windows)]
fn host() -> Box<dyn Source> {
    Box::new(host::Host::default())
}

#[cfg(not(windows))]
fn host() -> Box<dyn Source> {
    struct Nothing;
    impl Source for Nothing {
        fn pad(&mut self, _: usize) -> Option<PadState> {
            None
        }
        fn keys(&mut self) -> Vec<KeyState> {
            Vec::new()
        }
        fn rumble(&mut self, _: usize, _: Motors) {}
    }
    Box::new(Nothing)
}

static HUB: LazyLock<&'static Hub> = LazyLock::new(|| Hub::start(host()));

type This = *const Object;
type Out = *mut *const c_void;

/// The DLL's only export (ordinal 1): the input object, static and shared.
pub extern "system" fn GameInputCreate(out: *mut This) -> HRESULT {
    tracing::info!("GameInputCreate");
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null; the title passes room for a pointer.
    unsafe { out.write(&HUB.object) };
    S_OK
}

fn hub(this: This) -> &'static Hub {
    // SAFETY: `this` is a hub's object; hubs are never freed.
    unsafe { &*(*this).hub }
}

/// Clears a COM out pointer and returns `hr`.
fn fail(out: Out, hr: HRESULT) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a pointer.
        unsafe { out.write(std::ptr::null()) };
    }
    hr
}

/// Gives the title a reference to `reading`, or fails as with none.
fn hand(out: Out, reading: Option<&Ref>) -> HRESULT {
    match reading {
        _ if out.is_null() => E_INVALIDARG,
        Some(reading) => {
            // SAFETY: checked non-null; the title passes room for a pointer.
            unsafe { out.write(reading.share().cast()) };
            S_OK
        }
        None => fail(out, READING_NOT_FOUND),
    }
}

/// `QueryInterface` for an input object: `IID_IUnknown` and the object's own
/// interface id succeed (`AddRef`, `S_OK`); anything else clears `*out`.
fn query_iid<T>(
    this: *const T,
    iid: *const Guid,
    out: Out,
    retain: extern "system" fn(*const T) -> u32,
    self_iid: &Guid,
) -> HRESULT {
    if iid.is_null() || out.is_null() {
        return fail(out, E_POINTER);
    }
    // SAFETY: the caller passes a readable IID, checked non-null above.
    let asked = unsafe { *iid };
    if asked != IID_IUNKNOWN && asked != *self_iid {
        return fail(out, E_NOINTERFACE);
    }
    retain(this);
    // SAFETY: checked non-null; the caller passes room for a pointer.
    unsafe { out.write(this.cast()) };
    S_OK
}

extern "system" fn query_interface(this: *const c_void, iid: *const Guid, out: Out) -> HRESULT {
    query_iid(this, iid, out, add_ref, &IID_IGAMEINPUT)
}
extern "system" fn device_query(this: *const c_void, iid: *const Guid, out: Out) -> HRESULT {
    query_iid(this, iid, out, add_ref, &IID_IGAMEINPUT_DEVICE)
}
/// The input object and devices live as long as the process.
extern "system" fn add_ref(_: *const c_void) -> u32 {
    1
}

extern "system" fn timestamp(_: This) -> u64 {
    now()
}

extern "system" fn current_reading(
    this: This,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    let hub = hub(this);
    let state = hub.state();
    let newest = |i: usize| {
        state.slots[i]
            .history
            .back()
            .filter(|r| r.kind() & kind != 0)
    };
    let found = if device.is_null() {
        (0..DEVICES)
            .filter(|&i| state.slots[i].connected)
            .filter_map(newest)
            .max_by_key(|r| r.timestamp)
    } else {
        match hub.index(device) {
            None => return fail(out, E_INVALIDARG),
            Some(i) if !state.slots[i].connected => return fail(out, DEVICE_DISCONNECTED),
            Some(i) => newest(i),
        }
    };
    tracing::trace!(
        kind = format_args!("{kind:#x}"),
        device = ?hub.index(device),
        sequence = found.map(|r| r.sequence),
        "current reading"
    );
    hand(out, found)
}

/// The reading after (`forward`) or before `reference`: on `device`, or with
/// none given on any device, in time order (ties by device, then sequence).
fn relative(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
    forward: bool,
) -> HRESULT {
    if reference.is_null() {
        return fail(out, E_INVALIDARG);
    }
    let hub = hub(this);
    // SAFETY: a reading we handed out and the title still holds.
    let reference = unsafe { &*reference };
    let Some(index) = hub.index(reference.device) else {
        return fail(out, E_INVALIDARG);
    };
    if !device.is_null() && !std::ptr::eq(device, reference.device) {
        return fail(out, E_INVALIDARG);
    }
    let state = hub.state();
    if state.slots[index]
        .history
        .front()
        .is_none_or(|oldest| reference.sequence < oldest.sequence)
    {
        return fail(out, REFERENCE_TOO_OLD);
    }
    // On one device this order is its sequence order.
    let at = (reference.timestamp, index, reference.sequence);
    let devices = if device.is_null() {
        0..DEVICES
    } else {
        index..index + 1
    };
    let candidates = devices
        .flat_map(|i| {
            let history = &state.slots[i].history;
            history
                .iter()
                .map(move |r| ((r.timestamp, i, r.sequence), r))
        })
        .filter(|(_, r)| r.kind() & kind != 0);
    let found = if forward {
        candidates
            .filter(|(key, _)| *key > at)
            .min_by_key(|(key, _)| *key)
    } else {
        candidates
            .filter(|(key, _)| *key < at)
            .max_by_key(|(key, _)| *key)
    };
    hand(out, found.map(|(_, r)| r))
}
extern "system" fn next_reading(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    relative(this, reference, kind, device, out, true)
}
extern "system" fn previous_reading(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    relative(this, reference, kind, device, out, false)
}
/// The device's reading at `time`, among those still kept.
extern "system" fn temporal_reading(
    this: This,
    time: u64,
    device: *const Device,
    out: Out,
) -> HRESULT {
    let hub = hub(this);
    let Some(index) = hub.index(device) else {
        return fail(out, E_INVALIDARG);
    };
    let state = hub.state();
    let history = &state.slots[index].history;
    hand(out, history.iter().rev().find(|r| r.timestamp <= time))
}

static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

fn token(out: *mut u64) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null.
        unsafe { out.write(NEXT_TOKEN.fetch_add(1, Ordering::Relaxed)) };
    }
    S_OK
}

// shortcut: reading callbacks never run; the title registers none
// (it polls).
extern "system" fn register_reading(
    _: This,
    _device: *const c_void,
    _kind: u32,
    _threshold: f32,
    _context: *mut c_void,
    _callback: *const c_void,
    out: *mut u64,
) -> HRESULT {
    tracing::warn!("input reading callbacks are never called");
    token(out)
}

/// Device callbacks run on our poll thread when a device comes or goes.
/// Enumeration reports the devices already there: blocking on the caller's
/// thread before we return (the title registers holding a lock its callback
/// takes again), asynchronous on the next poll.
extern "system" fn register_device(
    this: This,
    device: *const Device,
    kind: u32,
    status: u32,
    enumeration: u32,
    context: *mut c_void,
    callback: Option<DeviceCallback>,
    out: *mut u64,
) -> HRESULT {
    let Some(callback) = callback else {
        return E_INVALIDARG;
    };
    let hub = hub(this);
    let registration = Registration {
        token: NEXT_TOKEN.fetch_add(1, Ordering::Relaxed),
        device: device as usize,
        kind,
        status,
        context: context as usize,
        callback,
    };
    let mut found = Vec::new();
    {
        let mut state = hub.state();
        if enumeration != 0 && status & STATUS_CONNECTED != 0 {
            let now = now();
            found = (0..DEVICES)
                .filter(|&i| state.slots[i].connected && registration.matches(&hub.devices, i))
                .map(|i| registration.event(i, now, STATUS_CONNECTED, 0))
                .collect();
        }
        tracing::info!(
            kind = format_args!("{kind:#x}"),
            enumeration,
            devices = found.len(),
            "input device callback"
        );
        state.callbacks.push(registration);
        if enumeration == ASYNC_ENUMERATION {
            state.pending.append(&mut found);
        }
    }
    if !out.is_null() {
        // SAFETY: checked non-null.
        unsafe { out.write(registration.token) };
    }
    // A blocking enumeration's callbacks have all run, on this thread,
    // before registration returns, even while another thread is delivering.
    if enumeration != 0 && enumeration != ASYNC_ENUMERATION {
        hub.deliver_blocking(found);
    } else {
        hub.deliver(found);
    }
    S_OK
}
/// Guide button and keyboard layout callbacks: neither ever changes here.
extern "system" fn register_other(
    _: This,
    _device: *const c_void,
    _context: *mut c_void,
    _callback: *const c_void,
    out: *mut u64,
) -> HRESULT {
    token(out)
}

fn forget(this: This, token: u64) {
    let mut state = hub(this).state();
    state.callbacks.retain(|r| r.token != token);
    state.pending.retain(|e| e.registration.token != token);
}
extern "system" fn stop_callback(this: This, token: u64) {
    forget(this, token);
}
/// Stops the callback, then waits up to `timeout` µs for a call of it that
/// is running; false on time-out, or from inside that callback (it cannot
/// wait for itself), as documented. Either way it is never called again.
extern "system" fn unregister_callback(this: This, token: u64, timeout: u64) -> bool {
    forget(this, token);
    let hub = hub(this);
    let me = std::thread::current().id();
    let deadline = Instant::now().checked_add(Duration::from_micros(timeout));
    let mut state = hub.state();
    loop {
        let running = state.running.iter().filter(|r| r.0 == token);
        if running.clone().any(|r| r.1 == me) {
            return false;
        }
        if running.count() == 0 {
            return true;
        }
        state = match deadline {
            None => hub
                .returned
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner),
            Some(deadline) => {
                let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                    return false;
                };
                let waited = hub.returned.wait_timeout(state, left);
                waited.unwrap_or_else(PoisonError::into_inner).0
            }
        };
    }
}

/// Dispatcher and aggregate device: no title of ours asks for them.
extern "system" fn create_dispatcher(_: This, out: Out) -> HRESULT {
    tracing::warn!("input dispatcher not implemented");
    fail(out, E_NOTIMPL)
}
extern "system" fn create_aggregate(_: This, _kind: u32, out: Out) -> HRESULT {
    tracing::warn!("aggregate input device not implemented");
    fail(out, E_NOTIMPL)
}
extern "system" fn find_by_id(this: This, id: *const [u8; 32], out: Out) -> HRESULT {
    if id.is_null() || out.is_null() {
        return fail(out, E_INVALIDARG);
    }
    // SAFETY: checked non-null; the title passes a device id.
    let id = unsafe { &*id };
    match hub(this).devices.iter().find(|d| d.info.id == *id) {
        Some(device) => {
            // SAFETY: checked non-null.
            unsafe { out.write((device as *const Device).cast()) };
            S_OK
        }
        None => fail(out, DEVICE_NOT_FOUND),
    }
}
/// Lookup by object, platform handle or platform string: our devices have
/// none of these.
extern "system" fn find_device(_: This, _value: *const c_void, out: Out) -> HRESULT {
    fail(out, DEVICE_NOT_FOUND)
}
extern "system" fn enable_oem_device(
    _: This,
    _vendor: u16,
    _product: u16,
    _interface: u8,
    _collection: u8,
) -> HRESULT {
    S_OK
}
extern "system" fn set_focus_policy(_: This, _policy: u32) {}

static INPUT: Table<22> = Table([
    query_interface as *const (),     // 0x00
    add_ref as *const (),             // 0x08 AddRef
    add_ref as *const (),             // 0x10 Release
    timestamp as *const (),           // 0x18
    current_reading as *const (),     // 0x20
    next_reading as *const (),        // 0x28
    previous_reading as *const (),    // 0x30
    temporal_reading as *const (),    // 0x38
    register_reading as *const (),    // 0x40
    register_device as *const (),     // 0x48
    register_other as *const (),      // 0x50 guide button
    register_other as *const (),      // 0x58 keyboard layout
    stop_callback as *const (),       // 0x60
    unregister_callback as *const (), // 0x68
    create_dispatcher as *const (),   // 0x70
    create_aggregate as *const (),    // 0x78
    find_by_id as *const (),          // 0x80
    find_device as *const (),         // 0x88 by object
    find_device as *const (),         // 0x90 by platform handle
    find_device as *const (),         // 0x98 by platform string
    enable_oem_device as *const (),   // 0xa0
    set_focus_policy as *const (),    // 0xa8
]);

extern "system" fn device_info(this: *const Device) -> *const DeviceInfo {
    // SAFETY: one of our devices.
    unsafe { &(*this).info }
}
extern "system" fn device_status(this: *const Device) -> u32 {
    // SAFETY: one of our devices.
    let device = unsafe { &*this };
    if device.hub().state().slots[device.index].connected {
        STATUS_CONNECTED
    } else {
        0
    }
}
// The title sends trigger levels in slot 0x50; all four reach the
// host, which drops the trigger motors (see `host::Host::rumble`).
extern "system" fn set_rumble(this: *const Device, params: *const [f32; 4]) {
    // SAFETY: one of our devices.
    let device = unsafe { &*this };
    if device.index >= PADS {
        return;
    }
    let [low, high, left_trigger, right_trigger] = if params.is_null() {
        [0.0; 4]
    } else {
        // SAFETY: checked non-null; four motor levels.
        unsafe { *params }
    };
    let speed = |level: f32| (level.clamp(0.0, 1.0) * f32::from(u16::MAX)) as u16;
    let motors = Motors {
        low: speed(low),
        high: speed(high),
        left_trigger: speed(left_trigger),
        right_trigger: speed(right_trigger),
    };
    lock(&device.hub().source).rumble(device.index, motors);
}
extern "system" fn unsupported<const SLOT: usize>(_: *const c_void) -> HRESULT {
    tracing::warn!(
        slot = format_args!("{SLOT:#x}"),
        "input device method not implemented"
    );
    E_NOTIMPL
}

static DEVICE: Table<21> = Table([
    device_query as *const (),        // 0x00
    add_ref as *const (),             // 0x08 AddRef
    add_ref as *const (),             // 0x10 Release
    device_info as *const (),         // 0x18
    device_status as *const (),       // 0x20
    unsupported::<0x28> as *const (), // battery
    unsupported::<0x30> as *const (), // force feedback
    unsupported::<0x38> as *const (), //
    unsupported::<0x40> as *const (), //
    unsupported::<0x48> as *const (), // haptics
    set_rumble as *const (),          // 0x50
    unsupported::<0x58> as *const (), // input synchronization
    unsupported::<0x60> as *const (), //
    unsupported::<0x68> as *const (), // power off
    unsupported::<0x70> as *const (), // raw device access
    unsupported::<0x78> as *const (), //
    unsupported::<0x80> as *const (), //
    unsupported::<0x88> as *const (), //
    unsupported::<0x90> as *const (), //
    unsupported::<0x98> as *const (), //
    unsupported::<0xa0> as *const (), //
]);

fn reading<'a>(this: *const Reading) -> &'a Reading {
    // SAFETY: a reading we handed out and the title still holds.
    unsafe { &*this }
}
extern "system" fn reading_query(this: *const c_void, iid: *const Guid, out: Out) -> HRESULT {
    query_iid(
        this.cast(),
        iid,
        out,
        reading_add_ref,
        &IID_IGAMEINPUT_READING,
    )
}
extern "system" fn reading_add_ref(this: *const Reading) -> u32 {
    reading(this).refs.fetch_add(1, Ordering::Relaxed) + 1
}
extern "system" fn reading_release(this: *const Reading) -> u32 {
    let left = release_ref(&reading(this).refs, "input reading");
    if left == Some(0) {
        // SAFETY: the last reference is gone; `Ref::new` boxed it.
        drop(unsafe { Box::from_raw(this.cast_mut()) });
    }
    left.unwrap_or(0)
}
extern "system" fn input_kind(this: *const Reading) -> u32 {
    reading(this).kind()
}
extern "system" fn sequence_number(this: *const Reading, kind: u32) -> u64 {
    let reading = reading(this);
    if reading.kind() & kind != 0 {
        reading.sequence
    } else {
        0
    }
}
extern "system" fn reading_timestamp(this: *const Reading) -> u64 {
    reading(this).timestamp
}
extern "system" fn reading_device(this: *const Reading, out: *mut *const Device) {
    if !out.is_null() {
        // SAFETY: checked non-null; devices need no reference.
        unsafe { out.write(reading(this).device) };
    }
}
extern "system" fn key_count(this: *const Reading) -> u32 {
    match &reading(this).value {
        Value::Keys(keys) => keys.len() as u32,
        Value::Pad(_) => 0,
    }
}
extern "system" fn key_state(this: *const Reading, count: u32, out: *mut KeyState) -> u32 {
    let Value::Keys(keys) = &reading(this).value else {
        return 0;
    };
    if out.is_null() {
        return 0;
    }
    let n = keys.len().min(count as usize);
    // SAFETY: the title passes room for `count` keys.
    unsafe { std::ptr::copy_nonoverlapping(keys.as_ptr(), out, n) };
    n as u32
}
extern "system" fn gamepad_state(this: *const Reading, out: *mut PadState) -> bool {
    match &reading(this).value {
        Value::Pad(state) if !out.is_null() => {
            // SAFETY: checked non-null.
            unsafe { out.write(*state) };
            true
        }
        _ => false,
    }
}
/// Kinds our devices do not have: no count, no state.
extern "system" fn nothing(_: *const c_void) -> usize {
    0
}

static READING: Table<25> = Table([
    reading_query as *const (),     // 0x00
    reading_add_ref as *const (),   // 0x08
    reading_release as *const (),   // 0x10
    input_kind as *const (),        // 0x18
    sequence_number as *const (),   // 0x20
    reading_timestamp as *const (), // 0x28
    reading_device as *const (),    // 0x30
    nothing as *const (),           // 0x38 raw report
    nothing as *const (),           // 0x40 controller axes, buttons, switches
    nothing as *const (),           // 0x48
    nothing as *const (),           // 0x50
    nothing as *const (),           // 0x58
    nothing as *const (),           // 0x60
    nothing as *const (),           // 0x68
    key_count as *const (),         // 0x70
    key_state as *const (),         // 0x78
    nothing as *const (),           // 0x80 mouse
    nothing as *const (),           // 0x88 touch
    nothing as *const (),           // 0x90
    nothing as *const (),           // 0x98 motion
    nothing as *const (),           // 0xa0 arcade stick
    nothing as *const (),           // 0xa8 flight stick
    gamepad_state as *const (),     // 0xb0
    nothing as *const (),           // 0xb8 racing wheel
    nothing as *const (),           // 0xc0 UI navigation
]);

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "gameinput",
        dlls: &["gameinput.dll"],
        // Titles import it by ordinal only.
        exports: vec![HleExport {
            name: "#1",
            address: GameInputCreate as *const () as usize,
        }],
    }
}

/// Host input: XInput pads, the keyboard while our process has focus.
#[cfg(windows)]
mod host {
    use std::time::{Duration, Instant};

    use windows_sys::Win32::System::Threading::GetCurrentProcessId;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, GetKeyState, GetKeyboardLayout, HKL, MAPVK_VK_TO_VSC_EX, MapVirtualKeyW,
        ToUnicodeEx,
    };
    use windows_sys::Win32::UI::Input::XboxController::{
        XINPUT_GAMEPAD, XINPUT_GAMEPAD_A, XINPUT_GAMEPAD_B, XINPUT_GAMEPAD_BACK,
        XINPUT_GAMEPAD_DPAD_DOWN, XINPUT_GAMEPAD_DPAD_LEFT, XINPUT_GAMEPAD_DPAD_RIGHT,
        XINPUT_GAMEPAD_DPAD_UP, XINPUT_GAMEPAD_LEFT_SHOULDER, XINPUT_GAMEPAD_LEFT_THUMB,
        XINPUT_GAMEPAD_RIGHT_SHOULDER, XINPUT_GAMEPAD_RIGHT_THUMB, XINPUT_GAMEPAD_START,
        XINPUT_GAMEPAD_X, XINPUT_GAMEPAD_Y, XINPUT_STATE, XINPUT_VIBRATION, XInputGetState,
        XInputSetState,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, GetWindowThreadProcessId,
    };

    use super::*;

    /// Asking XInput about an empty slot is slow; look again this often.
    const EMPTY_SLOT_RECHECK: Duration = Duration::from_secs(1);

    const BUTTONS: [(u16, u32); 14] = [
        (XINPUT_GAMEPAD_START, MENU),
        (XINPUT_GAMEPAD_BACK, VIEW),
        (XINPUT_GAMEPAD_A, A),
        (XINPUT_GAMEPAD_B, B),
        (XINPUT_GAMEPAD_X, X),
        (XINPUT_GAMEPAD_Y, Y),
        (XINPUT_GAMEPAD_DPAD_UP, UP),
        (XINPUT_GAMEPAD_DPAD_DOWN, DOWN),
        (XINPUT_GAMEPAD_DPAD_LEFT, LEFT),
        (XINPUT_GAMEPAD_DPAD_RIGHT, RIGHT),
        (XINPUT_GAMEPAD_LEFT_SHOULDER, LEFT_SHOULDER),
        (XINPUT_GAMEPAD_RIGHT_SHOULDER, RIGHT_SHOULDER),
        (XINPUT_GAMEPAD_LEFT_THUMB, LEFT_STICK),
        (XINPUT_GAMEPAD_RIGHT_THUMB, RIGHT_STICK),
    ];

    /// Raw values, no dead zone: titles apply their own.
    pub(super) fn convert(pad: &XINPUT_GAMEPAD) -> PadState {
        let stick = |v: i16| (f32::from(v) / 32767.0).max(-1.0);
        PadState {
            buttons: BUTTONS
                .iter()
                .filter(|(host, _)| pad.wButtons & host != 0)
                .fold(0, |all, (_, bit)| all | bit),
            left_trigger: f32::from(pad.bLeftTrigger) / 255.0,
            right_trigger: f32::from(pad.bRightTrigger) / 255.0,
            left_x: stick(pad.sThumbLX),
            left_y: stick(pad.sThumbLY),
            right_x: stick(pad.sThumbRX),
            right_y: stick(pad.sThumbRY),
        }
    }

    #[derive(Default)]
    pub(super) struct Host {
        empty_until: [Option<Instant>; PADS],
    }

    impl Source for Host {
        fn pad(&mut self, slot: usize) -> Option<PadState> {
            let empty = self.empty_until[slot].is_some_and(|until| Instant::now() < until);
            if !empty {
                let mut state = XINPUT_STATE::default();
                // SAFETY: room for the state.
                if unsafe { XInputGetState(slot as u32, &mut state) } == 0 {
                    self.empty_until[slot] = None;
                    return Some(convert(&state.Gamepad));
                }
                self.empty_until[slot] = Some(Instant::now() + EMPTY_SLOT_RECHECK);
            }
            // A DualSense is a HID device, not an XInput one; it plays slot 0
            // when no XInput pad answers there.
            if slot == 0
                && let Some(pad) = super::super::dualsense::latest()
            {
                return Some(PadState {
                    buttons: pad.buttons,
                    left_trigger: pad.left_trigger,
                    right_trigger: pad.right_trigger,
                    left_x: pad.left_x,
                    left_y: pad.left_y,
                    right_x: pad.right_x,
                    right_y: pad.right_y,
                });
            }
            None
        }

        fn keys(&mut self) -> Vec<KeyState> {
            #[cfg(feature = "diagnostics")]
            let mut down = scripted();
            #[cfg(not(feature = "diagnostics"))]
            let mut down = Vec::new();
            if focused() {
                // Left/right modifier keys only, as devices report them; no
                // mouse buttons (below 0x08).
                let held: Vec<u8> = (0x08..=0xFE_u8)
                    .filter(|vk| !(0x10..=0x12).contains(vk) && !down.contains(vk))
                    // SAFETY: plain queries.
                    .filter(|&vk| unsafe { GetAsyncKeyState(i32::from(vk)) } < 0)
                    .collect();
                down.extend(held);
            }
            let state = keyboard_state(&down, caps_lock());
            let layout = layout();
            down.iter()
                .map(|&vk| {
                    // SAFETY: plain query.
                    let scan_code = unsafe { MapVirtualKeyW(u32::from(vk), MAPVK_VK_TO_VSC_EX) };
                    let (code_point, dead_key) = character(vk, scan_code, &state, layout);
                    KeyState {
                        scan_code,
                        code_point,
                        virtual_key: vk,
                        dead_key,
                    }
                })
                .collect()
        }

        #[cfg(feature = "diagnostics")]
        fn scripted(&mut self) -> Vec<u8> {
            scripted()
        }

        // shortcut: trigger motors are dropped (XInput has none). The host
        // API that drives them without a redistributable is
        // Windows.Gaming.Input (WinRT `Gamepad.Vibration`), which needs WinRT
        // bindings this crate lacks outside diagnostics builds.
        fn rumble(&mut self, slot: usize, motors: Motors) {
            let speeds = XINPUT_VIBRATION {
                wLeftMotorSpeed: motors.low,
                wRightMotorSpeed: motors.high,
            };
            // SAFETY: a valid vibration; an empty slot just fails.
            unsafe { XInputSetState(slot as u32, &speeds) };
        }
    }

    /// Debugging aid: `VICTORY_KEYS=<seconds>:<key>,...` holds each key for
    /// 0.2 s that long after input starts, with or without focus. A key is a
    /// letter, a digit, or `Enter`, `Esc`, `Tab`, `Back`, `Up`, `Down`,
    /// `Left`, `Right` (the stand-in pad's table maps them to buttons, on the
    /// stand-in or, while one is connected, on the first host pad).
    ///
    /// `VICTORY_KEYS_FILE=<path>` also presses each key named on a line of
    /// that file, read while the title runs, so a person or a script can
    /// append keys as screens appear instead of guessing load times.
    #[cfg(feature = "diagnostics")]
    fn scripted() -> Vec<u8> {
        static SCRIPT: LazyLock<(Instant, Vec<(Duration, u8)>)> = LazyLock::new(|| {
            let list = std::env::var("VICTORY_KEYS").unwrap_or_default();
            let keys = list.split(',').filter_map(|entry| {
                let (at, key) = entry.split_once(':')?;
                Some((
                    Duration::try_from_secs_f64(at.trim().parse().ok()?).ok()?,
                    key_code(key.trim())?,
                ))
            });
            (Instant::now(), keys.collect())
        });
        let (start, keys) = &*SCRIPT;
        let now = start.elapsed();
        let mut down: Vec<u8> = keys
            .iter()
            .filter(|&&(at, _)| (at..at + KEY_HOLD).contains(&now))
            .map(|&(_, vk)| vk)
            .collect();
        down.extend(key_file());
        down
    }

    /// How long a scripted key stays down.
    #[cfg(feature = "diagnostics")]
    const KEY_HOLD: Duration = Duration::from_millis(200);
    /// How long `VICTORY_KEYS_FILE` waits between keys, so each reads as a
    /// separate press.
    #[cfg(feature = "diagnostics")]
    const KEY_GAP: Duration = Duration::from_millis(200);

    /// The key `VICTORY_KEYS_FILE` holds down now, reading the file at most
    /// every 0.1 s.
    #[cfg(feature = "diagnostics")]
    fn key_file() -> Option<u8> {
        struct File {
            path: Option<std::path::PathBuf>,
            feed: KeyFeed,
            read_at: Option<Instant>,
        }
        static FILE: LazyLock<Mutex<File>> = LazyLock::new(|| {
            Mutex::new(File {
                path: std::env::var_os("VICTORY_KEYS_FILE").map(Into::into),
                feed: KeyFeed::default(),
                read_at: None,
            })
        });
        let mut file = FILE.lock().unwrap_or_else(PoisonError::into_inner);
        let path = file.path.clone()?;
        let now = Instant::now();
        if file
            .read_at
            .is_none_or(|at| now - at >= Duration::from_millis(100))
        {
            file.read_at = Some(now);
            if let Ok(text) = std::fs::read_to_string(&path) {
                file.feed.feed(&text);
            }
        }
        file.feed.key(now)
    }

    /// Keys named one per line in a growing text, pressed one at a time.
    #[cfg(feature = "diagnostics")]
    #[derive(Default)]
    struct KeyFeed {
        /// Bytes of the text already turned into keys.
        read: usize,
        queue: VecDeque<u8>,
        /// The key pressed last and when.
        pressed: Option<(Instant, u8)>,
    }

    #[cfg(feature = "diagnostics")]
    impl KeyFeed {
        /// Queues the keys on the whole lines of `text` past those already
        /// read; a line still being written waits for its newline.
        fn feed(&mut self, text: &str) {
            let Some(new) = text.get(self.read..) else {
                return;
            };
            let whole = new.rfind('\n').map_or(0, |i| i + 1);
            self.read += whole;
            for name in new[..whole].lines().map(str::trim) {
                match key_code(name) {
                    Some(vk) => {
                        tracing::info!(key = name, "VICTORY_KEYS_FILE key queued");
                        self.queue.push_back(vk);
                    }
                    None if !name.is_empty() => {
                        tracing::warn!(key = name, "VICTORY_KEYS_FILE key unknown");
                    }
                    None => {}
                }
            }
        }

        /// The key down at `now`: each queued key for `KEY_HOLD`, then
        /// nothing for `KEY_GAP`.
        fn key(&mut self, now: Instant) -> Option<u8> {
            if let Some((at, vk)) = self.pressed {
                let since = now.saturating_duration_since(at);
                if since < KEY_HOLD {
                    return Some(vk);
                }
                if since < KEY_HOLD + KEY_GAP {
                    return None;
                }
                self.pressed = None;
            }
            let vk = self.queue.pop_front()?;
            self.pressed = Some((now, vk));
            Some(vk)
        }
    }

    /// The virtual key a scripted key name stands for.
    #[cfg(feature = "diagnostics")]
    fn key_code(name: &str) -> Option<u8> {
        Some(match name {
            "Enter" => 0x0D,
            "Esc" => 0x1B,
            "Tab" => 0x09,
            "Back" => 0x08,
            "Up" => 0x26,
            "Down" => 0x28,
            "Left" => 0x25,
            "Right" => 0x27,
            k if k.len() == 1 && k.as_bytes()[0].is_ascii_alphanumeric() => {
                k.as_bytes()[0].to_ascii_uppercase()
            }
            _ => return None,
        })
    }

    /// The keyboard state `ToUnicodeEx` reads with the keys `held` down:
    /// each held key, the generic Shift, Ctrl and Alt from either side's key
    /// (devices report sides only), and Caps Lock's toggle.
    fn keyboard_state(held: &[u8], caps_lock: bool) -> [u8; 256] {
        const DOWN: u8 = 0x80;
        const TOGGLED: u8 = 1;
        let mut state = [0u8; 256];
        for &vk in held {
            state[usize::from(vk)] = DOWN;
            let generic = match vk {
                0xA0 | 0xA1 => Some(0x10), // shift
                0xA2 | 0xA3 => Some(0x11), // control
                0xA4 | 0xA5 => Some(0x12), // alt
                _ => None,
            };
            if let Some(g) = generic {
                state[g] = DOWN;
            }
        }
        if caps_lock {
            state[0x14] |= TOGGLED;
        }
        state
    }

    fn caps_lock() -> bool {
        // SAFETY: plain query.
        let state = unsafe { GetKeyState(0x14) };
        state & 1 != 0
    }

    /// The foreground window's keyboard layout (the title's while it has
    /// focus; this thread's when there is no foreground window).
    fn layout() -> HKL {
        // SAFETY: plain queries; a null window gives thread 0, this thread.
        unsafe {
            GetKeyboardLayout(GetWindowThreadProcessId(
                GetForegroundWindow(),
                std::ptr::null_mut(),
            ))
        }
    }

    /// The character key `vk` types in `layout` with `state`: its code point
    /// (0 when it types none, or several at once), and whether it is a dead
    /// key, whose code point is then the accent's spacing form. Pending dead
    /// keys are not combined: the thread's keyboard state is left alone.
    fn character(vk: u8, scan_code: u32, state: &[u8; 256], layout: HKL) -> (u32, bool) {
        /// `ToUnicodeEx` flag: keep the keyboard state (Windows 10 1607 on).
        const KEEP_STATE: u32 = 4;
        let mut text = [0u16; 8];
        // SAFETY: the state is the 256 bytes the call reads; the buffer's
        // length is passed with it.
        let n = unsafe {
            ToUnicodeEx(
                u32::from(vk),
                scan_code & 0xFF,
                state.as_ptr(),
                text.as_mut_ptr(),
                text.len() as i32,
                KEEP_STATE,
                layout,
            )
        };
        let units = match n {
            0 => return (0, false),
            ..0 => &text[..1],
            n => &text[..(n as usize).min(text.len())],
        };
        let mut chars = char::decode_utf16(units.iter().copied());
        match (chars.next(), chars.next()) {
            (Some(Ok(c)), None) => (u32::from(c), n < 0),
            _ => (0, false),
        }
    }

    /// The foreground window is one of ours (the title's or our fallback).
    fn focused() -> bool {
        let mut process = 0;
        // SAFETY: plain queries; a null window gives process 0.
        unsafe {
            GetWindowThreadProcessId(GetForegroundWindow(), &mut process);
            process == GetCurrentProcessId()
        }
    }

    #[test]
    fn host_pads_convert_to_console_pads() {
        let pad = XINPUT_GAMEPAD {
            wButtons: XINPUT_GAMEPAD_A | XINPUT_GAMEPAD_START | XINPUT_GAMEPAD_DPAD_LEFT,
            bLeftTrigger: 255,
            sThumbLX: -32768,
            sThumbLY: 32767,
            ..Default::default()
        };
        let state = convert(&pad);
        assert_eq!(state.buttons, A | MENU | LEFT);
        assert_eq!(
            (
                state.left_trigger,
                state.left_x,
                state.left_y,
                state.right_x
            ),
            (1.0, -1.0, 1.0, 0.0)
        );
    }

    #[test]
    fn either_sides_modifier_holds_the_generic_one() {
        let state = keyboard_state(&[0xA1, 0xA2, 0x41], false);
        for vk in [0xA1, 0x10, 0xA2, 0x11, 0x41] {
            assert_eq!(state[vk], 0x80, "{vk:#x}");
        }
        for vk in [0xA0, 0xA3, 0x12, 0x14] {
            assert_eq!(state[vk], 0, "{vk:#x}");
        }
        assert_eq!(keyboard_state(&[], true)[0x14], 1);
    }

    #[test]
    fn keys_type_their_characters() {
        // Space and Enter type the same in every layout; F1 types nothing.
        let layout = layout();
        let typed = |vk: u8, held: &[u8]| {
            // SAFETY: plain query.
            let scan = unsafe { MapVirtualKeyW(u32::from(vk), MAPVK_VK_TO_VSC_EX) };
            character(vk, scan, &keyboard_state(held, false), layout)
        };
        assert_eq!(typed(0x20, &[0x20]), (0x20, false));
        assert_eq!(typed(0x0D, &[0x0D]), (0x0D, false));
        assert_eq!(typed(0x70, &[0x70]), (0, false));
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    fn key_file_presses_whole_lines_one_key_at_a_time() {
        let start = Instant::now();
        let at = |ms| start + Duration::from_millis(ms);
        let mut feed = KeyFeed::default();
        // The last line has no newline yet; an unknown name is skipped.
        let mut text = String::from("Down\nNope\n\nDo");
        feed.feed(&text);
        assert_eq!(feed.key(at(0)), Some(0x28));
        assert_eq!(feed.key(at(199)), Some(0x28));
        assert_eq!(feed.key(at(200)), None);
        assert_eq!(feed.key(at(400)), None, "nothing queued after the gap");
        text.push_str("wn\nEnter\nx\n");
        feed.feed(&text);
        feed.feed(&text);
        let pressed: Vec<_> = (400..2000)
            .step_by(100)
            .map(|ms| feed.key(at(ms)))
            .collect();
        let d = Some(0x28);
        let e = Some(0x0D);
        let x = Some(b'X');
        #[rustfmt::skip]
        assert_eq!(pressed, [
            d, d, None, None,
            e, e, None, None,
            x, x, None, None,
            None, None, None, None,
        ]);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::HleResolver;
    use crate::HostPolicy;
    use victory_loader::Provenance;
    use victory_pe::ImportSymbol;

    #[derive(Default)]
    struct FakeHost {
        pads: [Option<PadState>; PADS],
        keys: Vec<KeyState>,
        scripted: Vec<u8>,
        rumble: Vec<(usize, Motors)>,
    }

    struct Fake(Arc<Mutex<FakeHost>>);

    impl Source for Fake {
        fn pad(&mut self, slot: usize) -> Option<PadState> {
            self.0.lock().unwrap().pads[slot]
        }
        fn keys(&mut self) -> Vec<KeyState> {
            self.0.lock().unwrap().keys.clone()
        }
        fn scripted(&mut self) -> Vec<u8> {
            self.0.lock().unwrap().scripted.clone()
        }
        fn rumble(&mut self, slot: usize, motors: Motors) {
            self.0.lock().unwrap().rumble.push((slot, motors));
        }
    }

    fn fake() -> (&'static Hub, Arc<Mutex<FakeHost>>) {
        let host = Arc::new(Mutex::new(FakeHost::default()));
        let hub = Hub::new(Box::new(Fake(Arc::clone(&host))));
        hub.poll();
        (hub, host)
    }

    fn key(virtual_key: u8) -> KeyState {
        KeyState {
            scan_code: u32::from(virtual_key) + 0x100,
            virtual_key,
            ..Default::default()
        }
    }

    fn slot<F: Copy, T>(object: *const T, offset: usize) -> F {
        // SAFETY: our objects start with their table, which has the slot.
        let f = unsafe { *(*(object as *const *const *const ())).add(offset / 8) };
        // SAFETY: the caller names the slot's signature.
        unsafe { std::mem::transmute_copy(&f) }
    }

    /// Device callbacks seen: device index, current and previous status.
    type Log = Mutex<Vec<(usize, u32, u32)>>;

    extern "system" fn record(
        _token: u64,
        context: *mut c_void,
        device: *const Device,
        _timestamp: u64,
        current: u32,
        previous: u32,
    ) {
        // SAFETY: the test passes its log and one of the hub's devices.
        let (log, index) = unsafe { (&*(context as *const Log), (*device).index) };
        log.lock().unwrap().push((index, current, previous));
    }

    fn register(hub: &Hub, kind: u32, enumeration: u32, log: &Log) -> u64 {
        type Register = extern "system" fn(
            This,
            *const Device,
            u32,
            u32,
            u32,
            *mut c_void,
            Option<DeviceCallback>,
            *mut u64,
        ) -> HRESULT;
        let mut token = 0;
        let context = log as *const Log as *mut c_void;
        let register = slot::<Register, _>(&hub.object, 0x48);
        let hr = register(
            &hub.object,
            std::ptr::null(),
            kind,
            1,
            enumeration,
            context,
            Some(record),
            &mut token,
        );
        assert_eq!(hr, S_OK);
        assert_ne!(token, 0);
        token
    }

    fn result(hr: HRESULT, out: *const c_void) -> Result<*const Reading, HRESULT> {
        if hr == S_OK {
            Ok(out.cast())
        } else {
            assert!(out.is_null());
            Err(hr)
        }
    }

    fn current(hub: &Hub, kind: u32, device: usize) -> Result<*const Reading, HRESULT> {
        type Current = extern "system" fn(This, u32, *const Device, Out) -> HRESULT;
        let mut out = std::ptr::null();
        let hr = slot::<Current, _>(&hub.object, 0x20)(
            &hub.object,
            kind,
            &hub.devices[device],
            &mut out,
        );
        result(hr, out)
    }

    fn walk(hub: &Hub, offset: usize, from: *const Reading) -> Result<*const Reading, HRESULT> {
        type Walk = extern "system" fn(This, *const Reading, u32, *const Device, Out) -> HRESULT;
        let mut out = std::ptr::null();
        let device = reading(from).device;
        let hr =
            slot::<Walk, _>(&hub.object, offset)(&hub.object, from, GAMEPAD_KIND, device, &mut out);
        result(hr, out)
    }

    fn pad(reading: *const Reading) -> PadState {
        let mut state = PadState::default();
        let get =
            slot::<extern "system" fn(*const Reading, *mut PadState) -> bool, _>(reading, 0xb0);
        assert!(get(reading, &mut state));
        state
    }

    fn stamp(reading: *const Reading) -> (u64, u64) {
        let time = slot::<extern "system" fn(*const Reading) -> u64, _>(reading, 0x28);
        let sequence = slot::<extern "system" fn(*const Reading, u32) -> u64, _>(reading, 0x20);
        (time(reading), sequence(reading, GAMEPAD_KIND))
    }

    fn release(reading: *const Reading) {
        slot::<extern "system" fn(*const Reading) -> u32, _>(reading, 0x10)(reading);
    }

    #[test]
    fn ordinal_one_resolves_to_create() {
        let r = HleResolver::new(HostPolicy::none());
        assert_eq!(
            r.classify("GameInput.dll", &ImportSymbol::Ordinal(1)),
            Some(Provenance::Hle)
        );
        assert_eq!(r.classify("GameInput.dll", &ImportSymbol::Ordinal(2)), None);
    }

    #[test]
    fn query_interface_returns_iunknown_own_iid_and_retains_the_reading() {
        type Query = extern "system" fn(*const c_void, *const Guid, Out) -> HRESULT;
        let (hub, _) = fake();
        let reading = Ref::new(Reading {
            vtable: &READING,
            refs: AtomicU32::new(1),
            device: &hub.devices[FALLBACK],
            sequence: 1,
            timestamp: 0,
            value: Value::Pad(PadState::default()),
        });
        let reading_ptr = reading.0.as_ptr().cast_const();
        let cases = [
            (std::ptr::from_ref(&hub.object).cast(), &IID_IGAMEINPUT),
            (
                std::ptr::from_ref(&hub.devices[FALLBACK]).cast(),
                &IID_IGAMEINPUT_DEVICE,
            ),
            (reading_ptr.cast(), &IID_IGAMEINPUT_READING),
        ];
        for (object, self_iid) in cases {
            let query = slot::<Query, _>(object, 0);
            let mut out = std::ptr::null();
            assert_eq!(query(object, &IID_IUNKNOWN, &mut out), S_OK);
            assert_eq!(out, object);
            out = std::ptr::null();
            assert_eq!(query(object, self_iid, &mut out), S_OK);
            assert_eq!(out, object);
            // Another interface's id is not ours.
            let foreign = if *self_iid == IID_IGAMEINPUT {
                &IID_IGAMEINPUT_DEVICE
            } else {
                &IID_IGAMEINPUT
            };
            assert_eq!(query(object, foreign, &mut out), E_NOINTERFACE);
            assert!(out.is_null());
            out = object;
            assert_eq!(query(object, std::ptr::null(), &mut out), E_POINTER);
            assert!(out.is_null());
            assert_eq!(
                query(object, &IID_IUNKNOWN, std::ptr::null_mut()),
                E_POINTER
            );
        }
        // IUnknown once and own iid once; the reading started at 1.
        assert_eq!(reading.refs.load(Ordering::Relaxed), 3);
        drop(reading);
        let release = slot::<extern "system" fn(*const Reading) -> u32, _>(reading_ptr, 0x10);
        assert_eq!(release(reading_ptr), 1);
        assert_eq!(release(reading_ptr), 0);
    }

    #[test]
    fn pads_connect_report_and_disconnect() {
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, GAMEPAD_KIND, 2, &log);
        // Blocking enumeration ran before registration returned: with no
        // host pad the keyboard's stand-in is the only one.
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

        host.lock().unwrap().pads[1] = Some(PadState {
            buttons: A,
            left_x: 0.5,
            ..Default::default()
        });
        hub.poll();
        assert_eq!(
            log.lock().unwrap()[1..],
            [(1, STATUS_CONNECTED, 0), (FALLBACK, 0, STATUS_CONNECTED)]
        );

        let first = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(pad(first).buttons, A);
        assert_eq!(pad(first).left_x, 0.5);
        // No change, no new reading.
        hub.poll();
        let again = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(again, first);
        release(again);

        std::thread::sleep(Duration::from_millis(2));
        host.lock().unwrap().pads[1].as_mut().unwrap().buttons = B;
        hub.poll();
        let second = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(pad(second).buttons, B);
        let (t1, s1) = stamp(first);
        let (t2, s2) = stamp(second);
        assert!(t2 > t1 && s2 == s1 + 1);
        assert!(t2 <= now());

        // The title's edge check: previous of current, and back again.
        assert_eq!(walk(hub, 0x30, second), Ok(first));
        assert_eq!(walk(hub, 0x28, first), Ok(second));
        assert_eq!(walk(hub, 0x28, second), Err(READING_NOT_FOUND));
        assert_eq!(walk(hub, 0x30, first), Err(READING_NOT_FOUND));
        for r in [first, first, second, second] {
            release(r);
        }

        host.lock().unwrap().pads[1] = None;
        hub.poll();
        assert_eq!(
            log.lock().unwrap()[3..],
            [(1, 0, STATUS_CONNECTED), (FALLBACK, STATUS_CONNECTED, 0)]
        );
        assert_eq!(current(hub, GAMEPAD_KIND, 1), Err(DEVICE_DISCONNECTED));
        // The stand-in is back, at rest.
        let rest = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(pad(rest), PadState::default());
        release(rest);
    }

    #[test]
    fn with_no_device_readings_are_walked_across_devices_in_time_order() {
        type Walk = extern "system" fn(This, *const Reading, u32, *const Device, Out) -> HRESULT;
        let any = |hub: &Hub, offset, from, kind| {
            let mut out = std::ptr::null();
            let walk = slot::<Walk, _>(&hub.object, offset);
            result(
                walk(&hub.object, from, kind, std::ptr::null(), &mut out),
                out,
            )
        };
        let both = GAMEPAD_KIND | KEYBOARD_KIND;
        let tick = || std::thread::sleep(Duration::from_millis(2));
        let (hub, host) = fake();
        // The first poll: the stand-in pad and the keyboard, at one time.
        let stand_in = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        let no_keys = current(hub, KEYBOARD_KIND, KEYBOARD).unwrap();
        tick();
        host.lock().unwrap().pads[1] = Some(PadState::default());
        hub.poll();
        let pad_a = current(hub, GAMEPAD_KIND, 1).unwrap();
        tick();
        host.lock().unwrap().keys = vec![key(b'W')];
        hub.poll();
        let w = current(hub, KEYBOARD_KIND, KEYBOARD).unwrap();
        tick();
        host.lock().unwrap().pads[1].as_mut().unwrap().buttons = B;
        hub.poll();
        let pad_b = current(hub, GAMEPAD_KIND, 1).unwrap();

        let order = [stand_in, no_keys, pad_a, w, pad_b];
        for pair in order.windows(2) {
            assert_eq!(any(hub, 0x28, pair[0], both), Ok(pair[1]));
            assert_eq!(any(hub, 0x30, pair[1], both), Ok(pair[0]));
        }
        assert_eq!(any(hub, 0x30, stand_in, both), Err(READING_NOT_FOUND));
        assert_eq!(any(hub, 0x28, pad_b, both), Err(READING_NOT_FOUND));
        // Only gamepads: the keyboard's readings are skipped.
        assert_eq!(any(hub, 0x28, stand_in, GAMEPAD_KIND), Ok(pad_a));
        assert_eq!(any(hub, 0x28, pad_a, GAMEPAD_KIND), Ok(pad_b));
        assert_eq!(any(hub, 0x30, pad_a, GAMEPAD_KIND), Ok(stand_in));
        // Naming the device keeps the walk on it.
        assert_eq!(walk(hub, 0x30, pad_a), Err(READING_NOT_FOUND));
        for r in order {
            release(r);
        }
    }

    #[test]
    fn keys_feed_the_keyboard_and_the_stand_in_pad() {
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, KEYBOARD_KIND, 2, &log);
        assert_eq!(*log.lock().unwrap(), [(KEYBOARD, STATUS_CONNECTED, 0)]);

        let keys = vec![key(b'W'), key(0x0D), key(b'L'), key(b'C')];
        host.lock().unwrap().keys = keys.clone();
        hub.poll();
        let r = current(hub, KEYBOARD_KIND, KEYBOARD).unwrap();
        let count = slot::<extern "system" fn(*const Reading) -> u32, _>(r, 0x70);
        let state =
            slot::<extern "system" fn(*const Reading, u32, *mut KeyState) -> u32, _>(r, 0x78);
        assert_eq!(count(r), 4);
        let mut got = [KeyState::default(); 8];
        assert_eq!(state(r, 8, got.as_mut_ptr()), 4);
        assert_eq!(got[..4], keys[..]);
        release(r);

        let r = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(
            pad(r),
            PadState {
                buttons: A,
                left_y: 1.0,
                right_x: 1.0,
                right_trigger: 1.0,
                ..Default::default()
            }
        );
        release(r);
        // A keyboard reading is no gamepad reading.
        assert_eq!(current(hub, GAMEPAD_KIND, KEYBOARD), Err(READING_NOT_FOUND));
    }

    #[test]
    fn scripted_keys_press_the_first_host_pad_while_it_hides_the_stand_in() {
        let (hub, host) = fake();
        // A receiver answering for its switched-off pad: a pad at rest in
        // slot 1, a pad held left in slot 3.
        let held = PadState {
            left_x: -0.5,
            ..Default::default()
        };
        {
            let mut host = host.lock().unwrap();
            host.pads[1] = Some(PadState::default());
            host.pads[3] = Some(held);
            host.scripted = vec![0x0D, b'D'];
            host.keys = vec![key(0x0D), key(b'D')];
        }
        hub.poll();
        assert_eq!(
            current(hub, GAMEPAD_KIND, FALLBACK),
            Err(DEVICE_DISCONNECTED)
        );
        let r = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(
            pad(r),
            PadState {
                buttons: A,
                left_x: 1.0,
                ..Default::default()
            }
        );
        release(r);
        let r = current(hub, GAMEPAD_KIND, 3).unwrap();
        assert_eq!(pad(r), held);
        release(r);

        // Released, the pad reads what the host reports again.
        {
            let mut host = host.lock().unwrap();
            host.scripted.clear();
            host.keys.clear();
        }
        hub.poll();
        let r = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(pad(r), PadState::default());
        release(r);
    }

    #[test]
    fn async_enumeration_waits_for_the_poll_and_unregistering_stops_callbacks() {
        let (hub, host) = fake();
        let log = Log::default();
        let token = register(hub, GAMEPAD_KIND, ASYNC_ENUMERATION, &log);
        assert!(log.lock().unwrap().is_empty());
        hub.poll();
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

        let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(&hub.object, 0x68);
        assert!(unregister(&hub.object, token, 5000));
        host.lock().unwrap().pads[0] = Some(PadState::default());
        hub.poll();
        assert_eq!(log.lock().unwrap().len(), 1);
    }

    #[test]
    fn unregistering_waits_for_a_running_callback() {
        use std::sync::atomic::AtomicBool;
        static INSIDE: AtomicBool = AtomicBool::new(false);
        static LEAVE: AtomicBool = AtomicBool::new(false);
        static FROM_INSIDE: AtomicU32 = AtomicU32::new(0);
        extern "system" fn slow(
            token: u64,
            context: *mut c_void,
            _: *const Device,
            _: u64,
            _: u32,
            _: u32,
        ) {
            // From inside its own callback, unregistering fails at once.
            let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(context, 0x68);
            if !unregister(context.cast(), token, 0) {
                FROM_INSIDE.fetch_add(1, Ordering::SeqCst);
            }
            INSIDE.store(true, Ordering::SeqCst);
            while !LEAVE.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        let (hub, host) = fake();
        type Register = extern "system" fn(
            This,
            *const Device,
            u32,
            u32,
            u32,
            *mut c_void,
            Option<DeviceCallback>,
            *mut u64,
        ) -> HRESULT;
        let register = slot::<Register, _>(&hub.object, 0x48);
        let mut token = 0;
        let context = (&raw const hub.object).cast_mut().cast();
        assert_eq!(
            register(
                &hub.object,
                std::ptr::null(),
                GAMEPAD_KIND,
                1,
                0,
                context,
                Some(slow),
                &mut token
            ),
            S_OK
        );
        host.lock().unwrap().pads[1] = Some(PadState::default());
        let poller = std::thread::spawn(|| hub.poll());
        while !INSIDE.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(&hub.object, 0x68);
        assert!(
            !unregister(&hub.object, token, 5000),
            "times out while it runs"
        );
        let leave = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(50));
            LEAVE.store(true, Ordering::SeqCst);
        });
        let start = Instant::now();
        assert!(unregister(&hub.object, token, u64::MAX), "waits for it");
        assert!(start.elapsed() >= Duration::from_millis(40));
        assert_eq!(FROM_INSIDE.load(Ordering::SeqCst), 1);
        poller.join().unwrap();
        leave.join().unwrap();
    }

    #[test]
    fn callbacks_from_registration_and_poll_are_serialized() {
        use std::sync::atomic::AtomicBool;
        static ACTIVE: AtomicU32 = AtomicU32::new(0);
        static PEAK: AtomicU32 = AtomicU32::new(0);
        static CALLS: AtomicU32 = AtomicU32::new(0);
        static RELEASE: AtomicBool = AtomicBool::new(false);
        extern "system" fn tracked(
            _: u64,
            _: *mut c_void,
            _: *const Device,
            _: u64,
            _: u32,
            _: u32,
        ) {
            let active = ACTIVE.fetch_add(1, Ordering::SeqCst) + 1;
            PEAK.fetch_max(active, Ordering::SeqCst);
            CALLS.fetch_add(1, Ordering::SeqCst);
            while !RELEASE.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
            ACTIVE.fetch_sub(1, Ordering::SeqCst);
        }
        let (hub, host) = fake();
        type Register = extern "system" fn(
            This,
            *const Device,
            u32,
            u32,
            u32,
            *mut c_void,
            Option<DeviceCallback>,
            *mut u64,
        ) -> HRESULT;
        let register = slot::<Register, _>(&hub.object, 0x48);
        ACTIVE.store(0, Ordering::SeqCst);
        PEAK.store(0, Ordering::SeqCst);
        CALLS.store(0, Ordering::SeqCst);
        RELEASE.store(false, Ordering::SeqCst);
        // Blocking enumeration delivers its first callback on the registering
        // thread, which stays inside it while the poll thread runs.
        let registrar = std::thread::spawn(move || {
            let mut token = 0;
            register(
                &hub.object,
                std::ptr::null(),
                GAMEPAD_KIND,
                1,
                2,
                std::ptr::null_mut(),
                Some(tracked),
                &mut token,
            )
        });
        let deadline = Instant::now() + Duration::from_secs(2);
        while ACTIVE.load(Ordering::SeqCst) == 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_ne!(
            ACTIVE.load(Ordering::SeqCst),
            0,
            "registration delivered a callback"
        );
        let poller = std::thread::spawn(move || {
            host.lock().unwrap().pads[1] = Some(PadState::default());
            hub.poll();
        });
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(ACTIVE.load(Ordering::SeqCst), 1, "only one callback runs");
        RELEASE.store(true, Ordering::SeqCst);
        assert_eq!(registrar.join().unwrap(), S_OK);
        poller.join().unwrap();
        assert_eq!(PEAK.load(Ordering::SeqCst), 1);
        assert!(
            CALLS.load(Ordering::SeqCst) >= 2,
            "initial and hotplug both ran"
        );
    }

    /// A blocking enumeration registered while the poll thread is inside a
    /// callback runs its callbacks on the registering thread before it
    /// returns. The title registers holding a lock its callback takes: here
    /// the poll thread's callback is waiting for that lock, so handing the
    /// enumeration to the poll thread and waiting for it would never return.
    #[test]
    fn blocking_enumeration_runs_on_the_registering_thread_while_another_delivers() {
        use std::sync::atomic::AtomicBool;
        static HELD: AtomicU32 = AtomicU32::new(0);
        static SEEN: AtomicU32 = AtomicU32::new(0);
        static LOCKED: AtomicBool = AtomicBool::new(false);
        /// The title's lock, held by the registering thread.
        static TITLE: Mutex<()> = Mutex::new(());
        static SEEN_ON: Mutex<Option<ThreadId>> = Mutex::new(None);
        extern "system" fn hold(_: u64, _: *mut c_void, _: *const Device, _: u64, _: u32, _: u32) {
            HELD.fetch_add(1, Ordering::SeqCst);
            drop(TITLE.lock().unwrap_or_else(PoisonError::into_inner));
        }
        extern "system" fn seen(_: u64, _: *mut c_void, _: *const Device, _: u64, _: u32, _: u32) {
            *SEEN_ON.lock().unwrap() = Some(std::thread::current().id());
            SEEN.fetch_add(1, Ordering::SeqCst);
        }
        type Register = extern "system" fn(
            This,
            *const Device,
            u32,
            u32,
            u32,
            *mut c_void,
            Option<DeviceCallback>,
            *mut u64,
        ) -> HRESULT;
        let (hub, host) = fake();
        let register = slot::<Register, _>(&hub.object, 0x48);
        HELD.store(0, Ordering::SeqCst);
        SEEN.store(0, Ordering::SeqCst);
        LOCKED.store(false, Ordering::SeqCst);
        let mut token = 0;
        let no_context = std::ptr::null_mut();
        let any_device = std::ptr::null();
        assert_eq!(
            register(
                &hub.object,
                any_device,
                GAMEPAD_KIND,
                1,
                0,
                no_context,
                Some(hold),
                &mut token
            ),
            S_OK
        );
        // The title takes its lock, then registers once the poll thread is
        // inside `hold`, waiting for that lock.
        let registrar = std::thread::spawn(move || {
            let title = TITLE.lock().unwrap();
            LOCKED.store(true, Ordering::SeqCst);
            let deadline = Instant::now() + Duration::from_secs(2);
            while HELD.load(Ordering::SeqCst) == 0 && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(1));
            }
            let mut token = 0;
            let any_device = std::ptr::null();
            let r = register(
                &hub.object,
                any_device,
                GAMEPAD_KIND,
                1,
                2,
                std::ptr::null_mut(),
                Some(seen),
                &mut token,
            );
            let seen_at_return = SEEN.load(Ordering::SeqCst);
            drop(title);
            (r, seen_at_return, std::thread::current().id())
        });
        while !LOCKED.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        // Plugging a pad in runs `hold` on the poll thread.
        let poller = std::thread::spawn(move || {
            host.lock().unwrap().pads[1] = Some(PadState::default());
            hub.poll();
        });
        let deadline = Instant::now() + Duration::from_secs(2);
        while !registrar.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(registrar.is_finished(), "registration returned");
        let (r, seen_at_return, registering) = registrar.join().unwrap();
        assert_eq!(r, S_OK);
        assert_ne!(
            HELD.load(Ordering::SeqCst),
            0,
            "the poll thread was delivering"
        );
        assert_ne!(seen_at_return, 0, "its callbacks ran before it returned");
        assert_eq!(*SEEN_ON.lock().unwrap(), Some(registering));
        poller.join().unwrap();
    }

    #[test]
    fn devices_are_found_by_id_and_rumble_reaches_the_host() {
        let (hub, host) = fake();
        host.lock().unwrap().pads[2] = Some(PadState::default());
        hub.poll();
        let device = &hub.devices[2] as *const Device;
        let info = slot::<extern "system" fn(*const Device) -> *const u8, _>(device, 0x18);
        // SAFETY: the info has the id at +0x20.
        let id = unsafe { *(info(device).add(0x20) as *const [u8; 32]) };
        assert_ne!(id, [0; 32]);

        type Find = extern "system" fn(This, *const [u8; 32], Out) -> HRESULT;
        let find = slot::<Find, _>(&hub.object, 0x80);
        let mut out = std::ptr::null();
        assert_eq!(find(&hub.object, &id, &mut out), S_OK);
        assert_eq!(out, device.cast());
        assert_eq!(find(&hub.object, &[1; 32], &mut out), DEVICE_NOT_FOUND);
        assert!(out.is_null());

        type Rumble = extern "system" fn(*const Device, *const [f32; 4]);
        let rumble = slot::<Rumble, _>(device, 0x50);
        rumble(device, &[0.5, 1.0, 1.0, 1.0]);
        rumble(&hub.devices[FALLBACK], &[1.0; 4]);
        rumble(device, std::ptr::null());
        let full = Motors {
            low: 32767,
            high: 65535,
            left_trigger: 65535,
            right_trigger: 65535,
        };
        assert_eq!(
            host.lock().unwrap().rumble,
            [(2, full), (2, Motors::default())]
        );
    }

    #[test]
    fn trigger_rumble_reaches_the_host() {
        let (hub, host) = fake();
        host.lock().unwrap().pads[1] = Some(PadState::default());
        hub.poll();
        let device = &hub.devices[1] as *const Device;
        type Rumble = extern "system" fn(*const Device, *const [f32; 4]);
        let rumble = slot::<Rumble, _>(device, 0x50);
        // Out-of-range and NaN levels clamp or fall to off, like the grips.
        rumble(device, &[0.0, 0.0, 0.25, 2.0]);
        rumble(device, &[0.0, 0.0, -1.0, f32::NAN]);
        let expected = [
            (
                1,
                Motors {
                    left_trigger: 16383,
                    right_trigger: 65535,
                    ..Motors::default()
                },
            ),
            (1, Motors::default()),
        ];
        assert_eq!(host.lock().unwrap().rumble, expected);
    }

    /// A title releasing a reading once too often: the count stays 0 and
    /// nothing is freed again (the old `fetch_sub(1) - 1` overflowed).
    #[test]
    fn reading_release_past_zero_leaves_the_count_at_zero() {
        let reading = Box::into_raw(Box::new(Reading {
            vtable: &READING,
            refs: AtomicU32::new(0),
            device: std::ptr::null(),
            sequence: 0,
            timestamp: 0,
            value: Value::Keys(Vec::new()),
        }));
        assert_eq!(reading_release(reading), 0);
        // SAFETY: still allocated: `reading_release` freed nothing.
        assert_eq!(unsafe { &*reading }.refs.load(Ordering::Relaxed), 0);
        // SAFETY: boxed above and not freed.
        drop(unsafe { Box::from_raw(reading) });
    }
}
