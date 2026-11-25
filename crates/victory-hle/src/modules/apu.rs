//! The console's audio unit (`xapu.dll`): Opus packets or float PCM in,
//! float PCM out, optionally resampled with a pitch change.
//!
//! A title connects once for one kind of processing — decode Opus, decode
//! and convert, or convert PCM — gets a block of memory for its packets and
//! output, and then queues per-stream commands (activate, process, reset,
//! deactivate) and dequeues one result per command. We do each command when
//! it is queued, so its result is ready at once; the order of results is the
//! order of commands. The rate conversion itself lives in
//! [`super::apu_resample`], which is testable on every platform.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::Layout;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::Mutex;

use opus_pure::OpusDecoder;
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::Threading::{CreateEventW, SetEvent};

use super::apu_resample::Resampler;
use super::xasync::E_INVALIDARG;
use super::xgameruntime::{HRESULT, S_OK};
use crate::{HleModule, hle_export};

const E_NOTIMPL: HRESULT = 0x8000_4001_u32 as i32;
const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;
const QUEUE_FULL: HRESULT = 0x8AC8_1000_u32 as i32;
/// No result yet, which titles poll for.
pub(crate) const PENDING_RESULTS: HRESULT = 0x8AC8_1001_u32 as i32;
const INVALID_COMMAND: HRESULT = 0x8AC8_1002_u32 as i32;
const INVALID_STREAM: HRESULT = 0x8AC8_1003_u32 as i32;
const ALREADY_ACTIVE: HRESULT = 0x8AC8_1005_u32 as i32;
const BAD_PACKET: HRESULT = 0x8AC8_1008_u32 as i32;

/// XApuProcessType: decode, decode and resample, resample float PCM.
const DECODE_OPUS: u32 = 1;
const DECODE_CONVERT_OPUS: u32 = 2;
const CONVERT_PCM: u32 = 3;
/// The output lengths the hardware accepts when converting.
const OUTPUT_FRAMES: [u32; 5] = [256, 480, 512, 960, 1024];

const ACTIVATE: u32 = 1;
const DEACTIVATE: u32 = 2;
const PROCESS: u32 = 3;
const NOTIFY: u32 = 4;
const RESET: u32 = 5;

/// Command options: signal the connection's event, write out a partial frame.
const SIGNAL: u32 = 1;
const FLUSH: u32 = 2;
/// Connect options (XApuConnectOptions).
const DEINTERLEAVED_INPUT: u32 = 2;
const DEINTERLEAVED_OUTPUT: u32 = 4;
const RESULT_EVENT: u32 = 8;

/// The longest Opus packet duration at 48 kHz (120 ms).
const MAX_FRAMES: usize = 5760;

#[repr(C)]
pub struct ConnectInput {
    process: u32,
    streams: u32,
    commands_per_stream: u32,
    memory: u32,
    frames_per_output: u32,
    options: u32,
}

#[repr(C)]
pub struct ConnectOutput {
    memory: *mut c_void,
    result_event: HANDLE,
}

/// What every command starts with, and what its result echoes.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct CommandId {
    kind: u32,
    stream: u32,
    sequence: u32,
    options: u32,
    context: u64,
}

#[repr(C, packed)]
struct Activate {
    id: CommandId,
    channels: u32,
    init_length: u32,
    init: *const c_void,
    start_pitch: f32,
    processing_length: u32,
    processing: *mut c_void,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct Process {
    id: CommandId,
    pitch_ramp: f32,
    target_pitch: f32,
    first_frame: u32,
    frame_count: u32,
    input_length: u32,
    input: *const u8,
    output_length: u32,
    output: *mut u8,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct ApuResult {
    id: CommandId,
    state: u32,
    state_data: u64,
    usec: u64,
    error: HRESULT,
    frames: u32,
    output_length: u32,
    output: *mut u8,
    pitch: f32,
    input_length: u32,
    input: *const u8,
}

// SAFETY: the raw pointers are title addresses carried through, never
// dereferenced after the result is made.
unsafe impl Send for ApuResult {}

pub struct Connection {
    memory: *mut u8,
    layout: Layout,
    event: HANDLE,
    process: u32,
    /// The deinterleave midpoint of a converter's output buffer.
    frames_per_output: u32,
    options: u32,
    per_stream: u32,
    state: Mutex<State>,
}

/// The decoders and converters a stream owns: a decode connection needs a
/// decoder, a convert connection a resampler, decode-convert both.
struct Stream {
    decoder: Option<OpusDecoder>,
    resampler: Option<Resampler>,
    start_pitch: f32,
}

struct State {
    streams: Vec<Option<Stream>>,
    /// Results not yet dequeued, per stream.
    queued: Vec<u32>,
    results: VecDeque<ApuResult>,
    commands: u32,
    dequeued: u32,
}

fn lock(c: &Connection) -> std::sync::MutexGuard<'_, State> {
    c.state.lock().unwrap_or_else(|e| e.into_inner())
}

pub extern "system" fn XApuConnect(
    input: *const ConnectInput,
    output: *mut ConnectOutput,
    handle: *mut *mut Connection,
) -> HRESULT {
    if input.is_null() || output.is_null() || handle.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    let input = unsafe { &*input };
    if !(1..=500).contains(&input.streams)
        || !(1..=16).contains(&input.commands_per_stream)
        || input.memory == 0
    {
        return E_INVALIDARG;
    }
    if !matches!(
        input.process,
        DECODE_OPUS | DECODE_CONVERT_OPUS | CONVERT_PCM
    ) {
        tracing::warn!(
            process = input.process,
            "audio decoder: unknown processing kind"
        );
        return E_NOTIMPL;
    }
    // Plain decoding writes what the packet holds; a converter fills a
    // buffer of one of the durations the hardware lists.
    if input.process == DECODE_OPUS {
        if input.frames_per_output != 0 {
            return E_INVALIDARG;
        }
    } else if !OUTPUT_FRAMES.contains(&input.frames_per_output) {
        return E_INVALIDARG;
    }
    let Ok(layout) = Layout::from_size_align(input.memory as usize, 16) else {
        return E_INVALIDARG;
    };
    // shortcut: a converter's per-stream processing buffer is a host Vec in
    // `Resampler`, not a slice carved out of this title-visible block; the
    // block still holds the commands' inputs and outputs as on the console.
    // SAFETY: non-zero size.
    let memory = unsafe { std::alloc::alloc_zeroed(layout) };
    if memory.is_null() {
        return E_OUTOFMEMORY;
    }
    let event = if input.options & RESULT_EVENT != 0 {
        // SAFETY: plain auto-reset event, no name.
        unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) }
    } else {
        std::ptr::null_mut()
    };
    let streams = input.streams as usize;
    tracing::info!(
        streams,
        memory = input.memory,
        options = input.options,
        "audio decoder connected"
    );
    let connection = Box::new(Connection {
        memory,
        layout,
        event,
        process: input.process,
        frames_per_output: input.frames_per_output,
        options: input.options,
        per_stream: input.commands_per_stream,
        state: Mutex::new(State {
            streams: (0..streams).map(|_| None).collect(),
            queued: vec![0; streams],
            results: VecDeque::new(),
            commands: 0,
            dequeued: 0,
        }),
    });
    // SAFETY: checked non-null.
    unsafe {
        output.write(ConnectOutput {
            memory: memory.cast(),
            result_event: event,
        });
        handle.write(Box::into_raw(connection));
    }
    S_OK
}

pub extern "system" fn XApuDisconnect(handle: *mut Connection) -> HRESULT {
    if handle.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: made by `XApuConnect`; disconnected once.
    let connection = unsafe { Box::from_raw(handle) };
    if !connection.event.is_null() {
        // SAFETY: our event.
        unsafe { CloseHandle(connection.event) };
    }
    // SAFETY: allocated in `XApuConnect` with this layout.
    unsafe { std::alloc::dealloc(connection.memory, connection.layout) };
    S_OK
}

pub extern "system" fn XApuEnqueueCommand(
    handle: *mut Connection,
    command: *const CommandId,
    total: *mut u32,
) -> HRESULT {
    if handle.is_null() || command.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: a connection from `XApuConnect`; every command starts with an id.
    let (connection, id) = unsafe { (&*handle, command.read_unaligned()) };
    let mut state = lock(connection);
    let stream = id.stream as usize;
    if stream >= state.streams.len() {
        return INVALID_STREAM;
    }
    if state.queued[stream] >= connection.per_stream {
        return QUEUE_FULL;
    }
    let mut result = ApuResult {
        id,
        state: 0,
        state_data: 0,
        usec: 0,
        error: S_OK,
        frames: 0,
        output_length: 0,
        output: std::ptr::null_mut(),
        pitch: 1.0,
        input_length: 0,
        input: std::ptr::null(),
    };
    let stream_state = &mut state.streams[stream];
    match id.kind {
        ACTIVATE => {
            // SAFETY: an activate command is this struct.
            let activate = unsafe { command.cast::<Activate>().read_unaligned() };
            let channels = activate.channels as usize;
            tracing::debug!(
                stream,
                channels,
                process = connection.process,
                "APU stream activated"
            );
            if stream_state.is_some() {
                result.error = ALREADY_ACTIVE;
            } else if !(1..=2).contains(&channels) {
                result.error = E_INVALIDARG;
            } else {
                let decoder = match connection.process {
                    CONVERT_PCM => None,
                    _ => match OpusDecoder::new(48_000, channels) {
                        Ok(d) => Some(d),
                        Err(_) => {
                            result.error = E_INVALIDARG;
                            None
                        }
                    },
                };
                let resampler = match connection.process {
                    DECODE_OPUS => None,
                    _ => Some(Resampler::new(channels, activate.start_pitch)),
                };
                if result.error == S_OK {
                    *stream_state = Some(Stream {
                        decoder,
                        resampler,
                        start_pitch: activate.start_pitch,
                    });
                }
            }
        }
        DEACTIVATE => *stream_state = None,
        RESET => {
            if let Some(stream) = stream_state {
                let pitch = stream.start_pitch;
                if let Some(d) = &mut stream.decoder {
                    let _ = d.reset_state();
                }
                if let Some(r) = &mut stream.resampler {
                    r.reset(pitch);
                }
            }
        }
        NOTIFY => {}
        PROCESS => {
            // SAFETY: a process command is this struct.
            let command = unsafe { command.cast::<Process>().read_unaligned() };
            match stream_state {
                Some(stream) => {
                    match (
                        connection.process,
                        &mut stream.decoder,
                        &mut stream.resampler,
                    ) {
                        (DECODE_OPUS, Some(d), _) => {
                            process(d, connection.options, &command, &mut result)
                        }
                        (DECODE_CONVERT_OPUS, Some(d), Some(r)) => process_convert(
                            Some(d),
                            r,
                            connection.frames_per_output,
                            connection.options,
                            &command,
                            &mut result,
                        ),
                        (CONVERT_PCM, _, Some(r)) => process_convert(
                            None,
                            r,
                            connection.frames_per_output,
                            connection.options,
                            &command,
                            &mut result,
                        ),
                        _ => result.error = INVALID_STREAM,
                    }
                    if let Some(r) = &stream.resampler {
                        result.pitch = r.pitch();
                    }
                }
                None => result.error = INVALID_STREAM,
            }
        }
        _ => return INVALID_COMMAND,
    }
    state.queued[stream] += 1;
    state.commands = state.commands.wrapping_add(1);
    state.results.push_back(result);
    if !total.is_null() {
        // SAFETY: optional out parameter.
        unsafe { total.write(state.commands) };
    }
    if id.options & SIGNAL != 0 && !connection.event.is_null() {
        // SAFETY: our event.
        unsafe { SetEvent(connection.event) };
    }
    S_OK
}

/// Decodes one packet into the command's output buffer, keeping the frames
/// from `first_frame` on, at most `frame_count` of them.
fn process(decoder: &mut OpusDecoder, options: u32, command: &Process, result: &mut ApuResult) {
    let channels = decoder.channels();
    let (input, input_length) = (command.input, command.input_length);
    let (output, output_length) = (command.output, command.output_length);
    result.input = input;
    result.input_length = input_length;
    result.output = output;
    if input.is_null() || output.is_null() || input_length == 0 {
        result.error = E_INVALIDARG;
        return;
    }
    // SAFETY: the title's packet in its connection memory.
    let packet = unsafe { std::slice::from_raw_parts(input, input_length as usize) };
    let mut pcm = vec![0f32; MAX_FRAMES * channels];
    let decoded = match decoder.decode(packet, MAX_FRAMES, &mut pcm) {
        Ok(n) => n,
        Err(_) => {
            result.error = BAD_PACKET;
            return;
        }
    };
    let first = (command.first_frame as usize).min(decoded);
    let frames = (decoded - first)
        .min(command.frame_count as usize)
        .min(output_length as usize / (4 * channels));
    let kept = &pcm[first * channels..(first + frames) * channels];
    let mut out = vec![0f32; frames * channels];
    if options & DEINTERLEAVED_OUTPUT != 0 && channels == 2 {
        for (i, frame) in kept.as_chunks::<2>().0.iter().enumerate() {
            (out[i], out[frames + i]) = (frame[0], frame[1]);
        }
    } else {
        out.copy_from_slice(kept);
    }
    // Copied as bytes: the title's buffer need not be aligned for f32 (#839).
    // SAFETY: the title's output buffer holds `output_length` bytes, and
    // `frames` was bounded by it above.
    unsafe {
        std::ptr::copy_nonoverlapping(out.as_ptr().cast::<u8>(), output, out.len() * 4);
    }
    result.frames = frames as u32;
    result.output_length = (frames * channels * 4) as u32;
}

/// Decodes (when the connection decodes) and resamples one command's input
/// into float PCM at the pitch the title asked for.
fn process_convert(
    decoder: Option<&mut OpusDecoder>,
    resampler: &mut Resampler,
    max_frames: u32,
    options: u32,
    command: &Process,
    result: &mut ApuResult,
) {
    let channels = resampler.channels();
    let (input, input_length) = (command.input, command.input_length);
    let (output, output_length) = (command.output, command.output_length);
    result.input = input;
    result.input_length = input_length;
    result.output = output;
    if input.is_null() || output.is_null() || input_length == 0 {
        result.error = E_INVALIDARG;
        return;
    }
    // SAFETY: the title's buffer in its connection memory.
    let bytes = unsafe { std::slice::from_raw_parts(input, input_length as usize) };
    let pcm;
    if let Some(decoder) = decoder {
        let mut decoded = vec![0f32; MAX_FRAMES * channels];
        let n = match decoder.decode(bytes, MAX_FRAMES, &mut decoded) {
            Ok(n) => n,
            Err(_) => {
                result.error = BAD_PACKET;
                return;
            }
        };
        let first = (command.first_frame as usize).min(n);
        let frames = (n - first).min(command.frame_count as usize);
        pcm = decoded[first * channels..(first + frames) * channels].to_vec();
    } else {
        // ConvertPCM's frameCount is required and fixes the input length.
        let frames = command.frame_count as usize;
        let minimum = if channels == 2 { 8 } else { 4 };
        let multiple = if channels == 2 { 8 } else { 4 };
        if frames < minimum
            || frames > 960
            || !frames.is_multiple_of(multiple)
            || input_length as usize != frames * channels * 4
        {
            result.error = E_INVALIDARG;
            return;
        }
        let samples: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        pcm = if options & DEINTERLEAVED_INPUT != 0 && channels == 2 {
            let frames = samples.len() / 2;
            let mut interleaved = vec![0f32; samples.len()];
            for (i, frame) in interleaved.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                *frame = [samples[i], samples[frames + i]];
            }
            interleaved
        } else {
            samples
        };
    }
    let deinterleaved_output = options & DEINTERLEAVED_OUTPUT != 0 && channels == 2;
    let mut room = output_length as usize / (4 * channels);
    if deinterleaved_output {
        // The right channel starts at the midpoint; more frames than that
        // would write the left channel over it.
        room = room.min(max_frames as usize);
    }
    let mut converted = vec![0f32; room * channels];
    let frames = resampler.convert(
        &pcm,
        &mut converted,
        command.target_pitch,
        command.pitch_ramp,
        // FLUSH is a command option; `options` are the connection's.
        { command.id.options } & FLUSH != 0,
    );
    if deinterleaved_output {
        // The converter's midpoint is `maxFrameCountPerOutput`, even when
        // fewer frames were written (the docs spell this out).
        let midpoint = max_frames as usize;
        let len = midpoint + frames;
        if len * 4 > output_length as usize {
            result.error = E_INVALIDARG;
            return;
        }
        // SAFETY: the title's output buffer holds `output_length` bytes.
        let out = unsafe { std::slice::from_raw_parts_mut(output.cast::<f32>(), len) };
        for (i, frame) in converted[..frames * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .enumerate()
        {
            out[i] = frame[0];
            out[midpoint + i] = frame[1];
        }
    } else {
        // SAFETY: the title's output buffer holds `output_length` bytes.
        let out =
            unsafe { std::slice::from_raw_parts_mut(output.cast::<f32>(), frames * channels) };
        out.copy_from_slice(&converted[..frames * channels]);
    }
    result.frames = frames as u32;
    result.output_length = (frames * channels * 4) as u32;
    result.pitch = resampler.pitch();
}

pub extern "system" fn XApuDequeueResult(handle: *mut Connection, out: *mut ApuResult) -> HRESULT {
    if handle.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: a connection from `XApuConnect`.
    let mut state = lock(unsafe { &*handle });
    let Some(result) = state.results.pop_front() else {
        return PENDING_RESULTS;
    };
    let stream = result.id.stream as usize;
    state.queued[stream] -= 1;
    state.dequeued = state.dequeued.wrapping_add(1);
    // SAFETY: the title passes room for one result.
    unsafe { out.write_unaligned(result) };
    S_OK
}

pub extern "system" fn XApuGetResultCount(handle: *mut Connection, total: *mut u32) -> u32 {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: a connection from `XApuConnect`.
    let state = lock(unsafe { &*handle });
    if !total.is_null() {
        // SAFETY: optional out parameter.
        unsafe { total.write(state.dequeued) };
    }
    state.results.len() as u32
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "apu",
        dlls: &["xapu.dll"],
        exports: vec![
            hle_export!(XApuConnect),
            hle_export!(XApuDisconnect),
            hle_export!(XApuEnqueueCommand),
            hle_export!(XApuDequeueResult),
            hle_export!(XApuGetResultCount),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opus_pure::{Application, OpusEncoder};
    use windows_sys::Win32::Foundation::WAIT_OBJECT_0;
    use windows_sys::Win32::System::Threading::WaitForSingleObject;

    fn id(kind: u32, stream: u32) -> CommandId {
        CommandId {
            kind,
            stream,
            sequence: 7,
            options: 0,
            context: 0x1234,
        }
    }

    /// Output written to a buffer at any byte offset, as a title may pass
    /// one, is the same as to an aligned one (#839).
    #[test]
    fn decodes_into_an_unaligned_buffer() {
        let tone: Vec<f32> = (0..960 * 2)
            .map(|i| ((i / 2) as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.5)
            .collect();
        let mut encoder = OpusEncoder::new(48_000, 2, Application::Audio).unwrap();
        let mut packet = [0u8; 1500];
        let n = encoder.encode(&tone, 960, &mut packet).unwrap();
        let decode = |options: u32, at: usize| {
            let mut buffer = vec![0u8; 960 * 8 + 8];
            let command = Process {
                id: id(PROCESS, 0),
                pitch_ramp: 0.0,
                target_pitch: 1.0,
                first_frame: 0,
                frame_count: 960,
                input_length: n as u32,
                input: packet.as_ptr(),
                output_length: 960 * 8,
                output: buffer.as_mut_ptr().wrapping_add(at),
            };
            // SAFETY: a zeroed result is valid.
            let mut result: ApuResult = unsafe { std::mem::zeroed() };
            let mut decoder = OpusDecoder::new(48_000, 2).unwrap();
            process(&mut decoder, options, &command, &mut result);
            let (error, frames) = (result.error, result.frames);
            assert_eq!((error, frames), (S_OK, 960));
            buffer[at..at + 960 * 8].to_vec()
        };
        for options in [0, DEINTERLEAVED_OUTPUT] {
            let aligned = decode(options, 0);
            assert!(aligned.iter().any(|&b| b != 0));
            for at in 1..4 {
                assert!(decode(options, at) == aligned, "offset {at}");
            }
        }
    }

    #[test]
    fn decodes_opus_as_the_title_drives_it() {
        assert_eq!(size_of::<ApuResult>(), 0x50);
        // 256 mono/stereo streams, two commands each, as the title connects.
        let input = ConnectInput {
            process: DECODE_OPUS,
            streams: 256,
            commands_per_stream: 2,
            memory: 0x10_4000,
            frames_per_output: 0,
            options: 1,
        };
        let mut output = ConnectOutput {
            memory: std::ptr::null_mut(),
            result_event: std::ptr::null_mut(),
        };
        let mut handle = std::ptr::null_mut();
        assert_eq!(XApuConnect(&input, &mut output, &mut handle), S_OK);
        let base = output.memory.cast::<u8>();

        let activate = Activate {
            id: id(ACTIVATE, 3),
            channels: 2,
            init_length: 0,
            init: std::ptr::null(),
            start_pitch: 1.0,
            processing_length: 0,
            processing: std::ptr::null_mut(),
        };
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const activate).cast(), std::ptr::null_mut()),
            S_OK
        );

        // A 20 ms stereo packet of a 440 Hz tone.
        let tone: Vec<f32> = (0..960 * 2)
            .map(|i| ((i / 2) as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.5)
            .collect();
        let mut encoder = OpusEncoder::new(48_000, 2, Application::Audio).unwrap();
        let mut packet = [0u8; 1500];
        let n = encoder.encode(&tone, 960, &mut packet).unwrap();
        // SAFETY: inside the connection memory.
        unsafe { std::ptr::copy_nonoverlapping(packet.as_ptr(), base, n) };
        let process = Process {
            id: id(PROCESS, 3),
            pitch_ramp: 0.0,
            target_pitch: 1.0,
            first_frame: 0,
            frame_count: 960,
            input_length: n as u32,
            input: base,
            output_length: 0xf00 * 2,
            output: base.wrapping_add(0x14000),
        };
        let mut total = 0;
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const process).cast(), &mut total),
            S_OK
        );
        assert_eq!(total, 2);
        // Two results queued for stream 3: its queue is full.
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const process).cast(), &mut total),
            QUEUE_FULL
        );
        assert_eq!(XApuGetResultCount(handle, std::ptr::null_mut()), 2);

        let mut result = std::mem::MaybeUninit::<ApuResult>::zeroed();
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        // SAFETY: written above.
        let first = unsafe { result.assume_init() };
        let (kind, error, context) = (first.id.kind, first.error, first.id.context);
        assert_eq!((kind, error, context), (ACTIVATE, S_OK, 0x1234));
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        // SAFETY: written above.
        let decoded = unsafe { result.assume_init() };
        let (kind, error, frames, length) = (
            decoded.id.kind,
            decoded.error,
            decoded.frames,
            decoded.output_length,
        );
        assert_eq!((kind, error, frames, length), (PROCESS, S_OK, 960, 960 * 8));
        // SAFETY: the decoder wrote 960 stereo frames there.
        let pcm =
            unsafe { std::slice::from_raw_parts(base.wrapping_add(0x14000).cast::<f32>(), 1920) };
        let energy: f32 = pcm.iter().map(|s| s * s).sum::<f32>() / 1920.0;
        assert!(energy > 0.01, "decoded a tone, energy {energy}");
        assert_eq!(
            XApuDequeueResult(handle, result.as_mut_ptr()),
            PENDING_RESULTS
        );
        let mut dequeued = 0;
        assert_eq!(XApuGetResultCount(handle, &mut dequeued), 0);
        assert_eq!(dequeued, 2);

        // A process on a stream that was never activated fails in its result.
        let mut stray = process;
        stray.id.stream = 4;
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const stray).cast(), &mut total),
            S_OK
        );
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        // SAFETY: written above.
        let error = unsafe { result.assume_init() }.error;
        assert_eq!(error, INVALID_STREAM);
        assert_eq!(XApuDisconnect(handle), S_OK);
    }

    fn connect(process: u32, frames: u32) -> (*mut Connection, *mut u8) {
        connect_with(process, frames, 0)
    }

    fn connect_with(process: u32, frames: u32, options: u32) -> (*mut Connection, *mut u8) {
        let input = ConnectInput {
            process,
            streams: 1,
            commands_per_stream: 2,
            memory: 0x10_0000,
            frames_per_output: frames,
            options,
        };
        let mut output = ConnectOutput {
            memory: std::ptr::null_mut(),
            result_event: std::ptr::null_mut(),
        };
        let mut handle = std::ptr::null_mut();
        assert_eq!(XApuConnect(&input, &mut output, &mut handle), S_OK);
        (handle, output.memory.cast::<u8>())
    }

    fn activate(handle: *mut Connection, channels: u32, pitch: f32) {
        let command = Activate {
            id: id(ACTIVATE, 0),
            channels,
            init_length: 0,
            init: std::ptr::null(),
            start_pitch: pitch,
            processing_length: 0,
            processing: std::ptr::null_mut(),
        };
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const command).cast(), std::ptr::null_mut()),
            S_OK
        );
    }

    #[test]
    fn connects_for_both_conversion_kinds() {
        let mut output = ConnectOutput {
            memory: std::ptr::null_mut(),
            result_event: std::ptr::null_mut(),
        };
        let mut handle = std::ptr::null_mut();
        fn connect_as(
            input: &ConnectInput,
            output: &mut ConnectOutput,
            handle: &mut *mut Connection,
        ) -> HRESULT {
            XApuConnect(input, output, handle)
        }
        let input = |process, frames| ConnectInput {
            process,
            streams: 1,
            commands_per_stream: 1,
            memory: 0x1_0000,
            frames_per_output: frames,
            options: 0,
        };
        assert_eq!(
            connect_as(&input(DECODE_OPUS, 0), &mut output, &mut handle),
            S_OK
        );
        assert_eq!(XApuDisconnect(handle), S_OK);
        for frames in OUTPUT_FRAMES {
            for process in [DECODE_CONVERT_OPUS, CONVERT_PCM] {
                assert_eq!(
                    connect_as(&input(process, frames), &mut output, &mut handle),
                    S_OK
                );
                assert_eq!(XApuDisconnect(handle), S_OK);
            }
        }
        assert_eq!(
            connect_as(&input(DECODE_OPUS, 256), &mut output, &mut handle),
            E_INVALIDARG
        );
        assert_eq!(
            connect_as(&input(DECODE_CONVERT_OPUS, 0), &mut output, &mut handle),
            E_INVALIDARG
        );
        assert_eq!(
            connect_as(&input(CONVERT_PCM, 7), &mut output, &mut handle),
            E_INVALIDARG
        );
        assert_eq!(
            connect_as(&input(9, 256), &mut output, &mut handle),
            E_NOTIMPL
        );
    }

    #[test]
    fn deinterleaved_output_stops_at_the_channel_midpoint() {
        // A 256-frame midpoint, half pitch, and room for 1024 frames: the
        // left channel must not run on over the right one.
        let (handle, base) = connect_with(CONVERT_PCM, 256, DEINTERLEAVED_OUTPUT);
        activate(handle, 2, 0.5);
        let samples: Vec<f32> = (0..256)
            .flat_map(|i| [i as f32, 1000.0 + i as f32])
            .collect();
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        // SAFETY: inside the connection memory.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), base, bytes.len()) };
        let command = Process {
            id: id(PROCESS, 0),
            pitch_ramp: 0.0,
            target_pitch: 0.5,
            first_frame: 0,
            frame_count: 256,
            input_length: bytes.len() as u32,
            input: base,
            output_length: 1024 * 2 * 4,
            output: base.wrapping_add(0x1000),
        };
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const command).cast(), std::ptr::null_mut()),
            S_OK
        );
        let mut result = std::mem::MaybeUninit::<ApuResult>::zeroed();
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        // SAFETY: written above.
        let result = unsafe { result.assume_init() };
        let (error, frames) = (result.error, result.frames);
        assert_eq!((error, frames), (S_OK, 256));
        // SAFETY: the converter wrote both 256-float halves there.
        let pcm =
            unsafe { std::slice::from_raw_parts(base.wrapping_add(0x1000).cast::<f32>(), 512) };
        assert_eq!(pcm[..4], [0.0, 0.5, 1.0, 1.5]);
        assert_eq!(pcm[255], 127.5);
        assert_eq!(pcm[256..260], [1000.0, 1000.5, 1001.0, 1001.5]);
        assert_eq!(pcm[511], 1127.5);
        assert_eq!(XApuDisconnect(handle), S_OK);
    }

    #[test]
    fn converts_float_pcm_at_the_target_pitch() {
        let (handle, base) = connect(CONVERT_PCM, 256);
        activate(handle, 1, 1.0);
        let samples: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        // SAFETY: inside the connection memory.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), base, bytes.len()) };
        let command = Process {
            id: id(PROCESS, 0),
            pitch_ramp: 0.0,
            target_pitch: 2.0,
            first_frame: 0,
            frame_count: 16,
            input_length: bytes.len() as u32,
            input: base,
            output_length: 256 * 4,
            output: base.wrapping_add(0x1000),
        };
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const command).cast(), std::ptr::null_mut()),
            S_OK
        );
        let mut result = std::mem::MaybeUninit::<ApuResult>::zeroed();
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        // SAFETY: written above.
        let first = unsafe { result.assume_init() };
        let first_kind = first.id.kind;
        assert_eq!(first_kind, ACTIVATE);
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        // SAFETY: written above.
        let result = unsafe { result.assume_init() };
        let (error, frames, pitch, output_length) = (
            result.error,
            result.frames,
            result.pitch,
            result.output_length,
        );
        assert_eq!((error, frames, pitch, output_length), (S_OK, 8, 2.0, 8 * 4));
        // SAFETY: the converter wrote eight mono frames there.
        let pcm = unsafe { std::slice::from_raw_parts(base.wrapping_add(0x1000).cast::<f32>(), 8) };
        assert_eq!(pcm, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]);
        assert_eq!(XApuDisconnect(handle), S_OK);
    }

    #[test]
    fn decodes_and_converts_an_opus_stream() {
        let (handle, base) = connect(DECODE_CONVERT_OPUS, 960);
        activate(handle, 2, 1.0);
        let tone: Vec<f32> = (0..960 * 2)
            .map(|i| ((i / 2) as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.5)
            .collect();
        let mut encoder = OpusEncoder::new(48_000, 2, Application::Audio).unwrap();
        let mut packet = [0u8; 1500];
        let n = encoder.encode(&tone, 960, &mut packet).unwrap();
        // SAFETY: inside the connection memory.
        unsafe { std::ptr::copy_nonoverlapping(packet.as_ptr(), base, n) };
        let command = Process {
            id: id(PROCESS, 0),
            pitch_ramp: 0.0,
            target_pitch: 0.5,
            first_frame: 0,
            frame_count: 960,
            input_length: n as u32,
            input: base,
            output_length: 2048 * 2 * 4,
            output: base.wrapping_add(0x1000),
        };
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const command).cast(), std::ptr::null_mut()),
            S_OK
        );
        let mut result = std::mem::MaybeUninit::<ApuResult>::zeroed();
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        // SAFETY: written above.
        let first = unsafe { result.assume_init() };
        let first_kind = first.id.kind;
        assert_eq!(first_kind, ACTIVATE);
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        // SAFETY: written above.
        let result = unsafe { result.assume_init() };
        let (error, frames, pitch, output_length) = (
            result.error,
            result.frames,
            result.pitch,
            result.output_length,
        );
        assert_eq!(error, S_OK);
        assert!(frames > 960, "a half pitch writes more frames");
        assert_eq!(pitch, 0.5);
        assert_eq!(output_length, frames * 2 * 4);
        // SAFETY: the converter wrote `frames` stereo frames there.
        let pcm = unsafe {
            std::slice::from_raw_parts(base.wrapping_add(0x1000).cast::<f32>(), frames as usize * 2)
        };
        let energy: f32 = pcm.iter().map(|s| s * s).sum::<f32>() / pcm.len() as f32;
        assert!(energy > 0.01, "converted a tone, energy {energy}");
        assert_eq!(XApuDisconnect(handle), S_OK);
    }

    /// Where the audio unit tests put title output inside connection memory.
    const OUT: usize = 0x4000;

    struct Conn {
        handle: *mut Connection,
        base: *mut u8,
        event: HANDLE,
    }

    fn open(process: u32, frames: u32, options: u32, streams: u32, per_stream: u32) -> Conn {
        let input = ConnectInput {
            process,
            streams,
            commands_per_stream: per_stream,
            memory: 0x10_0000,
            frames_per_output: frames,
            options,
        };
        let mut output = ConnectOutput {
            memory: std::ptr::null_mut(),
            result_event: std::ptr::null_mut(),
        };
        let mut handle = std::ptr::null_mut();
        assert_eq!(XApuConnect(&input, &mut output, &mut handle), S_OK);
        Conn {
            handle,
            base: output.memory.cast(),
            event: output.result_event,
        }
    }

    fn close(c: Conn) {
        assert_eq!(XApuDisconnect(c.handle), S_OK);
    }

    fn enqueue<T>(handle: *mut Connection, command: &T) -> HRESULT {
        XApuEnqueueCommand(handle, (command as *const T).cast(), std::ptr::null_mut())
    }

    fn dequeue(handle: *mut Connection) -> ApuResult {
        let mut result = std::mem::MaybeUninit::<ApuResult>::zeroed();
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        // SAFETY: written by the successful dequeue above.
        unsafe { result.assume_init() }
    }

    /// Queues an activate and returns the error its result carries.
    fn activate_stream(c: &Conn, stream: u32, channels: u32) -> HRESULT {
        let command = Activate {
            id: id(ACTIVATE, stream),
            channels,
            init_length: 0,
            init: std::ptr::null(),
            start_pitch: 1.0,
            processing_length: 0,
            processing: std::ptr::null_mut(),
        };
        assert_eq!(enqueue(c.handle, &command), S_OK);
        dequeue(c.handle).error
    }

    /// Queues a stream command that is only an id and returns its result.
    fn simple(c: &Conn, kind: u32, stream: u32) -> ApuResult {
        assert_eq!(enqueue(c.handle, &id(kind, stream)), S_OK);
        dequeue(c.handle)
    }

    /// 20 ms of a 440 Hz tone, continuing from the `frame`th 20 ms.
    fn tone(channels: usize, frame: usize) -> Vec<f32> {
        (0..960)
            .flat_map(|i| {
                let t = (frame * 960 + i) as f32;
                let sample = (t * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.5;
                std::iter::repeat_n(sample, channels)
            })
            .collect()
    }

    /// `count` consecutive 20 ms packets from one encoder.
    fn packets(channels: usize, count: usize) -> Vec<Vec<u8>> {
        let mut encoder = OpusEncoder::new(48_000, channels, Application::Audio).unwrap();
        (0..count)
            .map(|frame| {
                let mut packet = [0u8; 1500];
                let n = encoder
                    .encode(&tone(channels, frame), 960, &mut packet)
                    .unwrap();
                packet[..n].to_vec()
            })
            .collect()
    }

    fn decode_command(c: &Conn, stream: u32, input_length: u32, first: u32, count: u32) -> Process {
        Process {
            id: id(PROCESS, stream),
            pitch_ramp: 0.0,
            target_pitch: 1.0,
            first_frame: first,
            frame_count: count,
            input_length,
            input: c.base,
            output_length: 0,
            output: c.base.wrapping_add(OUT),
        }
    }

    /// Decodes `packet` on `stream`, with `room` bytes of output, and returns
    /// the result and the floats written to the output buffer.
    fn decode(
        c: &Conn,
        stream: u32,
        packet: &[u8],
        first: u32,
        count: u32,
        room: u32,
    ) -> (ApuResult, Vec<f32>) {
        // SAFETY: inside the connection memory.
        unsafe { std::ptr::copy_nonoverlapping(packet.as_ptr(), c.base, packet.len()) };
        let mut command = decode_command(c, stream, packet.len() as u32, first, count);
        command.output_length = room;
        assert_eq!(enqueue(c.handle, &command), S_OK);
        let result = dequeue(c.handle);
        let floats = ({ result.output_length } / 4) as usize;
        // SAFETY: the decoder wrote `output_length` bytes there.
        let pcm =
            unsafe { std::slice::from_raw_parts(c.base.wrapping_add(OUT).cast::<f32>(), floats) };
        (result, pcm.to_vec())
    }

    fn energy(pcm: &[f32]) -> f32 {
        pcm.iter().map(|s| s * s).sum::<f32>() / pcm.len() as f32
    }

    fn try_connect(process: u32, frames: u32, streams: u32, per: u32, memory: u32) -> HRESULT {
        let input = ConnectInput {
            process,
            streams,
            commands_per_stream: per,
            memory,
            frames_per_output: frames,
            options: 0,
        };
        let mut output = ConnectOutput {
            memory: std::ptr::null_mut(),
            result_event: std::ptr::null_mut(),
        };
        let mut handle = std::ptr::null_mut();
        let hr = XApuConnect(&input, &mut output, &mut handle);
        if hr == S_OK {
            assert!(!handle.is_null());
            assert_eq!(XApuDisconnect(handle), S_OK);
        } else {
            assert!(handle.is_null(), "a failed connect leaves the handle alone");
        }
        hr
    }

    #[test]
    fn connect_rejects_null_arguments() {
        let input = ConnectInput {
            process: DECODE_OPUS,
            streams: 1,
            commands_per_stream: 1,
            memory: 0x1000,
            frames_per_output: 0,
            options: 0,
        };
        let mut output = ConnectOutput {
            memory: std::ptr::null_mut(),
            result_event: std::ptr::null_mut(),
        };
        let mut handle = std::ptr::null_mut();
        assert_eq!(
            XApuConnect(std::ptr::null(), &mut output, &mut handle),
            E_INVALIDARG
        );
        assert_eq!(
            XApuConnect(&input, std::ptr::null_mut(), &mut handle),
            E_INVALIDARG
        );
        assert_eq!(
            XApuConnect(&input, &mut output, std::ptr::null_mut()),
            E_INVALIDARG
        );
        assert!(handle.is_null());
        assert!(output.memory.is_null());
    }

    #[test]
    fn connect_validates_stream_command_and_memory_sizes() {
        let decode = |streams, per, memory| try_connect(DECODE_OPUS, 0, streams, per, memory);
        assert_eq!(decode(0, 1, 0x1000), E_INVALIDARG);
        assert_eq!(decode(501, 1, 0x1000), E_INVALIDARG);
        assert_eq!(decode(1, 1, 0x1000), S_OK);
        assert_eq!(decode(500, 1, 0x1000), S_OK);
        assert_eq!(decode(1, 0, 0x1000), E_INVALIDARG);
        assert_eq!(decode(1, 17, 0x1000), E_INVALIDARG);
        assert_eq!(decode(1, 16, 0x1000), S_OK);
        assert_eq!(decode(1, 1, 0), E_INVALIDARG);
        // The process kind is only judged once the sizes pass.
        assert_eq!(try_connect(9, 0, 0, 1, 0x1000), E_INVALIDARG);
        assert_eq!(try_connect(9, 0, 1, 1, 0x1000), E_NOTIMPL);
        assert_eq!(try_connect(0, 0, 1, 1, 0x1000), E_NOTIMPL);
    }

    #[test]
    fn connect_memory_is_aligned_zeroed_and_the_event_is_optional() {
        for memory in [1u32, 0x1001, 0x10_0000] {
            let input = ConnectInput {
                process: DECODE_OPUS,
                streams: 1,
                commands_per_stream: 1,
                memory,
                frames_per_output: 0,
                options: 0,
            };
            let mut output = ConnectOutput {
                memory: std::ptr::null_mut(),
                result_event: std::ptr::null_mut(),
            };
            let mut handle = std::ptr::null_mut();
            assert_eq!(XApuConnect(&input, &mut output, &mut handle), S_OK);
            assert_eq!(output.memory as usize % 16, 0);
            // SAFETY: the connection owns `memory` bytes there.
            let block =
                unsafe { std::slice::from_raw_parts(output.memory.cast::<u8>(), memory as usize) };
            assert!(block.iter().all(|&b| b == 0));
            assert!(output.result_event.is_null(), "no event unless asked");
            assert_eq!(XApuDisconnect(handle), S_OK);
        }
        let c = open(DECODE_OPUS, 0, RESULT_EVENT, 1, 1);
        assert!(!c.event.is_null());
        close(c);
    }

    #[test]
    fn activate_checks_channels_and_refuses_a_second_activation() {
        let c = open(DECODE_OPUS, 0, 0, 2, 16);
        assert_eq!(activate_stream(&c, 0, 0), E_INVALIDARG);
        assert_eq!(activate_stream(&c, 0, 3), E_INVALIDARG);
        // A rejected activation leaves the stream inactive.
        let packet = &packets(2, 1)[0];
        let (failed, _) = decode(&c, 0, packet, 0, 960, 960 * 8);
        assert_eq!({ failed.error }, INVALID_STREAM);

        assert_eq!(activate_stream(&c, 0, 2), S_OK);
        assert_eq!(activate_stream(&c, 0, 2), ALREADY_ACTIVE);
        assert_eq!(activate_stream(&c, 0, 1), ALREADY_ACTIVE);
        // The first activation stands: the stream still decodes stereo.
        let (ok, pcm) = decode(&c, 0, packet, 0, 960, 960 * 8);
        assert_eq!(({ ok.error }, { ok.frames }), (S_OK, 960));
        assert_eq!(pcm.len(), 1920);
        // Other streams are independent.
        assert_eq!(activate_stream(&c, 1, 1), S_OK);

        assert_eq!({ simple(&c, DEACTIVATE, 0).error }, S_OK);
        let (gone, _) = decode(&c, 0, packet, 0, 960, 960 * 8);
        assert_eq!({ gone.error }, INVALID_STREAM);
        assert_eq!(activate_stream(&c, 0, 2), S_OK);
        close(c);
    }

    #[test]
    fn decoding_is_deterministic_and_clamped_to_the_request() {
        let packet = &packets(2, 1)[0];
        let run = |first, count, room| {
            let c = open(DECODE_OPUS, 0, 0, 1, 2);
            assert_eq!(activate_stream(&c, 0, 2), S_OK);
            let (result, pcm) = decode(&c, 0, packet, first, count, room);
            assert_eq!(({ result.error }, { result.pitch }), (S_OK, 1.0));
            let frames = result.frames;
            assert_eq!({ result.output_length }, frames * 8);
            // Nothing lands beyond what the result reports.
            // SAFETY: inside the connection memory.
            let past = unsafe {
                std::slice::from_raw_parts(c.base.wrapping_add(OUT + frames as usize * 8), 64)
            };
            assert!(past.iter().all(|&b| b == 0));
            close(c);
            (frames, pcm)
        };
        let (frames, full) = run(0, 960, 960 * 8);
        assert_eq!(frames, 960);
        assert!(energy(&full) > 0.01);
        assert!(run(0, 960, 960 * 8).1 == full, "same packet, same samples");

        let (frames, tail) = run(100, 960, 960 * 8);
        assert_eq!(frames, 860);
        assert!(tail == full[200..]);
        let (frames, head) = run(0, 100, 960 * 8);
        assert_eq!(frames, 100);
        assert!(head == full[..200]);
        // `first_frame` past the end of the packet leaves nothing.
        assert_eq!(run(5000, 960, 960 * 8).0, 0);
        // Output room of 50 stereo frames, with a stray few bytes on top.
        let (frames, small) = run(0, 960, 50 * 8 + 3);
        assert_eq!(frames, 50);
        assert!(small == full[..100]);
    }

    #[test]
    fn deinterleaved_output_writes_each_channel_as_a_plane() {
        let packet = &packets(2, 1)[0];
        let run = |options| {
            let c = open(DECODE_OPUS, 0, options, 1, 2);
            assert_eq!(activate_stream(&c, 0, 2), S_OK);
            let (result, pcm) = decode(&c, 0, packet, 0, 960, 960 * 8);
            assert_eq!({ result.frames }, 960);
            close(c);
            pcm
        };
        let interleaved = run(0);
        let planar = run(DEINTERLEAVED_OUTPUT);
        assert_eq!(planar.len(), 1920);
        assert!(planar != interleaved);
        for i in 0..960 {
            assert_eq!(planar[i], interleaved[2 * i], "left {i}");
            assert_eq!(planar[960 + i], interleaved[2 * i + 1], "right {i}");
        }
    }

    #[test]
    fn mono_streams_decode_mono_and_downmix_a_stereo_packet() {
        let mono = &packets(1, 1)[0];
        let stereo = &packets(2, 1)[0];
        for options in [0, DEINTERLEAVED_OUTPUT] {
            let c = open(DECODE_OPUS, 0, options, 2, 4);
            assert_eq!(activate_stream(&c, 0, 1), S_OK);
            assert_eq!(activate_stream(&c, 1, 1), S_OK);
            let (result, direct) = decode(&c, 0, mono, 0, 960, 960 * 4);
            assert_eq!(({ result.error }, { result.frames }), (S_OK, 960));
            assert_eq!({ result.output_length }, 960 * 4);
            assert!(energy(&direct) > 0.01);
            // A stereo packet on a mono decoder is downmixed, not rejected.
            let (result, down) = decode(&c, 1, stereo, 0, 960, 960 * 4);
            assert_eq!(({ result.error }, { result.frames }), (S_OK, 960));
            assert_eq!(down.len(), 960);
            assert!(energy(&down) > 0.01);
            // Mono has no second plane, so the plane option changes nothing.
            if options != 0 {
                let plain = open(DECODE_OPUS, 0, 0, 1, 2);
                assert_eq!(activate_stream(&plain, 0, 1), S_OK);
                assert!(decode(&plain, 0, mono, 0, 960, 960 * 4).1 == direct);
                close(plain);
            }
            close(c);
        }
    }

    #[test]
    fn unusable_buffers_and_packets_fail_in_the_result_without_writing() {
        let c = open(DECODE_OPUS, 0, 0, 1, 16);
        assert_eq!(activate_stream(&c, 0, 2), S_OK);
        let packet = &packets(2, 1)[0];
        // SAFETY: inside the connection memory.
        unsafe { std::ptr::copy_nonoverlapping(packet.as_ptr(), c.base, packet.len()) };
        let good = decode_command(&c, 0, packet.len() as u32, 0, 960);
        let mut no_input = good;
        no_input.input = std::ptr::null();
        let mut no_output = good;
        no_output.output = std::ptr::null_mut();
        let mut empty = good;
        empty.input_length = 0;
        for mut command in [no_input, no_output, empty] {
            command.output_length = 960 * 8;
            assert_eq!(enqueue(c.handle, &command), S_OK);
            let result = dequeue(c.handle);
            assert_eq!(({ result.error }, { result.frames }), (E_INVALIDARG, 0));
        }

        // A packet the decoder refuses reports BAD_PACKET and the title's
        // buffer keeps what it held.
        let garbage = [0xFFu8; 3];
        // SAFETY: inside the connection memory.
        unsafe {
            std::ptr::write_bytes(c.base.wrapping_add(OUT), 0xA5, 4096);
            std::ptr::copy_nonoverlapping(garbage.as_ptr(), c.base, garbage.len());
        }
        let mut bad = decode_command(&c, 0, garbage.len() as u32, 0, 960);
        bad.output_length = 960 * 8;
        assert_eq!(enqueue(c.handle, &bad), S_OK);
        let result = dequeue(c.handle);
        assert_eq!(({ result.error }, { result.frames }), (BAD_PACKET, 0));
        assert_eq!({ result.output_length }, 0);
        // SAFETY: inside the connection memory.
        let kept = unsafe { std::slice::from_raw_parts(c.base.wrapping_add(OUT), 4096) };
        assert!(kept.iter().all(|&b| b == 0xA5));
        close(c);

        let c = open(CONVERT_PCM, 256, 0, 1, 16);
        assert_eq!(activate_stream(&c, 0, 1), S_OK);
        let mut convert = decode_command(&c, 0, 16, 0, 4);
        convert.output_length = 1024;
        let mut no_input = convert;
        no_input.input = std::ptr::null();
        let mut no_output = convert;
        no_output.output = std::ptr::null_mut();
        let mut empty = convert;
        empty.input_length = 0;
        for command in [no_input, no_output, empty] {
            assert_eq!(enqueue(c.handle, &command), S_OK);
            let result = dequeue(c.handle);
            assert_eq!(({ result.error }, { result.frames }), (E_INVALIDARG, 0));
        }
        close(c);
    }

    #[test]
    fn reset_returns_a_stream_to_its_start_state() {
        let [first, second] = &packets(2, 2)[..] else {
            unreachable!()
        };
        let c = open(DECODE_OPUS, 0, 0, 1, 16);
        // Resetting a stream nobody activated is harmless.
        let idle = simple(&c, RESET, 0);
        assert_eq!(({ idle.error }, { idle.id.kind }), (S_OK, RESET));
        assert_eq!(activate_stream(&c, 0, 2), S_OK);

        let (_, fresh) = decode(&c, 0, first, 0, 960, 960 * 8);
        let (_, next) = decode(&c, 0, second, 0, 960, 960 * 8);
        let (_, carried) = decode(&c, 0, first, 0, 960, 960 * 8);
        assert!(
            carried != fresh,
            "the decoder carries state between packets"
        );
        assert!(next != fresh);

        let reset = simple(&c, RESET, 0);
        assert_eq!(({ reset.error }, { reset.id.kind }), (S_OK, RESET));
        let (result, again) = decode(&c, 0, first, 0, 960, 960 * 8);
        assert_eq!(({ result.error }, { result.frames }), (S_OK, 960));
        assert!(
            again == fresh,
            "after a reset the first packet decodes as new"
        );
        close(c);
    }

    #[test]
    fn results_come_back_in_command_order_echoing_each_command() {
        let c = open(DECODE_OPUS, 0, 0, 2, 4);
        let mut pending = 0;
        assert_eq!(XApuGetResultCount(c.handle, &mut pending), 0);
        assert_eq!(pending, 0);
        assert_eq!(
            XApuDequeueResult(c.handle, std::ptr::null_mut()),
            E_INVALIDARG
        );
        let mut scratch = std::mem::MaybeUninit::<ApuResult>::zeroed();
        assert_eq!(
            XApuDequeueResult(c.handle, scratch.as_mut_ptr()),
            PENDING_RESULTS
        );

        let command_id = |kind, stream, sequence, options, context| CommandId {
            kind,
            stream,
            sequence,
            options,
            context,
        };
        let activate = |id| Activate {
            id,
            channels: 2,
            init_length: 0,
            init: std::ptr::null(),
            start_pitch: 1.0,
            processing_length: 0,
            processing: std::ptr::null_mut(),
        };
        let packet = &packets(2, 1)[0];
        // SAFETY: inside the connection memory.
        unsafe { std::ptr::copy_nonoverlapping(packet.as_ptr(), c.base, packet.len()) };
        let mut process = decode_command(&c, 1, packet.len() as u32, 0, 960);
        process.id = command_id(PROCESS, 1, 14, 0x40, 0xFEED_0000_0004);
        process.output_length = 960 * 8;

        let sent = [
            command_id(ACTIVATE, 0, 11, 0x10, 0xFEED_0000_0001),
            command_id(ACTIVATE, 1, 12, 0x20, 0xFEED_0000_0002),
            command_id(NOTIFY, 0, 13, 0x30, 0xFEED_0000_0003),
            process.id,
            command_id(DEACTIVATE, 1, 15, 0x50, 0xFEED_0000_0005),
        ];
        for (i, id) in sent.iter().enumerate() {
            let mut total = 0;
            let command = match id.kind {
                ACTIVATE => {
                    let command = activate(*id);
                    XApuEnqueueCommand(c.handle, (&raw const command).cast(), &mut total)
                }
                PROCESS => XApuEnqueueCommand(c.handle, (&raw const process).cast(), &mut total),
                _ => XApuEnqueueCommand(c.handle, id, &mut total),
            };
            assert_eq!(command, S_OK);
            assert_eq!(total, i as u32 + 1, "one result per command");
        }
        assert_eq!(XApuGetResultCount(c.handle, std::ptr::null_mut()), 5);

        for (i, id) in sent.iter().enumerate() {
            let result = dequeue(c.handle);
            let echoed = result.id;
            assert_eq!(
                (
                    echoed.kind,
                    echoed.stream,
                    echoed.sequence,
                    echoed.options,
                    echoed.context
                ),
                (id.kind, id.stream, id.sequence, id.options, id.context),
            );
            assert_eq!(({ result.error }, { result.state }), (S_OK, 0));
            if id.kind == PROCESS {
                assert_eq!({ result.input }, c.base.cast_const());
                assert_eq!({ result.input_length }, packet.len() as u32);
                assert_eq!({ result.output }, c.base.wrapping_add(OUT));
                assert_eq!({ result.frames }, 960);
            }
            let mut done = 0;
            let left = XApuGetResultCount(c.handle, &mut done);
            assert_eq!((left, done), (4 - i as u32, i as u32 + 1));
        }
        assert_eq!(
            XApuDequeueResult(c.handle, scratch.as_mut_ptr()),
            PENDING_RESULTS
        );
        close(c);
    }

    #[test]
    fn queue_depth_is_per_stream_and_dequeuing_frees_a_slot() {
        let c = open(DECODE_OPUS, 0, 0, 2, 2);
        let notify = |stream| id(NOTIFY, stream);
        assert_eq!(enqueue(c.handle, &notify(0)), S_OK);
        assert_eq!(enqueue(c.handle, &notify(0)), S_OK);
        assert_eq!(enqueue(c.handle, &notify(0)), QUEUE_FULL);
        // The other stream has its own depth.
        assert_eq!(enqueue(c.handle, &notify(1)), S_OK);
        assert_eq!(enqueue(c.handle, &notify(1)), S_OK);
        assert_eq!(enqueue(c.handle, &notify(1)), QUEUE_FULL);
        assert_eq!(XApuGetResultCount(c.handle, std::ptr::null_mut()), 4);

        // The oldest result is stream 0's; taking it frees stream 0 only.
        assert_eq!({ dequeue(c.handle).id.stream }, 0);
        assert_eq!(enqueue(c.handle, &notify(1)), QUEUE_FULL);
        assert_eq!(enqueue(c.handle, &notify(0)), S_OK);
        assert_eq!(enqueue(c.handle, &notify(0)), QUEUE_FULL);
        while XApuGetResultCount(c.handle, std::ptr::null_mut()) > 0 {
            dequeue(c.handle);
        }
        assert_eq!(enqueue(c.handle, &notify(1)), S_OK);
        close(c);
    }

    #[test]
    fn rejected_commands_are_not_counted_or_queued() {
        let c = open(DECODE_OPUS, 0, 0, 1, 1);
        let mut total = 0xDEAD;
        let unknown = id(99, 0);
        let zero = id(0, 0);
        let stray = id(NOTIFY, 1);
        for (command, error) in [
            (&unknown, INVALID_COMMAND),
            (&zero, INVALID_COMMAND),
            (&stray, INVALID_STREAM),
        ] {
            assert_eq!(XApuEnqueueCommand(c.handle, command, &mut total), error);
        }
        assert_eq!(total, 0xDEAD);
        assert_eq!(XApuGetResultCount(c.handle, std::ptr::null_mut()), 0);

        // None of that used the stream's single slot.
        assert_eq!(
            XApuEnqueueCommand(c.handle, &id(NOTIFY, 0), &mut total),
            S_OK
        );
        assert_eq!(total, 1);
        total = 0xDEAD;
        assert_eq!(
            XApuEnqueueCommand(c.handle, &id(NOTIFY, 0), &mut total),
            QUEUE_FULL
        );
        assert_eq!(total, 0xDEAD);
        assert_eq!(XApuGetResultCount(c.handle, std::ptr::null_mut()), 1);
        close(c);
    }

    #[test]
    fn exports_reject_null_arguments() {
        let c = open(DECODE_OPUS, 0, 0, 1, 1);
        let mut result = std::mem::MaybeUninit::<ApuResult>::zeroed();
        let mut total = 0xBEEF;
        assert_eq!(XApuDisconnect(std::ptr::null_mut()), E_INVALIDARG);
        assert_eq!(
            XApuEnqueueCommand(std::ptr::null_mut(), &id(NOTIFY, 0), &mut total),
            E_INVALIDARG
        );
        assert_eq!(
            XApuEnqueueCommand(c.handle, std::ptr::null(), &mut total),
            E_INVALIDARG
        );
        assert_eq!(
            XApuDequeueResult(std::ptr::null_mut(), result.as_mut_ptr()),
            E_INVALIDARG
        );
        assert_eq!(
            XApuDequeueResult(c.handle, std::ptr::null_mut()),
            E_INVALIDARG
        );
        assert_eq!(XApuGetResultCount(std::ptr::null_mut(), &mut total), 0);
        assert_eq!(total, 0xBEEF);
        // The total of enqueued commands is optional, as is the count's.
        assert_eq!(
            XApuEnqueueCommand(c.handle, &id(NOTIFY, 0), std::ptr::null_mut()),
            S_OK
        );
        assert_eq!(XApuGetResultCount(c.handle, std::ptr::null_mut()), 1);
        close(c);
    }

    #[test]
    fn the_signal_option_sets_the_auto_reset_result_event() {
        let c = open(DECODE_OPUS, 0, RESULT_EVENT, 1, 8);
        let signalled = || {
            // SAFETY: our connection's event.
            unsafe { WaitForSingleObject(c.event, 0) == WAIT_OBJECT_0 }
        };
        assert!(!signalled());
        assert_eq!(enqueue(c.handle, &id(NOTIFY, 0)), S_OK);
        assert!(!signalled(), "only a command that asks for it signals");

        let mut asking = id(NOTIFY, 0);
        asking.options = SIGNAL;
        assert_eq!(enqueue(c.handle, &asking), S_OK);
        assert!(signalled());
        assert!(!signalled(), "waiting reset the event");
        close(c);

        // Without a result event the option has nothing to signal.
        let c = open(DECODE_OPUS, 0, 0, 1, 8);
        assert!(c.event.is_null());
        assert_eq!(enqueue(c.handle, &asking), S_OK);
        close(c);
    }
}
