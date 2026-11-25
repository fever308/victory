//! Heaps, committed and placed resources, and resource sizes.
//!
//! Heaps and resources are host memory and a resource's GPU address is its
//! CPU address, as on the console; textures also get a host copy, and
//! buffers a host copy when a draw reads them (`shadow`).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ffi::c_void;
use std::ptr::NonNull;

use victory_formats::texture;

use crate::object::{
    self, E_INVALIDARG, E_OUTOFMEMORY, HEAP, HRESULT, Iid, Object, RESOURCE, S_OK, Table, This,
    guid,
};

/// The interfaces every device child also answers for.
pub(crate) const PAGEABLE_IIDS: [Iid; 3] = [
    guid(
        0x63ee58fb,
        0x1268,
        0x4835,
        [0x86, 0xda, 0xf0, 0x08, 0xce, 0x62, 0xf0, 0xd6],
    ),
    guid(
        0x905db94b,
        0xa00c,
        0x4140,
        [0x9d, 0xf5, 0x2b, 0x64, 0xca, 0x9e, 0xa3, 0x57],
    ),
    guid(
        0xc4fec28f,
        0x7966,
        0x4e95,
        [0x9f, 0x94, 0xf4, 0x31, 0xcb, 0x56, 0xc3, 0xb8],
    ),
];
const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x6b3b2502,
        0x6e51,
        0x45b3,
        [0x90, 0xee, 0x98, 0x84, 0x26, 0x5e, 0x8d, 0xf3],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];
const RESOURCE_IIDS: [Iid; 4] = [
    guid(
        0x696442be,
        0xa72e,
        0x4059,
        [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

const KIB_64: u64 = 64 << 10;
const MIB_4: u64 = 4 << 20;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapProperties {
    kind: u32,
    cpu_pages: u32,
    pool: u32,
    creation_nodes: u32,
    visible_nodes: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapDesc {
    size: u64,
    properties: HeapProperties,
    alignment: u64,
    flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResourceDesc {
    dimension: u32,
    alignment: u64,
    width: u64,
    height: u32,
    depth_or_array_size: u16,
    mip_levels: u16,
    format: u32,
    sample_count: u32,
    sample_quality: u32,
    layout: u32,
    flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AllocationInfo {
    size: u64,
    alignment: u64,
}

const _: () = assert!(size_of::<HeapProperties>() == 20);
const _: () = assert!(size_of::<HeapDesc>() == 48);
// placed_resource_words copies the two field spans on either side of the padding.
const _: () = assert!(std::mem::offset_of!(ResourceDesc, alignment) == 8);
const _: () = assert!(std::mem::offset_of!(ResourceDesc, flags) == 48);
const _: () = assert!(size_of::<ResourceDesc>() == 56);

const BUFFER: u32 = 1;
const TEXTURE_2D: u32 = 3;
const TEXTURE_3D: u32 = 4;

/// Bytes per element and the element's width in texels (4 for block
/// compression), by DXGI format number. `None` for 1-bit and video
/// formats. The console adds 116/117 (10-bit float color), 118-120 (16-bit
/// depth with 8-bit stencil, two planes: 3 bytes per texel), 189 (signed
/// 10-bit color), 190 (two 4-bit channels) and 191 (24/8 color).
pub(crate) fn element(format: u32) -> Option<(u64, u64)> {
    Some(match format {
        1..=4 => (16, 1),
        5..=8 => (12, 1),
        9..=22 => (8, 1),
        23..=47 | 67 | 87..=93 | 116 | 117 | 189 | 191 => (4, 1),
        118..=120 => (3, 1),
        48..=59 | 68 | 69 | 85 | 86 | 115 => (2, 1),
        60..=65 | 190 => (1, 1),
        70..=72 | 79..=81 => (8, 4),
        73..=78 | 82..=84 | 94..=99 => (16, 4),
        _ => return None,
    })
}

/// Bytes, width and height of one storage block in a linear texture footprint.
fn texture_element(format: u32) -> Option<(u64, u64, u64)> {
    // Packed RGB formats store two adjacent texels in each 32-bit word.
    if matches!(format, 68 | 69) {
        return Some((4, 2, 1));
    }
    let (bytes, block) = element(format)?;
    Some((bytes, block, block))
}

/// A texture's layout in console memory, as the console's layout library
/// gives it (`victory-hle`'s `xg` serves the same): the description's
/// layout field is `0x100 | swizzle mode`, 1 linear, 0 the library's pick.
pub(crate) fn texture_layout(desc: &ResourceDesc) -> Option<texture::Texture> {
    use texture::{Dimension, Swizzle, TextureDesc};
    const TEXTURE_1D: u32 = 2;
    const RENDER_TARGET: u32 = 1;
    const DEPTH: u32 = 2;
    let dimension = match desc.dimension {
        TEXTURE_1D => Dimension::Tex1d,
        TEXTURE_2D => Dimension::Tex2d,
        TEXTURE_3D => Dimension::Tex3d,
        _ => return None,
    };
    let swizzle = match desc.layout {
        0 if dimension == Dimension::Tex1d => Swizzle::LINEAR,
        0 if desc.flags & DEPTH != 0 => Swizzle::SW_64KB_Z_X,
        0 if desc.flags & RENDER_TARGET != 0 => Swizzle::SW_64KB_R_X,
        0 => Swizzle::SW_4KB_S,
        1 => Swizzle::LINEAR,
        l if l & !0xff == 0x100 => Swizzle(l & 0xff),
        _ => return None,
    };
    texture::Texture::new(&TextureDesc {
        dimension,
        width: u32::try_from(desc.width).ok()?,
        height: desc.height,
        depth_or_array: u32::from(desc.depth_or_array_size),
        mips: u32::from(desc.mip_levels),
        format: desc.format,
        samples: desc.sample_count.max(1),
        swizzle,
    })
    .ok()
}

/// Bytes and alignment a resource takes: buffers their width (64 KiB
/// aligned), textures their console layout (formats it does not know:
/// every mip of every array slice and sample, packed, 64 KiB aligned),
/// rounded up to the alignment.
fn footprint(desc: &ResourceDesc) -> Option<AllocationInfo> {
    if desc.dimension != BUFFER
        && let Some(t) = texture_layout(desc)
    {
        // As the console: the layout's own alignment unless more is asked
        // (4 KiB for 4 KiB modes, where PC says 64 KiB). The
        // title keeps these sizes in its texture tables, and its streaming
        // asserts when they disagree with the streamed layout.
        let alignment = desc.alignment.max(t.alignment);
        if !alignment.is_power_of_two() {
            return None;
        }
        return Some(AllocationInfo {
            size: t.size.checked_next_multiple_of(alignment)?,
            alignment,
        });
    }
    let bytes = if desc.dimension == BUFFER {
        desc.width
    } else {
        let (element, block_width, block_height) = texture_element(desc.format)?;
        let (depth, slices) = if desc.dimension == TEXTURE_3D {
            (u64::from(desc.depth_or_array_size), 1)
        } else {
            (1, u64::from(desc.depth_or_array_size))
        };
        let height = u64::from(desc.height);
        let mips = mip_count(desc);
        let mut chain = 0u64;
        for mip in 0..mips.min(64) {
            let w = (desc.width >> mip).max(1).div_ceil(block_width);
            let h = (height >> mip).max(1).div_ceil(block_height);
            let d = (depth >> mip).max(1);
            chain = chain.checked_add(w.checked_mul(h)?.checked_mul(d)?.checked_mul(element)?)?;
        }
        chain
            .checked_mul(slices)?
            .checked_mul(u64::from(desc.sample_count.max(1)))?
    };
    if bytes == 0 {
        return None;
    }
    let minimum = if desc.sample_count > 1 { MIB_4 } else { KIB_64 };
    let alignment = desc.alignment.max(minimum);
    if !alignment.is_power_of_two() {
        return None;
    }
    Some(AllocationInfo {
        size: bytes.checked_next_multiple_of(KIB_64)?,
        alignment,
    })
}

/// A format as the host has it. The console's 16-bit depth with stencil
/// (and its typeless views) has no PC format: the 24-bit ones stand in.
pub(crate) fn pc_format(format: u32) -> u32 {
    match format {
        118 => 45,
        119 => 46,
        120 => 47,
        other => other,
    }
}

/// The format a render target or unordered-access texture of `format` is
/// made in on the host, and that render target views and pipelines name:
/// `R16G16B16A16_FLOAT` (10) for `R9G9B9E5_SHAREDEXP` (67), which PC D3D12
/// forbids as a render target or unordered-access resource; else
/// [`pc_format`]. Textures only sampled keep `R9G9B9E5_SHAREDEXP`.
pub(crate) fn target_format(format: u32) -> u32 {
    const R9G9B9E5_SHAREDEXP: u32 = 67;
    const RGBA16_FLOAT: u32 = 10;
    let format = pc_format(format);
    if format == R9G9B9E5_SHAREDEXP {
        RGBA16_FLOAT
    } else {
        format
    }
}

/// The typeless format of a format's family (itself if it has none).
pub(crate) fn family(format: u32) -> u32 {
    const FAMILIES: [(u32, u32); 22] = [
        (1, 4),
        (5, 8),
        (9, 14),
        (15, 18),
        (19, 22),
        (23, 25),
        (27, 32),
        (33, 38),
        (39, 43),
        (44, 47),
        (48, 52),
        (53, 59),
        (60, 64),
        (70, 72),
        (73, 75),
        (76, 78),
        (79, 81),
        (82, 84),
        (90, 91),
        (92, 93),
        (94, 96),
        (97, 99),
    ];
    match format {
        87 => 90,
        88 => 92,
        f => FAMILIES
            .iter()
            .find(|&&(first, last)| (first..=last).contains(&f))
            .map_or(f, |&(first, _)| first),
    }
}

/// sRGB formats and their linear twins: 8-bit color, then BC1, BC2, BC3
/// and BC7.
pub(crate) const SRGB: [(u32, u32); 7] = [
    (29, 28),
    (91, 87),
    (93, 88),
    (72, 71),
    (75, 74),
    (78, 77),
    (99, 98),
];

/// The format a view of a title texture must name when its host copy is
/// typeless (see [`DEPTH_FORMATS`], [`SRGB`]): the title's own; `None` if
/// the host copy is typed.
#[cfg(windows)]
pub(crate) fn view_format(this: This) -> Option<u32> {
    let host = host_resource(this)?;
    let own = pc_format(resource(this).desc.format);
    // SAFETY: a live resource.
    let format = unsafe { host.GetDesc() }.Format.0 as u32;
    (format != own && format != target_format(own)).then_some(own)
}

/// Dimension, depth-or-array size, and sample count for a NULL-desc RTV/DSV
/// when the host copy is typeless ([`view_format`] is `Some`).
#[cfg(windows)]
pub(crate) fn view_shape(this: This) -> Option<(u32, u16, u32)> {
    if this.is_null() {
        return None;
    }
    let d = &resource(this).desc;
    Some((d.dimension, d.depth_or_array_size, d.sample_count.max(1)))
}

/// Depth formats with their typeless family: depth textures that shaders
/// read are made typeless on the host, which only allows views of
/// typeless depth resources.
pub(crate) const DEPTH_FORMATS: [(u32, u32); 4] = [(20, 19), (40, 39), (45, 44), (55, 53)];

/// Default concrete format for a typeless DXGI family. Depth multi-plane,
/// block-compressed, and 96-bit `R32G32B32` allow shader views only
/// (`None` when `unordered`). Device-specific typed-UAV support is the
/// caller's to check.
pub(crate) fn typeless_default_format(typeless: u32, unordered: bool) -> Option<u32> {
    // DXGI_FORMAT numbers: typeless → a common concrete member of the family.
    match typeless {
        // R24G8_TYPELESS → R24_UNORM_X8_TYPELESS
        44 => (!unordered).then_some(46),
        // R32G8X24_TYPELESS → R32_FLOAT_X8X24_TYPELESS
        19 => (!unordered).then_some(21),
        // BC1..BC7 typeless → UNORM / UF16 (SRV only)
        70 => (!unordered).then_some(71),
        73 => (!unordered).then_some(74),
        76 => (!unordered).then_some(77),
        79 => (!unordered).then_some(80),
        82 => (!unordered).then_some(83),
        94 => (!unordered).then_some(95),
        97 => (!unordered).then_some(98),
        // R32G32B32_TYPELESS → FLOAT (no typed UAV in D3D12)
        5 => (!unordered).then_some(6),
        // Color typeless → FLOAT / UNORM defaults
        1 => Some(2),   // R32G32B32A32
        9 => Some(10),  // R16G16B16A16
        15 => Some(16), // R32G32
        23 => Some(24), // R10G10B10A2
        27 => Some(28), // R8G8B8A8
        33 => Some(34), // R16G16
        39 => Some(41), // R32
        48 => Some(49), // R8G8
        53 => Some(54), // R16
        60 => Some(61), // R8
        90 => Some(87), // B8G8R8A8
        92 => Some(88), // B8G8R8X8
        _ => None,
    }
}

/// Concrete format for a null or `DXGI_FORMAT_UNKNOWN` SRV/UAV of a typeless
/// host texture. `title` is the console format when the host copy was promoted
/// from a typed resource (depth without deny-shader, or sRGB with UAV); `None`
/// when the title created the resource typeless. Maps sRGB → linear for UAVs
/// and depth → an SRV-capable plane format. (#438 VIC-BT-004)
pub(crate) fn null_view_format(
    host_typeless: u32,
    title: Option<u32>,
    unordered: bool,
) -> Option<u32> {
    const D16_UNORM: u32 = 55;
    const R16_UNORM: u32 = 56;
    if let Some(fmt) = title {
        if unordered {
            let linear = SRGB
                .iter()
                .find(|&&(srgb, _)| srgb == fmt)
                .map(|&(_, linear)| linear)
                .unwrap_or(fmt);
            if DEPTH_FORMATS
                .iter()
                .any(|&(typed, _)| typed == linear || typed == fmt)
            {
                return None;
            }
            return Some(linear);
        }
        if fmt == D16_UNORM {
            return Some(R16_UNORM);
        }
        if DEPTH_FORMATS.iter().any(|&(typed, _)| typed == fmt) {
            return typeless_default_format(host_typeless, false);
        }
        return Some(fmt);
    }
    typeless_default_format(host_typeless, unordered)
}

/// Mip levels, the full chain for 0.
fn mip_count(desc: &ResourceDesc) -> u32 {
    let depth = if desc.dimension == TEXTURE_3D {
        u64::from(desc.depth_or_array_size)
    } else {
        1
    };
    match desc.mip_levels {
        0 => {
            64 - desc
                .width
                .max(u64::from(desc.height))
                .max(depth)
                .max(1)
                .leading_zeros()
        }
        n => u32::from(n),
    }
}

/// Most mip levels the host gives up on a texture it cannot allocate.
const MAX_MIP_DROP: u32 = 3;

/// The mip levels every later host texture gives up, raised by each
/// allocation the host refuses with `E_OUTOFMEMORY`. A process that has
/// run out of memory for one texture will for the next (the failures
/// arrive in runs), so later textures start smaller instead of failing
/// first.
static PRESSURE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// How many top mip levels a host copy of `desc` can lose, at most `wanted`:
/// only plain sampled 2D textures lose any (render targets, depth and
/// unordered access keep their size, which draws and clears rely on), and
/// never the last level. Block-compressed textures keep a whole number of
/// blocks.
fn droppable_mips(desc: &ResourceDesc, wanted: u32) -> u32 {
    const PC_FLAGS: u32 = 0xf;
    const TEXTURE_2D: u32 = 3;
    if desc.dimension != TEXTURE_2D || desc.sample_count > 1 || desc.flags & PC_FLAGS != 0 {
        return 0;
    }
    let Some((_, block)) = element(desc.format) else {
        return 0;
    };
    let most = wanted.min(mip_count(desc).saturating_sub(1));
    (1..=most)
        .rev()
        .find(|&n| {
            let (w, h) = ((desc.width >> n).max(1), u64::from(desc.height >> n).max(1));
            block == 1 || (w % block == 0 && h % block == 0)
        })
        .unwrap_or(0)
}

/// A view's `(most_detailed, levels)` of a title texture, as the same
/// levels of a host copy that lacks the title's top `dropped` ones. A view
/// of every level (`u32::MAX`) stays one; one that named only dropped levels
/// shows the host's largest.
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn rebase_mip_range(most_detailed: u32, levels: u32, dropped: u32) -> (u32, u32) {
    let first = most_detailed.saturating_sub(dropped);
    if levels == u32::MAX {
        return (first, levels);
    }
    let end = most_detailed.saturating_add(levels).saturating_sub(dropped);
    (first, end.saturating_sub(first).max(1))
}

/// `desc` without its `dropped` largest mip levels.
fn without_top_mips(desc: &ResourceDesc, dropped: u32) -> ResourceDesc {
    if dropped == 0 {
        return *desc;
    }
    ResourceDesc {
        width: (desc.width >> dropped).max(1),
        height: (desc.height >> dropped).max(1),
        mip_levels: (mip_count(desc) - dropped) as u16,
        ..*desc
    }
}

/// Makes a host texture through `create`, which answers `Err` with the
/// host's HRESULT. When the host refuses for lack of memory
/// (`E_OUTOFMEMORY`), drops another top mip level of an eligible texture
/// (up to [`MAX_MIP_DROP`]) and tries again, raising the process-wide
/// [`PRESSURE`] so later textures start already smaller. Returns what
/// `create` made, the description it made it from, and how many levels
/// that lost against `desc`.
fn create_texture_with_auto_mip_drop<T>(
    desc: &ResourceDesc,
    mut create: impl FnMut(&ResourceDesc) -> Result<T, HRESULT>,
) -> Result<(T, ResourceDesc, u32), HRESULT> {
    use std::sync::atomic::Ordering::Relaxed;
    let mut dropped = droppable_mips(desc, PRESSURE.load(Relaxed).min(MAX_MIP_DROP));
    loop {
        let attempt = without_top_mips(desc, dropped);
        match create(&attempt) {
            Ok(made) => return Ok((made, attempt, dropped)),
            Err(E_OUTOFMEMORY) => {
                // The next single-level drop may be invalid for block-compressed
                // sizes (e.g. 36×36 BC1 cannot drop 1 or 2 levels but can drop
                // 3); search every remaining allowed count for the next valid.
                let more = (dropped + 1..=MAX_MIP_DROP)
                    .find_map(|n| {
                        let d = droppable_mips(desc, n);
                        (d > dropped).then_some(d)
                    })
                    .unwrap_or(dropped);
                if more <= dropped {
                    return Err(E_OUTOFMEMORY);
                }
                PRESSURE.fetch_max(more, Relaxed);
                tracing::warn!(
                    dropping = more,
                    width = desc.width,
                    height = desc.height,
                    mip_levels = mip_count(desc),
                    "host memory pressure: dropping {more} mips automatically"
                );
                dropped = more;
            }
            Err(e) => return Err(e),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlacedFootprint {
    pub(crate) offset: u64,
    pub(crate) format: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) depth: u32,
    pub(crate) row_pitch: u32,
}

impl PlacedFootprint {
    /// `(lines, row_bytes, total_bytes)` spanned in a buffer: every row
    /// pitched but the last.
    pub(crate) fn layout(&self) -> Option<(u64, u64, u64)> {
        let (element, block_width, block_height) = texture_element(self.format)?;
        let lines = u64::from(self.height)
            .div_ceil(block_height)
            .checked_mul(u64::from(self.depth))?;
        let row_bytes = u64::from(self.width)
            .div_ceil(block_width)
            .checked_mul(element)?;
        if lines == 0 {
            return Some((0, row_bytes, 0));
        }
        let total = u64::from(self.row_pitch)
            .checked_mul(lines - 1)?
            .checked_add(row_bytes)?;
        Some((lines, row_bytes, total))
    }

    /// The bytes it spans in a buffer: every row pitched but the last.
    pub(crate) fn bytes(&self) -> Option<u64> {
        self.layout().map(|(_, _, total)| total)
    }
}

const _: () = assert!(size_of::<PlacedFootprint>() == 32);

const NO_FOOTPRINT: PlacedFootprint = PlacedFootprint {
    offset: u64::MAX,
    format: 0,
    width: 0,
    height: 0,
    depth: 0,
    row_pitch: 0,
};

/// One subresource's upload footprint (offset 0), row count and unpadded row size.
fn subresource(desc: &ResourceDesc, index: u32) -> Option<(PlacedFootprint, u32, u64)> {
    const PITCH: u64 = 256;
    // NV12 and P010 copy footprints require an even-sized 2D texture with one mip.
    if matches!(desc.format, 103 | 104)
        && (desc.dimension != TEXTURE_2D
            || desc.mip_levels != 1
            || desc.sample_count != 1
            || !(1..=2048).contains(&desc.depth_or_array_size)
            || desc.width < 2
            || desc.width > u64::from(u32::MAX)
            || desc.height < 2
            || !desc.width.is_multiple_of(2)
            || !desc.height.is_multiple_of(2))
    {
        return None;
    }
    if desc.dimension == BUFFER {
        let footprint = PlacedFootprint {
            offset: 0,
            format: 0,
            width: u32::try_from(desc.width).ok()?,
            height: 1,
            depth: 1,
            row_pitch: u32::try_from(desc.width.checked_next_multiple_of(PITCH)?).ok()?,
        };
        return (index == 0).then_some((footprint, 1, desc.width));
    }
    let mips = mip_count(desc);
    let slices = if desc.dimension == TEXTURE_3D {
        1
    } else {
        u32::from(desc.depth_or_array_size).max(1)
    };
    let mip = index % mips;
    let plane = index / mips / slices;
    // Each plane contains all mips and array slices. Video chroma samples
    // contain two components and cover a 2x2 luma region.
    let (format, pitch_alignment, extent_shift) = match (desc.format, plane) {
        (19..=22 | 44..=47, 0) => (39, 512, mip), // R32_TYPELESS
        (19..=22 | 44..=47, 1) => (60, 512, mip), // R8_TYPELESS
        (103, 0) => (60, 512, 0),                 // NV12 Y: R8_TYPELESS
        (103, 1) => (48, 512, 1),                 // NV12 UV: R8G8_TYPELESS
        (104, 0) => (53, 512, 0),                 // P010 Y: R16_TYPELESS
        (104, 1) => (33, 512, 1),                 // P010 UV: R16G16_TYPELESS
        (format, 0) => (format, PITCH, mip),
        _ => return None,
    };
    let (element, block_width, block_height) = texture_element(format)?;
    let width = desc
        .width
        .checked_shr(extent_shift)?
        .max(1)
        .checked_next_multiple_of(block_width)?;
    let height = u64::from(desc.height)
        .checked_shr(extent_shift)?
        .max(1)
        .checked_next_multiple_of(block_height)?;
    let depth = if desc.dimension == TEXTURE_3D {
        u32::from(desc.depth_or_array_size).checked_shr(mip)?.max(1)
    } else {
        1
    };
    let row_size = (width / block_width).checked_mul(element)?;
    let footprint = PlacedFootprint {
        offset: 0,
        format,
        width: u32::try_from(width).ok()?,
        height: u32::try_from(height).ok()?,
        depth,
        row_pitch: u32::try_from(row_size.checked_next_multiple_of(pitch_alignment)?).ok()?,
    };
    Some((
        footprint,
        u32::try_from(height / block_height).ok()?,
        row_size,
    ))
}

/// Places subresources at 512-byte boundaries in an upload buffer.
/// Row pitch is aligned to 512 bytes for depth/stencil and planar video,
/// and 256 bytes otherwise. An unknown size gives an all-ones total.
pub(crate) extern "system" fn copyable_footprints(
    _: This,
    desc: *const ResourceDesc,
    first: u32,
    count: u32,
    base: u64,
    layouts: *mut PlacedFootprint,
    rows: *mut u32,
    row_sizes: *mut u64,
    total: *mut u64,
) {
    const PLACEMENT: u64 = 512;
    // SAFETY: the title passes a description (null is tolerated).
    let desc = unsafe { desc.as_ref() };
    // Keep offsets relative to `base`: D3D12 preserves an unaligned base
    // and aligns the distance to each following subresource.
    let mut end = 0u64;
    let mut known = desc.is_some();
    for i in 0..count {
        let one = desc
            .and_then(|d| first.checked_add(i).and_then(|index| subresource(d, index)))
            .and_then(|(mut footprint, row_count, row_size)| {
                let start = end.checked_next_multiple_of(PLACEMENT)?;
                footprint.offset = base.checked_add(start)?;
                let lines = u64::from(row_count).checked_mul(u64::from(footprint.depth))?;
                let preceding_rows =
                    u64::from(footprint.row_pitch).checked_mul(lines.saturating_sub(1))?;
                let span = preceding_rows.checked_add(row_size)?;
                let next = start.checked_add(span)?;
                Some((footprint, row_count, row_size, next))
            });
        let (layout, row_count, row_size) = match one {
            Some((footprint, row_count, row_size, next)) => {
                end = next;
                (footprint, row_count, row_size)
            }
            None => {
                known = false;
                (NO_FOOTPRINT, u32::MAX, u64::MAX)
            }
        };
        let i = i as usize;
        // SAFETY: each output is null or holds `count` entries.
        unsafe {
            if !layouts.is_null() {
                layouts.add(i).write(layout);
            }
            if !rows.is_null() {
                rows.add(i).write(row_count);
            }
            if !row_sizes.is_null() {
                row_sizes.add(i).write(row_size);
            }
        }
    }
    if !known {
        tracing::warn!(?desc, "no footprint for this resource");
    }
    if !total.is_null() {
        // SAFETY: checked non-null.
        unsafe { total.write(if known { end } else { u64::MAX }) };
    }
}

/// Device method: the footprint of resources placed one after another.
/// Returns a 16-byte value, through a hidden first argument. Invalid input
/// gives the documented size `u64::MAX`.
pub(crate) extern "system" fn allocation_info(
    out: *mut AllocationInfo,
    _: This,
    _visible: u32,
    count: u32,
    descs: *const ResourceDesc,
) -> *mut AllocationInfo {
    let descs = if descs.is_null() {
        &[][..]
    } else {
        // SAFETY: the title passes `count` descriptions.
        unsafe { std::slice::from_raw_parts(descs, count as usize) }
    };
    // SAFETY: the caller passes room for the value.
    unsafe { out.write(place(descs.iter(), |_, _| ())) };
    out
}

/// The newer description: the plain one, then a sampler feedback mip
/// region (width, height, depth).
#[repr(C)]
pub(crate) struct ResourceDesc1 {
    desc: ResourceDesc,
    mip_region: [u32; 3],
}
const _: () = assert!(size_of::<ResourceDesc1>() == 72);

/// Where one resource goes in [`allocation_info2`]'s placement.
#[repr(C)]
pub(crate) struct AllocationInfo1 {
    offset: u64,
    alignment: u64,
    size: u64,
}

/// Device method 0x360 (`ID3D12Device8`): [`allocation_info`] for newer
/// descriptions, also telling where each resource goes.
pub(crate) extern "system" fn allocation_info2(
    out: *mut AllocationInfo,
    _: This,
    _visible: u32,
    count: u32,
    descs: *const ResourceDesc1,
    each: *mut AllocationInfo1,
) -> *mut AllocationInfo {
    let descs = if descs.is_null() {
        &[][..]
    } else {
        // SAFETY: the title passes `count` descriptions.
        unsafe { std::slice::from_raw_parts(descs, count as usize) }
    };
    let total = place(descs.iter().map(|d| &d.desc), |i, info| {
        if !each.is_null() {
            // SAFETY: the title passes room for `count` entries.
            unsafe { each.add(i).write(info) };
        }
    });
    // SAFETY: the caller passes room for the value.
    unsafe { out.write(total) };
    out
}

/// Resources placed one after another: the whole footprint, telling `each`
/// where every resource goes. Unknown sizes give the documented `u64::MAX`.
fn place<'a>(
    descs: impl Iterator<Item = &'a ResourceDesc>,
    mut each: impl FnMut(usize, AllocationInfo1),
) -> AllocationInfo {
    let mut total = AllocationInfo {
        size: 0,
        alignment: KIB_64,
    };
    for (i, desc) in descs.enumerate() {
        let Some(one) = footprint(desc) else {
            tracing::warn!(?desc, "no size for this resource");
            each(
                i,
                AllocationInfo1 {
                    offset: u64::MAX,
                    alignment: 0,
                    size: u64::MAX,
                },
            );
            return AllocationInfo {
                size: u64::MAX,
                alignment: 0,
            };
        };
        tracing::debug!(?desc, ?one, "allocation info");
        let offset = total.size.next_multiple_of(one.alignment);
        each(
            i,
            AllocationInfo1 {
                offset,
                alignment: one.alignment,
                size: one.size,
            },
        );
        total.size = offset + one.size;
        total.alignment = total.alignment.max(one.alignment);
    }
    total
}

/// Zeroed host memory standing in for GPU memory.
pub(crate) struct Memory {
    ptr: NonNull<u8>,
    layout: Layout,
    /// Set when the memory came from the title's allocator, with these
    /// allocation attributes (it goes back there).
    attributes: Option<u64>,
}

pub type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
pub type FreeRoutine = extern "system" fn(*mut c_void, u64);
static ALLOCATOR: std::sync::OnceLock<(AllocRoutine, FreeRoutine)> = std::sync::OnceLock::new();

/// Committed resources get their memory from the title's general
/// allocator, as on the console, where titles hook it to place graphics
/// memory themselves (the title places its back buffers in a
/// block of its own and checks they fill it). The runtime names it before
/// the device exists.
pub fn use_allocator(alloc: AllocRoutine, free: FreeRoutine) {
    let _ = ALLOCATOR.set((alloc, free));
}

/// Maps the physical pages behind one range of title memory at another,
/// `(to, from, len)` (`from` 0 unmaps); false on failure.
pub type PageMapper = fn(u64, u64, u64) -> bool;
static PAGE_MAPPER: std::sync::OnceLock<PageMapper> = std::sync::OnceLock::new();

/// The runtime's page mapper, for the queues' page-mapping copies (memory
/// is the runtime's, as on the console).
pub fn use_page_mapper(mapper: PageMapper) {
    let _ = PAGE_MAPPER.set(mapper);
}

pub(crate) fn page_mapper() -> Option<PageMapper> {
    PAGE_MAPPER.get().copied()
}

/// Allocation attributes for resource memory: allocator id, page size
/// (bits 14-15: 2 MiB pages from 2 MiB up, else 64 KiB), alignment as log2
/// (bits 24-28) and memory type (bits 29-32: write-combined for GPU-only
/// heaps, cacheable for the CPU-visible ones).
/// shortcut: the console runtime's own choice of page size and memory type
/// is not known; this is the plausible one.
fn resource_attributes(info: AllocationInfo, heap_kind: u32) -> u64 {
    const GRAPHICS_ALLOCATOR: u64 = 128 << 16;
    const DEFAULT_HEAP: u32 = 1;
    let pages: u64 = if info.size >= 2 << 20 { 2 } else { 1 };
    let memory: u64 = if heap_kind == DEFAULT_HEAP { 1 } else { 3 };
    let alignment = u64::from(info.alignment.max(1).trailing_zeros());
    GRAPHICS_ALLOCATOR | pages << 14 | alignment << 24 | memory << 29
}

impl Memory {
    /// A committed resource's memory: from the title's allocator if the
    /// runtime named one, else our own; zeroed either way.
    fn for_resource(info: AllocationInfo, heap_kind: u32) -> Result<Memory, HRESULT> {
        let Some(&(alloc, _)) = ALLOCATOR.get() else {
            return Memory::new(info);
        };
        let layout = Layout::from_size_align(info.size as usize, info.alignment as usize)
            .map_err(|_| E_INVALIDARG)?;
        let attributes = resource_attributes(info, heap_kind);
        let ptr =
            NonNull::new(alloc(layout.size(), attributes).cast::<u8>()).ok_or(E_OUTOFMEMORY)?;
        // SAFETY: the allocator handed out `size` bytes.
        unsafe { ptr.as_ptr().write_bytes(0, layout.size()) };
        Ok(Memory {
            ptr,
            layout,
            attributes: Some(attributes),
        })
    }

    /// `size` zeroed bytes aligned to `alignment` (a power of two).
    pub(crate) fn zeroed(size: u64, alignment: u64) -> Result<Memory, HRESULT> {
        Memory::new(AllocationInfo { size, alignment })
    }

    pub(crate) fn address(&self) -> u64 {
        self.ptr.as_ptr() as u64
    }

    fn new(info: AllocationInfo) -> Result<Memory, HRESULT> {
        let layout = usize::try_from(info.size)
            .ok()
            .zip(usize::try_from(info.alignment).ok())
            .filter(|&(size, _)| size > 0)
            .and_then(|(size, align)| Layout::from_size_align(size, align).ok())
            .ok_or(E_INVALIDARG)?;
        // SAFETY: the size is non-zero.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) }).ok_or(E_OUTOFMEMORY)?;
        Ok(Memory {
            ptr,
            layout,
            attributes: None,
        })
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        match (self.attributes, ALLOCATOR.get()) {
            (Some(attributes), Some(&(_, free))) => free(self.ptr.as_ptr().cast(), attributes),
            // SAFETY: allocated in `new` with this layout.
            _ => unsafe { dealloc(self.ptr.as_ptr(), self.layout) },
        }
    }
}

#[repr(C)]
struct Heap {
    object: Object,
    desc: HeapDesc,
    memory: Memory,
}

pub(crate) extern "system" fn create_heap(
    _: This,
    desc: *const HeapDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(
        size = desc.size,
        kind = desc.properties.kind,
        flags = format_args!("{:#x}", desc.flags),
        "heap"
    );
    if desc.alignment != 0 && !desc.alignment.is_power_of_two() {
        return E_INVALIDARG;
    }
    let info = AllocationInfo {
        size: desc.size,
        alignment: desc.alignment.max(KIB_64),
    };
    let memory = match Memory::new(info) {
        Ok(memory) => memory,
        Err(hr) => return hr,
    };
    let heap = Box::new(Heap {
        object: Object::new(&HEAP_FUNCTIONS),
        desc,
        memory,
    });
    object::hand_out(HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn heap_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(HEAP, iid, object::is(iid, &HEAP_IIDS).then_some(this), out)
}
extern "system" fn heap_release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}
/// Returns `out`, as the x64 COM ABI requires for struct returns in `RAX`.
extern "system" fn heap_desc(this: This, out: *mut HeapDesc) -> *mut HeapDesc {
    if !out.is_null() {
        // SAFETY: `this` is a heap; the title passes room for a description.
        unsafe { out.write((*this.cast::<Heap>()).desc) };
    }
    out
}

/// A resource's memory: its own, or a range of a heap it holds a reference
/// to.
enum Backing {
    Committed(Memory),
    Placed {
        heap: This,
        offset: u64,
    },
    /// Memory the title mapped itself, at this address.
    Address(*mut u8),
}

impl Drop for Backing {
    fn drop(&mut self) {
        if let Backing::Placed { heap, .. } = *self {
            heap_release(heap);
        }
    }
}

#[repr(C)]
struct Resource {
    object: Object,
    desc: ResourceDesc,
    heap: HeapProperties,
    heap_flags: u32,
    backing: Backing,
    /// Textures also live on the host GPU (decision 0007); buffers do not
    /// yet. Made on first use (a view, a target, a copy, a barrier), not
    /// at creation: a texture the title makes and never has the GPU use
    /// costs no host memory (the world makes thousands; their host copies
    /// took about 9 GB, past an 8 GB card's budget, 2026-11-24).
    #[cfg(windows)]
    host: std::sync::OnceLock<Option<std::sync::Arc<HostTexture>>>,
    /// The creation state and clear value the host copy is made with.
    #[cfg(windows)]
    made_with: (u32, Option<[u32; 5]>),
    /// For a depth texture the title also writes as unordered access, the
    /// colour texture shaders write instead ([`Twin`]), made with the host
    /// copy.
    #[cfg(windows)]
    twin: std::sync::OnceLock<Option<std::sync::Arc<Twin>>>,
}

#[cfg(windows)]
impl Resource {
    /// The host copy, made now if this is its first use.
    ///
    /// Placed views (`texture_at`) never lose mips: they often alias a
    /// render-target or UAV that keeps every level. Dropping here would
    /// make the host subresource counts differ from that source; the SRV
    /// path would then fall back to uploading title memory, which does not
    /// hold the source's GPU writes. (`texture::alias` can also copy the
    /// retained source levels when counts differ, as a backstop.)
    fn host(&self, this: This) -> Option<&std::sync::Arc<HostTexture>> {
        self.host
            .get_or_init(|| {
                let (state, clear) = self.made_with;
                let clear = clear
                    .as_ref()
                    .map_or(std::ptr::null(), |c| std::ptr::from_ref(c).cast::<c_void>());
                let mip_drop = !placed_views().contains(&(this as usize));
                let (texture, state, dropped) = host_texture(&self.desc, state, clear, mip_drop)?;
                Some(std::sync::Arc::new(HostTexture {
                    texture,
                    state: state.into(),
                    born_seq: crate::host::birth_seq(),
                    dropped,
                }))
            })
            .as_ref()
    }

    /// The colour twin, made now if this is its first use: only a depth
    /// texture the title also writes as unordered access has one.
    fn twin(&self, this: This) -> Option<&std::sync::Arc<Twin>> {
        self.twin
            .get_or_init(|| {
                let depth = self.host(this)?;
                twin_texture(&self.desc, &depth.texture)
            })
            .as_ref()
    }
}

/// A title texture's host copy and the state it is in. The title's barriers
/// name console states (looser than the host's), so we transition from this
/// one. Lists that use the texture hold it until they are reset: the title
/// may release a resource whose commands it recorded but not yet submitted
/// (console memory outlives the object; a host texture must outlive the
/// lists that name it).
#[cfg(windows)]
pub(crate) struct HostTexture {
    pub(crate) texture: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    pub(crate) state: std::sync::atomic::AtomicU32,
    pub(crate) born_seq: u64,
    /// Top mip levels of the title's texture the host copy does not have
    /// (host memory pressure, see [`create_texture_with_auto_mip_drop`]):
    /// its subresource `i` is the title's `i + dropped`, per layer.
    pub(crate) dropped: u32,
}

/// The colour twin of a depth texture the title also writes as unordered
/// access. The console allows both on one texture; the host gives a depth
/// texture no unordered access, so shaders write a colour texture of the
/// same size and format family instead, and one is copied into the other
/// when a side is used after the other was written
/// (`HostList::twin_use`).
/// A depth texture with a stencil plane gets a twin for each plane (a
/// title's tiled lighting writes each pixel's material class into the
/// depth plane and masks into the stencil plane, then draws each class
/// with the depth test "equal"): the stencil plane's is copied through a
/// buffer, the depth plane's is drawn into the plane by a pass of ours
/// (`HostList::pass_plane`). Unordered access views of a multisampled
/// texture stay null.
/// shortcut: whole textures are copied, not the mips and slices written.
#[cfg(windows)]
pub(crate) struct Twin {
    /// The colour texture and the state it is in, as a host copy's.
    pub(crate) colour: std::sync::Arc<HostTexture>,
    /// Which side lacks what the other was written last, after the lists
    /// submitted so far: bit 0 the depth texture, bit 1 the twin.
    pub(crate) behind: std::sync::atomic::AtomicU32,
    /// For a depth texture with a stencil plane, `colour` is the twin of
    /// that plane (8 bits a texel) and this the buffer the two are copied
    /// through: no colour format shares a family with a depth-stencil
    /// format, but the host copies between a buffer and one plane.
    pub(crate) through: Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    /// For a depth texture with a stencil plane: the twin of its depth
    /// plane, 16 bits a texel for the console's 16-bit depth-stencil
    /// formats and a 32-bit float otherwise. No host colour format holds
    /// the bits of a 24-bit depth plane, so the two are not copied: a pass
    /// draws the twin into the plane.
    pub(crate) plane: Option<std::sync::Arc<HostTexture>>,
    /// Immutable shader-visible views of this plane's subresources. The lists
    /// referencing these descriptors keep the twin alive until they are reset.
    pub(crate) plane_views:
        std::sync::OnceLock<Option<windows::Win32::Graphics::Direct3D12::ID3D12DescriptorHeap>>,
}

#[cfg(windows)]
impl HostTexture {
    /// A host texture of our own making (a twin or a plane), in `state`.
    fn new(texture: windows::Win32::Graphics::Direct3D12::ID3D12Resource, state: u32) -> Self {
        Self {
            texture,
            state: state.into(),
            born_seq: crate::host::birth_seq(),
            // Twins and planes are made from the title's own description.
            dropped: 0,
        }
    }
}

#[cfg(windows)]
impl Drop for Twin {
    fn drop(&mut self) {
        TWINS[0].fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Colour twins alive, depth textures with unordered access left without
/// one, and the copies recorded into depth textures and into twins since
/// the last frame-rate line.
#[cfg(windows)]
pub(crate) static TWINS: [std::sync::atomic::AtomicU64; 4] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Depth planes drawn from their twins by the pass since the last
/// frame-rate line; and since the start, how often a depth plane's twin
/// lacked what was written to the plane (not given to it), and how often a
/// depth plane was used in the list that had written its twin, before the
/// pass (which runs when the list closes).
#[cfg(windows)]
pub(crate) static PLANES: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// The colour twin of title texture `this`, if it has one, made now if the
/// texture has not been used by the host yet.
#[cfg(windows)]
pub(crate) fn twin(this: This) -> Option<std::sync::Arc<Twin>> {
    if this.is_null() {
        return None;
    }
    resource(this).twin(this).cloned()
}

/// The twin's texture, for the unordered access views of `this`.
#[cfg(windows)]
pub(crate) fn twin_resource<'a>(
    this: This,
) -> Option<&'a windows::Win32::Graphics::Direct3D12::ID3D12Resource> {
    if this.is_null() {
        return None;
    }
    // SAFETY: the twin lives as long as the host copy, which lives as long
    // as the resource.
    resource(this)
        .twin(this)
        .map(|t| unsafe { &*std::ptr::from_ref(&t.colour.texture) })
}

/// The texture of the twin of the depth plane of `this`, a depth texture
/// with a stencil plane, for the unordered access views of that plane.
#[cfg(windows)]
pub(crate) fn twin_plane_resource<'a>(
    this: This,
) -> Option<&'a windows::Win32::Graphics::Direct3D12::ID3D12Resource> {
    if this.is_null() {
        return None;
    }
    let twin = resource(this).twin(this)?;
    // SAFETY: as in `twin_resource`.
    twin.plane
        .as_ref()
        .map(|p| unsafe { &*std::ptr::from_ref(&p.texture) })
}

/// The colour twin for a texture described by `desc` whose host copy is
/// `depth`, if the title asks for depth and unordered access both and the
/// host can copy between the two: the same size, mips and slices in the
/// depth format's typeless family (R16 for 16-bit depth, R32 for 32-bit),
/// or R8 for the stencil plane of a format that has one, with unordered
/// access only. A format with a stencil plane gets a twin for its depth
/// plane too, R16 for the console's 16-bit depth-stencil formats and R32
/// otherwise. `VICTORY_DEPTH_TWINS=0` makes none (such views stay null),
/// `VICTORY_DEPTH_TWINS=stencil` only those of formats with a stencil
/// plane, `VICTORY_DEPTH_TWINS=nodepth` none for the depth planes of such
/// formats.
#[cfg(windows)]
fn twin_texture(
    desc: &ResourceDesc,
    depth: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
) -> Option<std::sync::Arc<Twin>> {
    use std::sync::atomic::Ordering::Relaxed;
    use windows::Win32::Graphics::Direct3D12::*;
    const DEPTH_AND_UNORDERED: u32 = 6;
    const UNORDERED: u32 = 4;
    const UNORDERED_ACCESS: u32 = 0x8;
    // R32 and R16 typeless: the families of the depth formats of one plane.
    const ONE_PLANE: [u32; 2] = [39, 53];
    if desc.dimension == BUFFER || desc.flags & DEPTH_AND_UNORDERED != DEPTH_AND_UNORDERED {
        return None;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_DEPTH_TWINS").is_none_or(|v| v != "0")) {
        return None;
    }
    // R32G8X24 and R24G8 typeless: depth with a stencil plane, whose twin
    // is R8 typeless.
    const WITH_STENCIL: [u32; 2] = [19, 44];
    const R8_TYPELESS: u32 = 60;
    let typeless = family(pc_format(desc.format));
    let stencil = WITH_STENCIL.contains(&typeless);
    static STENCIL_ONLY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let stencil_only = *STENCIL_ONLY
        .get_or_init(|| std::env::var_os("VICTORY_DEPTH_TWINS").is_some_and(|v| v == "stencil"));
    if !stencil && stencil_only {
        return None;
    }
    if !(stencil || ONE_PLANE.contains(&typeless)) || desc.sample_count > 1 {
        TWINS[1].fetch_add(1, Relaxed);
        return None;
    }
    let host = crate::host::host()?;
    let mut d = *desc;
    d.layout = 0;
    d.alignment = 0;
    d.flags = UNORDERED;
    d.format = if stencil { R8_TYPELESS } else { typeless };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: the description has the host layout, as in `host_texture`;
    // the out pointer is ours.
    let made = unsafe {
        host.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            (&raw const d).cast(),
            D3D12_RESOURCE_STATES(UNORDERED_ACCESS as i32),
            None,
            &mut texture,
        )
    };
    if let Err(e) = made {
        tracing::warn!(error = %e, desc = ?d, "no colour twin for a depth texture");
    }
    let Some(texture) = texture else {
        TWINS[1].fetch_add(1, Relaxed);
        return None;
    };
    crate::host::name(&texture, || {
        format!("depth twin {}x{} f{}", d.width, d.height, d.format)
    });
    // The buffer a stencil plane and its twin are copied through: as big
    // as the plane's mips and slices laid out for a copy (they follow the
    // depth plane's in the texture's subresources).
    let through = if stencil {
        // SAFETY: a live texture.
        let of = unsafe { depth.GetDesc() };
        let planes = u32::from(of.MipLevels) * u32::from(of.DepthOrArraySize);
        let mut bytes = 0;
        // SAFETY: a plain query; the out pointer is ours.
        unsafe {
            host.device.GetCopyableFootprints(
                &of,
                planes,
                planes,
                0,
                None,
                None,
                None,
                Some(&mut bytes),
            );
        }
        let Some(buffer) = host.default_buffer(bytes) else {
            TWINS[1].fetch_add(1, Relaxed);
            return None;
        };
        Some(buffer)
    } else {
        None
    };
    // The twin of the depth plane of a format with a stencil plane: R16
    // typeless (53) for the console's 16-bit depth-stencil formats (118 to
    // 120), whose shaders write 16-bit steps, R32 typeless (39) otherwise.
    static NO_PLANE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let no_plane = *NO_PLANE
        .get_or_init(|| std::env::var_os("VICTORY_DEPTH_TWINS").is_some_and(|v| v == "nodepth"));
    let plane = if stencil && !no_plane {
        d.format = if (118..=120).contains(&desc.format) {
            53
        } else {
            39
        };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: as for the first twin.
        let made = unsafe {
            host.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                (&raw const d).cast(),
                D3D12_RESOURCE_STATES(UNORDERED_ACCESS as i32),
                None,
                &mut texture,
            )
        };
        if let Err(e) = made {
            tracing::warn!(error = %e, desc = ?d, "no twin for a depth plane");
        }
        texture.map(|texture| {
            crate::host::name(&texture, || {
                format!("depth plane twin {}x{} f{}", d.width, d.height, d.format)
            });
            std::sync::Arc::new(HostTexture::new(texture, UNORDERED_ACCESS))
        })
    } else {
        None
    };
    TWINS[0].fetch_add(1, Relaxed);
    Some(std::sync::Arc::new(Twin {
        colour: std::sync::Arc::new(HostTexture::new(texture, UNORDERED_ACCESS)),
        behind: std::sync::atomic::AtomicU32::new(0),
        through,
        plane,
        plane_views: Default::default(),
    }))
}

/// Moves the host texture to `wanted` (a host texture state); the host
/// resource and the state it leaves, if that is a change.
#[cfg(windows)]
pub(crate) fn host_transition(this: This, wanted: u32) -> Option<(*mut c_void, u32)> {
    use windows::core::Interface;
    let host = resource(this).host(this)?;
    let before = host
        .state
        .swap(wanted, std::sync::atomic::Ordering::Relaxed);
    (before != wanted).then(|| (host.texture.as_raw(), before))
}

/// The host GPU's copy of a title resource, if it has one.
#[cfg(windows)]
pub(crate) fn host_resource<'a>(
    this: This,
) -> Option<&'a windows::Win32::Graphics::Direct3D12::ID3D12Resource> {
    if this.is_null() {
        return None;
    }
    resource(this).host(this).map(|h| &h.texture)
}

/// A title texture's host copy, shared, for a list to hold.
#[cfg(windows)]
pub(crate) fn host_copy(this: This) -> Option<std::sync::Arc<HostTexture>> {
    if this.is_null() {
        return None;
    }
    resource(this).host(this).cloned()
}

/// How many top mip levels of a title texture its host copy lacks (made
/// now if this is its first use); 0 if it has none or no host copy.
#[cfg(windows)]
pub(crate) fn dropped_mips(this: This) -> u32 {
    if this.is_null() {
        return 0;
    }
    resource(this).host(this).map_or(0, |h| h.dropped)
}

/// A host texture like the title's: same description, in GPU memory,
/// without the console's layout and flags, the state it was made in
/// (what the transitions that follow start from), and how many top mip
/// levels it lost to host memory pressure. `mip_drop` is false for placed
/// views that may alias a full-mip render target or UAV.
#[cfg(windows)]
fn host_texture(
    desc: &ResourceDesc,
    state: u32,
    clear: *const c_void,
    mip_drop: bool,
) -> Option<(
    windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    u32,
    u32,
)> {
    use windows::Win32::Graphics::Direct3D12::*;
    const PC_FLAGS: u32 = 0xf; // render target, depth, UAV, no shader access
    if desc.dimension == BUFFER {
        return None;
    }
    let host = crate::host::host()?;
    let mut d = *desc;
    d.layout = 0;
    d.alignment = 0;
    d.flags &= PC_FLAGS;
    // PC refuses depth targets that are also unordered-access (the console
    // allows both; the title's shadow maps ask for it).
    // The host copy loses UAV use; depth textures of one plane get a colour
    // twin that shaders write instead ([`Twin`]).
    const DEPTH: u32 = 2;
    const UNORDERED: u32 = 4;
    if d.flags & DEPTH != 0 {
        d.flags &= !UNORDERED;
    }
    const RENDER_TARGET: u32 = 1;
    d.format = if d.flags & (RENDER_TARGET | UNORDERED) != 0 {
        target_format(d.format)
    } else {
        pc_format(d.format)
    };
    const DENY_SHADER_RESOURCE: u32 = 8;
    if d.flags & DENY_SHADER_RESOURCE == 0
        && let Some(&(_, typeless)) = DEPTH_FORMATS.iter().find(|&&(f, _)| f == d.format)
    {
        d.format = typeless;
    }
    // The host has no sRGB unordered access: such textures are typeless,
    // their unordered access views plain UNORM.
    if d.flags & UNORDERED != 0 && SRGB.iter().any(|&(srgb, _)| srgb == d.format) {
        d.format = family(d.format);
    }
    // The clear value names the format too: format, then color or depth
    // and stencil (5 words). The host takes it only when it names the
    // resource's own format or, for a depth target, the pair's other one,
    // so it is passed through as the title gave it and dropped only if
    // the host refuses the texture with it: that costs the texture's
    // initial contents, not the texture.
    let clear_value: Option<[u32; 5]> = (!clear.is_null()).then(|| {
        // SAFETY: the title passes a whole clear value.
        let mut value = unsafe { clear.cast::<[u32; 5]>().read_unaligned() };
        value[0] = if d.flags & (RENDER_TARGET | UNORDERED) != 0 {
            target_format(value[0])
        } else {
            pc_format(value[0])
        };
        value
    });
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    // Not every host GPU offers every use of every format (NVIDIA: no
    // unordered access to BGRA8, no B4G4R4A4 render targets, where the
    // console's AMD GPU has both). Try what the title asked for, then give
    // up the state, the clear value and the uses the host may refuse, in
    // that order; the views the host cannot make are skipped as for depth.
    // Within each attempt, `E_OUTOFMEMORY` may drop top mips first.
    let asked = crate::host::legal_for(d.flags, crate::host::texture_state(state));
    let tries = attempts(d.flags, asked);
    let mut tried: Vec<(u32, u32, Option<[u32; 5]>)> = Vec::with_capacity(tries.len());
    let mut texture: Option<ID3D12Resource> = None;
    let mut made_in = None;
    let mut dropped = 0u32;
    let mut last = None;
    let create_one = |trial: &ResourceDesc,
                      initial: u32,
                      try_clear: Option<[u32; 5]>,
                      out: &mut Option<ID3D12Resource>| {
        #[cfg(test)]
        if trial.width > synthetic_oom_above() {
            return Err(windows::core::Error::from(windows::core::HRESULT(
                E_OUTOFMEMORY,
            )));
        }
        // SAFETY: the description has the host layout (checked size, same
        // fields); the out pointer is ours.
        unsafe {
            host.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                std::ptr::from_ref(trial).cast(),
                D3D12_RESOURCE_STATES(initial as i32),
                // The same layout: format, then color or depth and stencil.
                try_clear.as_ref().map(|v| std::ptr::from_ref(v).cast()),
                out,
            )
        }
    };
    for (i, &(flags, wanted, keep_clear)) in tries.iter().enumerate() {
        let mut trial = d;
        trial.flags = flags;
        let initial = crate::host::legal_for(flags, wanted);
        // A clear value is only legal on render targets and depth
        // (VIC-BT-022): a non-null pOptimizedClearValue with neither flag
        // is E_INVALIDARG on PC D3D12.
        let try_clear = optimized_clear(flags, keep_clear, clear_value);
        // Permissions that leave the description unchanged are not asked twice.
        if tried.contains(&(flags, initial, try_clear)) {
            continue;
        }
        tried.push((flags, initial, try_clear));
        let result = if mip_drop {
            create_texture_with_auto_mip_drop(&trial, |attempt| {
                let mut made = None;
                create_one(attempt, initial, try_clear, &mut made)
                    .map(|()| made)
                    .map_err(|e| e.code().0)
            })
            .map_err(|code| windows::core::Error::from(windows::core::HRESULT(code)))
        } else {
            let mut made = None;
            create_one(&trial, initial, try_clear, &mut made).map(|()| (made, trial, 0u32))
        };
        match result {
            Ok((made, attempt, lost)) => {
                if i > 0 {
                    // The host refused what the title asked for: name what
                    // was given up, once for this texture.
                    tracing::warn!(
                        desc = ?d,
                        flags_asked = d.flags,
                        flags_made = flags,
                        state_asked = asked,
                        state_made = initial,
                        clear_asked = clear_value.map(|v| v[0]),
                        clear_dropped = clear_value.is_some() && try_clear.is_none(),
                        "host texture made only as a fallback"
                    );
                }
                texture = made;
                d = attempt;
                dropped = lost;
                made_in = Some(initial);
                break;
            }
            Err(e) => {
                texture = None;
                tracing::debug!(
                    error = %e,
                    attempt = i,
                    flags,
                    state = initial,
                    "host texture refused; trying a fallback"
                );
                last = Some((e, trial, initial, try_clear.map(|v| v[0])));
            }
        }
    }
    let (Some(texture), Some(initial)) = (texture, made_in) else {
        if let Some((e, trial, initial, clear)) = last {
            tracing::warn!(
                error = %e,
                desc = ?trial,
                state = initial,
                title_state = state,
                clear,
                "no host texture"
            );
        }
        return None;
    };
    crate::host::name(&texture, || {
        format!(
            "texture {}x{} f{} flags {:#x}",
            d.width, d.height, d.format, d.flags
        )
    });
    Some((texture, initial, dropped))
}

// Tests: host textures wider than this fail with `E_OUTOFMEMORY` before
// reaching the host, as an exhausted adapter would refuse them. Thread-local
// so parallel GPU tests are not refused by another test's threshold.
#[cfg(all(test, windows))]
std::thread_local! {
    static SYNTHETIC_OOM_ABOVE: std::cell::Cell<u64> = const { std::cell::Cell::new(u64::MAX) };
}

#[cfg(all(test, windows))]
fn synthetic_oom_above() -> u64 {
    SYNTHETIC_OOM_ABOVE.with(|c| c.get())
}

/// Sets the synthetic OOM width threshold for this thread and restores it
/// when dropped (including on panic).
#[cfg(all(test, windows))]
struct SyntheticOom {
    previous: u64,
}

#[cfg(all(test, windows))]
impl SyntheticOom {
    fn above(limit: u64) -> Self {
        let previous = SYNTHETIC_OOM_ABOVE.with(|c| {
            let previous = c.get();
            c.set(limit);
            previous
        });
        Self { previous }
    }
}

#[cfg(all(test, windows))]
impl Drop for SyntheticOom {
    fn drop(&mut self) {
        SYNTHETIC_OOM_ABOVE.with(|c| c.set(self.previous));
    }
}

/// PC D3D12 accepts `pOptimizedClearValue` only when the resource may be a
/// render target or a depth stencil. Passing a non-null clear with neither
/// flag (or for a buffer) fails with `E_INVALIDARG`; console titles can
/// still name a clear on resources whose host flags lose those bits.
#[cfg(any(windows, test))]
fn optimized_clear(flags: u32, keep: bool, clear: Option<[u32; 5]>) -> Option<[u32; 5]> {
    const RENDER_TARGET: u32 = 1;
    const DEPTH: u32 = 2;
    clear.filter(|_| keep && flags & (RENDER_TARGET | DEPTH) != 0)
}

/// The (flags, state, whether the clear value is kept) triples to try for a
/// host texture, most faithful first: what the title asked for, then in the
/// common state, then without the clear value, then without the permissions
/// the host may refuse. A texture the host cannot make as asked still exists
/// for the draws that read it; the log names what was lost. Nothing is
/// tried twice, permissions only go down, a depth target keeps its depth
/// permission, and `keep_clear` is false when the flags cannot take a clear.
#[cfg(any(windows, test))]
fn attempts(flags: u32, state: u32) -> Vec<(u32, u32, bool)> {
    const UNORDERED: u32 = 4;
    const RENDER_TARGET: u32 = 1;
    const DEPTH: u32 = 2;
    let mut out: Vec<(u32, u32, bool)> = Vec::with_capacity(7);
    for (f, s, keep) in [
        (flags, state, true),
        (flags, 0, true),
        (flags, state, false),
        (flags, 0, false),
        (flags & !UNORDERED, 0, true),
        (flags & !UNORDERED, 0, false),
        (flags & !UNORDERED & !RENDER_TARGET, 0, false),
    ] {
        // Never advertise a clear the host would refuse (VIC-BT-022).
        let triple = (f, s, keep && f & (RENDER_TARGET | DEPTH) != 0);
        if !out.contains(&triple) {
            out.push(triple);
        }
    }
    out
}

/// A texture's console memory and description; `None` for buffers.
pub(crate) fn texture_memory(this: This) -> Option<(*const u8, ResourceDesc)> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    (r.desc.dimension != BUFFER).then(|| (r.data().cast_const(), r.desc))
}

/// A buffer's host memory and size; `None` for textures.
pub(crate) fn buffer_bytes(this: This) -> Option<(*mut u8, u64)> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    (r.desc.dimension == BUFFER).then(|| (r.data(), r.desc.width))
}

impl Resource {
    fn data(&self) -> *mut u8 {
        match self.backing {
            Backing::Committed(ref memory) => memory.ptr.as_ptr(),
            // SAFETY: placement checked the range lies inside the heap.
            Backing::Placed { heap, offset } => unsafe {
                (*heap.cast::<Heap>())
                    .memory
                    .ptr
                    .as_ptr()
                    .add(offset as usize)
            },
            Backing::Address(at) => at,
        }
    }
}

fn resource<'a>(this: This) -> &'a Resource {
    // SAFETY: called from resource methods only.
    unsafe { &*this.cast::<Resource>() }
}

// The creation methods' own arguments, passed on.
#[allow(clippy::too_many_arguments)]
fn hand_out_resource(
    desc: ResourceDesc,
    heap: HeapProperties,
    heap_flags: u32,
    backing: Backing,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    #[cfg(not(windows))]
    let _ = (state, clear);
    // The clear value is the caller's only during the call.
    #[cfg(windows)]
    let made_with = (
        state,
        // SAFETY: the title passes a whole clear value (5 words) or null.
        (!clear.is_null()).then(|| unsafe { clear.cast::<[u32; 5]>().read_unaligned() }),
    );
    let resource = Box::new(Resource {
        object: Object::new(&RESOURCE_FUNCTIONS),
        desc,
        heap,
        heap_flags,
        backing,
        #[cfg(windows)]
        host: std::sync::OnceLock::new(),
        #[cfg(windows)]
        made_with,
        #[cfg(windows)]
        twin: std::sync::OnceLock::new(),
    });
    #[cfg(windows)]
    if desc.dimension == BUFFER {
        let at = std::ptr::from_ref::<Resource>(&resource) as usize;
        crate::shadow::register(at, resource.data() as u64, desc.width);
    } else if host_writes(&desc) {
        let at = std::ptr::from_ref::<Resource>(&resource) as usize;
        targets().insert(resource.data() as u64, at);
    }
    object::hand_out(RESOURCE, resource, iid, &RESOURCE_IIDS, out)
}

#[cfg(windows)]
impl Drop for Resource {
    fn drop(&mut self) {
        #[cfg(feature = "diagnostics")]
        if self
            .host
            .get()
            .and_then(Option::as_ref)
            .is_some_and(|h| std::sync::Arc::strong_count(h) > 1)
        {
            // Before lists held their textures, this freed a texture that
            // lists not yet run still named (a GPU page fault).
            static OUTLIVED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = OUTLIVED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                tracing::info!(
                    n,
                    desc = ?self.desc,
                    "a title texture was released while a list not yet reset names its host copy"
                );
            }
        }
        // Forget descriptor/staging/target entries before retiring the host
        // copy so no concurrent bindless sweep can observe it after retirement
        // snapshots active epochs.
        crate::texture::forget(std::ptr::from_ref::<Resource>(self) as usize);
        crate::descriptor::forget(std::ptr::from_ref::<Resource>(self) as usize);
        if self.desc.dimension != BUFFER && host_writes(&self.desc) {
            let mut targets = targets();
            let at = self.data() as u64;
            if targets.get(&at) == Some(&(std::ptr::from_ref::<Resource>(self) as usize)) {
                targets.remove(&at);
            }
        }
        if self.desc.dimension == BUFFER {
            crate::shadow::unregister(
                std::ptr::from_ref::<Resource>(self) as usize,
                self.data() as u64,
            );
        }
        // A running submission or list that recorded work after this host copy
        // was created may still read it.
        if let Some(host) = self.host.take().flatten() {
            let born = host.born_seq;
            crate::host::retire_born(host, born);
        }
    }
}

pub(crate) extern "system" fn create_committed(
    _: This,
    heap: *const HeapProperties,
    heap_flags: u32,
    desc: *const ResourceDesc,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes both descriptions (null is tolerated).
    let (Some(&heap), Some(&desc)) = (unsafe { (heap.as_ref(), desc.as_ref()) }) else {
        return E_INVALIDARG;
    };
    let Some(info) = footprint(&desc) else {
        tracing::warn!(?desc, "no size for this resource");
        return E_INVALIDARG;
    };
    tracing::debug!(
        dimension = desc.dimension,
        size = info.size,
        heap = heap.kind,
        "committed resource"
    );
    let memory = match Memory::for_resource(info, heap.kind) {
        Ok(memory) => memory,
        Err(hr) => return hr,
    };
    hand_out_resource(
        desc,
        heap,
        heap_flags,
        Backing::Committed(memory),
        state,
        clear,
        iid,
        out,
    )
}

/// Device method 0x3f8, inferred from the title's calls
/// (title+0x193e1b4: its back buffers, heap flag "allow display"): the
/// committed resource creation of the enhanced-barrier interfaces,
/// `(heap, heap flags, desc1, initial layout, clear value, protected
/// session, castable format count, castable formats, iid, out)`. The host
/// copy starts in the common state, as layouts are not tracked.
#[allow(clippy::too_many_arguments)] // the method's own arguments
pub(crate) extern "system" fn create_committed3(
    this: This,
    heap: *const HeapProperties,
    heap_flags: u32,
    desc: *const ResourceDesc,
    _layout: u32,
    clear: *const c_void,
    _protected: This,
    _castable_count: u32,
    _castable: *const u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    const COMMON: u32 = 0;
    create_committed(this, heap, heap_flags, desc, COMMON, clear, iid, out)
}

/// Registered page pools (64 KiB pages the title maps into tiled
/// resources by page index): first address and page count; a handle is an
/// index plus one, a removed pool leaves `None`.
static PAGE_POOLS: std::sync::Mutex<Vec<Option<(u64, u32)>>> = std::sync::Mutex::new(Vec::new());

fn page_pools() -> std::sync::MutexGuard<'static, Vec<Option<(u64, u32)>>> {
    PAGE_POOLS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Device method 0x250: `(address, page count, handle out)`.
pub(crate) extern "system" fn register_page_pool(
    _: This,
    address: u64,
    pages: u32,
    out: *mut usize,
) -> HRESULT {
    if out.is_null() || pages == 0 || !address.is_multiple_of(KIB_64) {
        return E_INVALIDARG;
    }
    tracing::debug!(address = format_args!("{address:#x}"), pages, "page pool");
    let mut pools = page_pools();
    pools.push(Some((address, pages)));
    // SAFETY: checked non-null; the title passes room for a handle.
    unsafe { out.write(pools.len()) };
    S_OK
}

/// Device method 0x258.
pub(crate) extern "system" fn unregister_page_pool(_: This, handle: usize) -> HRESULT {
    match page_pools().get_mut(handle.wrapping_sub(1)) {
        Some(pool @ Some(_)) => {
            *pool = None;
            S_OK
        }
        _ => E_INVALIDARG,
    }
}

/// Device method 0x428, one of the `ID3D12Device9` to `ID3D12Device11`
/// additions. From the title's call: a resource at a virtual
/// address the title mapped itself, `(address, desc1, initial barrier
/// layout, clear value, iid, out)`, as `CreatePlacedResourceX` with
/// enhanced barriers. GPU addresses are CPU addresses, so the resource is
/// a view of that memory. `D3D12_RESOURCE_DESC1` starts like the plain
/// description; its sampler feedback region is not needed.
pub(crate) extern "system" fn create_at_address(
    _: This,
    address: u64,
    desc: *const ResourceDesc,
    layout: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if address == 0 {
        return E_INVALIDARG;
    }
    tracing::debug!(
        address = format_args!("{address:#x}"),
        dimension = desc.dimension,
        width = desc.width,
        layout,
        "resource at an address"
    );
    const COMMON: u32 = 0;
    const DEFAULT: HeapProperties = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 1,
        visible_nodes: 1,
    };
    hand_out_resource(
        desc,
        DEFAULT,
        0,
        Backing::Address(address as *mut u8),
        COMMON,
        clear,
        iid,
        out,
    )
}

/// Textures the title placed in its own memory and views without a
/// resource object (device 0x230, the title's streamed and
/// interface textures), by address and description: a resource at that
/// address, made the first time a draw reads one.
/// shortcut: never released (one host texture per distinct address and
/// description viewed); release them when the title unmaps or reuses the
/// memory once streaming grows VRAM use.
#[cfg(windows)]
static PLACED_TEXTURES: std::sync::Mutex<std::collections::BTreeMap<(u64, [u32; 14]), usize>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// The resource for a texture placed at `address` with the resource
/// description `desc` (as words), made on first use.
#[cfg(windows)]
pub(crate) fn texture_at(address: u64, desc: &[u32; 14]) -> Option<This> {
    let mut map = PLACED_TEXTURES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&r) = map.get(&(address, *desc)) {
        return Some(r as This);
    }
    // SAFETY: 56 bytes in the description's layout.
    let d = unsafe { desc.as_ptr().cast::<ResourceDesc>().read_unaligned() };
    if d.dimension == BUFFER {
        return None;
    }
    let mut out = std::ptr::null();
    let made = create_at_address(
        std::ptr::null(),
        address,
        &d,
        0,
        std::ptr::null(),
        &RESOURCE_IIDS[0],
        &mut out,
    );
    if made != S_OK {
        return None;
    }
    // Register before the first host() so mip-drop pressure does not shrink
    // placed views that may alias a full-mip RT/UAV.
    placed_views().insert(out as usize);
    // create_at_address can succeed with no host texture (unsupported
    // format/flags): do not cache that failure, or the next call would
    // return Some with host == None (VIC-BT-027).
    if resource(out).host(out).is_none() {
        placed_views().remove(&(out as usize));
        object::release_boxed::<Resource>(out);
        return None;
    }
    map.insert((address, *desc), out as usize);
    Some(out)
}

/// The resources [`texture_at`] made (never released, as they are).
#[cfg(windows)]
fn placed_views() -> std::sync::MutexGuard<'static, std::collections::BTreeSet<usize>> {
    static PLACED: std::sync::Mutex<std::collections::BTreeSet<usize>> =
        std::sync::Mutex::new(std::collections::BTreeSet::new());
    PLACED.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether the host writes a texture described by `desc` (a render target,
/// depth or unordered-access texture): its contents exist on the host alone,
/// not in title memory.
#[cfg(windows)]
fn host_writes(desc: &ResourceDesc) -> bool {
    const TARGET_DEPTH_UNORDERED: u32 = 0x7;
    desc.dimension != BUFFER && desc.flags & TARGET_DEPTH_UNORDERED != 0
}

/// The title textures the host writes ([`host_writes`]), by the address of
/// their memory: what a view the title places in that memory in another
/// format shows ([`alias_of`]). A texture leaves the table before it is
/// freed.
#[cfg(windows)]
fn targets() -> std::sync::MutexGuard<'static, std::collections::BTreeMap<u64, usize>> {
    static TARGETS_AT: std::sync::Mutex<std::collections::BTreeMap<u64, usize>> =
        std::sync::Mutex::new(std::collections::BTreeMap::new());
    TARGETS_AT.lock().unwrap_or_else(|e| e.into_inner())
}

/// For `this`, a texture made for a view the title placed in memory
/// ([`texture_at`]): the title texture the host writes that lies in the same
/// memory, if there is one of the same size, levels and layers with texels of
/// the same width.
///
/// The console reads such memory through the view as it is: a title can
/// pack light as R9G9B9E5 into R32_UINT textures with compute shaders, which
/// cannot write R9G9B9E5, and sample them through R9G9B9E5 views placed at
/// the textures' addresses. On the host these are two textures and what the
/// shaders wrote exists in one of them only; the other was filled from title
/// memory, which nothing writes, so everything read through such a view was
/// black. The caller copies the written texture's texels into the view's
/// (`texture::alias`).
///
/// `VICTORY_ALIAS_COPY=0` (diagnostics builds) finds none, as before.
#[cfg(windows)]
pub(crate) fn alias_of(this: This) -> Option<This> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ON.get_or_init(|| {
        !cfg!(feature = "diagnostics")
            || std::env::var_os("VICTORY_ALIAS_COPY").is_none_or(|v| v != "0")
    });
    if this.is_null() || !on {
        return None;
    }
    let r = resource(this);
    // A level count of 0 asks for a full chain: not counted here.
    let plain = |d: &ResourceDesc| {
        d.dimension != BUFFER
            && d.mip_levels >= 1
            && d.depth_or_array_size >= 1
            && d.sample_count <= 1
    };
    if host_writes(&r.desc) || !plain(&r.desc) || !placed_views().contains(&(this as usize)) {
        return None;
    }
    // Under the table's lock: a texture leaves it before it is freed.
    let targets = targets();
    let other = *targets.get(&(r.data() as u64))? as This;
    let o = resource(other);
    const DEPTH: u32 = 0x2;
    let same = plain(&o.desc)
        && o.desc.dimension == r.desc.dimension
        && o.desc.width == r.desc.width
        && o.desc.height == r.desc.height
        && o.desc.mip_levels == r.desc.mip_levels
        && o.desc.depth_or_array_size == r.desc.depth_or_array_size
        && o.desc.flags & DEPTH == 0
        && target_format(o.desc.format) == o.desc.format
        && element(o.desc.format).is_some_and(|e| e.1 == 1 && Some(e) == element(r.desc.format));
    same.then_some(other)
}

pub(crate) extern "system" fn create_placed(
    _: This,
    heap: This,
    offset: u64,
    desc: *const ResourceDesc,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if heap.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title passes one of our heaps.
    let h = unsafe { &*heap.cast::<Heap>() };
    let fits = footprint(&desc).is_some_and(|info| {
        offset.is_multiple_of(info.alignment)
            && offset
                .checked_add(info.size)
                .is_some_and(|end| end <= h.desc.size)
    });
    if !fits {
        tracing::warn!(
            ?desc,
            offset,
            heap = h.desc.size,
            "placed resource does not fit"
        );
        return E_INVALIDARG;
    }
    object::add_ref(heap);
    let backing = Backing::Placed { heap, offset };
    hand_out_resource(
        desc,
        h.desc.properties,
        h.desc.flags,
        backing,
        state,
        clear,
        iid,
        out,
    )
}

extern "system" fn resource_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        RESOURCE,
        iid,
        object::is(iid, &RESOURCE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn resource_release(this: This) -> u32 {
    object::release_boxed::<Resource>(this)
}
/// Memory is always mapped (subresources are not told apart yet); a null
/// `out` only asks for residency.
extern "system" fn map(this: This, _sub: u32, _read: *const c_void, out: *mut *mut u8) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a pointer.
        unsafe { out.write(resource(this).data()) };
    }
    S_OK
}
extern "system" fn unmap(_: This, _sub: u32, _written: *const c_void) {}
/// Returns `out`, as the x64 COM ABI requires for struct returns in `RAX`.
extern "system" fn resource_desc(this: This, out: *mut ResourceDesc) -> *mut ResourceDesc {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(resource(this).desc) };
    }
    out
}
/// Device method `GetCustomHeapProperties`, returning its struct through a
/// hidden first argument: what a standard heap type is as a custom heap.
/// Memory is unified and cache-coherent (as the architecture query says),
/// so every type is in pool L0, default heaps are not CPU-visible and the
/// others are write-back.
pub(crate) extern "system" fn custom_heap_properties(
    out: *mut HeapProperties,
    _: This,
    nodes: u32,
    kind: u32,
) -> *mut HeapProperties {
    const CUSTOM: u32 = 4;
    const NOT_AVAILABLE: u32 = 1;
    const WRITE_BACK: u32 = 3;
    const L0: u32 = 1;
    let nodes = nodes.max(1);
    let cpu_pages = match kind {
        1 => NOT_AVAILABLE,
        2 | 3 => WRITE_BACK,
        _ => 0,
    };
    let properties = HeapProperties {
        kind: CUSTOM,
        cpu_pages,
        pool: if cpu_pages == 0 { 0 } else { L0 },
        creation_nodes: nodes,
        visible_nodes: nodes,
    };
    // SAFETY: the title passes room for the result (null is tolerated).
    if let Some(out) = unsafe { out.as_mut() } {
        *out = properties;
    }
    out
}

extern "system" fn gpu_address(this: This) -> u64 {
    let r = resource(this);
    let at = r.data() as u64;
    // The console's GPU addresses fit in 40 bits; its ray-tracing builder
    // refuses others (diagnostics: which memory breaks that).
    #[cfg(feature = "diagnostics")]
    if at >> 40 != 0 {
        static ONCE: [std::sync::Once; 2] = [std::sync::Once::new(), std::sync::Once::new()];
        let backing = match r.backing {
            Backing::Committed(ref m) => format!("committed, attributes {:?}", m.attributes),
            Backing::Placed { offset, .. } => format!("placed in a heap at {offset:#x}"),
            Backing::Address(_) => "at a title address".into(),
        };
        ONCE[usize::from(r.desc.dimension == BUFFER)].call_once(|| {
            tracing::warn!(address = format_args!("{at:#x}"), backing, desc = ?r.desc, "a GPU address above 40 bits")
        });
    }
    at
}
extern "system" fn heap_properties(
    this: This,
    properties: *mut HeapProperties,
    flags: *mut u32,
) -> HRESULT {
    let r = resource(this);
    // SAFETY: either pointer may be null; otherwise the title passes room.
    unsafe {
        if let Some(p) = properties.as_mut() {
            *p = r.heap;
        }
        if let Some(f) = flags.as_mut() {
            *f = r.heap_flags;
        }
    }
    S_OK
}

/// IUnknown 3, object 4, device child 1, heap 1.
static HEAP_FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<HEAP, 24>());
    f[0] = heap_query as *const ();
    f[2] = heap_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = heap_desc as *const (); // 0x40
    Table(f)
};

/// IUnknown 3, object 4, device child 1, resource 10 (7 as on PC, then 3
/// console ones).
static RESOURCE_FUNCTIONS: Table<32> = {
    let mut f = object::d3d12_object(object::missing_table::<RESOURCE, 32>());
    f[0] = resource_query as *const ();
    f[2] = resource_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = map as *const (); // 0x40
    f[9] = unmap as *const (); // 0x48
    f[10] = resource_desc as *const (); // 0x50
    f[11] = gpu_address as *const (); // 0x58
    f[14] = heap_properties as *const (); // 0x70
    Table(f)
};

#[cfg(test)]
pub(crate) fn test_buffer(width: u64) -> This {
    let heap = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 0,
        visible_nodes: 0,
    };
    let desc = ResourceDesc {
        dimension: BUFFER,
        alignment: 0,
        width,
        height: 1,
        depth_or_array_size: 1,
        mip_levels: 1,
        format: 0,
        sample_count: 1,
        sample_quality: 0,
        layout: 1,
        flags: 0,
    };
    let mut out = std::ptr::null();
    let hr = create_committed(
        std::ptr::null(),
        &heap,
        0,
        &desc,
        0,
        std::ptr::null(),
        &RESOURCE_IIDS[0],
        &mut out,
    );
    assert_eq!(hr, S_OK);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    #[test]
    fn a8_unorm_is_not_in_the_r8_typeless_family() {
        // DXGI_FORMAT values: 60 R8_TYPELESS, 61 R8_UNORM, 62 R8_UINT,
        // 63 R8_SNORM, 64 R8_SINT, 65 A8_UNORM. The R8 typeless family is
        // 60..=64; A8_UNORM has no typeless family and cannot be cast to R8,
        // so it must stand alone rather than fold into R8_TYPELESS.
        assert_eq!(family(60), 60);
        for f in 61..=64 {
            assert_eq!(family(f), 60);
        }
        assert_eq!(family(65), 65);
        // Boundaries: 59 belongs to the preceding family, 66 is unmapped.
        assert_eq!(family(59), 53);
        assert_eq!(family(66), 66);
    }

    /// The ladder of descriptions a host texture is tried in.
    #[test]
    fn the_first_attempt_is_what_the_title_asked_for() {
        assert_eq!(
            attempts(5, 4),
            [
                (5, 4, true),
                (5, 0, true),
                (5, 4, false),
                (5, 0, false),
                (1, 0, true),
                (1, 0, false),
                (0, 0, false),
            ]
        );
        assert_eq!(attempts(3, 0x10)[0], (3, 0x10, true));
    }

    #[test]
    fn nothing_is_tried_twice() {
        // Flags with neither RT nor DS never keep a clear (VIC-BT-022).
        assert_eq!(attempts(0, 0), [(0, 0, false)]);
        for flags in 0..16 {
            for state in [0, 4, 0x10] {
                let tries = attempts(flags, state);
                for (i, a) in tries.iter().enumerate() {
                    assert!(!tries[i + 1..].contains(a), "{flags} {state}: {tries:?}");
                }
            }
        }
    }

    #[test]
    fn a_depth_target_keeps_its_depth_permission() {
        for flags in [2, 3, 6, 7, 10] {
            assert!(attempts(flags, 0x10).iter().all(|&(f, _, _)| f & 2 != 0));
        }
    }

    /// VIC-BT-022: PC CreateCommittedResource rejects a non-null optimized
    /// clear unless ALLOW_RENDER_TARGET or ALLOW_DEPTH_STENCIL is set.
    #[test]
    fn optimized_clear_needs_render_target_or_depth() {
        let clear = Some([28u32, 0, 0, 0, 0]); // DXGI_FORMAT_R8G8B8A8_UNORM
        assert!(optimized_clear(0, true, clear).is_none());
        assert!(optimized_clear(4, true, clear).is_none()); // UAV only
        assert!(optimized_clear(8, true, clear).is_none()); // deny shader
        assert!(optimized_clear(1, true, clear).is_some()); // RT
        assert!(optimized_clear(2, true, clear).is_some()); // depth
        assert!(optimized_clear(3, true, clear).is_some()); // RT|depth
        assert!(optimized_clear(5, true, clear).is_some()); // RT|UAV
        assert!(optimized_clear(1, false, clear).is_none());
        assert!(optimized_clear(1, true, None).is_none());
        // Ladder steps without RT/DS must not advertise a kept clear.
        for &(f, _, keep) in &attempts(4, 0) {
            assert!(
                !keep || f & 3 != 0,
                "UAV-only ladder kept clear at flags {f:#x}"
            );
            assert!(optimized_clear(f, keep, clear).is_none() || f & 3 != 0);
        }
        for &(f, _, keep) in &attempts(0, 4) {
            assert!(!keep);
            assert!(optimized_clear(f, keep, clear).is_none());
        }
    }

    #[test]
    fn permissions_only_go_down_the_ladder() {
        for flags in 0..16 {
            let tries = attempts(flags, 4);
            assert!(tries.iter().all(|&(f, _, _)| f & !flags == 0));
            let mut least = flags;
            for &(f, _, _) in &tries {
                assert!(f & !least == 0, "{flags}: {tries:?}");
                least = f;
            }
        }
    }

    #[test]
    fn the_state_and_clear_go_before_the_permissions() {
        let tries = attempts(5, 4);
        let first_fewer = tries.iter().position(|&(f, _, _)| f != 5).unwrap();
        assert!(tries[..first_fewer].iter().any(|&(_, s, _)| s == 0));
        assert!(tries[..first_fewer].iter().any(|&(_, _, clear)| !clear));
        assert!(tries[first_fewer..].iter().all(|&(_, s, _)| s == 0));
    }

    type Release = extern "system" fn(This) -> u32;
    type Map = extern "system" fn(This, u32, *const c_void, *mut *mut u8) -> HRESULT;
    type Address = extern "system" fn(This) -> u64;

    const PROPERTIES: HeapProperties = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 0,
        visible_nodes: 0,
    };

    fn buffer(width: u64) -> ResourceDesc {
        ResourceDesc {
            dimension: BUFFER,
            alignment: 0,
            width,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 0,
            sample_count: 1,
            sample_quality: 0,
            layout: 1,
            flags: 0,
        }
    }

    fn texture(width: u64, height: u32, format: u32, mips: u16) -> ResourceDesc {
        ResourceDesc {
            dimension: 3,
            width,
            height,
            format,
            mip_levels: mips,
            layout: 0,
            ..buffer(0)
        }
    }

    fn heap(size: u64) -> This {
        let desc = HeapDesc {
            size,
            properties: PROPERTIES,
            alignment: 0,
            flags: 0,
        };
        let mut heap = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &desc, &HEAP_IIDS[0], &mut heap),
            S_OK
        );
        heap
    }

    #[test]
    fn heaps_are_zeroed_aligned_memory() {
        let heap = heap(1 << 20);
        // SAFETY: a heap we just made.
        let memory = unsafe { &(*heap.cast::<Heap>()).memory };
        assert_eq!(memory.ptr.as_ptr() as u64 % KIB_64, 0);
        let mut back = std::mem::MaybeUninit::<HeapDesc>::uninit();
        type GetDesc = extern "system" fn(This, *mut HeapDesc) -> *mut HeapDesc;
        let get_desc = slot::<GetDesc>(heap, 0x40);
        assert_eq!(get_desc(heap, back.as_mut_ptr()), back.as_mut_ptr());
        // SAFETY: written above.
        assert_eq!(unsafe { back.assume_init() }.size, 1 << 20);
        assert!(get_desc(heap, std::ptr::null_mut()).is_null());
        assert_eq!(slot::<Release>(heap, 0x10)(heap), 0);

        let bad = HeapDesc {
            size: 1 << 20,
            properties: PROPERTIES,
            alignment: 3,
            flags: 0,
        };
        let mut out = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &bad, &HEAP_IIDS[0], &mut out),
            E_INVALIDARG
        );
    }

    #[test]
    fn device_setup_heap_properties_and_resources_at_addresses() {
        let device = crate::device::device();
        // GetCustomHeapProperties (0xd0): the struct comes back through the
        // hidden first argument, which is also returned.
        type Custom =
            extern "system" fn(*mut HeapProperties, This, u32, u32) -> *mut HeapProperties;
        let mut p = PROPERTIES;
        let back = slot::<Custom>(device, 0xd0)(&mut p, device, 1, 2);
        assert_eq!(back, &raw mut p);
        assert_eq!(
            (
                p.kind,
                p.cpu_pages,
                p.pool,
                p.creation_nodes,
                p.visible_nodes
            ),
            (4, 3, 1, 1, 1),
            "upload: custom, write-back, L0"
        );
        slot::<Custom>(device, 0xd0)(&mut p, device, 1, 1);
        assert_eq!(p.cpu_pages, 1, "default: not CPU-visible");

        // 0x428: a buffer over memory the title owns, as it asks.
        type At = extern "system" fn(
            This,
            u64,
            *const ResourceDesc,
            u32,
            *const c_void,
            *const Iid,
            *mut This,
        ) -> HRESULT;
        let memory = vec![0u8; 0x1_0000];
        let address = memory.as_ptr() as u64;
        let mut r = std::ptr::null();
        let create = slot::<At>(device, 0x428);
        let desc = buffer(0x1_0000);
        let null = std::ptr::null();
        assert_eq!(
            create(
                device,
                address,
                &desc,
                u32::MAX,
                null,
                &RESOURCE_IIDS[0],
                &mut r
            ),
            0
        );
        assert_eq!(slot::<Address>(r, 0x58)(r), address);
        assert_eq!(slot::<Release>(r, 0x10)(r), 0);
        // The title's memory is still there after the release.
        assert_eq!(memory[0xffff], 0);
        assert_ne!(
            create(device, 0, &desc, u32::MAX, null, &RESOURCE_IIDS[0], &mut r),
            0
        );
    }

    #[test]
    fn sizes_cover_every_mip() {
        let size = |d: &ResourceDesc| footprint(d).map(|i| i.size);
        assert_eq!(size(&buffer(256)), Some(KIB_64));
        // Textures take their console layout and its alignment (4 KiB
        // standard swizzle when the description leaves it to the library):
        // 512x512 BC7 with 10 mips is 0x56000 bytes, 4 KiB aligned, as on
        // the console.
        let layout = |d: &ResourceDesc| texture_layout(d).unwrap().size;
        let rgba = texture(256, 256, 28, 0);
        assert_eq!(size(&rgba), Some(layout(&rgba).next_multiple_of(4096)));
        let bc7 = footprint(&texture(512, 512, 98, 10));
        let console = AllocationInfo {
            size: 0x56000,
            alignment: 0x1000,
        };
        assert_eq!(bc7, Some(console));
        assert!(layout(&rgba) >= 4 * (256 * 256 * 4) / 3, "every mip");
        // BC1 is 8 bytes per 4x4 block: 1024x1024, one mip = 512 KiB.
        assert_eq!(size(&texture(1024, 1024, 71, 1)), Some(512 << 10));
        assert_eq!(size(&texture(16, 16, 66, 1)), None, "1-bit format");
        // Linear layout (1) and an explicit mode (64 KiB display, 0x10a).
        let linear = ResourceDesc {
            layout: 1,
            ..texture(100, 60, 28, 1)
        };
        assert_eq!(
            texture_layout(&linear).unwrap().planes[0]
                .layout
                .desc
                .swizzle
                .0,
            0
        );
        let display = ResourceDesc {
            layout: 0x10a,
            ..texture(100, 60, 28, 1)
        };
        assert_eq!(
            texture_layout(&display).unwrap().planes[0]
                .layout
                .desc
                .swizzle
                .0,
            10
        );

        let mut info = AllocationInfo {
            size: 0,
            alignment: 0,
        };
        let descs = [buffer(100), texture(256, 256, 28, 0)];
        allocation_info(&mut info, std::ptr::null(), 0, 2, descs.as_ptr());
        assert_eq!(
            info,
            AllocationInfo {
                size: KIB_64 + size(&descs[1]).unwrap(),
                alignment: KIB_64
            }
        );
        allocation_info(&mut info, std::ptr::null(), 0, 1, &texture(1, 1, 999, 1));
        assert_eq!(info.size, u64::MAX);
    }

    #[cfg(windows)]
    #[test]
    fn a_missing_resource_has_no_default_view_format() {
        assert_eq!(view_format(std::ptr::null()), None);
    }

    /// Null / UNKNOWN SRV and UAV formats for typeless host copies, including
    /// textures `host_texture` promotes from typed console formats (VIC-BT-004).
    #[test]
    fn null_view_formats_for_typeless_and_promoted_textures() {
        const R8G8B8A8_TYPELESS: u32 = 27;
        const R8G8B8A8_UNORM: u32 = 28;
        const R8G8B8A8_UNORM_SRGB: u32 = 29;
        const R32_TYPELESS: u32 = 39;
        const D32_FLOAT: u32 = 40;
        const R32_FLOAT: u32 = 41;
        const R24G8_TYPELESS: u32 = 44;
        const D24_UNORM_S8_UINT: u32 = 45;
        const R24_UNORM_X8_TYPELESS: u32 = 46;
        const R16_TYPELESS: u32 = 53;
        const D16_UNORM: u32 = 55;
        const R16_UNORM: u32 = 56;

        // Title-created typeless: inherit a family default.
        assert_eq!(
            null_view_format(R8G8B8A8_TYPELESS, None, false),
            Some(R8G8B8A8_UNORM)
        );
        assert_eq!(
            null_view_format(R8G8B8A8_TYPELESS, None, true),
            Some(R8G8B8A8_UNORM)
        );
        assert_eq!(null_view_format(R24G8_TYPELESS, None, true), None);

        // sRGB + UAV promotes to typeless: SRV keeps sRGB, UAV is linear.
        assert_eq!(
            null_view_format(R8G8B8A8_TYPELESS, Some(R8G8B8A8_UNORM_SRGB), false),
            Some(R8G8B8A8_UNORM_SRGB)
        );
        assert_eq!(
            null_view_format(R8G8B8A8_TYPELESS, Some(R8G8B8A8_UNORM_SRGB), true),
            Some(R8G8B8A8_UNORM)
        );

        // Depth without deny-shader promotes to typeless: SRV uses a plane format.
        assert_eq!(
            null_view_format(R32_TYPELESS, Some(D32_FLOAT), false),
            Some(R32_FLOAT)
        );
        assert_eq!(
            null_view_format(R24G8_TYPELESS, Some(D24_UNORM_S8_UINT), false),
            Some(R24_UNORM_X8_TYPELESS)
        );
        assert_eq!(
            null_view_format(R16_TYPELESS, Some(D16_UNORM), false),
            Some(R16_UNORM)
        );
        assert_eq!(
            null_view_format(R32_TYPELESS, Some(D32_FLOAT), true),
            None,
            "depth has no typed UAV"
        );
    }

    #[test]
    fn upload_footprints_follow_the_pc_rules() {
        let null = std::ptr::null();
        let mut layouts = [NO_FOOTPRINT; 2];
        let (mut rows, mut sizes, mut total) = ([0; 2], [0; 2], 0);
        let mut footprints = |desc: &ResourceDesc, first, count| {
            let (l, r, s) = (layouts.as_mut_ptr(), rows.as_mut_ptr(), sizes.as_mut_ptr());
            copyable_footprints(null, desc, first, count, 0, l, r, s, &mut total);
            (layouts, rows, sizes, total)
        };
        // 100x60 RGBA8, 2 mips: 400-byte rows pitched to 512.
        let (l, r, s, t) = footprints(&texture(100, 60, 28, 2), 0, 2);
        assert_eq!((l[0].row_pitch, r[0], s[0]), (512, 60, 400));
        // Mip 1 (50x30, 200-byte rows) starts after 59 pitched rows and one
        // packed row, rounded up to 512.
        assert_eq!(l[1].offset, (512 * 59 + 400u64).next_multiple_of(512));
        assert_eq!((l[1].width, l[1].height, r[1]), (50, 30, 30));
        assert_eq!(t, l[1].offset + 256 * 29 + 200);
        // BC1: rows of 4x4 blocks, 8 bytes each.
        let (_, r, s, _) = footprints(&texture(64, 64, 71, 1), 0, 1);
        assert_eq!((r[0], s[0]), (16, 128));
        let (_, _, _, t) = footprints(&texture(64, 64, 71, 1), 1, 1);
        assert_eq!(t, u64::MAX, "no second subresource");
    }

    #[test]
    fn copyable_footprints_match_native_packed_and_unaligned_cases() {
        // DXGI 68/69 store pairs of texels in one 32-bit word. An odd mip
        // width of five therefore occupies three words (six footprint texels).
        for format in [68, 69] {
            let desc = texture(10, 6, format, 2);
            let (mut layout, mut rows, mut sizes, mut total) =
                ([NO_FOOTPRINT], [0u32], [0u64], 0u64);
            copyable_footprints(
                std::ptr::null(),
                &desc,
                1,
                1,
                0,
                layout.as_mut_ptr(),
                rows.as_mut_ptr(),
                sizes.as_mut_ptr(),
                &mut total,
            );
            assert_eq!(
                (layout[0].width, layout[0].height),
                (6, 3),
                "format {format}"
            );
            assert_eq!((rows[0], sizes[0]), (3, 12), "format {format}");
            assert_eq!(layout[0].row_pitch, 256, "format {format}");
            assert_eq!(total, 524, "format {format}");
            let odd_width = PlacedFootprint {
                width: 5,
                ..layout[0]
            };
            assert_eq!(odd_width.layout(), Some((3, 12, 524)));
        }

        let desc = texture(10, 6, 28, 2);
        for base in [0, 1] {
            let (mut layouts, mut rows, mut sizes, mut total) =
                ([NO_FOOTPRINT; 2], [0u32; 2], [0u64; 2], 0u64);
            copyable_footprints(
                std::ptr::null(),
                &desc,
                0,
                2,
                base,
                layouts.as_mut_ptr(),
                rows.as_mut_ptr(),
                sizes.as_mut_ptr(),
                &mut total,
            );
            assert_eq!(layouts[0].offset, base);
            assert_eq!(layouts[1].offset, base + 1536);
            assert_eq!(total, 2068);
        }
    }

    #[test]
    fn planar_video_copy_footprints() {
        // Native D3D12 reports both Y slices first, then both interleaved UV slices.
        for (format, y, uv, bytes, pitch) in [(103, 60, 48, 514, 1024), (104, 53, 33, 1028, 1536)] {
            let desc = ResourceDesc {
                depth_or_array_size: 2,
                ..texture(514, 258, format, 1)
            };
            let mut layouts = [NO_FOOTPRINT; 4];
            let (mut rows, mut sizes, mut total) = ([0; 4], [0; 4], 0);
            copyable_footprints(
                std::ptr::null(),
                &desc,
                0,
                4,
                512,
                layouts.as_mut_ptr(),
                rows.as_mut_ptr(),
                sizes.as_mut_ptr(),
                &mut total,
            );
            assert_eq!(
                layouts.map(|l| (l.format, l.width, l.height)),
                [(y, 514, 258), (y, 514, 258), (uv, 257, 129), (uv, 257, 129)]
            );
            assert_eq!(layouts.map(|l| l.row_pitch), [pitch; 4]);
            assert_eq!(rows, [258, 258, 129, 129]);
            assert_eq!(sizes, [bytes; 4]);
            let stride = u64::from(pitch);
            assert_eq!(
                layouts.map(|l| l.offset),
                [0, 258, 516, 645].map(|r| 512 + r * stride)
            );
            assert_eq!(total, 773 * stride + bytes);
            // Starting at UV describes only the requested range in the upload buffer.
            copyable_footprints(
                std::ptr::null(),
                &desc,
                2,
                1,
                512,
                layouts.as_mut_ptr(),
                rows.as_mut_ptr(),
                sizes.as_mut_ptr(),
                &mut total,
            );
            assert_eq!(
                (layouts[0].offset, layouts[0].format, rows[0], sizes[0]),
                (512, uv, 129, bytes)
            );
            assert_eq!(total, 128 * stride + bytes);
            assert!(subresource(&desc, 4).is_none());

            for invalid in [
                ResourceDesc {
                    dimension: TEXTURE_3D,
                    ..desc
                },
                ResourceDesc {
                    mip_levels: 0,
                    ..desc
                },
                ResourceDesc {
                    mip_levels: 2,
                    ..desc
                },
                ResourceDesc {
                    sample_count: 4,
                    ..desc
                },
                ResourceDesc { width: 513, ..desc },
                ResourceDesc {
                    height: 257,
                    ..desc
                },
                ResourceDesc {
                    depth_or_array_size: 2049,
                    ..desc
                },
            ] {
                assert!(subresource(&invalid, 0).is_none());
            }
        }
    }

    #[test]
    fn depth_stencil_footprints_follow_the_pc_plane_order() {
        // Native D3D12 reports the same layout for both depth/stencil families
        // and their view formats. Mips vary fastest, then array slices, then planes.
        let expected = [
            (512, 39, 16, 8),
            (4_608, 39, 8, 4),
            (6_656, 39, 16, 8),
            (10_752, 39, 8, 4),
            (12_800, 60, 16, 8),
            (16_896, 60, 8, 4),
            (18_944, 60, 16, 8),
            (23_040, 60, 8, 4),
        ];
        for format in (19..=22).chain(44..=47) {
            let desc = ResourceDesc {
                depth_or_array_size: 2,
                ..texture(16, 8, format, 2)
            };
            let mut layouts = [NO_FOOTPRINT; 8];
            let (mut rows, mut sizes, mut total) = ([0; 8], [0; 8], 0);
            copyable_footprints(
                std::ptr::null(),
                &desc,
                0,
                8,
                512,
                layouts.as_mut_ptr(),
                rows.as_mut_ptr(),
                sizes.as_mut_ptr(),
                &mut total,
            );

            assert_eq!(
                layouts.map(|l| (l.offset, l.format, l.width, l.height)),
                expected,
                "format {format}"
            );
            assert!(
                layouts.iter().all(|l| l.depth == 1 && l.row_pitch == 512),
                "format {format}"
            );
            assert_eq!(rows, [8, 4, 8, 4, 8, 4, 8, 4], "format {format}");
            assert_eq!(sizes, [64, 32, 64, 32, 16, 8, 16, 8], "format {format}");
            assert_eq!(total, 24_072, "format {format}");
            assert!(subresource(&desc, 8).is_none(), "format {format}");
        }
    }

    /// The title may view a texture as sRGB or linear whichever format it
    /// made it in, as the console allows: a shader reads 0x80 as 0.216
    /// through the sRGB view, 0.502 through the linear one, and in the
    /// texture's own format through a view without a description.
    #[cfg(windows)]
    #[test]
    fn srgb_and_linear_views_of_one_texture_read_their_own_curve() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        use windows::core::s;
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=3)), UAV(u0)"
Texture2D<float4> as_srgb : register(t0);
Texture2D<float4> as_linear : register(t1);
Texture2D<float4> as_own : register(t2);
RWByteAddressBuffer reads : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    int3 at = int3(1, 2, 0);
    float3 r = float3(as_srgb.Load(at).r, as_linear.Load(at).r, as_own.Load(at).r);
    reads.Store3(0, asuint(r));
}
"#;
        let mut blob: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; the out
        // pointer is ours.
        unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("views"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut blob,
                None,
            )
            .unwrap()
        };
        let blob = blob.unwrap();
        // SAFETY: the blob's own bytes.
        let code = unsafe {
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
        };
        // SAFETY: plain object creation; the root signature is inside the
        // shader.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, code).unwrap();
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            };
            let pso = host.device.CreateComputePipelineState(&desc).unwrap();
            (root, pso)
        };
        let host_buffer = |kind: D3D12_HEAP_TYPE, flags, state| {
            let mut b: Option<ID3D12Resource> = None;
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let heap = D3D12_HEAP_PROPERTIES {
                Type: kind,
                ..Default::default()
            };
            // SAFETY: plain creation of a buffer.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        state,
                        None,
                        &mut b,
                    )
                    .unwrap()
            };
            b.unwrap()
        };
        // The texture made in `own`, its memory filled with `block`: what
        // the shader reads through the title's views in `views` (sRGB,
        // linear) and through one without a description.
        let read = |own: u32, views: [u32; 2], block: &[u8]| -> [f32; 3] {
            let null = std::ptr::null();
            let mut image = null;
            let desc = texture(4, 4, own, 1);
            let iid = &RESOURCE_IIDS[0];
            assert_eq!(
                create_committed(null, &PROPERTIES, 0, &desc, 0, null.cast(), iid, &mut image),
                S_OK
            );
            // Every element alike, so the layout does not matter.
            let size = footprint(&desc).unwrap().size as usize;
            // SAFETY: the texture's memory, `size` bytes.
            let memory = unsafe { std::slice::from_raw_parts_mut(resource(image).data(), size) };
            for (i, b) in memory.iter_mut().enumerate() {
                *b = block[i % block.len()];
            }
            let mut slots = [0u64; 3 * 4];
            let base = slots.as_mut_ptr() as u64;
            let at = |i: u32| base + 32 * u64::from(i);
            // 2D, the default component mapping, one mip.
            let mapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            let srv = |format| [format, 4, mapping, 0, 0, 1, 0, 0, 0, 0];
            crate::descriptor::create_srv(null, image, &srv(views[0]), at(0));
            crate::descriptor::create_srv(null, image, &srv(views[1]), at(1));
            crate::descriptor::create_srv(null, image, null.cast(), at(2));
            let heap = crate::descriptor::host::HostHeap::new(false, 3).unwrap();
            for i in 0..3 {
                assert!(heap.mirror(i, at(i), true).is_some());
            }
            let out = host_buffer(
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            );
            let back = host_buffer(
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            let list = host.new_list().unwrap();
            crate::texture::upload(&list, image, host_resource(image).unwrap());
            const NON_PIXEL_SHADER_RESOURCE: u32 = 0x40;
            crate::list::ensure_state(&list, image, NON_PIXEL_SHADER_RESOURCE);
            // SAFETY: an open list; our pipeline, heap and live resources.
            unsafe {
                let l = &list.list;
                l.SetComputeRootSignature(&root);
                l.SetPipelineState(&pso);
                l.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
                l.SetComputeRootDescriptorTable(0, heap.heap.GetGPUDescriptorHandleForHeapStart());
                l.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
                l.Dispatch(1, 1, 1);
                l.ResourceBarrier(&[crate::host::transition(
                    &out,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                l.CopyBufferRegion(&back, 0, &out, 0, 12);
            }
            list.close();
            host.execute(&[&list]);
            let mut mapped = std::ptr::null_mut();
            // SAFETY: the copy finished (execute waits); it wrote 12 bytes.
            let read = unsafe {
                let range = D3D12_RANGE { Begin: 0, End: 12 };
                back.Map(0, Some(&range), Some(&mut mapped)).unwrap();
                mapped.cast::<[f32; 3]>().read_unaligned()
            };
            assert_eq!(slot::<Release>(image, 0x10)(image), 0);
            read
        };
        let to_linear = |c: f32| ((c + 0.055) / 1.055).powf(2.4);
        // 0x80 in every channel of 8-bit color; a BC1 block of one color,
        // red 16 of 31 (132 of 255 when expanded to 8 bits first).
        const RGBA: &[u8] = &[0x80];
        const BC1: &[u8] = &[0x00, 0x80, 0x00, 0x80, 0, 0, 0, 0];
        let cases = [
            (28, 29, 128.0 / 255.0, 0.002f32, RGBA),
            (87, 91, 128.0 / 255.0, 0.002, RGBA),
            (71, 72, 16.0 / 31.0, 0.01, BC1),
        ];
        let mut wrong = Vec::new();
        for (linear, srgb, unorm, near, block) in cases {
            for own in [linear, srgb] {
                let curve = to_linear(unorm);
                let wanted = [curve, unorm, if own == srgb { curve } else { unorm }];
                let got = read(own, [srgb, linear], block);
                if got.iter().zip(wanted).any(|(g, w)| (g - w).abs() > near) {
                    wrong.push(format!(
                        "texture {own}, views {srgb}/{linear}/none: read {got:?}, wanted {wanted:?}"
                    ));
                }
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    /// A 4x4 texture filled from a title buffer through our list and queue,
    /// then read back from the host GPU.
    ///
    /// Exercises buffer→texture staging via `defer_fill` (host-owned mapped
    /// upload). Under Wine+vkd3d that path must not use `defer_copy`, or
    /// `VirtualQuery` reporting Map memory as MEM_FREE skips the upload and
    /// this reads zeros (#518).
    #[cfg(windows)]
    #[test]
    fn buffer_data_reaches_the_host_texture() {
        use windows::Win32::Graphics::Direct3D12::*;
        let Some(host) = crate::host::host() else {
            eprintln!("Skipping texture readback: no usable host D3D12 device");
            return;
        };
        let null = std::ptr::null();
        let iid = &RESOURCE_IIDS[0];
        const COPY_DEST: u32 = 0x400;
        let (mut image, mut source) = (null, null);
        let rgba = texture(4, 4, 28, 1);
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &rgba,
                COPY_DEST,
                null.cast(),
                iid,
                &mut image
            ),
            S_OK
        );
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &buffer(1024),
                0,
                null.cast(),
                iid,
                &mut source
            ),
            S_OK
        );
        let (data, _) = buffer_bytes(source).unwrap();
        // SAFETY: the buffer holds 1024 bytes.
        let bytes = unsafe { std::slice::from_raw_parts_mut(data, 1024) };
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }

        let queue_iid = guid(
            0x0ec870a6,
            0x5d7e,
            0x4c22,
            [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
        );
        let allocator_iid = guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        let (mut queue, mut allocator, mut list) = (null, null, null);
        let queue_desc = [0u32; 4];
        assert_eq!(
            crate::queue::create(null, queue_desc.as_ptr().cast(), &queue_iid, &mut queue),
            S_OK
        );
        assert_eq!(
            crate::list::create_allocator(null, 0, &allocator_iid, &mut allocator),
            S_OK
        );
        assert_eq!(
            crate::list::create_list(null, 0, 0, allocator, null, &list_iid, &mut list),
            S_OK
        );

        #[repr(C)]
        struct Location {
            resource: This,
            kind: u32,
            footprint: PlacedFootprint,
        }
        let rows = PlacedFootprint {
            offset: 0,
            format: 28,
            width: 4,
            height: 4,
            depth: 1,
            row_pitch: 256,
        };
        let to = Location {
            resource: image,
            kind: 0,
            footprint: PlacedFootprint { offset: 0, ..rows },
        };
        let from = Location {
            resource: source,
            kind: 1,
            footprint: rows,
        };
        type Copy =
            extern "system" fn(This, *const Location, u32, u32, u32, *const Location, *const u8);
        slot::<Copy>(list, 0x80)(list, &to, 0, 0, 0, &from, null.cast());
        slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list);
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);

        // Read the host texture back into a readback buffer.
        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 1024,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();
        let host_image = host_resource(image).unwrap();
        let back = host.new_list().unwrap();
        let (_, before) = host_transition(image, 0x800).unwrap();
        // The upload list holds the texture until it is reset.
        assert!(std::sync::Arc::strong_count(&host_copy(image).unwrap()) > 2);
        use windows::core::Interface;
        let out_to = Location {
            resource: readback.as_raw().cast_const().cast(),
            kind: 1,
            footprint: rows,
        };
        let out_from = Location {
            resource: host_image.as_raw().cast_const().cast(),
            kind: 0,
            footprint: to.footprint,
        };
        // SAFETY: an open list, live resources, locations in the host layout.
        unsafe {
            back.list.ResourceBarrier(&[crate::host::transition(
                host_image,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            back.list.CopyTextureRegion(
                (&raw const out_to).cast(),
                0,
                0,
                0,
                (&raw const out_from).cast(),
                None,
            );
        }
        back.close();
        host.execute(&[&back]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy finished (execute waits); the buffer is 1024 bytes.
        let read = unsafe {
            readback
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: 1024,
                    }),
                    Some(&mut mapped),
                )
                .unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), 1024)
        };
        for row in 0..4 {
            let at = row * 256;
            assert_eq!(read[at..at + 16], bytes[at..at + 16], "row {row}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn copy_texture_region_reads_back_into_title_buffer() {
        if crate::host::host().is_none() {
            return;
        }
        let null = std::ptr::null();
        let iid = &RESOURCE_IIDS[0];
        let tex_desc = ResourceDesc {
            dimension: 3, // Texture2D
            alignment: 0,
            width: 4,
            height: 4,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 28, // R8G8B8A8_UNORM
            sample_count: 1,
            sample_quality: 0,
            layout: 0,
            flags: 0,
        };
        let (mut image, mut upload_buf, mut dst_aligned, mut dst_packed, mut dst_crop) =
            (null, null, null, null, null);
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &tex_desc,
                0,
                null.cast(),
                iid,
                &mut image
            ),
            S_OK
        );
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &buffer(1024),
                0,
                null.cast(),
                iid,
                &mut upload_buf
            ),
            S_OK
        );
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &buffer(2048),
                0,
                null.cast(),
                iid,
                &mut dst_aligned
            ),
            S_OK
        );
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &buffer(256),
                0,
                null.cast(),
                iid,
                &mut dst_packed
            ),
            S_OK
        );
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &buffer(1024),
                0,
                null.cast(),
                iid,
                &mut dst_crop
            ),
            S_OK
        );

        // Fill a 4x4 RGBA8 pattern (row_pitch 256): pixel (x, y) = [x*40+1, y*50+2, (x+y)*20+3, 0xff].
        let (up_ptr, _) = buffer_bytes(upload_buf).unwrap();
        // SAFETY: upload_buf holds 1024 bytes.
        let up = unsafe { std::slice::from_raw_parts_mut(up_ptr, 1024) };
        for y in 0..4usize {
            for x in 0..4usize {
                let at = y * 256 + x * 4;
                up[at..at + 4].copy_from_slice(&[
                    (x * 40 + 1) as u8,
                    (y * 50 + 2) as u8,
                    ((x + y) * 20 + 3) as u8,
                    0xff,
                ]);
            }
        }

        let queue_iid = guid(
            0x0ec870a6,
            0x5d7e,
            0x4c22,
            [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
        );
        let allocator_iid = guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        let (mut queue, mut allocator, mut list) = (null, null, null);
        let queue_desc = [0u32; 4];
        assert_eq!(
            crate::queue::create(null, queue_desc.as_ptr().cast(), &queue_iid, &mut queue),
            S_OK
        );
        assert_eq!(
            crate::list::create_allocator(null, 0, &allocator_iid, &mut allocator),
            S_OK
        );
        assert_eq!(
            crate::list::create_list(null, 0, 0, allocator, null, &list_iid, &mut list),
            S_OK
        );

        #[repr(C)]
        struct Location {
            resource: This,
            kind: u32,
            footprint: PlacedFootprint,
        }
        type Copy = extern "system" fn(
            This,
            *const Location,
            u32,
            u32,
            u32,
            *const Location,
            *const [u32; 6],
        );
        let copy = slot::<Copy>(list, 0x80);

        let full_rows = PlacedFootprint {
            offset: 0,
            format: 28,
            width: 4,
            height: 4,
            depth: 1,
            row_pitch: 256,
        };
        let tex_loc = Location {
            resource: image,
            kind: 0,
            footprint: PlacedFootprint {
                offset: 0,
                ..full_rows
            },
        };
        let up_loc = Location {
            resource: upload_buf,
            kind: 1,
            footprint: full_rows,
        };
        // 1) Buffer -> Texture upload.
        copy(list, &tex_loc, 0, 0, 0, &up_loc, std::ptr::null());

        // 2) Texture -> Buffer readback at non-zero offset (256), 256-aligned row pitch.
        let aligned_loc = Location {
            resource: dst_aligned,
            kind: 1,
            footprint: PlacedFootprint {
                offset: 256,
                ..full_rows
            },
        };
        copy(list, &aligned_loc, 0, 0, 0, &tex_loc, std::ptr::null());

        // 3) Texture -> Buffer readback with unaligned row_pitch (16 bytes, packed).
        let packed_loc = Location {
            resource: dst_packed,
            kind: 1,
            footprint: PlacedFootprint {
                offset: 32,
                format: 28,
                width: 4,
                height: 4,
                depth: 1,
                row_pitch: 16,
            },
        };
        copy(list, &packed_loc, 0, 0, 0, &tex_loc, std::ptr::null());

        // 4) Texture -> Buffer sub-box readback: 2x2 crop from (1, 1) to (3, 3).
        let crop_box = [1u32, 1, 0, 3, 3, 1];
        let crop_loc = Location {
            resource: dst_crop,
            kind: 1,
            footprint: PlacedFootprint {
                offset: 0,
                format: 28,
                width: 2,
                height: 2,
                depth: 1,
                row_pitch: 256,
            },
        };
        copy(list, &crop_loc, 0, 0, 0, &tex_loc, &crop_box);

        // 5) Console formats 118..=120 (3 B/texel) map to 4-byte PC formats
        // (45..=47): skipped cleanly without breaking Close() or touching dst.
        let mut dst_d16s8 = null;
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &buffer(1024),
                0,
                null.cast(),
                iid,
                &mut dst_d16s8
            ),
            S_OK
        );
        let d16s8_loc = Location {
            resource: dst_d16s8,
            kind: 1,
            footprint: PlacedFootprint {
                offset: 0,
                format: 118,
                width: 4,
                height: 4,
                depth: 1,
                row_pitch: 256,
            },
        };
        copy(list, &d16s8_loc, 0, 0, 0, &tex_loc, std::ptr::null());

        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);

        // Verify 256-aligned readback at offset 256 (leading 256 bytes untouched).
        let (al_ptr, _) = buffer_bytes(dst_aligned).unwrap();
        // SAFETY: dst_aligned holds 2048 bytes.
        let al = unsafe { std::slice::from_raw_parts(al_ptr, 2048) };
        assert!(
            al[..256].iter().all(|&b| b == 0),
            "bytes before offset touched"
        );
        for row in 0..4 {
            let at = 256 + row * 256;
            assert_eq!(
                al[at..at + 16],
                up[row * 256..row * 256 + 16],
                "aligned row {row}"
            );
        }

        // Verify unaligned packed readback (row_pitch 16, offset 32).
        let (pk_ptr, _) = buffer_bytes(dst_packed).unwrap();
        // SAFETY: dst_packed holds 256 bytes.
        let pk = unsafe { std::slice::from_raw_parts(pk_ptr, 256) };
        assert!(
            pk[..32].iter().all(|&b| b == 0),
            "bytes before packed offset touched"
        );
        for row in 0..4 {
            let at = 32 + row * 16;
            assert_eq!(
                pk[at..at + 16],
                up[row * 256..row * 256 + 16],
                "packed row {row}"
            );
        }

        // Verify 2x2 sub-box crop from (1, 1)..(3, 3).
        let (cr_ptr, _) = buffer_bytes(dst_crop).unwrap();
        // SAFETY: dst_crop holds 1024 bytes.
        let cr = unsafe { std::slice::from_raw_parts(cr_ptr, 1024) };
        for row in 0..2 {
            let src_at = (row + 1) * 256 + 4; // x=1, y=row+1
            let dst_at = row * 256;
            assert_eq!(
                cr[dst_at..dst_at + 8],
                up[src_at..src_at + 8],
                "crop row {row}"
            );
        }

        // Verify skipped 118..=120 readback left its buffer untouched.
        let (d16_ptr, _) = buffer_bytes(dst_d16s8).unwrap();
        // SAFETY: dst_d16s8 holds 1024 bytes.
        let d16 = unsafe { std::slice::from_raw_parts(d16_ptr, 1024) };
        assert!(
            d16.iter().all(|&b| b == 0),
            "mismatched format 118 buffer touched"
        );
    }

    #[test]
    fn committed_and_placed_resources_map_at_their_gpu_address() {
        let null = std::ptr::null();
        let mut committed = null;
        let hr = create_committed(
            null,
            &PROPERTIES,
            0,
            &buffer(256),
            0,
            null.cast(),
            &RESOURCE_IIDS[0],
            &mut committed,
        );
        assert_eq!(hr, S_OK);
        let mut back = std::mem::MaybeUninit::<ResourceDesc>::uninit();
        type GetDesc = extern "system" fn(This, *mut ResourceDesc) -> *mut ResourceDesc;
        let get_desc = slot::<GetDesc>(committed, 0x50);
        assert_eq!(get_desc(committed, back.as_mut_ptr()), back.as_mut_ptr());
        // SAFETY: written above.
        let read = unsafe { back.assume_init() };
        assert_eq!((read.dimension, read.width), (BUFFER, 256));
        assert!(get_desc(committed, std::ptr::null_mut()).is_null());
        let mut data = std::ptr::null_mut();
        assert_eq!(
            slot::<Map>(committed, 0x40)(committed, 0, null.cast(), &mut data),
            S_OK
        );
        assert_eq!(slot::<Address>(committed, 0x58)(committed), data as u64);
        // SAFETY: the buffer is at least 256 zeroed bytes.
        assert_eq!(unsafe { *data.add(255) }, 0);
        assert_eq!(slot::<Release>(committed, 0x10)(committed), 0);

        let heap = heap(4 * KIB_64);
        let place = |offset, desc: &ResourceDesc, out: &mut This| {
            create_placed(
                null,
                heap,
                offset,
                desc,
                0,
                null.cast(),
                &RESOURCE_IIDS[0],
                out,
            )
        };
        let mut placed = null;
        assert_eq!(place(KIB_64, &buffer(KIB_64), &mut placed), S_OK);
        // SAFETY: a heap we made.
        let base = unsafe { (*heap.cast::<Heap>()).memory.ptr.as_ptr() } as u64;
        assert_eq!(slot::<Address>(placed, 0x58)(placed), base + KIB_64);
        let mut other = null;
        assert_eq!(
            place(4 * KIB_64, &buffer(1), &mut other),
            E_INVALIDARG,
            "past the end"
        );
        assert_eq!(
            place(100, &buffer(1), &mut other),
            E_INVALIDARG,
            "misaligned"
        );
        // The resource keeps the heap alive.
        assert_eq!(slot::<Release>(heap, 0x10)(heap), 1);
        assert_eq!(slot::<Release>(placed, 0x10)(placed), 0);
    }

    #[test]
    fn r9g9b9e5_target_format_substitutes_rgba16f() {
        assert_eq!(pc_format(67), 67);
        assert_eq!(target_format(67), 10);
        assert_eq!(target_format(28), 28);
        assert_eq!(target_format(118), 45);
    }

    #[cfg(windows)]
    #[test]
    fn r9g9b9e5_render_target_and_uav_textures_create_on_host() {
        if crate::host::host().is_none() {
            return;
        }
        let tex_desc = |flags: u32| ResourceDesc {
            dimension: 3, // TEXTURE2D
            alignment: 0,
            width: 64,
            height: 64,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 67, // DXGI_FORMAT_R9G9B9E5_SHAREDEXP
            sample_count: 1,
            sample_quality: 0,
            layout: 0,
            flags,
        };
        for (flags, expected_host_format) in [(0, 67), (4, 10), (5, 10)] {
            let d = tex_desc(flags);
            let mut out = std::ptr::null();
            let hr = create_committed(
                std::ptr::null(),
                &PROPERTIES,
                0,
                &d,
                0,
                std::ptr::null(),
                &RESOURCE_IIDS[0],
                &mut out,
            );
            assert_eq!(hr, S_OK, "flags {flags}");
            let host = host_resource(out).expect("host texture created");
            // SAFETY: a live host texture.
            let got = unsafe { host.GetDesc() }.Format.0 as u32;
            assert_eq!(got, expected_host_format, "flags {flags}");
            assert_eq!(view_format(out), None, "flags {flags}");
            assert_eq!(slot::<Release>(out, 0x10)(out), 0);
        }
    }

    /// Tests that read or raise the process-wide pressure run one at a time
    /// and leave it as they found it.
    static PRESSURE_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_pressure<R>(level: u32, test: impl FnOnce() -> R) -> R {
        use std::sync::atomic::Ordering::Relaxed;
        let _one_at_a_time = PRESSURE_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        PRESSURE.store(level, Relaxed);
        let result = test();
        PRESSURE.store(0, Relaxed);
        result
    }

    #[test]
    fn only_plain_sampled_textures_lose_mips() {
        const RGBA8: u32 = 28;
        const BC1: u32 = 71;
        let plain = texture(2048, 2048, BC1, 12);
        assert_eq!(droppable_mips(&plain, 2), 2);
        assert_eq!(droppable_mips(&plain, 0), 0);
        let target = ResourceDesc { flags: 1, ..plain };
        let depth = ResourceDesc { flags: 2, ..plain };
        let unordered = ResourceDesc { flags: 4, ..plain };
        let multisampled = ResourceDesc {
            sample_count: 4,
            ..plain
        };
        for kept in [target, depth, unordered, multisampled, buffer(1 << 20)] {
            assert_eq!(droppable_mips(&kept, 3), 0, "{kept:?}");
        }
        // Never the last level.
        assert_eq!(droppable_mips(&texture(64, 64, RGBA8, 2), 3), 1);
        assert_eq!(droppable_mips(&texture(64, 64, RGBA8, 1), 3), 0);
        // A full chain (0) counts its levels.
        assert_eq!(droppable_mips(&texture(64, 64, RGBA8, 0), 3), 3);
        // Block-compressed textures keep whole blocks: 16 >> 2 is one block.
        assert_eq!(droppable_mips(&texture(16, 16, BC1, 5), 3), 2);
        assert_eq!(droppable_mips(&texture(12, 12, BC1, 4), 3), 0);
        assert_eq!(droppable_mips(&texture(12, 12, RGBA8, 4), 3), 3);
        // 36×36 BC1 cannot drop one or two levels, but three yields 4×4.
        assert_eq!(droppable_mips(&texture(36, 36, BC1, 6), 1), 0);
        assert_eq!(droppable_mips(&texture(36, 36, BC1, 6), 2), 0);
        assert_eq!(droppable_mips(&texture(36, 36, BC1, 6), 3), 3);
        // Console-only formats with no host element size stay as they are.
        assert_eq!(droppable_mips(&texture(64, 64, 0, 7), 3), 0);
    }

    #[test]
    fn out_of_memory_skips_invalid_block_drops_to_the_next_valid() {
        with_pressure(0, || {
            let mut attempts = Vec::new();
            // 36 → 18 → 9 are not whole BC blocks; 4×4 (drop 3) is.
            let (made, d, dropped) = create_texture_with_auto_mip_drop(
                &texture(36, 36, 71, 6),
                refuse_above(8, &mut attempts),
            )
            .unwrap();
            assert_eq!((made, dropped), (4, 3));
            assert_eq!((d.width, d.height, d.mip_levels), (4, 4, 3));
            let widths: Vec<_> = attempts.iter().map(|a| a.width).collect();
            assert_eq!(widths, [36, 4]);
        });
    }

    #[test]
    fn dropped_description_is_the_title_texture_without_its_top_levels() {
        let d = without_top_mips(&texture(2048, 1024, 71, 12), 3);
        assert_eq!((d.width, d.height, d.mip_levels), (256, 128, 9));
        let full = without_top_mips(&texture(64, 8, 28, 0), 2);
        assert_eq!((full.width, full.height, full.mip_levels), (16, 2, 5));
        let same = without_top_mips(&texture(64, 8, 28, 4), 0);
        assert_eq!((same.width, same.height, same.mip_levels), (64, 8, 4));
    }

    #[test]
    fn views_of_a_dropped_texture_keep_naming_the_same_levels() {
        assert_eq!(rebase_mip_range(0, u32::MAX, 2), (0, u32::MAX));
        assert_eq!(rebase_mip_range(3, u32::MAX, 2), (1, u32::MAX));
        assert_eq!(rebase_mip_range(2, 5, 2), (0, 5));
        assert_eq!(rebase_mip_range(3, 5, 2), (1, 5));
        // Levels 0 and 1 only: nothing of them is left, the largest stands in.
        assert_eq!(rebase_mip_range(0, 2, 2), (0, 1));
        // Level 1 to 4 loses level 1.
        assert_eq!(rebase_mip_range(1, 4, 2), (0, 3));
        assert_eq!(rebase_mip_range(0, 4, 0), (0, 4));
    }

    /// A synthetic host that refuses every texture wider than `limit`.
    fn refuse_above(
        limit: u64,
        attempts: &mut Vec<ResourceDesc>,
    ) -> impl FnMut(&ResourceDesc) -> Result<u64, HRESULT> + '_ {
        move |d| {
            attempts.push(*d);
            if d.width > limit {
                Err(E_OUTOFMEMORY)
            } else {
                Ok(d.width)
            }
        }
    }

    #[test]
    fn out_of_memory_drops_mips_and_raises_pressure_for_later_textures() {
        use std::sync::atomic::Ordering::Relaxed;
        with_pressure(0, || {
            let big = texture(2048, 2048, 71, 12);
            let mut attempts = Vec::new();
            let (made, d, dropped) =
                create_texture_with_auto_mip_drop(&big, refuse_above(512, &mut attempts)).unwrap();
            // 2048 and 1024 are refused; 512 (two levels down) fits.
            assert_eq!((made, dropped), (512, 2));
            assert_eq!((d.width, d.height, d.mip_levels), (512, 512, 10));
            let widths: Vec<_> = attempts.iter().map(|a| a.width).collect();
            assert_eq!(widths, [2048, 1024, 512]);
            assert_eq!(PRESSURE.load(Relaxed), 2);
            // The next texture starts at the raised level, so it is not
            // refused first.
            let mut attempts = Vec::new();
            let (_, d, dropped) = create_texture_with_auto_mip_drop(
                &texture(1024, 1024, 98, 11),
                refuse_above(512, &mut attempts),
            )
            .unwrap();
            assert_eq!((dropped, d.width, attempts.len()), (2, 256, 1));
            // A texture too small for that level loses what it can.
            let (_, d, dropped) = create_texture_with_auto_mip_drop(
                &texture(4, 4, 28, 2),
                refuse_above(u64::MAX, &mut Vec::new()),
            )
            .unwrap();
            assert_eq!((dropped, d.width, d.mip_levels), (1, 2, 1));
        });
    }

    #[test]
    fn auto_mip_drop_gives_up_after_three_levels() {
        use std::sync::atomic::Ordering::Relaxed;
        with_pressure(0, || {
            let mut attempts = Vec::new();
            let refused = create_texture_with_auto_mip_drop(
                &texture(2048, 2048, 71, 12),
                refuse_above(0, &mut attempts),
            );
            assert_eq!(refused.err(), Some(E_OUTOFMEMORY));
            assert_eq!(attempts.len(), 4, "the full size and three drops");
            assert_eq!(PRESSURE.load(Relaxed), MAX_MIP_DROP);
        });
    }

    #[test]
    fn render_targets_and_other_errors_are_not_retried() {
        use std::sync::atomic::Ordering::Relaxed;
        with_pressure(0, || {
            let target = ResourceDesc {
                flags: 1,
                ..texture(2048, 2048, 28, 12)
            };
            let mut attempts = Vec::new();
            let refused =
                create_texture_with_auto_mip_drop(&target, refuse_above(0, &mut attempts));
            assert_eq!(refused.err(), Some(E_OUTOFMEMORY));
            assert_eq!(attempts.len(), 1);
            assert_eq!(
                PRESSURE.load(Relaxed),
                0,
                "no pressure from a texture that cannot lose"
            );
            let mut calls = 0;
            let refused = create_texture_with_auto_mip_drop(
                &texture(2048, 2048, 28, 12),
                |_: &ResourceDesc| -> Result<(), HRESULT> {
                    calls += 1;
                    Err(E_INVALIDARG)
                },
            );
            assert_eq!(refused.err(), Some(E_INVALIDARG));
            assert_eq!(calls, 1);
            assert_eq!(PRESSURE.load(Relaxed), 0);
        });
    }

    #[test]
    fn without_pressure_a_texture_is_made_as_the_title_describes_it() {
        with_pressure(0, || {
            let big = texture(2048, 2048, 71, 12);
            let (_, d, dropped) =
                create_texture_with_auto_mip_drop(&big, refuse_above(u64::MAX, &mut Vec::new()))
                    .unwrap();
            assert_eq!(dropped, 0);
            assert_eq!((d.width, d.height, d.mip_levels), (2048, 2048, 12));
        });
    }

    #[cfg(windows)]
    #[test]
    fn host_texture_out_of_memory_drops_mips_instead_of_failing() {
        use std::sync::atomic::Ordering::Relaxed;
        if crate::host::host().is_none() {
            return;
        }
        with_pressure(0, || {
            let _oom = SyntheticOom::above(16);
            let make = |desc: &ResourceDesc| {
                let mut out = std::ptr::null();
                let hr = create_committed(
                    std::ptr::null(),
                    &PROPERTIES,
                    0,
                    desc,
                    0,
                    std::ptr::null(),
                    &RESOURCE_IIDS[0],
                    &mut out,
                );
                assert_eq!(hr, S_OK);
                out
            };
            let image = make(&texture(64, 64, 28, 7));
            let host = host_resource(image).expect("a host texture despite the refusals");
            // SAFETY: a live host texture.
            let d = unsafe { host.GetDesc() };
            assert_eq!((d.Width, d.Height, d.MipLevels), (16, 16, 5));
            assert_eq!(dropped_mips(image), 2);
            assert_eq!(PRESSURE.load(Relaxed), 2);
            // A render target keeps its size: it has no host copy to drop.
            let target = make(&ResourceDesc {
                flags: 1,
                ..texture(8, 8, 28, 4)
            });
            let host = host_resource(target).expect("a small target fits");
            // SAFETY: a live host texture.
            let d = unsafe { host.GetDesc() };
            assert_eq!((d.Width, d.MipLevels), (8, 4));
            assert_eq!(dropped_mips(target), 0);
            assert_eq!(slot::<Release>(image, 0x10)(image), 0);
            assert_eq!(slot::<Release>(target, 0x10)(target), 0);
        });
    }

    #[cfg(windows)]
    #[test]
    fn placed_views_keep_all_mips_under_pressure_so_aliases_match() {
        use std::sync::atomic::Ordering::Relaxed;
        if crate::host::host().is_none() {
            return;
        }
        // Pressure alone makes committed sampled textures start smaller;
        // placed views must not follow, or alias copies against a full-mip
        // RT/UAV refuse (mismatched host subresource counts).
        with_pressure(2, || {
            let mut committed = std::ptr::null();
            assert_eq!(
                create_committed(
                    std::ptr::null(),
                    &PROPERTIES,
                    0,
                    &texture(64, 64, 28, 7),
                    0,
                    std::ptr::null(),
                    &RESOURCE_IIDS[0],
                    &mut committed,
                ),
                S_OK
            );
            assert_eq!(dropped_mips(committed), 2);
            // SAFETY: ResourceDesc is 56 bytes / 14 words.
            let words = unsafe {
                std::ptr::from_ref(&texture(64, 64, 28, 7))
                    .cast::<[u32; 14]>()
                    .read_unaligned()
            };
            // Any non-zero address: the host copy is a separate committed resource.
            let mut memory = [0u8; 8];
            let placed =
                texture_at(memory.as_mut_ptr() as u64, &words).expect("placed view under pressure");
            let host = host_resource(placed).expect("placed host texture");
            // SAFETY: a live host texture.
            let d = unsafe { host.GetDesc() };
            assert_eq!(
                (d.Width, d.Height, d.MipLevels),
                (64, 64, 7),
                "placed views must keep every level so alias copies against a full-mip RT/UAV still match"
            );
            assert_eq!(dropped_mips(placed), 0);
            assert_eq!(PRESSURE.load(Relaxed), 2);
            assert_eq!(slot::<Release>(committed, 0x10)(committed), 0);
        });
    }

    #[cfg(windows)]
    #[test]
    fn synthetic_oom_threshold_is_restored_on_drop() {
        let previous = synthetic_oom_above();
        {
            let _oom = SyntheticOom::above(32);
            assert_eq!(synthetic_oom_above(), 32);
        }
        assert_eq!(synthetic_oom_above(), previous);
        // Nested scopes restore the outer limit, not the process default.
        let _outer = SyntheticOom::above(64);
        {
            let _inner = SyntheticOom::above(8);
            assert_eq!(synthetic_oom_above(), 8);
        }
        assert_eq!(synthetic_oom_above(), 64);
    }
}
