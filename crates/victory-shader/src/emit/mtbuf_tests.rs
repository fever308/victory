//! Synthetic normalized MTBUF loads and flat accesses to structured views.
//! gfx1036 measurements in #1060 found correctly rounded results for all
//! twenty normalized formats, including packed fields.

use super::{Emission, HostShader, StructuredBinding, ViewStride, emit_with_view_strides};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{Container, FourCC, io, ir, resource, structure};

// UNORM format and field widths from low to high. The next format is SNORM.
const FORMATS: &[(u8, &[u32])] = &[
    (1, &[8]),
    (7, &[16]),
    (14, &[8, 8]),
    (23, &[16, 16]),
    (30, &[11, 11, 10]),
    (37, &[10, 11, 11]),
    (44, &[2, 10, 10, 10]),
    (50, &[10, 10, 10, 2]),
    (56, &[8, 8, 8, 8]),
    (65, &[16, 16, 16, 16]),
];

/// Each thread reads one 16-byte row of a raw SRV, then writes its four
/// converted dwords to the corresponding row of a separate raw UAV.
fn load_shader(format: u8) -> HostShader {
    let words = [
        0x8F02_8601, // s_lshl_b32 s2, s1, 6 (group ID x times 64)
        0x4A00_0002, // v_add_nc_u32 v0, s2, v0 (plus thread ID x)
        0x3400_0084, // v_lshlrev_b32 v0, 4, v0 (byte offset)
        0xBE81_0380, // s_mov_b32 s1, 0 (high half of table address)
        0xF408_0200,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], 0
        0xF408_0300,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[0:1], 32
        0xE803_1000 | (u32::from(format) << 19),
        0x8002_0100, // tbuffer_load_format_xyzw v[1:4], v0, s[8:11], 0 offen
        0xE078_1000,
        0x8003_0100, // buffer_store_dwordx4 v[1:4], v0, s[12:15], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    buffer_shader(&words, Some(0), &|_, _| None).unwrap()
}

fn buffer_shader(
    words: &[u32],
    input_stride: Option<u32>,
    view_stride: ViewStride<'_>,
) -> Result<HostShader, crate::ShaderError> {
    let range = |kind, table_offset_dwords, stride| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords: 4,
        user_data: 0,
        slot: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 5,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: input_stride
            .map(|stride| range(ResourceType::Srv, 0, stride))
            .into_iter()
            .chain([range(ResourceType::Uav, 8, 0)])
            .collect(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 7 | 1 << 1, // group ID x and one user SGPR
            threads: [64, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    emit_test_program(&native, Some(view_stride))
}

fn emit_test_program(
    native: &NativeProgram,
    view_stride: Option<ViewStride<'_>>,
) -> Result<HostShader, crate::ShaderError> {
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, native).unwrap();
    let bytes = crate::dxbc::build(&[]);
    let container = Container::parse(&bytes).unwrap();
    emit_with_view_strides(
        Emission {
            c: &container,
            native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
        },
        true,
        view_stride,
    )
}

#[test]
fn raw_buffer_byte_and_halfword_loads_extend_at_every_byte_offset() {
    // One lane per byte offset. All four loads use the same address.
    let words = [
        0xbe81_0380, // upper half of table address
        0xf408_0200,
        0xfa00_0000, // SRV in s[8:11]
        0xf408_0300,
        0xfa00_0020, // UAV in s[12:15]
        0xe020_1000,
        0x8002_0100, // buffer_load_ubyte v1 at v0
        0xe024_1000,
        0x8002_0200, // buffer_load_sbyte v2
        0xe028_1000,
        0x8002_0300, // buffer_load_ushort v3
        0xe02c_1000,
        0x8002_0400, // buffer_load_sshort v4
        0x3400_0084, // output byte offset = lane * 16
        0xe078_1000,
        0x8003_0100, // store four result dwords
        0xbf81_0000,
    ];
    let shader = buffer_shader(&words, Some(0), &|_, _| None).unwrap();
    assert!(
        Container::parse(&shader.dxil)
            .unwrap()
            .part(FourCC::DXIL)
            .is_some()
    );

    #[cfg(windows)]
    {
        use super::host::{Dispatch, Host, Root, Stride, View};
        let _one = super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let bytes: Vec<u8> = (0u8..68)
            .map(|n| n.wrapping_mul(91).wrapping_add(17))
            .collect();
        let data: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(u32::from_le_bytes)
            .collect();
        let expected: Vec<u32> = bytes
            .windows(2)
            .take(64)
            .flat_map(|b| {
                let half = u16::from_le_bytes([b[0], b[1]]);
                [
                    u32::from(b[0]),
                    b[0] as i8 as i32 as u32,
                    u32::from(half),
                    half as i16 as i32 as u32,
                ]
            })
            .collect();
        let force_warp = std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
        for &(name, warp) in super::valu_tests::backends(force_warp) {
            let Ok(host) = Host::new(warp) else { continue };
            let (root, pso) = host.compute_pipeline(&shader).unwrap();
            let output = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(
                            0,
                            Root::Constants(vec![0; shader.user_data_dwords() as usize]),
                        )],
                        views: &[
                            View::Buffer {
                                data: &data,
                                stride: Stride::Raw,
                            },
                            View::RawUav(256),
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            assert_eq!(output[0], expected, "{name}");
            eprintln!("{name}: verified all 256 extension results");
        }
    }
}

#[test]
fn normalized_mtbuf_loads_need_no_int64() {
    for &(unorm, _) in FORMATS {
        for format in [unorm, unorm + 1] {
            let shader = load_shader(format);
            let c = Container::parse(&shader.dxil).unwrap();
            let sfi0 = c.part(FourCC::SFI0).unwrap().data;
            let flags = u64::from_le_bytes(sfi0.try_into().unwrap());
            assert_eq!(flags & 0x8000, 0, "format {format}, SFI0 {flags:#x}");
            #[cfg(windows)]
            if let Ok(mut validator) = super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                validator.validate(&mut shader.dxil.clone()).unwrap();
            }
        }
    }
}

#[cfg(windows)]
#[test]
fn normalized_mtbuf_loads_round_every_field_value() {
    use super::host::{Dispatch, Host, Root, Stride, View};
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    // `VICTORY_WARP=1` makes `Host::new(false)` WARP too: the shared list
    // then runs it once, labelled as such.
    let force_warp = std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
    let hosts: Vec<_> = super::valu_tests::backends(force_warp)
        .iter()
        .filter_map(|&(name, warp)| Host::new(warp).ok().map(|h| (name, h)))
        .collect();
    assert!(!hosts.is_empty(), "no SM 6.6 device");

    for &(unorm, widths) in FORMATS {
        let rows = 1u32 << widths.iter().max().unwrap();
        for signed in [false, true] {
            let format = unorm + u8::from(signed);
            let shader = load_shader(format);
            let mut data = Vec::new();
            let mut expected = Vec::new();
            for row in 0..rows {
                let mut packed = 0u64;
                let mut shift = 0;
                for (channel, &width) in widths.iter().enumerate() {
                    // Distinguish the channels and visit every field value.
                    let raw = (row + channel as u32 * 71) & ((1 << width) - 1);
                    packed |= u64::from(raw) << shift;
                    shift += width;
                    let max = (1i32 << (width - u32::from(signed))) - 1;
                    let k = if signed && raw > max as u32 {
                        raw as i32 - (1 << width)
                    } else {
                        raw as i32
                    };
                    let ratio = f64::from(k.max(-max)) / f64::from(max);
                    expected.push((ratio as f32).to_bits());
                }
                data.extend([packed as u32, (packed >> 32) as u32, 0, 0]);
            }
            for (name, host) in &hosts {
                let (root, pso) = host.compute_pipeline(&shader).unwrap();
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(
                                0,
                                Root::Constants(vec![0; shader.user_data_dwords() as usize]),
                            )],
                            views: &[
                                View::Buffer {
                                    data: &data,
                                    stride: Stride::Raw,
                                },
                                View::RawUav(4 * rows),
                            ],
                            samplers: &[],
                            groups: [rows / 64, 1, 1],
                        },
                    )
                    .unwrap();
                assert_eq!(out[0].len(), 4 * rows as usize, "{name}: format {format}");
                for (row, (got, want)) in out[0]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(expected.chunks_exact(widths.len()))
                    .enumerate()
                {
                    assert_eq!(
                        &got[..widths.len()],
                        want,
                        "{name}: format {format}, row {row}"
                    );
                }
            }
        }
    }
}

// Read dword 1 of each record, then copy it to a raw output at 4 * thread ID.
const STRUCTURED_READ: &[u32] = &[
    0xBE81_0380, // s_mov_b32 s1, 0
    0xF408_0200,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], 0
    0xF408_0300,
    0xFA00_0020, // s_load_dwordx4 s[12:15], s[0:1], 32
    0xE030_2004,
    0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen offset:4
    0x3400_0082, // v_lshlrev_b32 v0, 2, v0
    0xE070_1000,
    0x8003_0100, // buffer_store_dword v1, v0, s[12:15], 0 offen
    0xBF81_0000, // s_endpgm
];

#[test]
fn indexed_load_uses_the_view_stride_when_metadata_is_missing_or_invalid() {
    for metadata in [None, Some(0), Some(6), Some(2052)] {
        assert!(matches!(
            buffer_shader(STRUCTURED_READ, metadata, &|_, _| None),
            Err(crate::ShaderError::Unsupported {
                what: super::UNKNOWN_STRUCTURED_STRIDE,
                ..
            })
        ));
        for stride in [16, 32] {
            let called = std::cell::Cell::new(false);
            buffer_shader(STRUCTURED_READ, metadata, &|binding, required| {
                if !required {
                    return None;
                }
                called.set(true);
                assert_eq!(
                    binding,
                    StructuredBinding {
                        slot: 0,
                        offset_dwords: 0,
                        uav: false
                    }
                );
                Some(stride)
            })
            .unwrap();
            assert!(called.get());
        }
    }
}

#[cfg(windows)]
#[test]
fn undeclared_structured_stride_reads_two_view_strides() {
    use super::host::{Dispatch, Host, Root, Stride, View};

    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let force_warp = std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
    let hosts: Vec<_> = super::valu_tests::backends(force_warp)
        .iter()
        .filter_map(|&(name, warp)| match Host::new(warp) {
            Ok(host) => Some((name, host)),
            Err(error) => {
                eprintln!("{name}: {error}");
                None
            }
        })
        .collect();
    for stride in [16, 32] {
        let mut shader = buffer_shader(STRUCTURED_READ, Some(0), &|_, required| {
            required.then_some(stride)
        })
        .unwrap();
        let path = super::validator::Validator::default_path();
        if path.is_some() {
            super::validator::Validator::load(path.as_deref())
                .unwrap()
                .validate(&mut shader.dxil)
                .unwrap();
        }
        let expected: Vec<u32> = (0..64).map(|i| 0x1234_0000 + i * 61 + stride).collect();
        let mut data = vec![0xdead_beef; 64 * stride as usize / 4];
        for (record, value) in data.chunks_exact_mut(stride as usize / 4).zip(&expected) {
            record[1] = *value;
        }
        for (name, host) in &hosts {
            let (root, pso) = host.compute_pipeline(&shader).unwrap();
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(
                            0,
                            Root::Constants(vec![0; shader.user_data_dwords() as usize]),
                        )],
                        views: &[
                            View::Buffer {
                                data: &data,
                                stride: Stride::Structured(stride),
                            },
                            View::RawUav(64),
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            assert_eq!(out[0], expected, "{name}, stride {stride}");
            eprintln!("{name}: exact readback at stride {stride}");
        }
    }
}

#[test]
fn indexed_store_and_atomic_use_the_structured_uav_view_stride() {
    for opcode in [0x1C_u32, 0x32_u32] {
        let words = [
            0xBE81_0380, // s_mov_b32 s1, 0
            0xF408_0300,
            0xFA00_0020, // s_load_dwordx4 s[12:15], s[0:1], 32
            0xE000_2004 | (opcode << 18),
            0x8003_0000, // store / atomic_add v0, v0, s[12:15], 0 idxen offset:4
            0xBF81_0000, // s_endpgm
        ];
        assert!(buffer_shader(&words, None, &|_, _| None).is_err());
        let shader = buffer_shader(&words, None, &|binding, required| {
            if !required {
                return None;
            }
            assert_eq!(
                binding,
                StructuredBinding {
                    slot: 0,
                    offset_dwords: 8,
                    uav: true
                }
            );
            Some(16)
        })
        .unwrap();
        #[cfg(windows)]
        if let Some(path) = super::validator::Validator::default_path() {
            super::validator::Validator::load(Some(&path))
                .unwrap()
                .validate(&mut shader.dxil.clone())
                .unwrap();
        }
        assert!(!shader.dxil.is_empty());
    }
}

const STRUCTURED_BYTES: u32 = 288;
const STRUCTURED_STRIDES: [u32; 3] = [8, 12, 16];
const STRUCTURED_DESCRIPTORS: [u32; 5] = [
    0xbe81_0380, // s_mov_b32 s1, 0 (high half of table address)
    0xf408_0200,
    0xfa00_0000, // s_load_dwordx4 s[8:11], s[0:1], 0 (SRV)
    0xf408_0300,
    0xfa00_0020, // s_load_dwordx4 s[12:15], s[0:1], 32 (UAV)
];

fn structured_input() -> Vec<u8> {
    (0..STRUCTURED_BYTES).map(|i| (i * 37 + 19) as u8).collect()
}

fn v_mov(dst: u32, src: u32) -> u32 {
    0x7e00_0200 | dst << 17 | src
}

fn mubuf(op: u32, off: u32, vaddr: Option<u8>, returns: bool, data: u32, uav: bool) -> [u32; 2] {
    [
        0xe000_0000 | op << 18 | u32::from(returns) << 14 | u32::from(vaddr.is_some()) << 12 | off,
        0x8000_0000 | (2 + u32::from(uav)) << 16 | data << 8 | u32::from(vaddr.unwrap_or(0)),
    ]
}

fn structured_loads_shader(stride: u32) -> HostShader {
    let edge = 2 * stride - 1;
    let mut words = STRUCTURED_DESCRIPTORS.to_vec();
    for (desc, uav, out) in [(8, false, 160), (12, true, 208)] {
        // Both scalar and vector loads start on the element's last byte.
        words.extend([
            0xf424_0400 | (desc / 2),
            0xfa00_0000 | edge, // s_buffer_load_dwordx2 s[16:17], s[desc:desc+3], edge
            v_mov(1, 16),
            v_mov(2, 17),
        ]);
        // offen uses v0 (thread ID), so the byte offset remains dynamic.
        for k in 0..2 {
            words.extend(mubuf(0x1c, out + 4 * k, None, false, 1 + k, true));
        }
        words.extend(mubuf(0x0e, edge, Some(0), false, 1, uav));
        for k in 0..4 {
            words.extend(mubuf(0x1c, out + 8 + 4 * k, None, false, 1 + k, true));
        }
        words.extend(mubuf(0x0b, edge, Some(0), false, 1, uav)); // signed short
        words.extend(mubuf(0x1c, out + 24, None, false, 1, true));
        words.extend([v_mov(1, 255), 0x7654_3210]);
        words.extend(mubuf(0x25, edge, None, false, 1, uav)); // short_d16_hi
        words.extend(mubuf(0x1c, out + 28, None, false, 1, true));
        words.extend(mubuf(0x08, edge, None, false, 1, uav)); // unsigned byte
        words.extend(mubuf(0x1c, out + 32, None, false, 1, true));
        // MTBUF format 20 (R32_UINT) with the same dynamic flat offset.
        words.extend([
            0xe800_1000 | 20 << 19 | edge,
            0x8000_0100 | (desc / 4) << 16,
        ]);
        words.extend(mubuf(0x1c, out + 36, None, false, 1, true));
    }
    words.push(0xbf81_0000);
    emit_structured_program(stride, &words, 1, 5)
}

fn structured_dimensions_shader(stride: u32, uav: bool) -> HostShader {
    let mut words = STRUCTURED_DESCRIPTORS.to_vec();
    let (count_reg, out) = if uav { (14, 272) } else { (10, 256) };
    words.push(v_mov(1, count_reg));
    words.extend(mubuf(0x1c, out, None, false, 1, true));
    words.push(0xbf81_0000);
    emit_structured_program(stride, &words, 1, 2)
}

fn structured_atomics_shader(stride: u32, input: &[u8]) -> HostShader {
    let aligned_store = (124 / stride + 1) * stride - 4;
    let mut words = STRUCTURED_DESCRIPTORS.to_vec();
    words.extend([v_mov(1, 255), 0x1122_3344, v_mov(2, 255), 0x5566_7788]);
    words.extend(mubuf(0x1d, aligned_store, None, false, 1, true)); // dwordx2 across elements
    words.extend(mubuf(0x32, 116, Some(0), true, 1, true)); // atomic_add, return old
    words.extend(mubuf(0x1c, 260, None, false, 1, true));
    words.extend([
        v_mov(1, 255),
        0xaabb_ccdd,
        v_mov(2, 255),
        u32::from_le_bytes(input[120..124].try_into().unwrap()),
    ]);
    words.extend(mubuf(0x31, 120, None, true, 1, true)); // atomic_cmpswap, return old
    words.extend(mubuf(0x1c, 264, None, false, 1, true));
    words.push(0xbf81_0000); // s_endpgm
    emit_structured_program(stride, &words, 1, 3)
}

fn structured_unaligned_stores_shader(stride: u32) -> HostShader {
    let mut words = STRUCTURED_DESCRIPTORS.to_vec();
    words.extend([
        0x3408_0085, // v_lshlrev_b32 v4, 5, v0 (32 bytes per lane)
        v_mov(1, 255),
        0x1122_3344,
        v_mov(2, 255),
        0x5566_7788,
        v_mov(3, 255),
        0x99aa_bbcc,
    ]);
    words.extend(mubuf(0x1f, 31, Some(4), false, 1, true)); // unaligned dwordx3
    words.extend(mubuf(0x1a, 47, Some(4), false, 1, true)); // straddling short
    words.extend(mubuf(0x19, 51, Some(4), false, 1, true)); // byte from high half
    words.extend([v_mov(1, 255), 0xab]);
    // All four lanes update different bytes of the same dword.
    words.extend(mubuf(0x18, stride - 4, Some(0), false, 1, true));
    words.push(0xbf81_0000);
    emit_structured_program(stride, &words, 4, 5)
}

fn emit_structured_program(stride: u32, words: &[u32], threads: u32, vgprs: u16) -> HostShader {
    emit_test_program(&structured_program(stride, words, threads, vgprs), None).unwrap()
}

fn structured_program(stride: u32, words: &[u32], threads: u32, vgprs: u16) -> NativeProgram {
    let range = |kind, table_offset_dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords: 4,
        user_data: 0,
        slot: 0,
    };
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![range(ResourceType::Srv, 0), range(ResourceType::Uav, 8)],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

#[test]
fn flat_and_indexed_accesses_share_the_bound_view_stride() {
    let mut words = STRUCTURED_DESCRIPTORS.to_vec();
    words.extend([
        0xf420_0404,
        0xfa00_000b, // s_buffer_load_dword s16, s[8:11], 11
        v_mov(1, 16),
        0xe030_2004,
        0x8002_0200,  // buffer_load_dword v2, v0, s[8:11], 0 idxen offset:4
        v_mov(3, 10), // SRV record count
    ]);
    words.extend(mubuf(0x1f, 143, None, false, 1, true)); // unaligned store of v[1:3]
    words.push(0xbf81_0000);

    let stride = 12;
    let listed = emit_structured_program(stride, &words, 1, 4);
    let specialized = emit_test_program(
        &structured_program(0, &words, 1, 4),
        Some(&|binding, _| {
            assert_eq!(binding.slot, 0);
            assert_eq!(binding.offset_dwords, if binding.uav { 8 } else { 0 });
            Some(stride)
        }),
    )
    .unwrap();
    assert!(specialized.dxil == listed.dxil);
}

#[test]
fn flat_store_keeps_declared_stride_four_raw() {
    let mut words = STRUCTURED_DESCRIPTORS.to_vec();
    words.extend(mubuf(0x1c, 12, None, false, 0, true));
    words.push(0xbf81_0000);

    let raw = emit_structured_program(0, &words, 1, 1);
    let bound_stride = |binding: StructuredBinding, _| binding.uav.then_some(4);
    let declared =
        emit_test_program(&structured_program(4, &words, 1, 1), Some(&bound_stride)).unwrap();
    let specialized =
        emit_test_program(&structured_program(0, &words, 1, 1), Some(&bound_stride)).unwrap();
    assert!(declared.dxil == raw.dxil);
    assert!(specialized.dxil != raw.dxil);
}

#[test]
fn flat_buffer_accesses_on_structured_views_translate() {
    let input = structured_input();
    for stride in STRUCTURED_STRIDES {
        for shader in [
            structured_loads_shader(stride),
            structured_dimensions_shader(stride, false),
            structured_dimensions_shader(stride, true),
            structured_atomics_shader(stride, &input),
            structured_unaligned_stores_shader(stride),
        ] {
            assert!(
                Container::parse(&shader.dxil)
                    .unwrap()
                    .part(FourCC::DXIL)
                    .is_some()
            );
            #[cfg(windows)]
            if let Some(path) = super::validator::Validator::default_path() {
                let mut validator = super::validator::Validator::load(Some(&path)).unwrap();
                validator.validate(&mut shader.dxil.clone()).unwrap();
            }
        }
    }
}

#[cfg(windows)]
#[test]
fn flat_structured_loads_preserve_bytes() {
    run_structured_case(
        false,
        |stride, _| structured_loads_shader(stride),
        |stride, input, expected| {
            let edge = (2 * stride - 1) as usize;
            let scalar_edge = edge & !3;
            for out in [160, 208] {
                expected[out..out + 8].copy_from_slice(&input[scalar_edge..scalar_edge + 8]);
                expected[out + 8..out + 24].copy_from_slice(&input[edge..edge + 16]);
                let short = i16::from_le_bytes(input[edge..edge + 2].try_into().unwrap());
                expected[out + 24..out + 28].copy_from_slice(&i32::from(short).to_le_bytes());
                let high_half = (u32::from(short as u16) << 16) | 0x3210;
                expected[out + 28..out + 32].copy_from_slice(&high_half.to_le_bytes());
                expected[out + 32..out + 36].copy_from_slice(&u32::from(input[edge]).to_le_bytes());
                expected[out + 36..out + 40].copy_from_slice(&input[edge..edge + 4]);
            }
        },
    );
}

#[cfg(windows)]
#[test]
fn flat_structured_srv_dimensions_preserve_count() {
    run_structured_case(
        false,
        |stride, _| structured_dimensions_shader(stride, false),
        |stride, _, expected| {
            let count = STRUCTURED_BYTES / stride - 1;
            expected[256..260].copy_from_slice(&count.to_le_bytes());
        },
    );
}

#[cfg(windows)]
#[test]
fn flat_structured_uav_dimensions_preserve_count() {
    // AMD's DXIL compiler faults on SM 6.6 heap-handle GetDimensions for structured UAVs
    // even with DXC-matching DXIL. dxil.dll validates the shader and WARP checks the exact count.
    run_structured_case(
        true,
        |stride, _| structured_dimensions_shader(stride, true),
        |stride, _, expected| {
            let count = STRUCTURED_BYTES / stride;
            expected[272..276].copy_from_slice(&count.to_le_bytes());
        },
    );
}

#[cfg(windows)]
#[test]
fn flat_structured_atomics_preserve_bytes() {
    run_structured_case(
        false,
        structured_atomics_shader,
        |stride, input, expected| {
            let aligned_store = ((124 / stride + 1) * stride - 4) as usize;
            expected[aligned_store..aligned_store + 4]
                .copy_from_slice(&0x1122_3344u32.to_le_bytes());
            expected[aligned_store + 4..aligned_store + 8]
                .copy_from_slice(&0x5566_7788u32.to_le_bytes());
            let old = u32::from_le_bytes(input[116..120].try_into().unwrap());
            expected[116..120].copy_from_slice(&old.wrapping_add(0x1122_3344).to_le_bytes());
            expected[120..124].copy_from_slice(&0xaabb_ccddu32.to_le_bytes());
            expected[260..264].copy_from_slice(&input[116..120]);
            expected[264..268].copy_from_slice(&input[120..124]);
        },
    );
}

#[cfg(windows)]
#[test]
fn flat_structured_unaligned_stores_preserve_bytes() {
    run_structured_case(
        false,
        |stride, _| structured_unaligned_stores_shader(stride),
        |stride, _, expected| {
            let values = [
                0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55, 0xcc, 0xbb, 0xaa, 0x99,
            ];
            for lane in 0..4 {
                let base = lane * 32;
                expected[base + 31..base + 43].copy_from_slice(&values);
                expected[base + 47..base + 49].copy_from_slice(&[0x44, 0x33]);
                expected[base + 51] = 0x22;
            }
            expected[(stride - 4) as usize..stride as usize].fill(0xab);
        },
    );
}

#[cfg(windows)]
fn run_structured_case(
    warp_only: bool,
    build: impl Fn(u32, &[u8]) -> HostShader,
    expect: impl Fn(u32, &[u8], &mut [u8]),
) {
    use super::host::Host;

    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let force_warp = warp_only || std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
    let hosts: Vec<_> = super::valu_tests::backends(force_warp)
        .iter()
        .filter_map(|&(label, warp)| match Host::new(warp) {
            Ok(host) => Some((label, host)),
            Err(err) => {
                eprintln!("{label}: {err}");
                None
            }
        })
        .collect();
    let input = structured_input();
    for stride in STRUCTURED_STRIDES {
        let shader = build(stride, &input);
        let mut expected = input.clone();
        expect(stride, &input, &mut expected);
        for (label, host) in &hosts {
            let actual = dispatch_structured(host, &shader, stride, &input);
            assert_eq!(actual.len(), expected.len());
            for (i, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(actual, expected, "{label}, stride {stride}, byte {i}");
            }
        }
    }
}

/// Bind a structured UAV, which the shared test runner's UAV variants do not support.
#[cfg(windows)]
fn dispatch_structured(
    host: &super::host::Host,
    shader: &HostShader,
    stride: u32,
    input: &[u8],
) -> Vec<u8> {
    use std::mem::ManuallyDrop;
    use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::DXGI_SAMPLE_DESC};
    use windows::core::Interface;

    let device = &host.device;
    let size = u64::from(STRUCTURED_BYTES);
    assert_eq!(input.len(), size as usize);
    let (root, pso) = host.compute_pipeline(shader).unwrap();
    // SAFETY: the descriptors name live buffers of the stated size. The
    // buffers, heap and command allocator all outlive the dispatch and copy.
    unsafe {
        let make = |heap, state, flags| {
            let mut resource: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: heap,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: size,
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
                    },
                    state,
                    None,
                    &mut resource,
                )
                .unwrap();
            resource.unwrap()
        };
        let src = make(
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            D3D12_RESOURCE_FLAG_NONE,
        );
        let dst = make(
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        );
        let readback = make(
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_FLAG_NONE,
        );
        let mut mapped = std::ptr::null_mut();
        src.Map(0, None, Some(&mut mapped)).unwrap();
        std::ptr::copy_nonoverlapping(input.as_ptr(), mapped.cast::<u8>(), input.len());
        src.Unmap(0, None);

        let heap: ID3D12DescriptorHeap = device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
            .unwrap();
        let mut at = heap.GetCPUDescriptorHandleForHeapStart();
        let elements = STRUCTURED_BYTES / stride;
        device.CreateShaderResourceView(
            &src,
            Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        NumElements: elements - 1,
                        StructureByteStride: stride,
                        ..Default::default()
                    },
                },
                ..Default::default()
            }),
            at,
        );
        at.ptr += device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
            as usize;
        device.CreateUnorderedAccessView(
            &dst,
            None::<&ID3D12Resource>,
            Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        NumElements: elements,
                        StructureByteStride: stride,
                        ..Default::default()
                    },
                },
                ..Default::default()
            }),
            at,
        );

        let queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default())
            .unwrap();
        let alloc: ID3D12CommandAllocator = device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .unwrap();
        let list: ID3D12GraphicsCommandList = device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso)
            .unwrap();
        let transition = |before, after| {
            list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: ManuallyDrop::new(Some(std::ptr::read(&dst))),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: before,
                        StateAfter: after,
                    }),
                },
                ..Default::default()
            }]);
        };
        list.CopyBufferRegion(&dst, 0, &src, 0, size);
        transition(
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        list.SetDescriptorHeaps(&[Some(heap.clone())]);
        list.SetComputeRootSignature(&root);
        let constants = vec![0u32; shader.user_data_dwords() as usize];
        list.SetComputeRoot32BitConstants(0, constants.len() as u32, constants.as_ptr().cast(), 0);
        list.Dispatch(1, 1, 1);
        transition(
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        list.CopyBufferRegion(&readback, 0, &dst, 0, size);
        list.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.Signal(&fence, 1).unwrap();
        while fence.GetCompletedValue() < 1 {
            device.GetDeviceRemovedReason().unwrap();
            std::thread::yield_now();
        }
        readback.Map(0, None, Some(&mut mapped)).unwrap();
        let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), size as usize).to_vec();
        readback.Unmap(0, None);
        bytes
    }
}
