//! Unit tests of `lower`, a child module so `lower.rs` stays under the asset policy's size limit.

use super::{Lowered, Numeric, Parts, buffer_format, f16_nan_widen_bits, half_shift, lower};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

#[test]
fn structured_buffer_properties_match_dword_record_alignment() {
    for (stride, srv) in [(4, 0x000c), (12, 0x020c), (16, 0x020c)] {
        assert_eq!(
            super::Lower::buffer_props(Some(stride), false),
            (srv, stride)
        );
        assert_eq!(
            super::Lower::buffer_props(Some(stride), true),
            (srv | 0x1000, stride)
        );
    }
}

/// The blocks of `words` (a wave32 program) whose `s_barrier` loses its
/// thread-group sync.
fn fence_only(words: &[u32]) -> Vec<usize> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = ir::build(&code, false, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let mut out: Vec<_> = super::divergent_barriers(&p, &st, &Default::default())
        .into_iter()
        .collect();
    out.sort();
    out
}

// v_cmp_gt_u32 vcc, v0, v1; s_and_saveexec_b32 s0, vcc_lo: a per-lane if.
const PER_LANE_IF: [u32; 2] = [0x7D88_0300, 0xBE80_3C6A];
// s_or_b32 exec_lo, exec_lo, s0: the if's end.
const END_IF: u32 = 0x887E_007E;

/// A barrier inside a per-lane `if` runs in some lanes only.
#[test]
fn a_barrier_under_a_per_lane_branch_is_fence_only() {
    let words = [
        PER_LANE_IF[0],
        PER_LANE_IF[1],
        0xBF88_0001, // s_cbranch_execz over the barrier
        0xBF8A_0000, // s_barrier
        END_IF,
        S_ENDPGM,
    ];
    assert_eq!(fence_only(&words).len(), 1);
}

/// A per-lane `if` that ends before a uniform `if` around the barrier
/// leaves the barrier to the uniform condition alone: every thread of
/// the group reaches it or none does, so it keeps its sync.
#[test]
fn a_barrier_after_a_reconverged_branch_keeps_its_sync() {
    let words = [
        PER_LANE_IF[0],
        PER_LANE_IF[1],
        0xBF88_0001, // s_cbranch_execz over the move
        0x7E04_0300, // v_mov_b32 v2, v0
        END_IF,
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF85_0001, // s_cbranch_scc1 over the barrier
        0xBF8A_0000, // s_barrier
        S_ENDPGM,
    ];
    assert_eq!(fence_only(&words), Vec::<usize>::new());
}

/// A wave32 exit mask grown while lanes leave a loop at different
/// iterations, counted after the loop: every lane is in the mask (32),
/// not each lane's own bit (1). Issue #465.
#[test]
fn loop_exit_mask_counted_after_the_loop_is_wave_wide() {
    let words: &[u32] = &[
        0x4A02_0081, // v_add_nc_u32 v1, 1, v0  (lane i loops i + 1 times)
        0xBE8A_0380, // s_mov_b32 s10, 0
        0x4E02_0281, // loop: v_subrev_nc_u32 v1, 1, v1
        0x7D84_0280, // v_cmp_eq_u32 vcc_lo, 0, v1
        0x880A_0A6A, // s_or_b32 s10, vcc_lo, s10
        0x8A7E_0A7E, // s_andn2_b32 exec_lo, exec_lo, s10
        0xBF89_FFFB, // s_cbranch_execnz loop
        0x887E_0A7E, // s_or_b32 exec_lo, exec_lo, s10
        0xBE89_0F0A, // s_bcnt1_i32_b32 s9, s10
        0x7E04_0209, // v_mov_b32 v2, s9
        S_ENDPGM,
    ];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [32, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, false, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let lowered = lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
    .expect("lower");
    // The count is a ballot of the reconverged wave, not a per-lane 0/1.
    assert!(lowered.wave_ops, "exit-mask count must ballot the wave");
    assert!(
        lowered.mask_ballots_after_loop > 0,
        "exit-mask count must ballot after the loop, not inside it"
    );
}

/// A mask written before a branch and counted inside it still ballots at
/// the definition: `reaches(def, def)` alone must not count as a loop.
#[test]
fn a_mask_before_a_branch_is_not_balloted_as_after_a_loop() {
    let words: &[u32] = &[
        0x7D88_008A, // v_cmp_gt_u32 vcc_lo, 10, v0
        0xBE94_036A, // s_mov_b32 s20, vcc_lo
        0xBE86_3C6A, // s_and_saveexec_b32 s6, vcc_lo
        0xBF88_0001, // s_cbranch_execz over the count
        0xBE89_0F14, // s_bcnt1_i32_b32 s9, s20
        0x7E04_0209, // v_mov_b32 v2, s9
        0x881E_067E, // s_or_b32 exec_lo, exec_lo, s6
        S_ENDPGM,
    ];
    let lowered = lower_words(words, 1 << 1).expect("lower");
    assert_eq!(
        lowered.mask_ballots_after_loop, 0,
        "acyclic mask defs must keep definition-site ballots"
    );
}

/// Wave64 form of [`loop_exit_mask_counted_after_the_loop_is_wave_wide`].
#[test]
fn wave64_loop_exit_mask_counted_after_the_loop_is_wave_wide() {
    let words: &[u32] = &[
        0x4A02_0081, // v_add_nc_u32 v1, 1, v0
        0xBE8A_0480, // s_mov_b64 s[10:11], 0
        0x4E02_0281, // loop: v_subrev_nc_u32 v1, 1, v1
        0x7D84_0280, // v_cmp_eq_u32 vcc, 0, v1
        0x888A_0A6A, // s_or_b64 s[10:11], vcc, s[10:11]
        0x8AFE_0A7E, // s_andn2_b64 exec, exec, s[10:11]
        0xBF89_FFFB, // s_cbranch_execnz loop
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0xBE89_100A, // s_bcnt1_i32_b64 s9, s[10:11]
        0x7E04_0209, // v_mov_b32 v2, s9
        S_ENDPGM,
    ];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [64, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, true, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let lowered = lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
    .expect("lower");
    assert!(lowered.wave_ops, "exit-mask count must ballot the wave");
    assert!(
        lowered.mask_ballots_after_loop > 0,
        "exit-mask count must ballot after the loop, not inside it"
    );
}

/// Lowers a synthetic compute program with these input registers and
/// returns the lowered result.
fn lower_words(words: &[u32], rsrc2: u32) -> Result<super::Lowered, crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2,
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, false, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
}

/// Whether a synthetic compute program translates with these input registers.
fn lowers(words: &[u32], rsrc2: u32) -> Result<(), crate::ShaderError> {
    lower_words(words, rsrc2).map(|_| ())
}

#[test]
fn med3_f32_emits_dxil_isnan_instead_of_fcmp_uno() {
    // v_med3_f32 v0, v1, v2, v3 (VOP3 opcode 0x157)
    let src0 = 256 + 1u32;
    let src1 = 256 + 2u32;
    let src2 = 256 + 3u32;
    let words = [
        0x35 << 26 | (0x157u32 << 16),
        src0 | (src1 << 9) | (src2 << 18),
        S_ENDPGM,
    ];
    let lowered = lower_words(&words, 0).unwrap();
    assert!(
        lowered.has_isnan_op,
        "v_med3_f32 must emit dx.op.isSpecialFloat.f32 (IsNaN)"
    );
    assert!(
        !lowered.has_fcmp_uno,
        "v_med3_f32 must not emit fcmp uno (folded away by some drivers)"
    );
}

/// VOP3 `v_add_f32` with output modifier `omod` in bits 28:27 of word 1.
fn v_add_f32_omod(omod: u8) -> [u32; 3] {
    let src0 = 256 + 1u32;
    let src1 = 256 + 2u32;
    [
        // VOP3 opcode 0x103 = VOP2 v_add_f32 (0x003) at the VOP2 base.
        0x35 << 26 | (0x103u32 << 16),
        src0 | (src1 << 9) | (u32::from(omod) & 3) << 27,
        S_ENDPGM,
    ]
}

#[test]
fn write_f_omod_flushes_neg_zero_with_fcmp_oeq() {
    // Plain add has no signed-zero flush; omod must compare against +0.
    let plain = lower_words(&v_add_f32_omod(0), 0).unwrap();
    assert!(
        !plain.has_fcmp_oeq,
        "v_add_f32 without omod must not emit fcmp oeq"
    );
    for omod in 1..=3 {
        let lowered = lower_words(&v_add_f32_omod(omod), 0).unwrap();
        assert!(
            lowered.has_fcmp_oeq,
            "v_add_f32 omod={omod} must emit fcmp oeq to flush -0.0"
        );
    }
}

/// A pixel shader that exports `v0` as a render-target colour, with
/// `input_addr` enabling its input VGPRs.
fn lower_ps(input_addr: u32) -> Result<Lowered, crate::ShaderError> {
    use super::super::container::SigElement;
    // exp mrt0 v0, v0, v0, v0 (done); s_endpgm
    let words = [0xF800_180Fu32, 0, S_ENDPGM];
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 1,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs: vec![],
            input_addr,
            rsrc2: Default::default(),
            db_shader_control: 0,
            patches: vec![],
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let target = SigElement {
        name: "SV_Target".into(),
        index: 0,
        kind: 16,
        comp: 5,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0,
        half: false,
    };
    let p = ir::build(&native.code, false, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![target],
        prims: vec![],
        mesh_stride: 0,
    })
}

#[test]
fn pixel_sample_coverage_input_reads_dx_op_coverage() {
    let with = lower_ps(1 << 14).unwrap();
    assert!(
        with.has_coverage_op,
        "SampleCoverage must lower to dx.op.coverage.i32 (opcode 91)"
    );
    let without = lower_ps(0).unwrap();
    assert!(!without.has_coverage_op);
}

#[test]
fn buffer_cache_invalidations_lower_as_no_ops() {
    // MUBUF opcode 0x071/0x072 with no data operand: the title asks the
    // host to invalidate its cached view of a buffer.
    assert!(
        lowers(&[0xE1C4_0000, 0, S_ENDPGM], 0).is_ok(),
        "buffer_gl0_inv"
    );
    assert!(
        lowers(&[0xE1C8_0000, 0, S_ENDPGM], 0).is_ok(),
        "buffer_gl1_inv"
    );
}

/// VIC-BT-042: fused `v_fma_f16` keeps `256h×256h + (-65504h) = 32h`;
/// unfused `v_mad_f16` / `v_mac_f16` overflow the product to +inf. All
/// three opcodes must decode and lower; host execution is in `valu_tests`.
#[test]
fn f16_fma_and_mad_mac_lower() {
    // Exact fused sum is finite; the product alone exceeds max finite f16.
    let (a, b, c) = (256.0f64, 256.0f64, -65504.0f64);
    assert_eq!(a * b + c, 32.0);
    assert!(a * b > 65504.0);

    let v = |n: u8| 256u32 + u32::from(n);
    let vop3 = |op: u16, vdst: u8, src0: u32, src1: u32, src2: u32| -> [u32; 2] {
        [
            0x35 << 26 | (op as u32 & 0x3FF) << 16 | u32::from(vdst),
            (src0 & 0x1FF) | (src1 & 0x1FF) << 9 | (src2 & 0x1FF) << 18,
        ]
    };
    let names = |words: &[u32]| -> Vec<&'static str> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        crate::gfx10::decode(&code)
            .unwrap()
            .iter()
            .map(|i| i.name().unwrap())
            .collect()
    };

    // v_fma_f16 v0, v1, v2, v3 (VOP3 0x34b)
    let mut fma = Vec::new();
    fma.extend_from_slice(&vop3(0x34b, 0, v(1), v(2), v(3)));
    fma.push(S_ENDPGM);
    assert_eq!(names(&fma), ["v_fma_f16", "s_endpgm"]);
    lowers(&fma, 0).expect("v_fma_f16");

    // v_mad_f16 v0, v1, v2, v3 (VOP3 0x203)
    let mut mad = Vec::new();
    mad.extend_from_slice(&vop3(0x203, 0, v(1), v(2), v(3)));
    mad.push(S_ENDPGM);
    assert_eq!(names(&mad), ["v_mad_f16", "s_endpgm"]);
    lowers(&mad, 0).expect("v_mad_f16");

    // v_mac_f16 v0, v1, v2 (VOP3 encoding of VOP2 0x023 → 0x123);
    // seed v0 from v3 so the accumulator is defined.
    let mut mac = Vec::new();
    mac.push(0x7E00_0303); // v_mov_b32 v0, v3
    mac.extend_from_slice(&vop3(0x123, 0, v(1), v(2), 0));
    mac.push(S_ENDPGM);
    assert_eq!(names(&mac), ["v_mov_b32", "v_mac_f16", "s_endpgm"]);
    lowers(&mac, 0).expect("v_mac_f16");
}

/// Lowers a pixel program with one target: whether it translates.
fn lowers_ps(words: &[u32]) -> Result<(), crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 4,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs: vec![],
            rsrc2: Default::default(),
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        },
        code,
    };
    let p = ir::build(&native.code, false, 4).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let target = crate::emit::container::SigElement {
        name: "SV_Target".into(),
        index: 0,
        kind: 16,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0,
        half: false,
    };
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![target],
        prims: vec![],
        mesh_stride: 0,
    })
    .map(|_| ())
}

/// Lowers a compute program of constants that allocates 512 bytes of LDS
/// (`COMPUTE_PGM_RSRC2`'s granule field): whether it translates.
fn lowers_lds(words: &[u32]) -> Result<(), crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 15,
            threads: [32, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, false, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
    .map(|_| ())
}

/// `v_permlanex16_b32` reads a lane of the *other* row of 16 in the
/// lane's 32-lane half, the pick from nibble i of {src2, src1}. Before
/// the rule its write was missing and the LDS store consuming it failed
/// the whole translation.
#[test]
fn permlanex16_reads_a_lane_of_the_other_row() {
    let words = [
        // The selector: lane i of a row reads lane i of the other.
        0xBE80_03FF,
        0x7654_3210, // s_mov_b32 s0, 0x76543210
        0xBE81_03FF,
        0xFEDC_BA98, // s_mov_b32 s1, 0xfedcba98
        // v0 is the thread id; lane i reads the lane its nibble names.
        0xD778_0001,
        0x0004_0100, // v_permlanex16_b32 v1, v0, s0, s1
        0x7E04_0280, // v_mov_b32 v2, 0
        0xD834_0000,
        0x0000_0102, // ds_write_b32 v2, v1
        S_ENDPGM,
    ];
    lowers_lds(&words).unwrap();
}

#[test]
fn a_compressed_export_unpacks_two_f16_channels_per_dword() {
    // A compr export: each source dword packs two f16 channels, the
    // low half first, so four channels come from two VGPRs.
    let words = [
        0x7E00_0282, // v_mov_b32 v0, 2
        0x7E02_0282, // v_mov_b32 v1, 2
        0xF800_0C0F,
        0x0000_0100, // exp mrt0, v0, v0, v1, v1 done compr
        S_ENDPGM,
    ];
    lowers_ps(&words).unwrap();
}

/// Lowers a compute program with the root parameters `layout` (in user
/// SGPRs from s0 on) and the used resources `used`.
fn lower_with(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
) -> Result<Lowered, crate::ShaderError> {
    lower_wave(words, layout, used, false)
}

/// [`lower_with`] in wave64 if `wave64`.
fn lower_wave(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    wave64: bool,
) -> Result<Lowered, crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 256,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            // The layout's user SGPRs (at least two), 512 bytes of LDS.
            rsrc2: 1 << 15 | sgprs.max(2) << 1,
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, wave64, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
}

/// Root parameter `slot` in user SGPRs from `user_data` on.
fn root(root: RootType, user_data: u8, slot: u8) -> RootParam {
    RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: if root == RootType::Descriptor { 2 } else { 1 },
        base_register: 0,
        user_data,
        slot,
    }
}

#[test]
fn a_scalar_carry_chain_reaches_the_select() {
    // The high dword of a 64-bit add or subtract, then a select on the
    // carry (borrow) the high half leaves in SCC, and a branch on it
    // (only code with an effect needs its operands).
    let add = [
        0xBE80_03C1, // s_mov_b32 s0, -1
        0xBE81_0380, // s_mov_b32 s1, 0
        0x8000_8100, // s_add_u32 s0, s0, 1
        0x8201_8001, // s_addc_u32 s1, s1, 0
        0x8502_0001, // s_cselect_b32 s2, s1, s0
        0xBF07_8002, // s_cmp_lg_u32 s2, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];
    let sub = [
        0xBE80_0380, // s_mov_b32 s0, 0
        0xBE81_0380, // s_mov_b32 s1, 0
        0x8080_8100, // s_sub_u32 s0, s0, 1
        0x8281_8001, // s_subb_u32 s1, s1, 0
        0x8502_0001, // s_cselect_b32 s2, s1, s0
        0xBF07_8002, // s_cmp_lg_u32 s2, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];

    for words in [add, sub] {
        lowers(&words, 0).unwrap();
    }
}

/// `v_mad_u32_u16` / `v_mad_i32_i16` (with op_sel halves) and `v_lerp_u8`
/// lower (#405). An LDS store keeps the result live.
#[test]
fn mad_32_16_and_lerp_u8_lower() {
    let vop3 = |op: u16, opsel: u8, vdst: u8, s0: u32, s1: u32, s2: u32| -> [u32; 2] {
        [
            0x35 << 26 | (op as u32 & 0x3FF) << 16 | (opsel as u32 & 0xF) << 11 | vdst as u32,
            (s0 & 0x1FF) | (s1 & 0x1FF) << 9 | (s2 & 0x1FF) << 18,
        ]
    };
    let (v0, v1, v2) = (256u32, 257, 258);
    let store = |op_words: [u32; 2]| {
        let mut words = vec![
            0x7E00_0283, // v_mov_b32 v0, 3
            0x7E02_0285, // v_mov_b32 v1, 5
            0x7E04_0287, // v_mov_b32 v2, 7
        ];
        words.extend_from_slice(&op_words);
        words.extend_from_slice(&[
            0x7E06_0280, // v_mov_b32 v3, 0
            0xD834_0000,
            0x0000_0003, // ds_write_b32 v3, v0
            S_ENDPGM,
        ]);
        lowers_lds(&words).unwrap();
    };
    store(vop3(0x373, 0, 0, v0, v1, v2)); // v_mad_u32_u16
    store(vop3(0x373, 0b0011, 0, v0, v1, v0)); // opsel hi halves
    store(vop3(0x375, 0, 0, v0, v1, v2)); // v_mad_i32_i16
    store(vop3(0x14d, 0, 0, v0, v1, v2)); // v_lerp_u8
}

/// VOP3 integer `clamp` is unsigned saturation on carry/non-carry add/sub
/// (VIC-BT-010). The LDS store forces the result live: without the fix,
/// `write_i` leaves it missing and the store refuses translation.
#[test]
fn integer_vop3_clamp_lowers_add_and_sub() {
    let v1 = 256 + 1u32;
    let v2 = 256 + 2u32;
    let store = |op_words: [u32; 2]| {
        let mut words = vec![
            0x7E02_0281, // v_mov_b32 v1, 1
            0x7E04_0282, // v_mov_b32 v2, 2
        ];
        words.extend_from_slice(&op_words);
        words.extend_from_slice(&[
            0x7E06_0280, // v_mov_b32 v3, 0
            0xD834_0000,
            0x0000_0003, // ds_write_b32 v3, v0
            S_ENDPGM,
        ]);
        lowers_lds(&words).unwrap();
    };
    // VOP2-as-VOP3: opcode + 0x100 (bare 0x025 is VOPC v_cmp_lg_f64).
    // v_add_nc_u32 / v_sub_nc_u32 / v_subrev_nc_u32 with clamp.
    for op in [0x125u16, 0x126, 0x127] {
        store([0x35 << 26 | (op as u32) << 16 | 1 << 15, v1 | v2 << 9]);
    }
    // v_add_co_u32 / v_sub_co_u32 / v_subrev_co_u32 with clamp (carry → s16).
    for op in [0x30fu16, 0x310, 0x319] {
        store([
            0x35 << 26 | (op as u32) << 16 | 1 << 15 | 16 << 8,
            v1 | v2 << 9,
        ]);
    }
}

/// `s_cmov_b32`, which the scalar offset trace follows, lowers too: the
/// source where SCC is set, the destination's old value where it is
/// clear.
#[test]
fn a_conditional_scalar_move_lowers() {
    let words = [
        0xBE82_0380, // s_mov_b32 s2, 0
        0xBF06_8002, // s_cmp_eq_u32 s2, 0
        0xBE82_0581, // s_cmov_b32 s2, 1
        0xBF07_8002, // s_cmp_lg_u32 s2, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];
    lowers(&words, 0).unwrap();
}

fn perm_features(selector: u32) -> u64 {
    // Synthetic compute shader:
    // v0 is the thread ID and can supply a dynamic selector.
    // Store the result in LDS to keep the permutation live.
    let words = [
        0x7E02_02FFu32,
        0x817F_34AB,
        0x7E04_02FF,
        0xFE01_56CD,
        0xD744_0003,
        257 | (258 << 9) | (selector << 18),
        0x3408_0082,
        0xD834_0000,
        0x0000_0304,
        S_ENDPGM,
    ];

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 15,
            threads: [64, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, false, 256).unwrap();

    assert!(
        p.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(|i| i.name() == "v_perm_b32")
    );

    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();

    let io = io::raise(&p, &native).unwrap();

    let emitted = lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
    .unwrap();

    let container = crate::Container::parse(&emitted.dxil).unwrap();

    let features = container.part(crate::FourCC::SFI0).unwrap();

    u64::from_le_bytes(features.data.try_into().unwrap())
}

#[test]
fn perm_constant_selector_needs_no_int64() {
    assert_eq!(perm_features(128) & 0x8000, 0);
}

#[test]
fn perm_dynamic_selector_needs_no_int64() {
    assert_eq!(
        perm_features(256) & 0x8000,
        0,
        "dynamic v_perm_b32 must not require INT64"
    );
}

#[test]
fn s_buffer_load_truncates_each_address_term_before_adding() {
    // Hardware masks soffset and imm_offset with & ~3 independently
    // before adding (VIC-BT-025): soffset=3 and offset:3 load byte 0, not 6.
    let program = |soffset_imm: u32, imm: u32| {
        let w0 = 0x3d << 26 | 8 << 18 | 13 << 6 | 4; // s13 from s[8:11]
        let w1 = 12 << 25 | (imm & 0x1f_ffff);
        vec![
            0xBE81_0380, // s_mov_b32 s1, 0
            0xF408_0200,
            0xFA00_0000,               // s_load_dwordx4 s[8:11], s[0:1], null
            0xBE8C_0380 | soffset_imm, // s_mov_b32 s12, imm
            w0,
            w1,          // s_buffer_load_dword s13, s[8:11], s12 offset:imm
            0x7E02_020D, // v_mov_b32 v1, s13
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ]
    };
    let layout = vec![root(RootType::Table, 0, 0)];
    let used = vec![UsedResource {
        kind: ResourceType::Cbv,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    }];
    let bytes = |so, off| {
        lower_with(&program(so, off), layout.clone(), used.clone())
            .unwrap()
            .smem_buffer_bytes
    };
    assert_eq!(bytes(3, 3), vec![0], "3+3 must not carry to byte 4");
    assert_eq!(bytes(0, 0), vec![0]);
    assert_eq!(bytes(4, 0), vec![4]);
    assert_eq!(bytes(0, 4), vec![4]);
    assert_eq!(bytes(5, 7), vec![8], "(5&~3)+(7&~3) = 8");
}

#[test]
fn scalar_loads_through_a_heap_buffer_that_vector_loads_read_use_a_buffer_view() {
    // A descriptor from the resource heap (s0) at an index from a root
    // constant (s1), read by a scalar buffer load, then used by nothing
    // else, by a vector buffer load, or by a vector buffer store. A slot
    // that vector loads read is a buffer view (an SRV), so the scalar
    // load must not read it through a constant buffer handle (a
    // descriptor of the wrong type). A store makes it a UAV, which a
    // read-only view would not match either: that case is unchanged.
    // Heap-direct scalar loads alone still use a CBV handle; index 0 is
    // skipped (shortcut: this title's dual-read null texture SRV; #485).
    #[derive(Clone, Copy)]
    enum Also {
        Nothing,
        Load,
        Store,
    }
    let program = |also: Also| {
        let mut words = vec![
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0x8F06_8501, // s_lshl_b32 s6, s1, 5
            0xF408_0202,
            0x0C00_0000, // s_load_dwordx4 s[8:11], s[4:5], s6
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xF420_0304,
            0xFA00_0000, // s_buffer_load_dword s12, s[8:11], 0x0
            0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
            0x7E02_020C, // v_mov_b32 v1, s12
        ];
        match also {
            Also::Nothing => {}
            Also::Load => words.extend([
                0xE030_1000,
                0x8002_0200, // buffer_load_dword v2, v0, s[8:11], 0 offen
                0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
                0x4A02_0302, // v_add_nc_u32 v1, v2, v1
            ]),
            Also::Store => words.extend([
                0xE070_1000,
                0x8002_0100, // buffer_store_dword v1, v0, s[8:11], 0 offen
            ]),
        }
        words.extend([
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ]);
        words
    };
    let layout = || {
        vec![
            root(RootType::ResourceHeap, 0, 0),
            root(RootType::Constants, 1, 1),
        ]
    };
    let cbuffers = |also| {
        lower_with(&program(also), layout(), vec![])
            .unwrap()
            .cbuffer_handles
    };
    assert_eq!(cbuffers(Also::Nothing), 1, "no other use: a CBV");
    assert_eq!(cbuffers(Also::Load), 0, "buffer_load: an SRV");
    assert_eq!(cbuffers(Also::Store), 1, "buffer_store: unchanged");
    // Untyped heap-direct (#485 null-slot skip) must still lower.
    assert!(
        lower_with(&program(Also::Nothing), layout(), vec![]).is_ok(),
        "heap-direct scalar CBV with null-slot skip"
    );
    // A listed ResourceHeap CBV keeps a live load at index 0 (no skip).
    let listed = vec![UsedResource {
        kind: ResourceType::Cbv,
        root: RootType::ResourceHeap,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 0,
        user_data: 0,
        slot: 0,
    }];
    assert_eq!(
        lower_with(&program(Also::Nothing), layout(), listed)
            .unwrap()
            .cbuffer_handles,
        1,
        "listed ResourceHeap CBV at slot 0 stays a CBV"
    );
}

#[test]
fn a_descriptor_selected_by_scc_picks_its_heap_index() {
    // Two buffer descriptors loaded from the table at s0, one picked by
    // a root constant (s1) with s_cselect_b64 per half, and loaded
    // through by index and stored to LDS so the load is needed.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0xF408_0302,
        0xFA00_0010, // s_load_dwordx4 s[12:15], s[4:5], 0x10
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0x8590_0C08, // s_cselect_b64 s[16:17], s[8:9], s[12:13]
        0x8592_0E0A, // s_cselect_b64 s[18:19], s[10:11], s[14:15]
        0xE030_2000,
        0x8004_0100, // buffer_load_dword v1, v0, s[16:19], 0 idxen
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let layout = || vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
    // Both are structured buffers of dwords in the table's range.
    let used = || {
        vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }]
    };
    lower_with(&words, layout(), used()).unwrap();
}

/// `v_mov_b32 vdst, src0` (`src0`: 128 is the inline zero).
fn v_mov(vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | 0x01 << 9 | (src0 as u32 & 0x1FF)
}

/// `ds_write_b32 v[addr], v[data]`.
fn ds_write_b32(addr: u8, data: u8) -> [u32; 2] {
    [0xD834_0000, u32::from(addr) | u32::from(data) << 8]
}

/// A buffer access: MUBUF (`mtbuf` false, opcode in bits 18-25, a
/// resource and optional `idxen`/`offen`) or MTBUF (`mtbuf` true, opcode
/// in bits 15-18, the format in bits 19-25), `srsrc` the first of four
/// resource SGPRs, `vdata`/`vaddr` VGPRs and `offset` a byte offset.
#[allow(clippy::too_many_arguments)]
fn buffer(
    opcode: u16,
    mtbuf: bool,
    format: u8,
    offset: u16,
    offen: bool,
    idxen: bool,
    vdata: u8,
    vaddr: u8,
    srsrc: u8,
) -> [u32; 2] {
    let class = if mtbuf { 0x3A } else { 0x38 };
    let op_shift = if mtbuf { 16 } else { 18 };
    [
        class << 26
            // MUBUF opcodes are 8 bits (0x00..0xff); MTBUF uses 4 bits
            // split across the two words.
            | u32::from(opcode & if mtbuf { 0x7 } else { 0xFF }) << op_shift
            | u32::from(format) << 19
            | u32::from(offen) << 12
            | u32::from(idxen) << 13
            | u32::from(offset),
        // soffset is the inline zero, TFE and GLC clear; MTBUF keeps
        // the top bit of its opcode here.
        0x80 << 24
            | u32::from(if mtbuf { opcode >> 3 } else { 0 }) << 21
            | u32::from(srsrc >> 2) << 16
            | u32::from(vdata) << 8
            | u32::from(vaddr),
    ]
}

/// A program that reads the descriptor at table entry 0 into `s[8:11]`,
/// runs `fetch` (VGPRs from 0, its address in v3) and writes its VGPRs
/// to LDS: the three positions of a vertex fetch, or four channels.
fn fetch_program(fetch: [u32; 2], channels: u8) -> Vec<u32> {
    let mut words = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000,              // s_load_dwordx4 s[8:11], s[4:5], null
        v_mov(3, 128),            // v_mov_b32 v3, 0 (the fetch's address)
        v_mov(channels + 1, 128), // v_mov_b32 v5, 0 (the LDS address)
    ];
    words.extend_from_slice(&fetch);
    for k in 0..channels {
        words.extend_from_slice(&ds_write_b32(channels + 1, k));
    }
    words.push(S_ENDPGM);
    words
}

/// The descriptor at table entry 0: a typed or raw SRV of `stride` (0:
/// none), read as `dwords` dwords.
fn fetch_tables(stride: u32, dwords: u8) -> (Vec<RootParam>, Vec<UsedResource>) {
    let layout = vec![root(RootType::Table, 0, 0)];
    let used = vec![UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride,
        dwords,
        user_data: 0,
        slot: 0,
    }];
    (layout, used)
}

#[test]
fn every_buffer_format_names_its_bits_and_numbers() {
    // Every number names a format, 1 to 77, and no number past that
    // does: the element sizes the typed fetches read follow from these.
    for format in 1..=77u8 {
        let (bits, _) = buffer_format(format).unwrap_or_else(|| panic!("format {format}"));
        let total: u32 = bits.iter().sum();
        assert_eq!(total % 8, 0, "format {format}: {bits:?} is whole bytes");
        assert!(
            total == 8 || total == 16 || total >= 32,
            "format {format}: {bits:?} is a dword or a byte/half inside one"
        );
    }
    assert_eq!(buffer_format(78), None);
    // The vertex fetch shapes: float3 is 12 bytes (16_16_16_16 comes
    // first, at 65), half2 and unorm8x4 are 4.
    let bytes = |f| -> u32 { buffer_format(f).unwrap().0.iter().sum::<u32>() / 8 };
    assert_eq!(bytes(74), 12, "32_32_32_FLOAT");
    assert_eq!(bytes(24), 4, "16_16_SNORM");
    assert_eq!(bytes(56), 4, "8_8_8_8_UNORM");
}

#[test]
fn a_typed_fetch_inside_a_dword_reads_the_pair_around_it() {
    // `tbuffer_load_format_xyz` at byte offset 33: the console reads
    // the bytes at 33, so the host reads the dword-aligned pair and
    // shifts (a dword read promised at 33 can return zeros).
    let odd = buffer(0x002, true, 74, 33, false, false, 0, 3, 8);
    let (layout, used) = fetch_tables(0, 4);
    let l = lower_with(&fetch_program(odd, 3), layout, used).unwrap();
    assert_eq!(l.unaligned_typed_loads, 1);
    // The same fetch on a dword boundary is one plain read.
    let even = buffer(0x002, true, 74, 32, false, false, 0, 3, 8);
    let (layout, used) = fetch_tables(0, 4);
    let l = lower_with(&fetch_program(even, 3), layout, used).unwrap();
    assert_eq!(l.unaligned_typed_loads, 0);
}

#[test]
fn a_four_dword_typed_fetch_inside_a_dword_loads_at_most_four_words_at_a_time() {
    // `tbuffer_load_format_xyzw` of format 77 at byte offset 33 needs
    // five aligned words; one raw load returns four and then the
    // status, so the fifth is read by a second load (the load closure
    // asserts no load asks for more than four).
    let odd = buffer(0x003, true, 77, 33, false, false, 0, 3, 8);
    let (layout, used) = fetch_tables(0, 4);
    let l = lower_with(&fetch_program(odd, 4), layout, used).unwrap();
    assert_eq!(l.unaligned_typed_loads, 1);
}

#[test]
fn a_halfword_typed_fetch_at_byte_3_reads_the_next_dword() {
    // `tbuffer_load_format_x` of 16_UINT at byte offset 3 straddles
    // two dwords: `(lo >> 24) | (hi << 8)`. VIC-BT-019 / #440.
    let odd = buffer(0x000, true, 11, 3, false, false, 0, 3, 8);
    let (layout, used) = fetch_tables(0, 4);
    let l = lower_with(&fetch_program(odd, 1), layout, used).unwrap();
    assert_eq!(l.unaligned_typed_loads, 1);
    let even = buffer(0x000, true, 11, 0, false, false, 0, 3, 8);
    let (layout, used) = fetch_tables(0, 4);
    let l = lower_with(&fetch_program(even, 1), layout, used).unwrap();
    assert_eq!(l.unaligned_typed_loads, 0);
}

#[test]
fn a_typed_fetch_fills_missing_channels_with_zero() {
    // `tbuffer_load_format_xyzw` of 32_FLOAT has one format channel;
    // the other three are 0 (not (0,0,0,1)). VIC-BT-019 / #440.
    let fetch = buffer(0x003, true, 22, 0, false, false, 0, 3, 8);
    let (layout, used) = fetch_tables(0, 4);
    let l = lower_with(&fetch_program(fetch, 4), layout, used).unwrap();
    assert!(!l.dxil.is_empty());
    // The old path wrote float 1.0 into W; a single-channel 32_FLOAT
    // fetch has no other reason to emit that constant.
    assert!(
        !l.dxil
            .windows(4)
            .any(|w| w == 1.0f32.to_bits().to_le_bytes()),
        "missing-channel fill must not be float 1.0"
    );
}

#[test]
fn typed_float_widen_keeps_snan_payload() {
    // 16_FLOAT sNaN 0x7C01 widens to 0x7F802000, not quieted
    // 0x7FC02000. VIC-BT-019 / #440.
    assert_eq!(f16_nan_widen_bits(0x7C01), 0x7F80_2000);
    assert_eq!(f16_nan_widen_bits(0xFC01), 0xFF80_2000);
    // A 16_FLOAT typed fetch lowers through the keep-sNaN path.
    let fetch = buffer(0x000, true, 13, 0, false, false, 0, 3, 8);
    let (layout, used) = fetch_tables(0, 4);
    let l = lower_with(&fetch_program(fetch, 1), layout, used).unwrap();
    assert!(l.widens_f16, "non-NaN values still use legacyF16ToF32");
}

#[test]
fn a_typed_fetch_of_no_format_is_refused() {
    // Format 0 is not a format: the fetch must stop rather than read
    // whatever the descriptor holds.
    let fetch = buffer(0x002, true, 0, 0, true, false, 0, 3, 8);
    let (layout, used) = fetch_tables(0, 4);
    let err = match lower_with(&fetch_program(fetch, 3), layout, used) {
        Err(e) => e,
        Ok(_) => panic!("a fetch of format 0 lowered"),
    };
    assert!(
        matches!(
            err,
            crate::ShaderError::Unsupported {
                what: "a buffer format",
                ..
            }
        ),
        "{err:?}"
    );
}

/// `v_mov_b32 vdst, literal`.
fn v_mov_lit(vdst: u8, lit: u32) -> [u32; 2] {
    [0x7E00_0000 | (vdst as u32) << 17 | 0x01 << 9 | 255, lit]
}

/// Seeds `v[reg]` then runs a d16 buffer load into it, writing the
/// packed result VGPRs to LDS so they stay live.
fn d16_buffer_program(opcode: u16, mtbuf: bool, format: u8, reg: u8, seed: u32) -> Vec<u32> {
    let channels = match opcode & 3 {
        0 => 1u8, // x
        1 => 1,   // xy → one packed VGPR
        2 => 2,   // xyz → two packed VGPRs
        _ => 2,   // xyzw
    };
    let mut words = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
    ];
    words.extend_from_slice(&v_mov_lit(reg, seed));
    words.push(v_mov(3, 128)); // v3 = 0 (address / index)
    words.extend_from_slice(&buffer(opcode, mtbuf, format, 0, false, !mtbuf, reg, 3, 8));
    words.push(v_mov(5, 128)); // LDS address
    for k in 0..channels {
        words.extend_from_slice(&ds_write_b32(5, reg + k));
    }
    words.push(S_ENDPGM);
    words
}

#[test]
fn packed_d16_buffer_loads_keep_an_odd_high_half() {
    // VIC-BT-015: buffer_load_format_d16_x / _xyz leave vdst[31:16] intact
    // on the trailing VGPR; a full pair (xy) replaces the whole dword.
    // MTBUF names the format; MUBUF typed d16 uses the same odd-half merge
    // (covered end-to-end on Windows by the waterfall host test).
    let (layout, used) = fetch_tables(0, 4);
    for (label, opcode, format) in [
        ("d16_x", 0x008u16, 13u8), // 16_FLOAT
        ("d16_xy", 0x009, 29),     // 16_16_FLOAT
        ("d16_xyz", 0x00a, 65),    // 16_16_16_16_UNORM
    ] {
        let want = u32::from(opcode != 0x009);
        let l = lower_with(
            &d16_buffer_program(opcode, true, format, 0, 0xDEAD_0000),
            layout.clone(),
            used.clone(),
        )
        .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(l.d16_kept_high_halves, want, "{label}");
    }
    // MUBUF buffer_load_format_d16_*: same merge on the float typed view.
    for (label, opcode) in [("d16_x", 0x080u16), ("d16_xy", 0x081), ("d16_xyz", 0x082)] {
        let want = u32::from(opcode != 0x081);
        let l = lower_with(
            &d16_buffer_program(opcode, false, 0, 0, 0xDEAD_0000),
            layout.clone(),
            used.clone(),
        )
        .unwrap_or_else(|e| panic!("mubuf {label}: {e}"));
        assert_eq!(l.d16_kept_high_halves, want, "mubuf {label}");
    }
}

#[test]
fn tbuffer_d16_integer_formats_saturate_to_16_bits() {
    // VIC-BT-015: tbuffer_load_format_d16_x of 32_UINT / 32_SINT saturates
    // to 16 bits rather than converting through f16.
    let (layout, used) = fetch_tables(0, 4);
    for format in [20u8, 21] {
        // Seed the high half so the odd-channel merge is exercised too.
        let l = lower_with(
            &d16_buffer_program(0x008, true, format, 0, 0xDEAD_0000),
            layout.clone(),
            used.clone(),
        )
        .unwrap_or_else(|e| panic!("d16_x format {format}: {e}"));
        assert_eq!(
            l.d16_kept_high_halves, 1,
            "format {format}: odd d16 channel keeps the high half"
        );
    }
}

/// `image_load` with `dmask` and optional `d16` into `v[vdata]`, address in
/// `v[vaddr]`, resource `s[srsrc:srsrc+7]`.
fn image_load(dmask: u8, d16: bool, vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    let dim = 1u32; // SQ_RSRC_IMG_2D
    [
        0xF000_0000 | u32::from(dmask) << 8 | dim << 3,
        u32::from(d16) << 31
            | (u32::from(srsrc) >> 2 & 0x1F) << 16
            | u32::from(vdata) << 8
            | u32::from(vaddr),
    ]
}

/// Seeds `v[reg]` then `image_load` d16 with `dmask` into it; writes the
/// packed result to LDS so it stays live.
fn d16_image_load_program(dmask: u8, reg: u8, seed: u32) -> Vec<u32> {
    let channels = dmask.count_ones().div_ceil(2);
    let mut words = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0
    ];
    // v_mov_b32 v[reg], seed
    words.extend_from_slice(&[0x7E00_0000 | (reg as u32) << 17 | 0x01 << 9 | 255, seed]);
    words.push(v_mov(2, 128)); // v2 = 0 (x)
    words.push(v_mov(3, 128)); // v3 = 0 (y)
    words.extend_from_slice(&image_load(dmask, true, reg, 2, 12));
    words.push(v_mov(5, 128)); // LDS address
    for k in 0..channels as u8 {
        words.extend_from_slice(&ds_write_b32(5, reg + k));
    }
    words.push(S_ENDPGM);
    words
}

#[test]
fn packed_d16_image_loads_keep_an_odd_high_half() {
    // VIC-BT-017: a one-channel d16 image_load leaves vdst[31:16] intact;
    // a full pair (dmask:0x3) replaces the whole dword.
    let (layout, used) = texture_tables();
    for (label, dmask, want) in [
        ("dmask1", 0x1u8, 1u32),
        ("dmask3", 0x3, 0),
        ("dmask7", 0x7, 1),
    ] {
        let l = lower_with(
            &d16_image_load_program(dmask, 0, 0x5A5A_0000),
            layout.clone(),
            used.clone(),
        )
        .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(l.d16_kept_high_halves, want, "{label}");
    }
}

/// Tables of 2D textures at s0 and samplers at s1.
fn texture_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
    let range = |kind, user_data, slot| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: if kind == ResourceType::Sampler { 4 } else { 8 },
        user_data,
        slot,
    };
    (
        vec![root(RootType::Table, 0, 0), root(RootType::Table, 1, 1)],
        vec![
            range(ResourceType::Srv, 0, 0),
            range(ResourceType::Sampler, 1, 1),
        ],
    )
}

#[test]
fn waterfall_loops_over_per_lane_descriptors_sample_with_each_lanes_own() {
    use crate::resource::tests::{
        WATERFALL_FETCHED_WAVE32, WATERFALL_FETCHED_WAVE64, WATERFALL_PICKED_WAVE32,
    };
    for (words, wave64) in [
        (WATERFALL_FETCHED_WAVE32, false),
        (WATERFALL_FETCHED_WAVE64, true),
        (WATERFALL_PICKED_WAVE32, false),
    ] {
        let (layout, used) = texture_tables();
        let l = lower_wave(words, layout, used, wave64).unwrap();
        // The loop's reads of the first lane became each lane's own
        // dwords: no wave operation is left.
        assert_eq!(l.nonuniform_handles, 1);
        assert!(!l.wave_ops);
    }
}

/// A wave64 descriptor picked per lane by `v_cndmask_b32` follows both
/// dwords of its mask: with `vcc_hi` cleared after the compare, lanes
/// 32-63 take the other descriptor, which the compare alone does not
/// say.
#[test]
fn a_wave64_per_lane_choice_follows_both_halves_of_its_mask() {
    use crate::resource::tests::{WATERFALL_FETCHED_WAVE64, WATERFALL_PICKED_WAVE32};
    let at = |w: &[u32], word: u32| w.iter().position(|&x| x == word).unwrap();
    // The picks of the wave32 program (up to its `s_mov_b32 s20,
    // exec_lo`), then the wave64 loop (from its `s_mov_b64 s[20:21],
    // exec`): the same registers.
    let picks = &WATERFALL_PICKED_WAVE32[..at(WATERFALL_PICKED_WAVE32, 0xBE94_037E)];
    let tail = &WATERFALL_FETCHED_WAVE64[at(WATERFALL_FETCHED_WAVE64, 0xBE94_047E)..];
    let whole = [picks, tail].concat();
    let mut cleared = whole.clone();
    // s_mov_b32 vcc_hi, 0, after v_cmp_gt_u32 vcc, 16, v0.
    cleared.insert(at(&whole, 0x7D88_0090) + 1, 0xBEEB_0380);
    let lower = |w: &[u32]| {
        let (layout, used) = texture_tables();
        lower_wave(w, layout, used, true)
    };
    let both = lower(&whole).unwrap();
    assert_eq!(both.nonuniform_handles, 1);
    // The pair (the compare's low dword, a cleared high one) is no mask
    // the structure tracks: refused, not traced as the compare's choice.
    let refused = lower(&cleared).map(|_| ()).unwrap_err();
    assert!(refused.to_string().contains("untracked mask"), "{refused}");
}

/// A select on a mask the same in every lane picks one source for all of
/// them, so it needs neither a lane index nor a wave operation.
#[test]
fn selects_on_masks_the_same_in_every_lane_need_no_lane_index() {
    let select = [
        0xD501_0001,
        258 | (259 << 9) | (2 << 18), // v_cndmask_b32 v1, v2, v3, s2 (or s[2:3])
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1: the select is needed
        S_ENDPGM,
    ];
    let cases: [(&[u32], bool); 6] = [
        (&[0xBF06_0100, 0x8502_80C1], false), // s_cmp_eq_u32; s_cselect_b32 s2, -1, 0
        (&[0xBF06_0100, 0x8582_80C1], true),  // s_cmp_eq_u32; s_cselect_b64 s[2:3], -1, 0
        (&[0xBE82_03C1], false),              // s_mov_b32 s2, -1
        (&[0xBE82_0380], false),              // s_mov_b32 s2, 0
        (&[0xBE82_04C1], true),               // s_mov_b64 s[2:3], -1
        (&[0xBE82_0480], true),               // s_mov_b64 s[2:3], 0
    ];
    for (prefix, wave64) in cases {
        let words = [prefix, &select].concat();
        let l = lower_wave(
            &words,
            (0..4).map(|i| root(RootType::Constants, i, i)).collect(),
            vec![],
            wave64,
        )
        .unwrap_or_else(|e| panic!("{words:x?} wave64={wave64}: {e}"));
        assert!(!l.wave_ops, "{words:x?} wave64={wave64}");
    }
}

/// [`texture_tables`] with a raw buffer UAV at dword 16 of the first.
pub(crate) fn texture_and_uav_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
    let (layout, mut used) = texture_tables();
    used.push(UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 16,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    });
    (layout, used)
}

#[test]
fn waterfall_loops_over_descriptors_joined_per_lane_sample_with_each_lanes_own() {
    use crate::resource::tests::{WATERFALL_JOINED_MIXED_WAVE32, WATERFALL_JOINED_WAVE32};
    for words in [WATERFALL_JOINED_WAVE32, WATERFALL_JOINED_MIXED_WAVE32] {
        let (layout, used) = texture_and_uav_tables();
        let l = lower_wave(words, layout, used, false).unwrap();
        // One image handle, from the join's per-lane heap index.
        assert_eq!(l.nonuniform_handles, 1);
        assert!(!l.wave_ops);
    }
}

#[test]
fn descriptors_joined_by_nested_control_flow_lower_to_one_uniform_handle() {
    // The view table at s0 (2D textures at entries 0 and 1, a raw UAV at
    // dword 16) and a root constant at s1: entry 0 if s1 is 0, else
    // entry 1 if it is 1, else entry 0 again, the inner if/else joined
    // before the outer one. The join's edges are a load and a join.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xBE88_03FF,
        0x0000_0092, // s_mov_b32 s8, 0x92 (an immediate sampler)
        0xBE89_0380, // s_mov_b32 s9, 0
        0xBE8A_0380, // s_mov_b32 s10, 0
        0xBE8B_0380, // s_mov_b32 s11, 0
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF84_0003, // s_cbranch_scc0 -> 0x38
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xBF82_0008, // s_branch -> 0x58
        0xBF06_8101, // 0x38: s_cmp_eq_u32 s1, 1
        0xBF84_0003, // s_cbranch_scc0 -> 0x4c
        0xF40C_0302,
        0xFA00_0020, // s_load_dwordx8 s[12:19], s[4:5], 0x20
        0xBF82_0002, // s_branch -> 0x54
        0xF40C_0302,
        0xFA00_0000, // 0x4c: s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xBE86_0301, // 0x54: s_mov_b32 s6, s1
        0x7E04_02F0, // 0x58: v_mov_b32 v2, 0.5
        0x7E06_02F0, // v_mov_b32 v3, 0.5
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        S_ENDPGM,
    ];
    let (_, mut used) = texture_and_uav_tables();
    used.retain(|u| u.user_data == 0);
    let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
    let l = lower_with(&words, layout, used).unwrap();
    // One image handle from the joins' heap index, the same in all lanes.
    assert_eq!(l.nonuniform_handles, 0);
    #[cfg(windows)]
    if let Ok(mut v) = super::super::validator::Validator::load(
        super::super::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut l.dxil.clone())
            .expect("DXIL validation failed");
    }
}

#[test]
fn level_zero_sample_with_offsets_lowers() {
    // image_sample_lz_o: the offset word first, then the coordinates.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xBE88_03FF,
        0x0000_0092, // s_mov_b32 s8, 0x92 (an immediate sampler)
        0xBE89_0380, // s_mov_b32 s9, 0
        0xBE8A_0380, // s_mov_b32 s10, 0
        0xBE8B_0380, // s_mov_b32 s11, 0
        0x7E04_02FF,
        0x0000_3F01, // v_mov_b32 v2, 0x3f01: offsets x 1, y -1
        0x7E06_02F0, // v_mov_b32 v3, 0.5
        0x7E08_02F0, // v_mov_b32 v4, 0.5
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF0DC_0F08,
        0x0043_0C02, // image_sample_lz_o v[12:15], v[2:4], s[12:19], s[8:11] dmask:0xf dim:2D
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        S_ENDPGM,
    ];
    let (_, mut used) = texture_and_uav_tables();
    used.retain(|u| u.user_data == 0);
    let layout = vec![root(RootType::Table, 0, 0)];
    #[cfg_attr(not(windows), allow(unused_variables))]
    let l = lower_with(&words, layout, used).unwrap();
    #[cfg(windows)]
    if let Ok(mut v) = super::super::validator::Validator::load(
        super::super::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut l.dxil.clone())
            .expect("DXIL validation failed");
    }
}

/// VIC-BT-009: the same immediate sampler words used non-comparison first
/// then comparison (or the reverse) allocate two static slots so each
/// annotateHandle matches `dx.resources` sampler kind.
#[test]
fn immediate_sampler_comparison_and_plain_get_distinct_slots() {
    let setup = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xBE88_03FF,
        0x0000_0092, // s_mov_b32 s8, 0x92 (an immediate sampler)
        0xBE89_0380, // s_mov_b32 s9, 0
        0xBE8A_0380, // s_mov_b32 s10, 0
        0xBE8B_0380, // s_mov_b32 s11, 0
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ];
    // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:2D
    let sample_lz = [0x7E04_02F0, 0x7E06_02F0, 0xF09C_0F08, 0x0043_0C02];
    // image_sample_c_lz v12, v[2:4], … dmask:0x1 (z-ref then coords)
    let sample_c_lz = [
        0x7E04_02F0,
        0x7E06_02F0,
        0x7E08_02F0,
        0xF0BC_0108,
        0x0043_0C02,
    ];
    let store = [
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        S_ENDPGM,
    ];
    let lower_mix = |plain_first: bool| {
        let mut words = setup.to_vec();
        if plain_first {
            words.extend_from_slice(&sample_lz);
            words.extend_from_slice(&sample_c_lz);
        } else {
            words.extend_from_slice(&sample_c_lz);
            words.extend_from_slice(&sample_lz);
        }
        words.extend_from_slice(&store);
        let (_, mut used) = texture_and_uav_tables();
        used.retain(|u| u.user_data == 0);
        let layout = vec![root(RootType::Table, 0, 0)];
        lower_with(&words, layout, used).unwrap()
    };
    for plain_first in [true, false] {
        let l = lower_mix(plain_first);
        assert_eq!(
            l.samplers.len(),
            2,
            "plain_first={plain_first}: need distinct static slots"
        );
        assert_eq!(l.samplers[0], l.samplers[1]);
        assert_eq!(
            l.cmp_samplers.count_ones(),
            1,
            "plain_first={plain_first}: one comparison slot"
        );
        let cmp_slot = l.cmp_samplers.trailing_zeros();
        let plain_slot = if cmp_slot == 0 { 1 } else { 0 };
        assert_eq!(
            l.cmp_samplers >> plain_slot & 1,
            0,
            "plain_first={plain_first}"
        );
        #[cfg(windows)]
        if let Ok(mut v) = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut l.dxil.clone())
                .unwrap_or_else(|e| panic!("plain_first={plain_first}: {e}"));
        }
    }
    // Same-kind reuse still shares one slot.
    let mut words = setup.to_vec();
    words.extend_from_slice(&sample_lz);
    words.extend_from_slice(&sample_lz);
    words.extend_from_slice(&store);
    let (_, mut used) = texture_and_uav_tables();
    used.retain(|u| u.user_data == 0);
    let l = lower_with(&words, vec![root(RootType::Table, 0, 0)], used).unwrap();
    assert_eq!(l.samplers.len(), 1);
    assert_eq!(l.cmp_samplers, 0);
}

/// A compute program that sets `defined` VGPRs from v2 on (`offsets` first:
/// the texel offset word) and samples 2D with the MIMG opcode `op`.
fn gradient_sample(op: u32, a16: bool, defined: usize, offsets: bool) -> Vec<u32> {
    let mut words = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xBE88_03FF,
        0x0000_0092, // s_mov_b32 s8, 0x92 (an immediate sampler)
        0xBE89_0380, // s_mov_b32 s9, 0
        0xBE8A_0380, // s_mov_b32 s10, 0
        0xBE8B_0380, // s_mov_b32 s11, 0
    ];
    for r in 0..defined as u32 {
        if r == 0 && offsets {
            words.extend([0x7E04_02FF, 0x0000_3F01]); // v_mov_b32 v2, 0x3f01: x 1, y -1
        } else {
            words.push(0x7E00_02F0 | (2 + r) << 17); // v_mov_b32 v(2+r), 0.5
        }
    }
    words.extend([
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF000_0F08 | (op & 0x7F) << 18 | op >> 7,
        0x0043_0C02 | u32::from(a16) << 30, // v[12:15], v2.., s[12:19], s[8:11]
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        S_ENDPGM,
    ]);
    words
}

fn lower_gradient_sample(
    op: u32,
    a16: bool,
    defined: usize,
    offsets: bool,
) -> Result<Lowered, crate::ShaderError> {
    let words = gradient_sample(op, a16, defined, offsets);
    let (_, mut used) = texture_and_uav_tables();
    used.retain(|u| u.user_data == 0);
    let layout = vec![root(RootType::Table, 0, 0)];
    lower_with(&words, layout, used)
}

/// The address-list indices the sample read, each once.
fn address_reads(l: &Lowered) -> Vec<usize> {
    let mut reads = l.image_address_reads.clone();
    reads.sort_unstable();
    reads.dedup();
    reads
}

#[test]
fn gradient_sample_reads_exactly_the_packed_address_dwords() {
    // 2D image_sample_d (0x22) and image_sample_d_g16 (0xa2).
    for (op, a16, dwords) in [
        (0x22, false, 6),
        (0x22, true, 5),
        (0xA2, false, 4),
        (0xA2, true, 3),
    ] {
        let l = lower_gradient_sample(op, a16, dwords, false)
            .unwrap_or_else(|e| panic!("op {op:#x} a16 {a16}: {e:?}"));
        let want: Vec<usize> = (0..dwords).collect();
        assert_eq!(address_reads(&l), want, "op {op:#x} a16 {a16}");
    }
}

#[test]
fn offset_gradient_samples_shift_every_address_by_the_offset_word() {
    // image_sample_d_o (0x32), image_sample_d_o_g16 (0xb2),
    // image_sample_d_cl_o (0x33), image_sample_d_cl_o_g16 (0xb3).
    for (op, a16, dwords) in [
        (0x32, false, 7),
        (0x32, true, 6),
        (0xB2, false, 5),
        (0xB2, true, 4),
        (0x33, false, 8),
        (0x33, true, 7),
        (0xB3, false, 6),
        (0xB3, true, 5),
    ] {
        let l = lower_gradient_sample(op, a16, dwords, true)
            .unwrap_or_else(|e| panic!("op {op:#x} a16 {a16}: {e:?}"));
        let want: Vec<usize> = (0..dwords).collect();
        assert_eq!(address_reads(&l), want, "op {op:#x} a16 {a16}");
    }
}

#[test]
fn gradient_halves_widen_only_for_g16_and_coordinate_halves_only_for_a16() {
    let plain = lower_gradient_sample(0x22, false, 6, false).unwrap();
    assert!(!plain.widens_f16);
    let a16 = lower_gradient_sample(0x22, true, 5, false).unwrap();
    assert!(a16.widens_f16);
    let g16 = lower_gradient_sample(0xA2, false, 4, false).unwrap();
    assert!(g16.widens_f16);
    let both = lower_gradient_sample(0xA2, true, 3, false).unwrap();
    assert!(both.widens_f16);
}

#[test]
fn gradient_sample_offsets_validate_as_dxil() {
    for (op, a16, dwords) in [(0x32, false, 7), (0xB2, true, 4)] {
        #[cfg_attr(not(windows), allow(unused_variables))]
        let l = lower_gradient_sample(op, a16, dwords, true).unwrap();
        #[cfg(windows)]
        if let Ok(mut v) = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut l.dxil.clone())
                .expect("DXIL validation failed");
        }
    }
}

#[test]
fn biased_sample_reads_whole_bias_vgpr_and_packed_a16_coords() {
    // 2D image_sample_b (0x25) and image_sample_b_o (0x35): bias (and
    // the offset word) stay whole VGPRs; a16 only packs the coordinates.
    for (op, a16, dwords, offsets) in [
        (0x25, false, 3, false),
        (0x25, true, 2, false),
        (0x35, false, 4, true),
        (0x35, true, 3, true),
        (0x26, false, 4, false),
        (0x26, true, 3, false),
        (0x36, false, 5, true),
        (0x36, true, 4, true),
    ] {
        let l = lower_gradient_sample(op, a16, dwords, offsets)
            .unwrap_or_else(|e| panic!("op {op:#x} a16 {a16}: {e:?}"));
        let want: Vec<usize> = (0..dwords).collect();
        assert_eq!(address_reads(&l), want, "op {op:#x} a16 {a16}");
        assert_eq!(l.widens_f16, a16, "op {op:#x} a16 {a16} f16 widen");
    }
}

#[test]
fn biased_a16_sample_validates_as_dxil() {
    for (op, dwords, offsets) in [(0x25, 2, false), (0x35, 3, true)] {
        #[cfg_attr(not(windows), allow(unused_variables))]
        let l = lower_gradient_sample(op, true, dwords, offsets).unwrap();
        #[cfg(windows)]
        if let Ok(mut v) = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut l.dxil.clone())
                .expect("DXIL validation failed");
        }
    }
}

/// A compute program that loads the table of views at s0, sets the
/// listed VGPRs to literals, runs one MIMG instruction on the first view
/// (2D unless `dim` says otherwise) with the sampler `0x92` and stores
/// the first result dword. The address starts at v2.
fn image_words(op: u32, dmask: u32, dim: u32, a16: bool, addr: &[u32]) -> Vec<u32> {
    let mut words = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xBE88_03FF,
        0x0000_0092, // s_mov_b32 s8, 0x92 (an immediate sampler)
        0xBE89_0380, // s_mov_b32 s9, 0
        0xBE8A_0380, // s_mov_b32 s10, 0
        0xBE8B_0380, // s_mov_b32 s11, 0
    ];
    for (k, &value) in addr.iter().enumerate() {
        // v_mov_b32 v(2 + k), literal
        words.extend([0x7E00_02FF | (2 + k as u32) << 17, value]);
    }
    words.extend([
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF000_0000 | op << 18 | dmask << 8 | dim << 3,
        0x0043_0C02 | u32::from(a16) << 30, // v[12:15], v[2:], s[12:19], s[8:11]
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        S_ENDPGM,
    ]);
    words
}

/// [`image_words`] lowered; on Windows with `dxil.dll`, also validated.
fn lower_image(words: &[u32]) -> Result<Lowered, crate::ShaderError> {
    let (_, mut used) = texture_and_uav_tables();
    used.retain(|u| u.user_data == 0);
    let layout = vec![root(RootType::Table, 0, 0)];
    let l = lower_with(words, layout, used)?;
    #[cfg(windows)]
    if let Ok(mut v) = super::super::validator::Validator::load(
        super::super::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut l.dxil.clone())
            .expect("DXIL validation failed");
    }
    Ok(l)
}

/// The integer-constant arguments of each call of `dx.op.<function>`.
fn calls<'a>(l: &'a Lowered, function: &str) -> Vec<&'a [Option<i32>]> {
    let name = format!("dx.op.{function}");
    l.dx_ops
        .iter()
        .filter(|(n, _)| *n == name)
        .map(|(_, args)| args.as_slice())
        .collect()
}

fn called(l: &Lowered, function: &str) -> bool {
    !calls(l, function).is_empty()
}

fn refused(words: &[u32]) -> String {
    match lower_image(words) {
        Err(crate::ShaderError::Unsupported { what, .. }) => what.to_string(),
        Err(e) => panic!("{e}"),
        Ok(_) => panic!("lowered"),
    }
}

const F_HALF: u32 = 0x3F00_0000;
const F_ONE: u32 = 0x3F80_0000;

#[test]
fn comparison_gathers_lower_to_gather_cmp() {
    // image_gather4_c_lz: the reference, then the coordinates.
    let words = image_words(0x4F, 0x1, 1, false, &[F_ONE, F_HALF, F_HALF]);
    let l = lower_image(&words).unwrap();
    assert!(called(&l, "textureGatherCmp.f32"));
    assert!(!called(&l, "textureGather.f32"));
    // Any one channel, through the same instruction.
    for dmask in [0x2, 0x4, 0x8] {
        let words = image_words(0x4F, dmask, 1, false, &[F_ONE, F_HALF, F_HALF]);
        assert!(called(
            &lower_image(&words).unwrap(),
            "textureGatherCmp.f32"
        ));
    }
    // 2D array: the slice is a third coordinate.
    let words = image_words(0x4F, 0x1, 5, false, &[F_ONE, F_HALF, F_HALF, 0]);
    assert!(called(
        &lower_image(&words).unwrap(),
        "textureGatherCmp.f32"
    ));
}

#[test]
fn comparison_gathers_take_programmable_offsets_after_the_offset_word() {
    // image_gather4_c_lz_o: offsets x 2, y -1, then the reference.
    let words = image_words(0x5F, 0x1, 1, false, &[0x3F02, F_ONE, F_HALF, F_HALF]);
    let l = lower_image(&words).unwrap();
    assert!(called(&l, "textureGatherCmp.f32"));
    // Any other offset word lowers the same way, in range or not
    // (the host takes -8..7 and the title's are 6-bit).
    let words = image_words(0x5F, 0x2, 1, false, &[0x3F3F, F_ONE, F_HALF, F_HALF]);
    assert!(called(
        &lower_image(&words).unwrap(),
        "textureGatherCmp.f32"
    ));
}

#[test]
fn comparison_gathers_of_unsupported_forms_are_refused() {
    // Several channels, a cube with an offset and a 3D view.
    let words = image_words(0x4F, 0x3, 1, false, &[F_ONE, F_HALF, F_HALF]);
    assert_eq!(
        refused(&words),
        "a comparison sample of 16-bit coordinates or several channels"
    );
    let words = image_words(
        0x5F,
        0x1,
        3,
        false,
        &[0x3F02, F_ONE, F_HALF, F_HALF, F_HALF],
    );
    assert_eq!(refused(&words), "a texel offset on a cube");
    let words = image_words(0x4F, 0x1, 2, false, &[F_ONE, F_HALF, F_HALF, F_HALF]);
    assert_eq!(refused(&words), "a gather of a view other than 2D or cube");
}

#[test]
fn sample_offset_variants_lower_with_the_offset_word_first() {
    // Offset words (x 1, y -1) lead every address.
    let off = 0x3F01;
    let cases: [(u32, u32, &str, &[u32], &str); 5] = [
        // image_sample_l_o: offset, coordinates, level.
        (
            0x34,
            0xF,
            "l_o",
            &[off, F_HALF, F_HALF, F_ONE],
            "sampleLevel.f32",
        ),
        // image_sample_c_lz_o: offset, reference, coordinates.
        (
            0x3F,
            0x1,
            "c_lz_o",
            &[off, F_ONE, F_HALF, F_HALF],
            "sampleCmpLevelZero.f32",
        ),
        // image_sample_lz_o: offset, coordinates.
        (0x37, 0xF, "lz_o", &[off, F_HALF, F_HALF], "sampleLevel.f32"),
        // image_sample_d_o: offset, x gradients, y gradients, coordinates.
        (
            0x32,
            0xF,
            "d_o",
            &[off, 0, 0, 0, 0, F_HALF, F_HALF],
            "sampleGrad.f32",
        ),
        // image_sample_o: offset, coordinates (the level from the quad).
        (0x30, 0xF, "o", &[off, F_HALF, F_HALF], "sampleGrad.f32"),
    ];
    for (op, dmask, label, addr, function) in cases {
        let words = image_words(op, dmask, 1, false, addr);
        let l = lower_image(&words).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert!(called(&l, function), "{label}");
    }
}

#[test]
fn a16_leaves_the_offset_and_gradients_whole_and_packs_the_coordinates() {
    // image_sample_d_o with a16: offset, four f32 gradients, then one
    // dword of two f16 coordinates.
    let packed = 0x3800_3800; // (0.5, 0.5) as f16
    let words = image_words(0x32, 0xF, 1, true, &[0x3F01, 0, 0, 0, 0, packed]);
    let l = lower_image(&words).unwrap();
    assert_eq!(calls(&l, "sampleGrad.f32")[0][7..9], [Some(1), Some(-1)]);
    // image_sample_l_o: offset, then coordinates and level packed in
    // pairs, so three f16 values take two dwords.
    let words = image_words(0x34, 0xF, 1, true, &[0x3F01, packed, 0x3C00]);
    let l = lower_image(&words).unwrap();
    assert_eq!(calls(&l, "sampleLevel.f32")[0][7..9], [Some(1), Some(-1)]);
}

#[test]
fn sample_offsets_must_be_constants_in_the_dxil_range() {
    let at = |offset: u32| image_words(0x37, 0xF, 1, false, &[offset, F_HALF, F_HALF]);
    for (offset, ok) in [
        (0x0007, true),
        (0x3F38, true),
        (0x0008, false),
        (0x3F37, false),
    ] {
        let r = lower_image(&at(offset));
        assert_eq!(r.is_ok(), ok, "{offset:#x}");
    }
    // x from the thread id: not a constant.
    let mut words = at(0);
    let at_mov = words.iter().position(|&w| w == 0x7E04_02FF).unwrap();
    words.splice(at_mov..at_mov + 2, [0x7E04_0300]); // v_mov_b32 v2, v0
    assert_eq!(refused(&words), "a sample offset that is not constant");
    // No texel offsets on cube samples.
    let words = image_words(0x37, 0xF, 3, false, &[0x0001, F_HALF, F_HALF, F_HALF]);
    assert_eq!(refused(&words), "a texel offset on a cube");
}

/// A pixel program: the view table at s0, literals in v2 on, one MIMG
/// instruction, and the result exported to the target.
fn lower_pixel_image(op: u32, dmask: u32, addr: &[u32]) -> Result<Lowered, crate::ShaderError> {
    let mut words = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xBE88_03FF,
        0x0000_0092, // s_mov_b32 s8, 0x92 (an immediate sampler)
        0xBE89_0380, // s_mov_b32 s9, 0
        0xBE8A_0380, // s_mov_b32 s10, 0
        0xBE8B_0380, // s_mov_b32 s11, 0
    ];
    for (k, &value) in addr.iter().enumerate() {
        words.extend([0x7E00_02FF | (2 + k as u32) << 17, value]);
    }
    words.extend([
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF000_0000 | op << 18 | dmask << 8 | 1 << 3,
        0x0043_0C02, // v[12:15], v[2:], s[12:19], s[8:11]
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xF800_080F,
        0x0F0E_0D0C, // exp mrt0 v12, v13, v14, v15 done
        S_ENDPGM,
    ]);
    let (_, mut used) = texture_and_uav_tables();
    used.retain(|u| u.user_data == 0);
    let layout = vec![root(RootType::Table, 0, 0)];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 32,
        layout,
        used,
        io: StageIo::Ps {
            inputs: vec![],
            input_addr: 0,
            rsrc2: Default::default(),
            db_shader_control: 0,
            patches: vec![],
        },
        code,
    };
    let p = ir::build(&native.code, false, 32).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native).unwrap();
    let target = crate::emit::container::SigElement {
        name: "SV_Target".into(),
        index: 0,
        kind: 16,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0,
        half: false,
    };
    let l = lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![target],
        prims: vec![],
        mesh_stride: 0,
    })?;
    #[cfg(windows)]
    if let Ok(mut v) = super::super::validator::Validator::load(
        super::super::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut l.dxil.clone())
            .expect("DXIL validation failed");
    }
    Ok(l)
}

#[test]
fn implicit_level_sample_offsets_lower_in_pixel_shaders() {
    let off = 0x3F01;
    // image_sample_b_o: offset, bias, coordinates.
    let l = lower_pixel_image(0x35, 0xF, &[off, 0, F_HALF, F_HALF]).unwrap();
    assert!(called(&l, "sampleBias.f32"));
    assert_eq!(calls(&l, "sampleBias.f32")[0][7..9], [Some(1), Some(-1)]);
    // image_sample_c_o: offset, reference, coordinates.
    let l = lower_pixel_image(0x38, 0x1, &[off, F_ONE, F_HALF, F_HALF]).unwrap();
    assert!(called(&l, "sampleCmp.f32"));
    // image_sample_o: offset, coordinates.
    let l = lower_pixel_image(0x30, 0xF, &[off, F_HALF, F_HALF]).unwrap();
    assert!(called(&l, "sample.f32"));
}

#[test]
fn scalar_bit_tests_lower() {
    // Each tests one bit of a constant into SCC and branches on it (only
    // code with an effect needs its operands).
    let tail = [
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE84_0380, // s_mov_b32 s4, 0
        S_ENDPGM,
    ];
    let bit_tests: [&[u32]; 4] = [
        &[0xBE81_0387, 0xBF0C_8301], // s_mov_b32 s1, 7; s_bitcmp0_b32 s1, 3
        &[0xBE81_0387, 0xBF0D_8301], // s_mov_b32 s1, 7; s_bitcmp1_b32 s1, 3
        &[
            0xBE80_0387, // s_mov_b32 s0, 7
            0xBE81_0380, // s_mov_b32 s1, 0
            0xBF0E_8300, // s_bitcmp0_b64 s[0:1], 3
        ],
        &[
            0xBE80_0387, // s_mov_b32 s0, 7
            0xBE81_0380, // s_mov_b32 s1, 0
            0xBF0F_8300, // s_bitcmp1_b64 s[0:1], 3
        ],
    ];
    for head in bit_tests {
        let words: Vec<u32> = head.iter().chain(&tail).copied().collect();
        lowers(&words, 0).unwrap();
    }
}

#[test]
fn lds_subdword_reads_and_writes_lower() {
    // Allocate 128 dwords (512 bytes) of LDS via COMPUTE_PGM_RSRC2[23:15] = 1.
    let ds = |op: u32, off: u32, addr: u32, data0: u32, dst: u32| -> [u32; 2] {
        [
            0xD800_0000 | (op << 18) | (off & 0xFFFF),
            (dst << 24) | (data0 << 8) | addr,
        ]
    };
    let mut words = vec![
        0x7E00_0280, // v_mov_b32 v0, 0  (base address)
        0x7E02_0281, // v_mov_b32 v1, 1  (value)
    ];
    // Writes at odd byte and 2-byte offsets, then all sub-dword read forms
    // (including _d16 / _d16_hi which preserve the untouched half of vdst).
    for inst in [
        ds(0x01e, 1, 0, 1, 0), // ds_write_b8 v0, v1 offset:1
        ds(0x01f, 2, 0, 1, 0), // ds_write_b16 v0, v1 offset:2
        ds(0x0a0, 3, 0, 1, 0), // ds_write_b8_d16_hi v0, v1 offset:3
        ds(0x0a1, 4, 0, 1, 0), // ds_write_b16_d16_hi v0, v1 offset:4
        ds(0x039, 1, 0, 0, 2), // ds_read_i8 v2, v0 offset:1
        ds(0x03a, 1, 0, 0, 2), // ds_read_u8 v2, v0 offset:1
        ds(0x03b, 2, 0, 0, 2), // ds_read_i16 v2, v0 offset:2
        ds(0x03c, 2, 0, 0, 2), // ds_read_u16 v2, v0 offset:2
        ds(0x0a2, 1, 0, 0, 2), // ds_read_u8_d16 v2, v0 offset:1
        ds(0x0a3, 1, 0, 0, 2), // ds_read_u8_d16_hi v2, v0 offset:1
        ds(0x0a4, 1, 0, 0, 2), // ds_read_i8_d16 v2, v0 offset:1
        ds(0x0a5, 1, 0, 0, 2), // ds_read_i8_d16_hi v2, v0 offset:1
        ds(0x0a6, 2, 0, 0, 2), // ds_read_u16_d16 v2, v0 offset:2
        ds(0x0a7, 2, 0, 0, 2), // ds_read_u16_d16_hi v2, v0 offset:2
        ds(0x00d, 8, 0, 2, 0), // ds_write_b32 v0, v2 offset:8 (keep reads live)
    ] {
        words.extend_from_slice(&inst);
    }
    words.push(S_ENDPGM);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 15,
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, false, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
    .unwrap();
}

#[test]
fn a_missing_value_stays_missing_through_two_live_joins() {
    let mut words = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
        0xBF85_0001, // skip the first overwrite
        0xBE86_0380, // s_mov_b32 s6, 0
        0xBF06_8101, // s_cmp_eq_u32 s1, 1 (s1 = group id Y)
        0xBF85_0001, // skip the second overwrite
        0xBE86_0381, // s_mov_b32 s6, 1
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001, // use the result to branch
        0xBE83_0380,
        S_ENDPGM,
    ];
    let group_ids_xy = 3 << 7;
    // Compare to a direct read so future timestamp support works in both cases.
    let refusal = |code: &[u32]| match lowers(code, group_ids_xy) {
        Ok(()) => None,
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(error) => panic!("{error}"),
    };
    let direct = [&words[..2], &words[8..]].concat();
    assert_eq!(refusal(&words), refusal(&direct));
    words[8] = 0xBF07_8000; // read the group id instead; the timestamp is dead
    lowers(&words, group_ids_xy).expect("dead timestamp");
}

/// A constant lane is wrapped by its value: lane 40 is lane 8 in wave32
/// and 40 in wave64, whatever its place in the constant pool.
#[test]
fn a_constant_lane_wraps_by_its_value_not_its_pool_place() {
    let mut m = super::Module::default();
    // Constants interned first, so lane 40's place in the pool is a
    // small index, not 40 (nor 8 or 40 once wrapped).
    for k in 100..103 {
        m.i32(k);
    }
    let lane = m.i32(40);
    let wave32 = super::wave_lane(&mut m, 0, lane, false);
    let wave64 = super::wave_lane(&mut m, 0, lane, true);
    assert_eq!(m.int_value(wave32), Some(8));
    assert_eq!(m.int_value(wave64), Some(40));
}

#[test]
fn v_readlane_and_writelane_mask_lane_and_ignore_exec() {
    // v_writelane_b32 under narrowed EXEC ignores EXEC and writes target lane.
    // v_readlane_b32 with lane > 31 masks to lane & 31.
    let words = [
        0xBE80_0380, // s_mov_b32 s0, 0
        0xBEFE_0380, // s_mov_b32 exec_lo, 0
        0xD761_0000,
        0x0001_0000, // v_writelane_b32 v0, s0, 0 (VOP3 0x361)
        0xD760_0000,
        0x0001_5100, // v_readlane_b32 s0, v0, 40 (lane 40 inline const 168) (VOP3 0x360)
        S_ENDPGM,
    ];
    let layout = || vec![];
    let used = || vec![];
    #[cfg_attr(not(windows), allow(unused_variables))]
    let l = lower_with(&words, layout(), used()).unwrap();
    #[cfg(windows)]
    if let Ok(mut v) = crate::emit::validator::Validator::load(
        crate::emit::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut l.dxil.clone()).unwrap();
    }
}

#[test]
fn buffer_formats_by_number() {
    // The numbers the disassembler names 8_8_8_8_UNORM, 16_16_SNORM,
    // 2_10_10_10_UNORM, 16_FLOAT, 32_32_32_32_FLOAT.
    assert_eq!(buffer_format(56), Some((&[8, 8, 8, 8][..], Numeric::Unorm)));
    assert_eq!(buffer_format(24), Some((&[16, 16][..], Numeric::Snorm)));
    assert_eq!(
        buffer_format(50),
        Some((&[10, 10, 10, 2][..], Numeric::Unorm))
    );
    assert_eq!(buffer_format(13), Some((&[16][..], Numeric::Float)));
    assert_eq!(
        buffer_format(77),
        Some((&[32, 32, 32, 32][..], Numeric::Float))
    );
    // Invalid, past the table.
    assert_eq!(buffer_format(0), None);
    assert_eq!(buffer_format(78), None);
}

#[test]
fn packed_buffer_formats_by_number() {
    // LLVM's `UfmtGFX10`: 10_11_11 is 30-36, 11_11_10 37-43,
    // 10_10_10_2 44-49. Channels from the lowest bits: R11G11B10_FLOAT
    // is 10_11_11 (Mesa), 10_10_10_2 has the 2-bit channel first.
    assert_eq!(buffer_format(30), Some((&[11, 11, 10][..], Numeric::Unorm)));
    assert_eq!(buffer_format(36), Some((&[11, 11, 10][..], Numeric::Float)));
    assert_eq!(buffer_format(37), Some((&[10, 11, 11][..], Numeric::Unorm)));
    assert_eq!(buffer_format(43), Some((&[10, 11, 11][..], Numeric::Float)));
    assert_eq!(
        buffer_format(44),
        Some((&[2, 10, 10, 10][..], Numeric::Unorm))
    );
    assert_eq!(
        buffer_format(45),
        Some((&[2, 10, 10, 10][..], Numeric::Snorm))
    );
    assert_eq!(
        buffer_format(46),
        Some((&[2, 10, 10, 10][..], Numeric::Uscaled))
    );
    assert_eq!(
        buffer_format(47),
        Some((&[2, 10, 10, 10][..], Numeric::Sscaled))
    );
    assert_eq!(
        buffer_format(48),
        Some((&[2, 10, 10, 10][..], Numeric::Uint))
    );
    assert_eq!(
        buffer_format(49),
        Some((&[2, 10, 10, 10][..], Numeric::Sint))
    );
}

/// A half's value (sign, 5-bit exponent biased 15, 10-bit mantissa), as
/// `legacyF16ToF32` widens it.
fn half(h: u32) -> f32 {
    let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 31, h & 1023);
    let v = match e {
        0 => m as f32 * 2f32.powi(-24),
        31 if m == 0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1024 + m) as f32 * 2f32.powi(e as i32 - 25),
    };
    if s == 1 { -v } else { v }
}

#[test]
fn small_floats_widen_as_halves() {
    // Unsigned 11- and 10-bit floats: a 5-bit exponent biased 15 over a
    // 6- or 5-bit mantissa, no sign; values from the D3D/GL spec.
    let f11 = |x: u32| half(x << half_shift(11));
    let f10 = |x: u32| half(x << half_shift(10));
    assert_eq!(f11(0), 0.0);
    assert_eq!(f11(0x001), 2f32.powi(-20)); // smallest denormal
    assert_eq!(f11(0x03F), 63.0 * 2f32.powi(-20)); // largest denormal
    assert_eq!(f11(0x040), 2f32.powi(-14)); // smallest normal
    assert_eq!(f11(0x3C0), 1.0);
    assert_eq!(f11(0x3E0), 1.5);
    assert_eq!(f11(0x7BF), 65024.0); // largest
    assert_eq!(f11(0x7C0), f32::INFINITY);
    assert!(f11(0x7C1).is_nan());
    assert_eq!(f10(0x001), 2f32.powi(-19));
    assert_eq!(f10(0x020), 2f32.powi(-14));
    assert_eq!(f10(0x1E0), 1.0);
    assert_eq!(f10(0x3DF), 64512.0);
    assert_eq!(f10(0x3E0), f32::INFINITY);
    assert!(f10(0x3FF).is_nan());
    // A 16-bit float is a half already, sign included.
    assert_eq!(half(0xBC00 << half_shift(16)), -1.0);
}

#[test]
fn constant_64_bit_compares_lower_without_int64() {
    for (vopc, branch) in [
        (0x7DC0_0080u32, 0xBF86_0001u32), // v_cmp_f_u64 vcc, ...; s_cbranch_vccnz
        (0x7D4E_0080, 0xBF86_0001),       // v_cmp_t_i64 vcc, ...; s_cbranch_vccnz
        (0x7DE0_0080, 0xBF89_0001),       // v_cmpx_f_u64 exec, ...; s_cbranch_execnz
        (0x7D6E_0080, 0xBF89_0001),       // v_cmpx_t_i64 exec, ...; s_cbranch_execnz
    ] {
        let words = [
            0x7E00_0280u32, // v_mov_b32 v0, 0
            0x7E02_0280,    // v_mov_b32 v1, 0
            vopc,
            branch,
            0xBE80_0380, // s_mov_b32 s0, 0
            S_ENDPGM,
        ];
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, false, 2).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let emitted = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        })
        .unwrap();
        let sfi0 = crate::dxbc::Container::parse(&emitted.dxil)
            .unwrap()
            .part(crate::dxbc::FourCC::SFI0)
            .unwrap()
            .data;
        let features = u64::from_le_bytes(sfi0.try_into().unwrap());
        assert_eq!(features & 0x8000, 0, "unexpected INT64 feature bit");
    }
}

/// `LaneExit::Branch` with `taken == not_taken` must lower as a jump.
/// Calling `edge` twice for the same `(from, to)` would overwrite the
/// first critical-edge block in the map and leave `fill_phis` missing a
/// CondBr predecessor (VIC-BT-007).
#[test]
fn identical_branch_arms_lower_as_a_jump() {
    let words = [
        0xBF06_0100u32, // s_cmp_eq_u32 s0, s1
        0xBF85_0001,    // s_cbranch_scc1 -> end
        0xBE82_0300,    // s_mov_b32 s2, s0
        S_ENDPGM,
    ];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 0,
            threads: [1, 1, 1],
        }),
        code: code.clone(),
    };
    let p = ir::build(&code, false, 256).unwrap();
    let mut st = structure::structure(&p).unwrap();
    let mut collapsed = false;
    for exit in &mut st.exits {
        if let structure::LaneExit::Branch {
            taken, not_taken, ..
        } = exit
        {
            *not_taken = *taken;
            collapsed = true;
        }
    }
    assert!(
        collapsed,
        "fixture must expose a LaneExit::Branch to force identical arms"
    );
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let lowered = lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
    .expect("degenerate Branch arms must lower without PHI/edge map errors");
    assert!(!lowered.dxil.is_empty());
}

/// Binary32 FMA with a round-to-odd sticky bit when the fused result is
/// not exact — the same sequence `fmad_round_to_odd` emits in IR.
fn f32_fma_round_to_odd(a: f32, b: f32, c: f32) -> f32 {
    let r = a.mul_add(b, c);
    let err = a.mul_add(b, -r) + c;
    let mut bits = r.to_bits();
    let sticky = err.to_bits() & 0x7FFF_FFFF != 0;
    if sticky && bits & 0x7FFF_FFFF < 0x7F80_0000 && bits & 1 == 0 {
        let abs = bits & 0x7FFF_FFFF;
        if abs == 0 {
            bits = if err.to_bits() & 0x8000_0000 == 0 {
                0x1
            } else {
                0x8000_0001
            };
        } else {
            // Same sign of residual and `r` → away from zero (+1 to bits).
            let same_sign = (err.to_bits() ^ bits) & 0x8000_0000 == 0;
            bits = if same_sign { bits + 1 } else { bits - 1 };
        }
    }
    f32::from_bits(bits)
}

/// Nearest-even binary16 of an f64, by search over the half values so it
/// does not share the lowering's bit arithmetic.
fn f16_nearest_f64(x: f64) -> u16 {
    if x.is_nan() {
        return 0x7E00;
    }
    let sign = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    if a >= 65520.0 {
        return sign | 0x7C00;
    }
    let f16_to_f64 = |h: u16| -> f64 {
        let exp = (h >> 10) & 0x1F;
        let mant = h & 0x3FF;
        match exp {
            0 => {
                if mant == 0 {
                    0.0
                } else {
                    (mant as f64) * f64::from(2f32.powi(-24))
                }
            }
            31 => f64::INFINITY,
            e => (1.0 + (mant as f64) / 1024.0) * 2f64.powi(e as i32 - 15),
        }
    };
    let (mut lo, mut hi) = (0u16, 0x7BFF);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if f16_to_f64(mid) <= a {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let below = f16_to_f64(lo);
    let above = if lo == 0x7BFF {
        65536.0
    } else {
        f16_to_f64(lo + 1)
    };
    let pick = if a - below < above - a || (a - below == above - a && lo % 2 == 0) {
        lo
    } else {
        lo + 1
    };
    sign | pick
}

/// Single-round `a*b + c` to binary16 (f64 FMA is exact enough here:
/// every finite f32 product fits in f64, and the addend is f32).
fn fma_f16_single_round(a: f32, b: f32, c: f32) -> u16 {
    let exact = f64::from(a).mul_add(f64::from(b), f64::from(c));
    f16_nearest_f64(exact)
}

/// Nearest-even binary16 of an f32 (via f64, which is exact for f32).
fn f16_nearest_f32(x: f32) -> u16 {
    f16_nearest_f64(f64::from(x))
}

#[test]
fn fma_mix_f16_round_to_odd_matches_single_round() {
    // Midpoint ties where f32 FMA alone then f16 RNE double-rounds the
    // wrong way (VIC-BT-041): exact is just above 1 + 2^-11.
    let above_mid = [
        (1.0f32, 1.0f32, f32::from_bits(0x3A00_0001)),
        (1.0, 1.0, 2.0f32.powi(-11) + 2.0f32.powi(-30)),
        (-1.0, 1.0, -(2.0f32.powi(-11) + 2.0f32.powi(-30))),
        (2.0, 0.5, f32::from_bits(0x3A00_0001)),
    ];
    for (a, b, c) in above_mid {
        let want = fma_f16_single_round(a, b, c);
        let got = f16_nearest_f32(f32_fma_round_to_odd(a, b, c));
        let double = f16_nearest_f32(a.mul_add(b, c));
        assert_ne!(
            double, want,
            "case should exercise double-rounding: {a:?}*{b:?}+{c:?}"
        );
        assert_eq!(
            got, want,
            "RTO path {a:?}*{b:?}+{c:?} = {got:#06x}, want {want:#06x} (double-round {double:#06x})"
        );
    }
    // Just below the same midpoint: f32 FMA alone already ties correctly,
    // but always-OR-1 sticky would cross the midpoint the wrong way.
    let below_mid = [
        (1.0f32, 1.0f32, 2.0f32.powi(-11) - 2.0f32.powi(-30)),
        (-1.0, 1.0, -(2.0f32.powi(-11) - 2.0f32.powi(-30))),
        (2.0, 0.5, 2.0f32.powi(-11) - 2.0f32.powi(-30)),
    ];
    for (a, b, c) in below_mid {
        let want = fma_f16_single_round(a, b, c);
        let got = f16_nearest_f32(f32_fma_round_to_odd(a, b, c));
        let naive = f16_nearest_f32(f32::from_bits(a.mul_add(b, c).to_bits() | 1));
        assert_ne!(
            naive, want,
            "case should catch always-OR-1: {a:?}*{b:?}+{c:?}"
        );
        assert_eq!(
            got, want,
            "below-mid RTO {a:?}*{b:?}+{c:?} = {got:#06x}, want {want:#06x} (naive {naive:#06x})"
        );
    }
    // Exact and ordinary results still match single-round.
    let ordinary = [
        (2.0f32, 3.0f32, 1.0f32),
        (1.0, 1.0, 1.0),
        (0.0, 5.0, 1.5),
        (-2.0, 3.0, 0.25),
        (1.0, 1.0, 0.0),
    ];
    for (a, b, c) in ordinary {
        let want = fma_f16_single_round(a, b, c);
        let got = f16_nearest_f32(f32_fma_round_to_odd(a, b, c));
        assert_eq!(got, want, "ordinary {a:?}*{b:?}+{c:?}");
    }
}

#[test]
fn fma_mix_f16_lowers() {
    // VOP3P v_fma_mixlo_f16 / v_fma_mixhi_f16 v0, v1, v2, v3 (all f32).
    let src0 = 256 + 1u32;
    let src1 = 256 + 2u32;
    let src2 = 256 + 3u32;
    for op in [0x021u32, 0x022] {
        let words = [
            0x33 << 26 | (op << 16),
            src0 | (src1 << 9) | (src2 << 18),
            S_ENDPGM,
        ];
        lowers(&words, 0).unwrap_or_else(|e| panic!("op {op:#x}: {e}"));
    }
}
