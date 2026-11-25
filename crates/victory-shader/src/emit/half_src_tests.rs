//! Lowering coverage for 16-bit integer inline float constants (VIC-BT-021).
//! Hardware keeps the f32 bits; f16 ops alone decode them as f16.
//! Host execution of the u16 case lives in `valu_tests` (Windows).

use super::lower::{self, Parts};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
/// Inline constant code for `1.0` (f32 bits `0x3F80_0000`, f16 bits `0x3C00`).
const INLINE_F32_ONE: u16 = 242;

const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

fn lower_body(body: &[u32]) -> lower::Lowered {
    let mut words = Vec::new();
    words.push(s_mov_b32(5, 0x80));
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, i as u16));
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    lower::lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower")
}

/// `v_add_nc_u16 v0, 1.0, v1` must keep the f32 bits of the inline constant
/// (`0x3F80_0000`, low half 0), not the f16 encoding (`0x3C00`).
#[test]
fn u16_inline_float_keeps_f32_bits() {
    let body = vop3(0x303, 0, INLINE_F32_ONE, 256 + 1, 0);
    let l = lower_body(&body);
    assert!(
        l.has_f32_one_imm,
        "u16 inline 1.0 must emit f32 bits 0x3F80_0000"
    );
    assert!(
        !l.has_f16_one_imm,
        "u16 inline 1.0 must not emit f16 bits 0x3C00"
    );
}

/// Control: `v_add_f16 v0, 1.0, v1` still decodes the inline as f16 `0x3C00`.
#[test]
fn f16_inline_float_keeps_f16_bits() {
    let body = vop3(0x132, 0, INLINE_F32_ONE, 256 + 1, 0);
    let l = lower_body(&body);
    assert!(
        l.has_f16_one_imm,
        "f16 inline 1.0 must emit f16 bits 0x3C00"
    );
    assert!(
        !l.has_f32_one_imm,
        "f16 inline 1.0 must not emit f32 bits 0x3F80_0000"
    );
}
