//! Title memory by address (`resource::MEMORY`) in a compute program: an
//! access whose address traces to no root parameter reads through the
//! 64 KiB page directory instead of being refused.

use super::{SLOT_FAULTS, SLOT_MEMORY, lower};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// A compute program of `words` whose one root parameter is a two-dword
/// table at `s0`; everything else the code reads is title memory.
fn native_compute(words: &[u32]) -> NativeProgram {
    let slot = 1;
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![RootParam {
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: 0,
            user_data: 16,
            slot,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Cbv,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 16,
            slot,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 4 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn lower_compute(native: &NativeProgram) -> lower::Lowered {
    let p =
        ir::build_with(&native.code, native.wave64, native.vgprs, true).expect("ir::build_with");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, native).expect("io::raise");
    lower::lower(&lower::Parts {
        native,
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

#[test]
#[ignore = "WIP: the synthetic load is dead and never lowered; the dump run is the evidence"]
fn a_scalar_load_from_an_untracked_address_reads_the_directory() {
    let words: [u32; 5] = [
        0xF400_0081, // s_load_dword s2, s[2:3], 0: no parameter holds
        0xFA00_0000, // s2, so the address is title memory at it
        0xBF06_8002, // s_cmp_eq_u32 s2, 0 (the dword is read)
        0xBF85_0000, // s_cbranch_scc1 0
        0xBF81_0000, // s_endpgm
    ];
    let lowered = lower_compute(&native_compute(&words));
    assert!(lowered.root_srvs.contains(&SLOT_MEMORY));
    assert_eq!(lowered.root_uavs, [SLOT_FAULTS]);
}

#[test]
#[ignore = "WIP: the synthetic atomic is not lowered yet; the dump run is the evidence"]
fn a_global_atomic_whose_address_is_loaded_from_memory_reads_the_directory() {
    let words: [u32; 9] = [
        0xF400_0081, // s_load_dword s2, s[2:3], 0
        0xFA00_0000,
        0xF400_00C1, // s_load_dword s3, s[2:3], 4
        0xFA00_0004,
        0x7E00_0202, // v_mov_b32 v0, s2 (a pointer loaded from memory)
        0x7E02_0203, // v_mov_b32 v1, s3
        0xDCC9_8000, // global_atomic_add v2, v[0:1], v3, off glc
        0x027D_0300,
        0xBF81_0000, // s_endpgm
    ];
    let lowered = lower_compute(&native_compute(&words));
    assert!(lowered.root_srvs.contains(&SLOT_MEMORY));
    assert_eq!(lowered.root_uavs, [SLOT_FAULTS]);
}
