//! `image_bvh*_intersect_ray` lowers as a miss: the shader translates, and
//! the descriptor the ray query names is never traced or read.

use super::lower;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const SLOT: u8 = 3;

/// A compute shader whose lanes query `node = 4 * lane` with the ray query
/// at `bvh` (its opcode and address width) and store the four results
/// through a root UAV. `load_descriptor` takes the BVH descriptor from
/// memory instead of building it from constants.
fn program(bvh_words: [u32; 2], load_descriptor: bool) -> Vec<u32> {
    let mut words = vec![
        0xF404_0200, // s_load_dwordx2 s[8:9], s[0:1], 0
        0xFA00_0000,
        0xBE8A_03C0, // s_mov_b32 s10, 64
        0xBE8B_0380, // s_mov_b32 s11, 0
    ];
    if load_descriptor {
        words.extend([
            0xF408_0300, // s_load_dwordx4 s[12:15], s[0:1], 0x10
            0xFA00_0010,
        ]);
    } else {
        words.extend([
            0xBE8C_0380, // s_mov_b32 s12, 0
            0xBE8D_0380, // s_mov_b32 s13, 0
            0xBE8E_0380, // s_mov_b32 s14, 0
            0xBE8F_0380, // s_mov_b32 s15, 0
        ]);
    }
    words.push(0x3402_0082); // v_lshlrev_b32 v1, 2, v0
    // Ray extent, origin, direction and inverse direction.
    words.extend((2..=12).map(|n| 0x7E00_0280 | (n << 17))); // v_mov_b32 vn, 0
    if load_descriptor {
        words.push(0xBF8C_C07F); // s_waitcnt lgkmcnt(0)
    }
    words.extend(bvh_words); // image_bvh*_intersect_ray v[16:19], v[1:12], s[12:15]
    words.push(0xBF8C_3F70); // s_waitcnt vmcnt(0)
    for k in 0..4 {
        // buffer_store_dword v(16 + k), off, s[8:11], 0 offset:4 * k
        words.extend([0xE070_0000 | (4 * k), 0x8002_0000 | ((16 + k) << 8)]);
    }
    words.push(0xBF81_0000); // s_endpgm
    words
}

const BVH32: [u32; 2] = [0xF198_8F01, 0x0003_1001];
const BVH64: [u32; 2] = [0xF19C_8F01, 0x0003_1001];

fn translate(words: &[u32]) -> Result<lower::Lowered, ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 20,
        layout: vec![RootParam {
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: 0,
            user_data: 16,
            slot: SLOT,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 16,
            slot: SLOT,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 4 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build_with(&native.code, native.wave64, native.vgprs, true)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    lower::lower(&lower::Parts {
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
}

#[test]
fn decoded_operands_match_the_ray_query_layout() {
    use crate::gfx10::{Fields, Operand, decode};
    for (words, addr, name) in [
        (BVH32, 11, "image_bvh_intersect_ray"),
        (BVH64, 12, "image_bvh64_intersect_ray"),
    ] {
        let code: Vec<u8> = words
            .iter()
            .chain(&[0xBF81_0000])
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let inst = &decode(&code).unwrap()[0];
        assert_eq!(inst.name(), Some(name));
        let Fields::Image {
            data,
            addr: a,
            rsrc,
            ..
        } = inst.fields(true)
        else {
            panic!("not an image instruction");
        };
        assert_eq!(data, Operand::Vector { reg: 16, dwords: 4 });
        assert_eq!(
            a,
            [Operand::Vector {
                reg: 1,
                dwords: addr
            }]
        );
        assert_eq!(
            rsrc,
            Operand::Scalar {
                code: 12,
                dwords: 4
            }
        );
    }
}

#[test]
fn image_bvh_intersect_ray_lowers_as_miss() {
    translate(&program(BVH32, false)).expect("image_bvh_intersect_ray translates");
}

#[test]
fn image_bvh64_intersect_ray_lowers_as_miss() {
    translate(&program(BVH64, false)).expect("image_bvh64_intersect_ray translates");
}

#[test]
fn bvh_descriptor_loaded_from_memory_is_never_read() {
    translate(&program(BVH32, true)).expect("a descriptor only a ray query uses translates");
}
