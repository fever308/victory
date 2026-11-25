//! End-to-end tests of comparison gathers and the texel-offset sample
//! variants: synthetic GFX10 programs lowered to DXIL, validated by
//! `dxil.dll` when present, and run on the host GPU and WARP against a 4x4
//! texture whose red channel is `10 * y + x`.

use windows::Win32::Graphics::Direct3D12::*;

use super::host::View;
use super::waterfall_tests::{build_any, point_sampler, run_with_sampler};

const SAMPLE_AT: u32 = 0x3ECC_CCCD; // 0.4: 1.6 texels, so 1.1 after the gather's half texel

/// A 32-thread program that samples or gathers texture 0 at the address
/// `addr` (VGPRs from v2) with the sampler table's first sampler and stores
/// the four result registers at `16 * (lane & 7)`.
fn program(op: u32, dmask: u32, addr: &[u32]) -> Vec<u32> {
    let mut words = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x3620_0087, // v_and_b32 v16, 7, v0
        0x3420_2084, // v_lshlrev_b32 v16, 4, v16
    ];
    for (k, &value) in addr.iter().enumerate() {
        // v_mov_b32 v(2 + k), literal
        words.extend([0x7E00_02FF | (2 + k as u32) << 17, value]);
    }
    words.extend([
        0xBF8C_C07F,                                  // s_waitcnt lgkmcnt(0)
        0xF000_0000 | op << 18 | dmask << 8 | 1 << 3, // dim:2D
        0x0043_0C02,                                  // v[12:15], v[2:], s[12:19], s[8:11]
        0xBF8C_3F70,                                  // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xE070_1004,
        0x800A_0D10, // buffer_store_dword v13, v16, s[40:43], 0 offen offset:4
        0xE070_1008,
        0x800A_0E10, // buffer_store_dword v14, v16, s[40:43], 0 offen offset:8
        0xE070_100C,
        0x800A_0F10, // buffer_store_dword v15, v16, s[40:43], 0 offen offset:12
        0xBF81_0000, // s_endpgm
    ]);
    words
}

/// Runs `words` against the test texture with a comparison sampler or a
/// point sampler and returns the four results of each backend's lane 0.
fn results(words: &[u32], comparison: bool) -> Vec<(&'static str, [f32; 4])> {
    let (shader, _) = build_any(words, false);
    let texels: Vec<[f32; 4]> = (0..16)
        .map(|i| [(10 * (i / 4) + i % 4) as f32, 0.0, 0.0, 1.0])
        .collect();
    // Comparison sampling needs a format with comparison support (R32F, a
    // depth texture's view). RGBA32F has none on NVIDIA or WARP; WARP
    // compared anyway, NVIDIA returned the texels uncompared.
    let reds: Vec<f32> = texels.iter().map(|t| t[0]).collect();
    let texture = || {
        if comparison {
            View::DepthTexture(4, 4, &reds)
        } else {
            View::Texture(4, 4, &texels)
        }
    };
    let views = [texture(), texture(), View::RawUav(32)];
    let filter = if comparison {
        D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT
    } else {
        D3D12_FILTER_MIN_MAG_MIP_POINT
    };
    let out = run_with_sampler(&shader, &views, point_sampler(filter));
    // Neither hardware nor WARP may offer SM 6.6; preserve that skip.
    if out.is_empty() {
        return Vec::new();
    }
    out.into_iter()
        .map(|(label, words)| {
            let at = |i: usize| f32::from_bits(words[i]);
            (label, [at(0), at(1), at(2), at(3)])
        })
        .collect()
}

#[test]
fn comparison_gather_compares_the_four_texels_in_gather_order() {
    // image_gather4_c_lz, reference then coordinates. The footprint is
    // texels (1,2), (2,2), (2,1), (1,1) = 21, 22, 12, 11; the reference
    // passes where it is at most the texel.
    let reference = 15.5f32.to_bits();
    let words = program(0x4F, 0x1, &[reference, SAMPLE_AT, SAMPLE_AT]);
    for (label, got) in results(&words, true) {
        assert_eq!(got, [1.0, 1.0, 0.0, 0.0], "{label}");
    }
}

#[test]
fn comparison_gather_applies_its_programmable_offsets() {
    // image_gather4_c_lz_o: offset word (x in bits 0-5, y in 8-13), the
    // reference, then the coordinates.
    let cases = [
        // (+1, +1): texels 32, 33, 23, 22.
        (0x0101, 25.5f32, [1.0, 1.0, 0.0, 0.0]),
        // (-1, -1): texels 10, 11, 1, 0.
        (0x3F3F, 10.5, [0.0, 1.0, 0.0, 0.0]),
    ];
    for (offset, reference, want) in cases {
        let words = program(
            0x5F,
            0x1,
            &[offset, reference.to_bits(), SAMPLE_AT, SAMPLE_AT],
        );
        for (label, got) in results(&words, true) {
            assert_eq!(got, want, "{label}: offset {offset:#x}");
        }
    }
}

#[test]
fn sample_offsets_move_the_texel_read() {
    // Point sampling at 1.6 texels reads texel (1, 1); the offset word
    // (+1, -1) moves it to (2, 0), whose red is 2.
    let offset = 0x3F01;
    let level = 0;
    let words = program(0x34, 0xF, &[offset, SAMPLE_AT, SAMPLE_AT, level]);
    for (label, got) in results(&words, false) {
        assert_eq!(got, [2.0, 0.0, 0.0, 1.0], "{label}: image_sample_l_o");
    }
    let words = program(0x37, 0xF, &[offset, SAMPLE_AT, SAMPLE_AT]);
    for (label, got) in results(&words, false) {
        assert_eq!(got, [2.0, 0.0, 0.0, 1.0], "{label}: image_sample_lz_o");
    }
}

#[test]
fn comparison_sample_offsets_move_the_texel_compared() {
    // image_sample_c_lz_o with (+1, +1): texel (2, 2) = 22.
    for (reference, want) in [(21.5f32, 1.0f32), (22.5, 0.0)] {
        let words = program(
            0x3F,
            0x1,
            &[0x0101, reference.to_bits(), SAMPLE_AT, SAMPLE_AT],
        );
        for (label, got) in results(&words, true) {
            assert_eq!(got[0], want, "{label}: reference {reference}");
        }
    }
}
