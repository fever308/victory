//! The DXBC container: a header, then a table of FourCC-tagged parts.
//!
//! ```text
//! 0x00  "DXBC"
//! 0x04  digest[16]
//! 0x14  version major u16, minor u16
//! 0x18  total size u32
//! 0x1C  part count u32
//! 0x20  part offsets u32[count]
//! part: fourcc[4], size u32, data[size]
//! ```

use crate::ShaderError;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FourCC(pub [u8; 4]);

impl FourCC {
    pub const DXIL: FourCC = FourCC(*b"DXIL");
    pub const STAT: FourCC = FourCC(*b"STAT");
    pub const RTS0: FourCC = FourCC(*b"RTS0");
    pub const PSV0: FourCC = FourCC(*b"PSV0");
    pub const ISG1: FourCC = FourCC(*b"ISG1");
    pub const OSG1: FourCC = FourCC(*b"OSG1");
    /// Per-primitive outputs of a mesh shader.
    pub const PSG1: FourCC = FourCC(*b"PSG1");
    pub const SFI0: FourCC = FourCC(*b"SFI0");
    pub const HASH: FourCC = FourCC(*b"HASH");
    pub const ILDB: FourCC = FourCC(*b"ILDB");
    pub const ILDN: FourCC = FourCC(*b"ILDN");
    /// The part that carries an [`Fspx`] precompiled compute module.
    pub const ACCC: FourCC = FourCC(*b"ACCC");
}

impl std::fmt::Debug for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl std::fmt::Display for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for &b in &self.0 {
            let c = if b.is_ascii_graphic() { b as char } else { '.' };
            write!(f, "{c}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Part<'a> {
    pub fourcc: FourCC,
    /// Offset of the part header within the container.
    pub offset: usize,
    pub data: &'a [u8],
}

#[derive(Debug, Clone)]
pub struct Container<'a> {
    pub digest: [u8; 16],
    pub version: (u16, u16),
    pub parts: Vec<Part<'a>>,
}

const HEADER_SIZE: usize = 0x20;
const MAX_PARTS: usize = 256;

impl<'a> Container<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, ShaderError> {
        if data.len() < HEADER_SIZE || &data[0..4] != b"DXBC" {
            return Err(ShaderError::NotDxbc);
        }
        let digest = data[4..20].try_into().expect("16 bytes");
        let version = (u16_at(data, 0x14)?, u16_at(data, 0x16)?);
        let total = u32_at(data, 0x18)? as usize;
        if total > data.len() || total < HEADER_SIZE {
            return Err(ShaderError::Malformed("container size disagrees with data"));
        }
        let data = &data[..total];
        let count = u32_at(data, 0x1C)? as usize;
        if count > MAX_PARTS {
            return Err(ShaderError::Malformed("implausible part count"));
        }
        let table_end = HEADER_SIZE + 4 * count;
        if table_end > data.len() {
            return Err(ShaderError::Malformed("part table out of bounds"));
        }
        let mut parts = Vec::with_capacity(count);
        for i in 0..count {
            let offset = u32_at(data, HEADER_SIZE + 4 * i)? as usize;
            if offset < table_end {
                return Err(ShaderError::Malformed(
                    "part starts inside the header or part table",
                ));
            }
            let fourcc = FourCC(
                data.get(offset..offset + 4)
                    .ok_or(ShaderError::Malformed("part header out of bounds"))?
                    .try_into()
                    .expect("4 bytes"),
            );
            let size = u32_at(data, offset + 4)? as usize;
            let body = data
                .get(offset + 8..offset + 8 + size)
                .ok_or(ShaderError::Malformed("part data out of bounds"))?;
            parts.push(Part {
                fourcc,
                offset,
                data: body,
            });
        }
        Ok(Container {
            digest,
            version,
            parts,
        })
    }

    pub fn part(&self, fourcc: FourCC) -> Option<&Part<'a>> {
        self.parts.iter().find(|p| p.fourcc == fourcc)
    }

    /// The FSPX compute module in this container's `ACCC` part, if it has one.
    pub fn fspx(&self) -> Option<Fspx<'a>> {
        Fspx::parse(self.part(FourCC::ACCC)?.data).ok()
    }
}

/// A precompiled compute module (`FSPX`), as the console's AMD upscaler passes it:
/// kept whole in a DXBC `ACCC` part, or handed to the device as separate kinds.
///
/// ```text
/// 0x00  "FSPX"
/// 0x04  version u32
/// 0x08  entry count u32
/// 0x0C  total size u32
/// 0x10  entries[count]: kind u32, offset u32, size u32, reserved u32
/// ```
///
/// Kinds we read: 0 (dispatch state), 1 (not understood yet), 6 (the GFX10 program) and 7
/// (resource and hash records). Others are skipped.
#[derive(Debug, Clone)]
pub struct Fspx<'a> {
    pub version: u32,
    pub count: u32,
    pub total_size: u32,
    pub dispatch_state: Option<&'a [u8]>,
    pub kind1: Option<&'a [u8]>,
    pub program: Option<&'a [u8]>,
    pub records: Option<&'a [u8]>,
}

/// What an [`Fspx`]'s dispatch state (kind 0) says about how its program runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchInfo {
    pub threads: [u32; 3],
    pub wave32: bool,
    /// The program's `COMPUTE_PGM_RSRC1` and `COMPUTE_PGM_RSRC2` words.
    pub rsrc1: u32,
    pub rsrc2: u32,
}

impl<'a> Fspx<'a> {
    pub const MAGIC: &'static [u8; 4] = b"FSPX";
    const HEADER_SIZE: usize = 0x10;
    const ENTRY_SIZE: usize = 16;
    /// Kind 0 is at least this long: it is read up to its `rsrc2` word.
    pub const DISPATCH_STATE_SIZE: usize = 28;
    const MAX_ENTRIES: usize = 64;

    pub fn parse(data: &'a [u8]) -> Result<Self, ShaderError> {
        if !data.starts_with(Self::MAGIC) {
            return Err(ShaderError::Malformed("not an FSPX module"));
        }
        let version = u32_at(data, 4)?;
        let count = u32_at(data, 8)?;
        let total_size = u32_at(data, 12)?;
        let total = total_size as usize;
        if total > data.len() || total < Self::HEADER_SIZE {
            return Err(ShaderError::Malformed("FSPX size disagrees with data"));
        }
        let data = &data[..total];
        if count as usize > Self::MAX_ENTRIES
            || Self::HEADER_SIZE + count as usize * Self::ENTRY_SIZE > total
        {
            return Err(ShaderError::Malformed("FSPX entry table out of bounds"));
        }
        let mut fspx = Fspx {
            version,
            count,
            total_size,
            dispatch_state: None,
            kind1: None,
            program: None,
            records: None,
        };
        for i in 0..count as usize {
            let at = Self::HEADER_SIZE + i * Self::ENTRY_SIZE;
            let kind = u32_at(data, at)?;
            let offset = u32_at(data, at + 4)? as usize;
            let size = u32_at(data, at + 8)? as usize;
            let body = offset
                .checked_add(size)
                .and_then(|end| data.get(offset..end))
                .ok_or(ShaderError::Malformed("FSPX entry out of bounds"))?;
            match kind {
                0 => fspx.dispatch_state = Some(body),
                1 => fspx.kind1 = Some(body),
                6 => fspx.program = Some(body),
                7 => fspx.records = Some(body),
                _ => {}
            }
        }
        Ok(fspx)
    }

    /// The module's thread group, wave size and resource words, from its dispatch state.
    pub fn dispatch_info(&self) -> Option<DispatchInfo> {
        dispatch_info(self.dispatch_state?)
    }

    /// A module made of the four kinds, in that order after the entry table (version 1), as the
    /// device hands them over separately.
    pub fn assemble(kind0: &[u8], kind1: &[u8], kind6: &[u8], kind7: &[u8]) -> Vec<u8> {
        const KINDS: [u32; 4] = [0, 1, 6, 7];
        let bodies = [kind0, kind1, kind6, kind7];
        let table = Self::HEADER_SIZE + KINDS.len() * Self::ENTRY_SIZE;
        let total = table + bodies.iter().map(|b| b.len()).sum::<usize>();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(Self::MAGIC);
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(KINDS.len() as u32).to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        let mut offset = table;
        for (kind, body) in KINDS.iter().zip(bodies) {
            out.extend_from_slice(&kind.to_le_bytes());
            out.extend_from_slice(&(offset as u32).to_le_bytes());
            out.extend_from_slice(&(body.len() as u32).to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            offset += body.len();
        }
        for body in bodies {
            out.extend_from_slice(body);
        }
        out
    }
}

/// The dispatch state (kind 0) of an FSPX module: words 3 and 4 hold the thread group (x and y
/// in word 3, z and the wave32 flag, bit 20, in word 4), words 5 and 6 the resource words.
pub fn dispatch_info(state: &[u8]) -> Option<DispatchInfo> {
    if state.len() < Fspx::DISPATCH_STATE_SIZE {
        return None;
    }
    let word = |i: usize| u32_at(state, 4 * i).ok();
    let (w3, w4) = (word(3)?, word(4)?);
    Some(DispatchInfo {
        threads: [w3 & 0xffff, w3 >> 16, w4 & 0xffff],
        wave32: w4 & (1 << 20) != 0,
        rsrc1: word(5)?,
        rsrc2: word(6)?,
    })
}

/// The FSPX module in `bytes`, if it is one bare or in the `ACCC` part of a DXBC container.
pub fn inspect_fspx(bytes: &[u8]) -> Option<Fspx<'_>> {
    if bytes.starts_with(Fspx::MAGIC) {
        return Fspx::parse(bytes).ok();
    }
    Container::parse(bytes).ok()?.fspx()
}

/// Every container stored as is in `data`, found by its magic, as its offset
/// and its bytes (`total size` long), in file order. A `DXBC` inside one of
/// a found container's parts (a container kept in a part, say) is part of
/// that container, not another one. Its header is not such a place: a magic
/// in bytes before a real container may read its header from the real one,
/// and must not hide it. Offsets for which `skip` is true are not looked at
/// (bytes that belong to something else, such as a packed record).
pub fn find_all(data: &[u8], skip: impl Fn(usize) -> bool) -> Vec<(usize, &[u8])> {
    let mut out = Vec::new();
    // Parts of the containers found so far that may still hold a magic.
    let mut parts: Vec<std::ops::Range<usize>> = Vec::new();
    for offset in memchr::memmem::find_iter(data, b"DXBC") {
        parts.retain(|p| p.end > offset);
        if parts.iter().any(|p| p.contains(&offset)) || skip(offset) {
            continue;
        }
        let Ok(container) = Container::parse(&data[offset..]) else {
            continue;
        };
        // `parse` checked the size against the data.
        let total = u32_at(data, offset + 0x18).expect("parsed header") as usize;
        parts.extend(container.parts.iter().map(|p| {
            let start = offset + p.offset + 8;
            start..start + p.data.len()
        }));
        out.push((offset, &data[offset..offset + total]));
    }
    out
}

pub(crate) fn u16_at(data: &[u8], at: usize) -> Result<u16, ShaderError> {
    data.get(at..at + 2)
        .map(|b| u16::from_le_bytes(b.try_into().expect("2 bytes")))
        .ok_or(ShaderError::Malformed("truncated"))
}

pub(crate) fn u32_at(data: &[u8], at: usize) -> Result<u32, ShaderError> {
    data.get(at..at + 4)
        .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
        .ok_or(ShaderError::Malformed("truncated"))
}

pub(crate) fn u64_at(data: &[u8], at: usize) -> Result<u64, ShaderError> {
    data.get(at..at + 8)
        .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
        .ok_or(ShaderError::Malformed("truncated"))
}

/// Builds a container from parts (test helper).
#[cfg(test)]
pub fn build(parts: &[(FourCC, &[u8])]) -> Vec<u8> {
    let table = HEADER_SIZE + 4 * parts.len();
    let total = table + parts.iter().map(|(_, d)| 8 + d.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"DXBC");
    out.extend_from_slice(&[0; 16]);
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(parts.len() as u32).to_le_bytes());
    let mut offset = table;
    for (_, d) in parts {
        out.extend_from_slice(&(offset as u32).to_le_bytes());
        offset += 8 + d.len();
    }
    for (fourcc, d) in parts {
        out.extend_from_slice(&fourcc.0);
        out.extend_from_slice(&(d.len() as u32).to_le_bytes());
        out.extend_from_slice(d);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_parts() {
        let bytes = build(&[(FourCC::SFI0, &[1, 2, 3, 4]), (FourCC::HASH, &[9; 20])]);
        let c = Container::parse(&bytes).unwrap();
        assert_eq!(c.parts.len(), 2);
        assert_eq!(c.part(FourCC::SFI0).unwrap().data, &[1, 2, 3, 4]);
        assert_eq!(c.part(FourCC::HASH).unwrap().data.len(), 20);
        assert!(c.part(FourCC::DXIL).is_none());
    }

    #[test]
    fn container_in_a_part_is_not_another_container() {
        let inner = build(&[(FourCC::SFI0, &[1, 2, 3, 4])]);
        let outer = build(&[(FourCC::ILDB, &inner), (FourCC::HASH, &[9; 20])]);
        let other = build(&[(FourCC::STAT, &[7; 12])]);
        let mut data = b"gap".to_vec();
        let outer_at = data.len();
        data.extend_from_slice(&outer);
        data.extend_from_slice(b"gap");
        let other_at = data.len();
        data.extend_from_slice(&other);
        let inner_at = outer_at + memchr::memmem::find(&outer[4..], b"DXBC").unwrap() + 4;
        // The nested one parses on its own.
        assert!(Container::parse(&data[inner_at..]).is_ok());

        let found = find_all(&data, |_| false);
        let offsets: Vec<_> = found.iter().map(|&(o, _)| o).collect();
        assert_eq!(offsets, [outer_at, other_at]);
        assert_eq!(found[0].1, &outer[..]);
        assert_eq!(found[1].1, &other[..]);

        // Skipping the outer one exposes the one it holds.
        let found = find_all(&data, |o| o == outer_at);
        let offsets: Vec<_> = found.iter().map(|&(o, _)| o).collect();
        assert_eq!(offsets, [inner_at, other_at]);
        assert_eq!(found[0].1, &inner[..]);
    }

    #[test]
    fn header_reading_into_a_container_does_not_hide_it() {
        // A magic 21 bytes before a real container: its digest and version
        // are the junk, its size and part count come from the real header
        // (0x43, the real "C", and 0 parts), so it parses, with no parts.
        let real = build(&[(FourCC::SFI0, &[1, 2, 3, 4])]);
        let mut data = b"DXBC not a container ".to_vec();
        let real_at = data.len();
        data.extend_from_slice(&real);
        assert_eq!(Container::parse(&data).unwrap().parts.len(), 0);

        let offsets: Vec<_> = find_all(&data, |_| false).iter().map(|&(o, _)| o).collect();
        assert_eq!(offsets, [0, real_at]);
    }

    #[test]
    fn finds_nothing_in_noise() {
        assert!(find_all(b"", |_| false).is_empty());
        assert!(find_all(b"DXBC", |_| false).is_empty());
        let mut bytes = build(&[(FourCC::SFI0, &[0; 8])]);
        bytes.pop(); // the stated size runs past the end
        assert!(find_all(&bytes, |_| false).is_empty());
    }

    fn dispatch_state(threads: [u32; 3], wave32: bool) -> Vec<u8> {
        let mut state = vec![0u8; Fspx::DISPATCH_STATE_SIZE];
        let w3 = threads[0] | threads[1] << 16;
        let w4 = threads[2] | u32::from(wave32) << 20;
        state[12..16].copy_from_slice(&w3.to_le_bytes());
        state[16..20].copy_from_slice(&w4.to_le_bytes());
        state[20..24].copy_from_slice(&0xc0u32.to_le_bytes());
        state[24..28].copy_from_slice(&0x400u32.to_le_bytes());
        state
    }

    #[test]
    fn fspx_assembled_from_kinds_parses_back() {
        let k0 = dispatch_state([8, 16, 1], true);
        let bytes = Fspx::assemble(&k0, &[1; 20], &[0xaa; 48], &[0xbb; 16]);
        let fspx = Fspx::parse(&bytes).unwrap();
        assert_eq!((fspx.version, fspx.count), (1, 4));
        assert_eq!(fspx.total_size as usize, bytes.len());
        assert_eq!(fspx.dispatch_state.unwrap(), &k0[..]);
        assert_eq!(fspx.kind1.unwrap(), &[1; 20]);
        assert_eq!(fspx.program.unwrap(), &[0xaa; 48]);
        assert_eq!(fspx.records.unwrap(), &[0xbb; 16]);
        let info = fspx.dispatch_info().unwrap();
        assert_eq!(info.threads, [8, 16, 1]);
        assert!(info.wave32);
        assert_eq!((info.rsrc1, info.rsrc2), (0xc0, 0x400));
    }

    #[test]
    fn fspx_wave64_and_wide_thread_groups() {
        let k0 = dispatch_state([0xffff, 2, 0x1234], false);
        let bytes = Fspx::assemble(&k0, &[], &[], &[]);
        let info = Fspx::parse(&bytes).unwrap().dispatch_info().unwrap();
        assert_eq!(info.threads, [0xffff, 2, 0x1234]);
        assert!(!info.wave32);
        // A short dispatch state says nothing.
        let bytes = Fspx::assemble(&k0[..27], &[], &[], &[]);
        assert!(Fspx::parse(&bytes).unwrap().dispatch_info().is_none());
    }

    #[test]
    fn fspx_is_found_bare_and_in_an_accc_part() {
        let fspx = Fspx::assemble(&dispatch_state([64, 1, 1], false), &[], &[0x55; 8], &[]);
        let bare = inspect_fspx(&fspx).unwrap();
        assert_eq!(bare.program.unwrap(), &[0x55; 8]);

        let dxbc = build(&[(FourCC::ACCC, &fspx), (FourCC::HASH, &[7; 16])]);
        let wrapped = inspect_fspx(&dxbc).unwrap();
        assert_eq!(wrapped.dispatch_info().unwrap().threads, [64, 1, 1]);
        assert_eq!(Container::parse(&dxbc).unwrap().fspx().unwrap().version, 1);

        let plain = build(&[(FourCC::SFI0, &[1, 2, 3, 4])]);
        assert!(inspect_fspx(&plain).is_none());
        let other_accc = build(&[(FourCC::ACCC, b"not fspx at all")]);
        assert!(inspect_fspx(&other_accc).is_none());
    }

    #[test]
    fn malformed_fspx_is_refused() {
        let good = Fspx::assemble(&[0; 28], &[], &[1; 8], &[]);
        assert!(Fspx::parse(&good).is_ok());
        // The stated size runs past the data.
        assert!(Fspx::parse(&good[..good.len() - 1]).is_err());
        // An entry that runs past the module, or whose end overflows.
        for size in [0x1000u32, u32::MAX] {
            let mut bad = good.clone();
            bad[0x10 + 8..0x10 + 12].copy_from_slice(&size.to_le_bytes());
            assert!(Fspx::parse(&bad).is_err());
        }
        // A table that does not fit.
        let mut bad = good.clone();
        bad[8..12].copy_from_slice(&1000u32.to_le_bytes());
        assert!(Fspx::parse(&bad).is_err());
        assert!(Fspx::parse(b"FSPX").is_err());
        assert!(Fspx::parse(b"DXBC0123456789abcdef").is_err());
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(Container::parse(b"nope").unwrap_err(), ShaderError::NotDxbc);
        let mut bytes = build(&[(FourCC::SFI0, &[0; 8])]);
        let last = bytes.len() - 1;
        bytes[0x20] = 0xFF; // corrupt the first part offset
        assert!(Container::parse(&bytes).is_err());
        bytes.truncate(last);
        assert!(Container::parse(&bytes).is_err());
    }

    #[test]
    fn rejects_parts_starting_inside_the_header_or_table() {
        let mut bytes = build(&[(FourCC([0; 4]), &[]), (FourCC::HASH, &[])]);
        let second_entry = HEADER_SIZE + 4;
        // A zero digest and a zero first FourCC make both bogus sizes zero,
        // so a part at any offset in the header reads back as "valid".
        for offset in [0u32, second_entry as u32] {
            bytes[second_entry..second_entry + 4].copy_from_slice(&offset.to_le_bytes());
            assert_eq!(
                Container::parse(&bytes).unwrap_err(),
                ShaderError::Malformed("part starts inside the header or part table"),
            );
        }
    }

    #[test]
    fn rejects_a_part_table_longer_than_the_container() {
        let mut bytes = build(&[(FourCC::SFI0, &[1, 2, 3, 4])]);
        bytes[0x1C..0x20].copy_from_slice(&7u32.to_le_bytes());
        assert_eq!(
            Container::parse(&bytes).unwrap_err(),
            ShaderError::Malformed("part table out of bounds"),
        );
    }
}
