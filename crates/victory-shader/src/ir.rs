//! One-to-one SSA form of a native program (decision 0003, step 1).
//!
//! Every machine instruction becomes one [`Inst`], except scheduling-only
//! ones (`s_nop`, `s_waitcnt*`, `s_inst_prefetch`...) and redundant conditional
//! branches.
//! Each register dword is a variable: a read names the [`Value`] it sees, a
//! write defines a new one, and [`Phi`]s merge values where control flow
//! joins. Nothing is raised: an instruction keeps its opcode and decoded
//! fields, so its exact semantics stay with it.
//!
//! Implicit operands are explicit: SCC, M0, and EXEC, which every vector
//! instruction reads. A write to a VGPR also reads the register's old value,
//! because lanes disabled in EXEC keep it.
//!
//! M0-relative VGPR access (`v_movrels/d/sd_b32`, `v_swaprel_b32`) names its
//! operand's whole possible range: from the encoded register up to the
//! program's VGPR count, all read (and all written, for a destination).

use crate::ShaderError;
use crate::gfx10::{self, Fields, Flow, Instruction, Operand};

/// One register dword. Scalar codes as in [`Operand::Scalar`]: 0-105 SGPRs,
/// 106/107 VCC, 108-123 trap temporaries, 124 M0, 126/127 EXEC. Codes from
/// [`PARKED`] up are scalars the compiler parked in VGPR lanes (`unspill`):
/// no machine instruction names them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Reg {
    S(u8),
    Scc,
    V(u8),
}

pub const VCC: u8 = 106;
pub const M0: u8 = 124;
pub const EXEC: u8 = 126;
/// The first scalar code past the machine's (0-127): see [`Reg`].
pub const PARKED: u8 = 128;

/// An SSA value: the content of one register dword (SCC: one bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Value(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    pub wave64: bool,
    /// VGPRs v0 up to this exist; bounds M0-relative access.
    pub vgprs: u16,
    /// In code order, then copies made to give every loop one entry. Block 0
    /// is the entry, which can be an empty block before the code.
    pub blocks: Vec<Block>,
    /// Registers read before any write, with their value on entry: what the
    /// hardware or a called fetch routine set up (user SGPRs, system values,
    /// vertex inputs).
    pub inputs: Vec<(Reg, Value)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    /// Byte offset of the block's first machine instruction (shared by a
    /// block and its copies). Zero for an empty entry block.
    pub offset: usize,
    pub preds: Vec<usize>,
    pub phis: Vec<Phi>,
    pub insts: Vec<Inst>,
    pub exit: Exit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    Jump(usize),
    /// `s_cbranch_*`, the block's last instruction.
    Branch {
        taken: usize,
        not_taken: usize,
    },
    /// `s_endpgm` and its variants.
    End,
    /// `s_setpc_b64` to an address held in registers, e.g. the handover to
    /// the next half of a merged hardware stage.
    Indirect,
}

impl Exit {
    fn successor_at(&self, index: usize) -> Option<usize> {
        match (*self, index) {
            (Exit::Jump(b), 0) | (Exit::Branch { taken: b, .. }, 0) => Some(b),
            (Exit::Branch { not_taken, .. }, 1) => Some(not_taken),
            _ => None,
        }
    }

    fn successor_iter(&self) -> impl Iterator<Item = usize> {
        self.successor_at(0).into_iter().chain(self.successor_at(1))
    }

    pub fn successors(&self) -> Vec<usize> {
        match *self {
            Exit::Jump(b) => vec![b],
            Exit::Branch { taken, not_taken } => vec![taken, not_taken],
            Exit::End | Exit::Indirect => vec![],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phi {
    pub reg: Reg,
    pub value: Value,
    /// One per predecessor, in [`Block::preds`] order.
    pub args: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inst {
    pub machine: Instruction,
    pub fields: Fields,
    /// Every register dword read, with the value it holds.
    pub reads: Vec<(Reg, Value)>,
    /// Every register dword written, with the value it receives.
    pub writes: Vec<(Reg, Value)>,
}

impl Inst {
    pub fn name(&self) -> &'static str {
        self.machine.name().unwrap_or("")
    }

    /// The value `reg` holds when this instruction reads it.
    pub fn read(&self, reg: Reg) -> Option<Value> {
        self.reads.iter().find(|r| r.0 == reg).map(|r| r.1)
    }

    /// The value this instruction writes to `reg`.
    pub fn write(&self, reg: Reg) -> Option<Value> {
        self.writes.iter().find(|w| w.0 == reg).map(|w| w.1)
    }
}

/// The first user SGPR that the spill table of root arguments stands for.
const SPILL_FIRST: u8 = 16;

#[derive(Clone, Copy, PartialEq, Eq)]
enum SpillTag {
    None,
    Low(u32),
    High,
}

fn spill_scalar(o: &Operand) -> Option<(u8, u8)> {
    match *o {
        Operand::Scalar { code, dwords } if code < 106 => Some((code, dwords.max(1))),
        _ => None,
    }
}

fn spill_const(o: &Operand) -> Option<u32> {
    match *o {
        Operand::Literal(x) => Some(x),
        Operand::Const(c) => crate::resource::inline(c),
        Operand::Scalar { code: 125, .. } => Some(0),
        _ => None,
    }
}

fn step_spill_tag(tag: &mut [SpillTag; 106], f: &Fields, name: &str) {
    use gfx10::Fields::{Smem, Sop, Valu};

    let mut written: Vec<(u8, u8)> = Vec::new();
    match f {
        Sop {
            dst: Some(d),
            src,
            imm,
            ..
        } => {
            if let Some((dc, dn)) = spill_scalar(d) {
                match (name, dn, src.as_slice()) {
                    ("s_mov_b32", 1, [x]) => {
                        tag[dc as usize] =
                            spill_scalar(x).map_or(SpillTag::None, |(c, _)| tag[c as usize]);
                    }
                    ("s_mov_b64", 2, [x]) if usize::from(dc) + 1 < 106 => {
                        let pair = match spill_scalar(x) {
                            Some((c, _)) if usize::from(c) + 1 < 106 => {
                                [tag[c as usize], tag[c as usize + 1]]
                            }
                            _ => [SpillTag::None, SpillTag::None],
                        };
                        tag[dc as usize] = pair[0];
                        tag[dc as usize + 1] = pair[1];
                    }
                    ("s_and_b32", 1, [x, y]) => {
                        let masked = |reg: &Operand, mask: &Operand| match (spill_scalar(reg), mask)
                        {
                            (Some((c, _)), Operand::Literal(m)) => {
                                tag[c as usize] == SpillTag::High && *m <= 0xFFFF
                            }
                            (Some((c, _)), Operand::Const(k)) => {
                                tag[c as usize] == SpillTag::High && (129..=192).contains(k)
                            }
                            _ => false,
                        };
                        tag[dc as usize] = if masked(x, y) || masked(y, x) {
                            SpillTag::High
                        } else {
                            SpillTag::None
                        };
                    }
                    ("s_add_u32" | "s_add_i32", 1, [x, y]) => {
                        let added = |reg: &Operand, rhs: &Operand| match (
                            spill_scalar(reg),
                            spill_const(rhs),
                        ) {
                            (Some((c, _)), Some(k)) => match tag[c as usize] {
                                SpillTag::Low(off) => Some(SpillTag::Low(off.wrapping_add(k))),
                                _ => None,
                            },
                            _ => None,
                        };
                        tag[dc as usize] = added(x, y)
                            .or_else(|| added(y, x))
                            .unwrap_or(SpillTag::None);
                    }
                    ("s_addk_i32", 1, _) => {
                        tag[dc as usize] = match (tag[dc as usize], *imm) {
                            (SpillTag::Low(off), Some(k)) => {
                                SpillTag::Low(off.wrapping_add(i32::from(k as i16) as u32))
                            }
                            _ => SpillTag::None,
                        };
                    }
                    ("s_addc_u32", 1, [x, y]) => {
                        let carried = |reg: &Operand, rhs: &Operand| {
                            spill_scalar(reg)
                                .is_some_and(|(c, _)| tag[c as usize] == SpillTag::High)
                                && spill_const(rhs) == Some(0)
                        };
                        tag[dc as usize] = if carried(x, y) || carried(y, x) {
                            SpillTag::High
                        } else {
                            SpillTag::None
                        };
                    }
                    _ => written.push((dc, dn)),
                }
            }
        }
        Smem { data: Some(d), .. } => written.extend(spill_scalar(d)),
        Valu(v) => {
            written.extend(v.dst.as_ref().and_then(spill_scalar));
            written.extend(v.carry.as_ref().and_then(spill_scalar));
        }
        _ => {}
    }
    for (c, n) in written {
        for k in c..c.saturating_add(n).min(106) {
            tag[k as usize] = SpillTag::None;
        }
    }
}

/// Rewrites loads from the spill table (`s[0:1]`, the high dword masked to a
/// 40- or 48-bit address, through copies and constant offsets across control
/// flow where the pointer holds on every predecessor) into `s_mov_b32` from
/// the user SGPRs the table's dwords stand for.
fn spill_loads(
    list: Vec<Instruction>,
    wave64: bool,
    code_len: usize,
) -> Result<Vec<Instruction>, ShaderError> {
    use gfx10::Fields::Smem;
    if list.is_empty() {
        return Ok(list);
    }

    let mut flows = Vec::with_capacity(list.len());
    let mut leaders = vec![0usize];
    for (i, inst) in list.iter().enumerate() {
        let flow = gfx10::control(inst, code_len, |at| {
            list[..list.partition_point(|i| i.offset < at)].last()
        })?;
        if let Flow::Jump(t) | Flow::Branch(t) = flow {
            leaders.push(t);
        }
        if flow != Flow::Next
            && let Some(next) = list.get(i + 1)
        {
            leaders.push(next.offset);
        }
        flows.push(flow);
    }
    leaders.sort_unstable();
    leaders.dedup();

    let nblocks = leaders.len();
    let mut ranges: Vec<std::ops::Range<usize>> = Vec::with_capacity(nblocks);
    let mut start = 0usize;
    for b in 0..nblocks {
        let next_off = leaders.get(b + 1).copied();
        let end = match next_off {
            Some(off) => list[start..].partition_point(|i| i.offset < off) + start,
            None => list.len(),
        };
        ranges.push(start..end);
        start = end;
    }

    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); nblocks];
    for (b, range) in ranges.iter().enumerate() {
        let Some(last) = range.end.checked_sub(1).filter(|&i| i >= range.start) else {
            continue;
        };
        match flows[last] {
            Flow::Next => {
                if b + 1 < nblocks {
                    succs[b].push(b + 1);
                }
            }
            Flow::Jump(t) => {
                if let Ok(tb) = leaders.binary_search(&t) {
                    succs[b].push(tb);
                }
            }
            Flow::Branch(t) => {
                if let Ok(tb) = leaders.binary_search(&t) {
                    succs[b].push(tb);
                }
                if b + 1 < nblocks {
                    succs[b].push(b + 1);
                }
            }
            _ => {}
        }
    }

    let mut entry = [SpillTag::None; 106];
    entry[0] = SpillTag::Low(0);
    entry[1] = SpillTag::High;
    let mut in_tags: Vec<Option<[SpillTag; 106]>> = vec![None; nblocks];
    in_tags[0] = Some(entry);

    let max_rounds = nblocks.saturating_mul(4).max(8);
    for _ in 0..max_rounds {
        let mut changed = false;
        for b in 0..nblocks {
            let Some(mut tag) = in_tags[b] else {
                continue;
            };
            for inst in &list[ranges[b].clone()] {
                let f = inst.fields(wave64);
                let name = inst.name().unwrap_or("");
                step_spill_tag(&mut tag, &f, name);
            }
            for &s in &succs[b] {
                match &mut in_tags[s] {
                    None => {
                        in_tags[s] = Some(tag);
                        changed = true;
                    }
                    Some(cur) => {
                        for r in 0..106 {
                            if cur[r] != SpillTag::None && cur[r] != tag[r] {
                                cur[r] = SpillTag::None;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    let mut out = Vec::with_capacity(list.len());
    let mut iter = list.into_iter();
    for (b, range) in ranges.into_iter().enumerate() {
        let mut tag = in_tags[b].unwrap_or([SpillTag::None; 106]);
        for _ in range {
            let inst = iter.next().expect("range within list");
            let f = inst.fields(wave64);
            let name = inst.name().unwrap_or("");
            if let Smem {
                data: Some(ref d),
                ref base,
                offset,
                ref soffset,
                ..
            } = f
                && name.starts_with("s_load_dword")
                && let Some((a, 2)) = spill_scalar(base)
                && usize::from(a) + 1 < 106
                && let SpillTag::Low(base_off) = tag[a as usize]
                && tag[a as usize + 1] == SpillTag::High
                && matches!(
                    *soffset,
                    Operand::Scalar { code: 125, .. } | Operand::Const(128)
                )
                && let Some((dst, n)) = spill_scalar(d)
            {
                let total = i64::from(base_off) + i64::from(offset);
                if total >= 0 && total % 4 == 0 {
                    let first = u32::from(SPILL_FIRST) + (total as u32) / 4;
                    if first + u32::from(n) <= 105 {
                        for k in 0..n {
                            let mov = 0xBE80_0000
                                | u32::from(dst + k) << 16
                                | 3 << 8
                                | (first + u32::from(k));
                            let mut bytes = mov.to_le_bytes().to_vec();
                            bytes.extend(0xBF81_0000u32.to_le_bytes()); // s_endpgm
                            let mut one = gfx10::decode(&bytes)?.remove(0);
                            one.offset = inst.offset;
                            out.push(one);
                        }
                        for k in 0..n {
                            if usize::from(dst + k) < 106 {
                                tag[(dst + k) as usize] = SpillTag::None;
                            }
                        }
                        continue;
                    }
                }
            }
            step_spill_tag(&mut tag, &f, name);
            out.push(inst);
        }
    }
    Ok(out)
}

/// Scalar registers the compiler parked in lanes of a VGPR, as scalar moves.
///
/// When it runs out of SGPRs the console compiler saves one in a lane of a
/// spare VGPR (`v_writelane_b32 vN, sX, K`) and takes it back later
/// (`v_readlane_b32 sX, vN, K`). A VGPR whose every use in the program is one
/// of those two forms with a constant lane is such a store: lane K of it is a
/// scalar register by another name. Each (VGPR, lane) gets a register code
/// from [`PARKED`] up, which no machine word can hold, and the save and the
/// restore become `s_mov_b32` to and from it. Returns those moves (index in
/// `list`, destination code, source code) in order, for `build_with` to put
/// the codes into the decoded fields. They are then copies for every later
/// stage: a parked descriptor is traced to its load, and nothing is left of a
/// cross-lane operation, which a console wave run as two host waves could not
/// do where not every lane runs.
///
/// A register pair parked in two neighbouring lanes of a wave64 program (a
/// lane mask, a 64-bit address) moves as a pair (`s_mov_b64`) when every save
/// and every restore of the two lanes come together, so a parked lane mask
/// comes back as the mask it is; otherwise its halves come back as two
/// scalars. The two halves of a save or a restore may be a few instructions
/// apart (the compiler schedules other work between them).
///
/// Nothing changes when a lane is read that no instruction saves into (it
/// would read what the hardware left in the VGPR), or when the codes run out.
fn unspill(
    list: &mut [Instruction],
    leaders: &[usize],
    wave64: bool,
    lanes: u8,
    vgprs: u16,
) -> Vec<(usize, u8, u8)> {
    /// `v_writelane_b32 vN, sX, K` / `v_readlane_b32 sX, vN, K`: (save, N, X, K).
    /// The hardware takes the lane modulo the wave size.
    fn parked(inst: &Instruction, fields: &Fields, wave: u32) -> Option<(bool, u8, u8, u32)> {
        let Fields::Valu(v) = fields else { return None };
        if v.abs != 0
            || v.neg != 0
            || v.opsel != 0
            || v.clamp
            || v.omod != 0
            || v.sdwa.is_some()
            || v.dpp.is_some()
        {
            return None;
        }
        let lane = |op: &Operand| match *op {
            Operand::Const(c @ 128..=192) => Some((u32::from(c) - 128) % wave),
            _ => None,
        };
        match (inst.name()?, v.dst?, v.src.as_slice()) {
            (
                "v_writelane_b32",
                Operand::Vector { reg, dwords: 1 },
                [Operand::Scalar { code, dwords: 1 }, k],
            ) if *code < 105 => Some((true, reg, *code, lane(k)?)),
            (
                "v_readlane_b32",
                Operand::Scalar { code, dwords: 1 },
                [Operand::Vector { reg, dwords: 1 }, k],
            ) if code < 105 => Some((false, *reg, code, lane(k)?)),
            _ => None,
        }
    }
    let wave = 32 * u32::from(lanes);
    // VGPRs used by anything but parking, and per instruction its parking and
    // the scalar codes it reads and writes.
    let mut other = [false; 256];
    let mut parks: Vec<Option<(bool, u8, u8, u32)>> = Vec::with_capacity(list.len());
    let mut written: Vec<Vec<u8>> = Vec::with_capacity(list.len());
    let mut read: Vec<Vec<u8>> = Vec::with_capacity(list.len());
    for inst in list.iter() {
        let Some(name) = inst.name() else {
            return Vec::new();
        };
        let fields = inst.fields(wave64);
        let park = parked(inst, &fields, wave);
        let Ok(effects) = effects(inst, &fields, name, lanes, vgprs) else {
            return Vec::new();
        };
        let (mut to, mut from) = (Vec::new(), Vec::new());
        if let Some((reads, writes)) = effects {
            to.extend(writes.iter().filter_map(|r| match *r {
                Reg::S(c) => Some(c),
                _ => None,
            }));
            from.extend(reads.iter().filter_map(|r| match *r {
                Reg::S(c) => Some(c),
                _ => None,
            }));
            // An M0-relative read (`v_movrels_b32`) names every VGPR from its
            // base up; the compiler indexes a small array with it, never the
            // lanes it parks scalars in, so it does not make those VGPRs
            // ordinary ones. (Relative writes still do.)
            let relative = name == "v_movrels_b32";
            let vector_reads = reads.into_iter().filter(|_| !relative);
            for reg in vector_reads.chain(writes) {
                if let Reg::V(r) = reg
                    && park.is_none_or(|p| p.1 != r)
                {
                    other[usize::from(r)] = true;
                }
            }
        }
        parks.push(park);
        written.push(to);
        read.push(from);
    }
    for park in &mut parks {
        if park.is_some_and(|p| other[usize::from(p.1)]) {
            *park = None;
        }
    }
    let mut slots: Vec<(u8, u32)> = parks.iter().flatten().map(|p| (p.1, p.3)).collect();
    slots.sort_unstable();
    slots.dedup();
    if slots.is_empty() {
        return Vec::new();
    }
    // A lane read that nothing saves into holds what the hardware put there.
    let saved = |slot: (u8, u32)| parks.iter().flatten().any(|p| p.0 && (p.1, p.3) == slot);
    if !slots.iter().all(|&s| saved(s)) {
        return Vec::new();
    }
    let leader = |i: usize| leaders.binary_search(&list[i].offset).is_ok();
    // The instruction that completes the pair begun at `i`: the same kind of
    // move of the next register and the next lane, later in the same block.
    // A save is moved to the second half, so nothing between may write either
    // register or touch the lanes. A restore is moved to the first, so nothing
    // between may read or write the second register, write the first, or save
    // into the lanes.
    let partner = |i: usize| -> Option<usize> {
        let (save, reg, code, lane) = parks[i]?;
        let mut j = i + 1;
        while j < list.len() && !leader(j) {
            if parks[j] == Some((save, reg, code + 1, lane + 1)) {
                return Some(j);
            }
            let lanes_touched = parks[j]
                .is_some_and(|p| (save || p.0) && p.1 == reg && (p.3 == lane || p.3 == lane + 1));
            if lanes_touched
                || written[j].contains(&code)
                || written[j].contains(&(code + 1))
                || !save && read[j].contains(&(code + 1))
            {
                return None;
            }
            j += 1;
        }
        None
    };
    // Lane pairs (register, lower lane) whose every move comes as a pair. Only
    // wave64 programs have them: a wave32 mask is one dword, and a pair move of
    // it with its neighbour would hide it.
    let mut pairs: Vec<(u8, u32)> = Vec::new();
    for &(reg, lane) in &slots {
        if !wave64
            || pairs.iter().any(|&(r, l)| r == reg && l + 1 == lane)
            || !slots.contains(&(reg, lane + 1))
        {
            continue;
        }
        let at = |l: u32| move |p: &(bool, u8, u8, u32)| (p.1, p.3) == (reg, l);
        let lower: Vec<usize> = (0..list.len())
            .filter(|&i| parks[i].as_ref().is_some_and(at(lane)))
            .collect();
        let upper = parks.iter().flatten().filter(|p| at(lane + 1)(p)).count();
        let partners: Option<Vec<usize>> = lower.iter().map(|&i| partner(i)).collect();
        if let Some(mut partners) = partners {
            partners.sort_unstable();
            partners.dedup();
            if partners.len() == lower.len() && upper == lower.len() {
                pairs.push((reg, lane));
            }
        }
    }
    // Codes for the parked scalars, from `PARKED` up to 254 (255 stays unused
    // so that the code after a parked one always exists).
    let mut next = usize::from(PARKED);
    let mut codes: Vec<((u8, u32), u8)> = Vec::new();
    for &(reg, lane) in &pairs {
        if next + 1 > 254 {
            return Vec::new();
        }
        codes.push(((reg, lane), next as u8));
        codes.push(((reg, lane + 1), next as u8 + 1));
        next += 2;
    }
    for &slot in &slots {
        if codes.iter().any(|c| c.0 == slot) {
            continue;
        }
        if next > 254 {
            return Vec::new();
        }
        codes.push((slot, next as u8));
        next += 1;
    }
    let code_of = |reg: u8, lane: u32| codes.iter().find(|c| c.0 == (reg, lane)).map(|c| c.1);
    /// What becomes of a parking instruction: a move (pair, destination,
    /// source), or `None` when the other half of its pair does the move.
    type Edit = Option<(bool, u8, u8)>;
    let mut edits: Vec<(usize, Edit)> = Vec::new();
    for (i, park) in parks.iter().enumerate() {
        let Some((save, reg, code, lane)) = *park else {
            continue;
        };
        let Some(slot) = code_of(reg, lane) else {
            return Vec::new();
        };
        let (dst, src) = if save { (slot, code) } else { (code, slot) };
        if pairs.contains(&(reg, lane)) {
            let Some(j) = partner(i) else {
                return Vec::new();
            };
            // A save moves the pair where its second half is saved; a
            // restore where its first half is restored.
            let (at, gone) = if save { (j, i) } else { (i, j) };
            edits.push((at, Some((true, dst, src))));
            edits.push((gone, None));
        } else if lane == 0 || !pairs.contains(&(reg, lane - 1)) {
            edits.push((i, Some((false, dst, src))));
        }
        // else: the upper half of a pair, moved with the lower.
    }
    // `s_mov_b32` / `s_mov_b64 s0, s0` (SOP1, opcodes 3 and 4) and `s_nop 0`:
    // the parked codes do not fit the word and are put into the fields.
    let mut moves = Vec::new();
    for (i, edit) in edits {
        let word = match edit {
            Some((wide, dst, src)) => {
                moves.push((i, dst, src));
                0xBE80_0000 | (3 + u32::from(wide)) << 8
            }
            None => 0xBF80_0000,
        };
        let mut bytes = word.to_le_bytes().to_vec();
        bytes.extend(0xBF81_0000u32.to_le_bytes()); // s_endpgm
        let Ok(mut one) = gfx10::decode(&bytes) else {
            return Vec::new();
        };
        let mut one = one.remove(0);
        one.offset = list[i].offset;
        list[i] = one;
    }
    moves.sort_unstable();
    moves
}

/// Decodes `code` and builds its SSA form; `vgprs` is the program's VGPR
/// count. Fails on instructions the IR cannot express yet
/// ([`ShaderError::Unsupported`]).
pub fn build(code: &[u8], wave64: bool, vgprs: u16) -> Result<Program, ShaderError> {
    build_with(code, wave64, vgprs, false)
}

/// [`build`]; with `spilled`, loads from the root-argument spill table that
/// `s[0:1]` points to read the user SGPRs the table stands for (table dword
/// `k` is user SGPR `16 + k`), as plain register copies.
pub fn build_with(
    code: &[u8],
    wave64: bool,
    vgprs: u16,
    spilled: bool,
) -> Result<Program, ShaderError> {
    let mut list = gfx10::decode(code)?;
    if spilled {
        list = spill_loads(list, wave64, code.len())?;
    }
    let lanes = 1 + u8::from(wave64);
    let isa = |offset, reason| ShaderError::Isa { offset, reason };

    let mut flows = Vec::with_capacity(list.len());
    let mut leaders = vec![0];
    for (i, inst) in list.iter().enumerate() {
        let flow = gfx10::control(inst, code.len(), |at| {
            list[..list.partition_point(|i| i.offset < at)].last()
        })?;
        if let Flow::Jump(t) | Flow::Branch(t) = flow {
            leaders.push(t);
        }
        if flow != Flow::Next
            && let Some(next) = list.get(i + 1)
        {
            leaders.push(next.offset);
        }
        flows.push(flow);
    }
    leaders.sort_unstable();
    leaders.dedup();
    let parked = unspill(&mut list, &leaders, wave64, lanes, vgprs);
    // When the first instruction is a loop header, give SSA a separate
    // entry edge to carry the initial register values into its phis.
    let preheader = usize::from(
        flows
            .iter()
            .any(|flow| matches!(flow, Flow::Jump(0) | Flow::Branch(0))),
    );
    let block_at = |offset: usize| {
        leaders
            .binary_search(&offset)
            .map(|b| b + preheader)
            .map_err(|_| isa(offset, "jump into a block"))
    };

    let mut blocks: Vec<Block> = leaders
        .iter()
        .map(|&offset| Block {
            offset,
            preds: Vec::new(),
            phis: Vec::new(),
            insts: Vec::new(),
            exit: Exit::End,
        })
        .collect();
    if preheader != 0 {
        blocks.insert(
            0,
            Block {
                offset: 0,
                preds: Vec::new(),
                phis: Vec::new(),
                insts: Vec::new(),
                exit: Exit::Jump(1),
            },
        );
    }
    let next_offsets: Vec<Option<usize>> = (0..list.len())
        .map(|i| list.get(i + 1).map(|n| n.offset))
        .collect();
    for (n, ((machine, flow), next)) in list.into_iter().zip(flows).zip(next_offsets).enumerate() {
        let b = leaders.partition_point(|&l| l <= machine.offset) - 1 + preheader;
        let name = machine.name().ok_or(ShaderError::Unsupported {
            offset: machine.offset,
            what: "unknown opcode",
        })?;
        let mut fields = machine.fields(wave64);
        // A move to or from a scalar parked in a VGPR lane (`unspill`): its
        // codes are ones the machine word cannot hold.
        if let Ok(k) = parked.binary_search_by_key(&n, |m| m.0)
            && let Fields::Sop {
                dst: Some(Operand::Scalar { code: to, .. }),
                src,
                ..
            } = &mut fields
            && let Some(Operand::Scalar { code: from, .. }) = src.first_mut()
        {
            (*to, *from) = (parked[k].1, parked[k].2);
        }
        let fallthrough = || block_at(next.ok_or(isa(machine.offset, "falls off the code"))?);
        // A conditional SOPP branch to fallthrough has no register effects
        // and contributes one edge to the join. Calls also use Flow::Branch
        // but write a return address.
        if machine.encoding == gfx10::Encoding::Sopp
            && matches!(flow, Flow::Branch(target) if Some(target) == next)
        {
            blocks[b].exit = Exit::Jump(fallthrough()?);
            continue;
        }
        let ends = flow != Flow::Next
            || next.is_none_or(|n| n != machine.offset && leaders.binary_search(&n).is_ok());
        if ends {
            blocks[b].exit = match flow {
                Flow::Next => Exit::Jump(fallthrough()?),
                Flow::Jump(t) => Exit::Jump(block_at(t)?),
                Flow::Branch(t) => {
                    let taken = block_at(t)?;
                    let not_taken = fallthrough()?;
                    // The same successor needs only one incoming SSA edge.
                    if taken == not_taken {
                        Exit::Jump(taken)
                    } else {
                        Exit::Branch { taken, not_taken }
                    }
                }
                Flow::Stop if name == "s_setpc_b64" => Exit::Indirect,
                Flow::Stop => Exit::End,
            };
        }
        if let Some((mut reads, writes)) = effects(&machine, &fields, name, lanes, vgprs)? {
            // Lanes disabled in EXEC keep the old value.
            for w in writes.iter().filter(|w| matches!(w, Reg::V(_))) {
                if !reads.contains(w) {
                    reads.push(*w);
                }
            }
            let unset = |regs: Vec<Reg>| regs.into_iter().map(|r| (r, NONE)).collect();
            blocks[b].insts.push(Inst {
                machine,
                fields,
                reads: unset(reads),
                writes: unset(writes),
            });
        }
    }
    prune_unreachable(&mut blocks);
    link(&mut blocks);
    split_irreducible(&mut blocks)?;

    Ok(rename(blocks, wave64, vgprs))
}

/// Drops blocks not reachable from entry block `0` and remaps remaining
/// [`Exit`] targets, keeping reachable blocks in their original order.
fn prune_unreachable(blocks: &mut Vec<Block>) {
    if blocks.is_empty() {
        return;
    }
    let mut reachable = vec![false; blocks.len()];
    let mut stack = vec![0];
    reachable[0] = true;
    while let Some(b) = stack.pop() {
        for s in blocks[b].exit.successors() {
            if !reachable[s] {
                reachable[s] = true;
                stack.push(s);
            }
        }
    }
    if reachable.iter().all(|&r| r) {
        return;
    }
    let mut remap = vec![usize::MAX; blocks.len()];
    let mut next = 0;
    for (i, &keep) in reachable.iter().enumerate() {
        if keep {
            remap[i] = next;
            next += 1;
        }
    }
    let mut idx = 0;
    blocks.retain(|_| {
        let keep = reachable[idx];
        idx += 1;
        keep
    });
    for block in blocks.iter_mut() {
        match &mut block.exit {
            Exit::Jump(t) => *t = remap[*t],
            Exit::Branch { taken, not_taken } => {
                *taken = remap[*taken];
                *not_taken = remap[*not_taken];
            }
            Exit::End | Exit::Indirect => {}
        }
    }
}

/// Rename register reads and writes after changing edges in a reducible graph.
fn rename(mut blocks: Vec<Block>, wave64: bool, vgprs: u16) -> Program {
    let mut order = reverse_postorder(&blocks);
    // `link` records every CFG predecessor, including from blocks never
    // reached from entry. Renaming and liveness only visit the reachable
    // set (`order`), so unreachable preds leave empty `outs` and would
    // insert `NONE` into phi.args (and panic `remove_trivial_phis`).
    if order.len() != blocks.len() {
        prune_unreachable(&mut blocks);
        link(&mut blocks);
        order = reverse_postorder(&blocks);
    }
    debug_assert_eq!(order.len(), blocks.len());
    let (live_in, live_out) = liveness(&blocks, &order);
    let mut values = 0u32;
    let mut new_value = || {
        values += 1;
        Value(values - 1)
    };
    let mut inputs = Vec::new();
    let mut outs: Vec<Vec<(Reg, Value)>> = vec![Vec::new(); blocks.len()];
    let mut current = [NONE; REGS];
    for &b in &order {
        current.fill(NONE);
        let block = &mut blocks[b];
        for reg in members(&live_in[b]) {
            current[index(reg)] = match block.preds[..] {
                [] => {
                    let v = new_value();
                    inputs.push((reg, v));
                    v
                }
                // A single predecessor comes earlier in reverse postorder.
                [p] => lookup(&outs[p], reg),
                _ => {
                    let value = new_value();
                    block.phis.push(Phi {
                        reg,
                        value,
                        args: Vec::new(),
                    });
                    value
                }
            };
        }
        for inst in &mut block.insts {
            for (reg, v) in &mut inst.reads {
                *v = current[index(*reg)];
                debug_assert_ne!(*v, NONE, "{reg:?} read at {:#x}", inst.machine.offset);
            }
            for (reg, v) in &mut inst.writes {
                *v = new_value();
                current[index(*reg)] = *v;
            }
        }
        outs[b] = members(&live_out[b])
            .map(|reg| (reg, current[index(reg)]))
            .collect();
    }
    for block in &mut blocks {
        for phi in &mut block.phis {
            phi.args = block
                .preds
                .iter()
                .map(|&p| lookup(&outs[p], phi.reg))
                .collect();
            debug_assert!(!phi.args.contains(&NONE), "{phi:?}");
        }
    }
    remove_trivial_phis(&mut blocks, values);
    Program {
        wave64,
        vgprs,
        blocks,
        inputs,
    }
}

/// Duplicate a join for each proven (predecessor, join, destination) edge.
/// The instructions still execute on that path and define fresh SSA values.
pub(crate) fn thread_edges(
    p: &Program,
    edges: &[(usize, usize, usize)],
) -> Result<Program, ShaderError> {
    let mut blocks = p.blocks.clone();
    for block in &mut blocks {
        block.phis.clear();
    }
    for &(pred, join, dest) in edges {
        let mut copy = blocks[join].clone();
        copy.exit = Exit::Jump(dest);
        let target = blocks.len();
        match &mut blocks[pred].exit {
            Exit::Jump(t) => *t = target,
            Exit::Branch { taken, not_taken } => {
                if *taken == join {
                    *taken = target;
                }
                if *not_taken == join {
                    *not_taken = target;
                }
            }
            _ => unreachable!("threading an incoming edge"),
        }
        blocks.push(copy);
    }
    prune_unreachable(&mut blocks);
    link(&mut blocks);
    split_irreducible(&mut blocks)?;
    Ok(rename(blocks, p.wave64, p.vgprs))
}

/// Placeholder before renaming.
const NONE: Value = Value(u32::MAX);
/// SGPR codes 0-127, SCC, VGPRs 0-255.
/// Scalar codes 0-255 (from [`PARKED`] up: parked scalars), SCC, VGPRs 0-255.
const REGS: usize = 256 + 1 + 256;
type RegSet = [u64; REGS.div_ceil(64)];

fn index(reg: Reg) -> usize {
    match reg {
        Reg::S(c) => usize::from(c),
        Reg::Scc => 256,
        Reg::V(r) => 257 + usize::from(r),
    }
}

fn reg_at(i: usize) -> Reg {
    match i {
        0..256 => Reg::S(i as u8),
        256 => Reg::Scc,
        _ => Reg::V((i - 257) as u8),
    }
}

fn insert(set: &mut RegSet, reg: Reg) {
    let i = index(reg);
    set[i / 64] |= 1 << (i % 64);
}

fn contains(set: &RegSet, reg: Reg) -> bool {
    let i = index(reg);
    set[i / 64] >> (i % 64) & 1 != 0
}

/// In [`Reg`] order, which is also index order.
fn members(set: &RegSet) -> impl Iterator<Item = Reg> + '_ {
    (0..REGS)
        .filter(|&i| set[i / 64] >> (i % 64) & 1 != 0)
        .map(reg_at)
}

/// `outs` is sorted by register, and holds every register live out.
fn lookup(outs: &[(Reg, Value)], reg: Reg) -> Value {
    outs.binary_search_by_key(&reg, |&(r, _)| r)
        .map_or(NONE, |i| outs[i].1)
}

pub(crate) fn reverse_postorder(blocks: &[Block]) -> Vec<usize> {
    let mut seen = vec![false; blocks.len()];
    let mut post = Vec::with_capacity(blocks.len());
    let mut stack = vec![(0, 0)];
    seen[0] = true;
    while let Some(top) = stack.last_mut() {
        let (b, i) = *top;
        top.1 += 1;
        match blocks[b].exit.successor_at(i) {
            Some(s) if !seen[s] => {
                seen[s] = true;
                stack.push((s, 0));
            }
            Some(_) => {}
            None => {
                post.push(b);
                stack.pop();
            }
        }
    }
    post.reverse();
    post
}

/// Recomputes every block's predecessors from the exits.
fn link(blocks: &mut [Block]) {
    for block in blocks.iter_mut() {
        block.preds.clear();
    }
    for b in 0..blocks.len() {
        for s in blocks[b].exit.successor_iter() {
            blocks[s].preds.push(b);
        }
    }
}

/// An edge to an earlier block in reverse postorder whose target does not
/// dominate its source, if any: there the control flow is irreducible.
/// `index` is each block's position in `order`.
pub(crate) fn irreducible_edge(
    blocks: &[Block],
    order: &[usize],
    index: &[usize],
) -> Option<(usize, usize)> {
    let &entry = order.first()?;
    let idom = idoms(blocks, order, index);
    for &b in order {
        for s in blocks[b].exit.successor_iter() {
            if index[s] > index[b] {
                continue;
            }
            let mut d = b;
            while d != s && d != entry {
                d = idom[d];
            }
            if d != s {
                return Some((b, s));
            }
        }
    }
    None
}

/// Each block's immediate dominator (the entry's is itself, unreachable
/// blocks' `usize::MAX`); `order` is reverse postorder, `index` each
/// block's position in it.
pub(crate) fn idoms(blocks: &[Block], order: &[usize], index: &[usize]) -> Vec<usize> {
    let mut idom = vec![usize::MAX; blocks.len()];
    let Some(&entry) = order.first() else {
        return idom;
    };
    idom[entry] = entry;
    let intersect = |idom: &[usize], mut a: usize, mut b: usize| {
        while a != b {
            while index[a] > index[b] {
                a = idom[a];
            }
            while index[b] > index[a] {
                b = idom[b];
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &order[1..] {
            let mut new = usize::MAX;
            for &q in blocks[b].preds.iter().filter(|&&q| idom[q] != usize::MAX) {
                new = if new == usize::MAX {
                    q
                } else {
                    intersect(&idom, q, new)
                };
            }
            if idom[b] != new {
                idom[b] = new;
                changed = true;
            }
        }
    }
    idom
}

/// Node splitting, before SSA. While a loop is entered at several blocks
/// (a compiler-rotated loop whose peeled first test jumps into its body),
/// the entry first in reverse postorder stays the header, and another
/// entry's outside predecessors move to a copy of everything that entry
/// reaches in the loop before the header. The copies run the same
/// instructions, so behaviour is unchanged.
fn split_irreducible(blocks: &mut Vec<Block>) -> Result<(), ShaderError> {
    let limit = 4 * blocks.len() + 64;
    loop {
        let order = reverse_postorder(blocks);
        let mut index = vec![usize::MAX; blocks.len()];
        for (i, &b) in order.iter().enumerate() {
            index[b] = i;
        }
        let Some((from, _)) = irreducible_edge(blocks, &order, &index) else {
            return Ok(());
        };
        let found = multi_entry_loop(blocks, &vec![true; blocks.len()]);
        let Some((region, mut entries)) = found.filter(|_| blocks.len() < limit) else {
            return Err(ShaderError::Unsupported {
                offset: blocks[from].offset,
                what: "irreducible control flow",
            });
        };
        entries.sort_by_key(|&e| index[e]);
        let (header, entry) = (entries[0], entries[1]);

        // What `entry` reaches in the loop before the header, and its copies.
        let mut copy = vec![usize::MAX; blocks.len()];
        let mut copied = Vec::new();
        let mut stack = vec![entry];
        while let Some(b) = stack.pop() {
            if copy[b] != usize::MAX {
                continue;
            }
            copy[b] = blocks.len() + copied.len();
            copied.push(b);
            for s in blocks[b].exit.successor_iter() {
                if region[s] && s != header {
                    stack.push(s);
                }
            }
        }
        let retarget = |exit: &mut Exit, to: &dyn Fn(usize) -> usize| match exit {
            Exit::Jump(s) => *s = to(*s),
            Exit::Branch { taken, not_taken } => {
                *taken = to(*taken);
                *not_taken = to(*not_taken);
            }
            Exit::End | Exit::Indirect => {}
        };
        for &b in &copied {
            let mut block = blocks[b].clone();
            retarget(&mut block.exit, &|s| {
                if copy[s] == usize::MAX { s } else { copy[s] }
            });
            blocks.push(block);
        }
        let outside: Vec<usize> = blocks[entry]
            .preds
            .iter()
            .copied()
            .filter(|&p| !region[p])
            .collect();
        for p in outside {
            retarget(&mut blocks[p].exit, &|s| {
                if s == entry { copy[entry] } else { s }
            });
        }
        link(blocks);
    }
}

/// A loop among the `inside` blocks entered at more than one block, with
/// its entries. Loops nested in a single-entry loop are searched with its
/// header removed, so one is found whenever the control flow is irreducible.
fn multi_entry_loop(blocks: &[Block], inside: &[bool]) -> Option<(Vec<bool>, Vec<usize>)> {
    let reach = |from: usize, forward: bool| {
        let mut seen = vec![false; blocks.len()];
        let mut stack = vec![from];
        while let Some(b) = stack.pop() {
            let mut visit = |n: usize| {
                if inside[n] && !seen[n] {
                    seen[n] = true;
                    stack.push(n);
                }
            };
            if forward {
                blocks[b].exit.successor_iter().for_each(&mut visit);
            } else {
                blocks[b].preds.iter().copied().for_each(&mut visit);
            }
        }
        seen
    };
    let mut done = vec![false; blocks.len()];
    for v in (0..blocks.len()).filter(|&v| inside[v]) {
        if done[v] {
            continue;
        }
        // Blocks on a cycle through `v`; `v` itself only if there is one.
        let (ahead, behind) = (reach(v, true), reach(v, false));
        let region: Vec<bool> = ahead.iter().zip(&behind).map(|(a, b)| a & b).collect();
        if !region[v] {
            continue;
        }
        let members = (0..blocks.len()).filter(|&b| region[b]);
        for b in members.clone() {
            done[b] = true;
        }
        let entries: Vec<usize> = members
            .filter(|&b| blocks[b].preds.iter().any(|&p| !region[p]))
            .collect();
        match entries[..] {
            [] => {}
            [header] => {
                let mut nested = region;
                nested[header] = false;
                if let Some(found) = multi_entry_loop(blocks, &nested) {
                    return Some(found);
                }
            }
            _ => return Some((region, entries)),
        }
    }
    None
}

/// Registers live into and out of each block.
fn liveness(blocks: &[Block], order: &[usize]) -> (Vec<RegSet>, Vec<RegSet>) {
    let empty = [0; REGS.div_ceil(64)];
    let mut uses = vec![empty; blocks.len()];
    let mut defs = vec![empty; blocks.len()];
    for (b, block) in blocks.iter().enumerate() {
        for inst in &block.insts {
            for &(reg, _) in &inst.reads {
                if !contains(&defs[b], reg) {
                    insert(&mut uses[b], reg);
                }
            }
            for &(reg, _) in &inst.writes {
                insert(&mut defs[b], reg);
            }
        }
    }
    let mut live_in = vec![empty; blocks.len()];
    let mut live_out = vec![empty; blocks.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for &b in order.iter().rev() {
            let mut out = empty;
            for s in blocks[b].exit.successor_iter() {
                for (o, i) in out.iter_mut().zip(&live_in[s]) {
                    *o |= i;
                }
            }
            let mut inn = empty;
            for w in 0..inn.len() {
                inn[w] = uses[b][w] | out[w] & !defs[b][w];
            }
            if inn != live_in[b] || out != live_out[b] {
                (live_in[b], live_out[b]) = (inn, out);
                changed = true;
            }
        }
    }
    (live_in, live_out)
}

/// Phis are placed for every live register at every join; those whose
/// arguments are all one value (or the phi itself) are replaced by it.
fn remove_trivial_phis(blocks: &mut [Block], values: u32) {
    remove_trivial_phis_steps(blocks, values);
}

fn remove_trivial_phis_steps(blocks: &mut [Block], values: u32) -> usize {
    fn find(replaced: &mut [Value], v: Value, steps: &mut usize) -> Value {
        let mut root = v;
        while replaced[root.0 as usize] != root {
            *steps += 1;
            root = replaced[root.0 as usize];
        }
        let mut cur = v;
        while replaced[cur.0 as usize] != root {
            *steps += 1;
            let next = replaced[cur.0 as usize];
            replaced[cur.0 as usize] = root;
            cur = next;
        }
        root
    }

    let mut replaced: Vec<Value> = (0..values).map(Value).collect();
    let mut steps = 0usize;
    let mut changed = true;
    while changed {
        changed = false;
        for block in blocks.iter_mut() {
            block.phis.retain(|phi| {
                let mut only = None;
                for &a in &phi.args {
                    let a = find(&mut replaced, a, &mut steps);
                    if a == phi.value || Some(a) == only {
                        continue;
                    }
                    if only.is_some() {
                        return true;
                    }
                    only = Some(a);
                }
                let Some(v) = only else { return true };
                replaced[phi.value.0 as usize] = v;
                changed = true;
                false
            });
        }
    }
    for block in blocks {
        for phi in &mut block.phis {
            for a in &mut phi.args {
                *a = find(&mut replaced, *a, &mut steps);
            }
        }
        for inst in &mut block.insts {
            for (_, v) in &mut inst.reads {
                *v = find(&mut replaced, *v, &mut steps);
            }
        }
    }
    steps
}

/// Scheduling hints, performance counters and tracing: no effect on results.
const DROPPED: &[&str] = &[
    "s_nop",
    "s_waitcnt",
    "s_waitcnt_vscnt",
    "s_waitcnt_vmcnt",
    "s_waitcnt_expcnt",
    "s_waitcnt_lgkmcnt",
    "s_waitcnt_depctr",
    "s_wait_idle",
    "s_inst_prefetch",
    "s_clause",
    "s_setprio",
    "s_sleep",
    "s_ttracedata",
    "s_ttracedata_imm",
    "s_incperflevel",
    "s_decperflevel",
    "v_nop",
    "v_pipeflush",
    "v_clrexcp",
];

/// Registers read, registers written.
type Effects = (Vec<Reg>, Vec<Reg>);

/// The registers `inst` of `p` reads as operands: its reads without the old
/// value of a VGPR it only writes.
pub(crate) fn operand_reads(p: &Program, inst: &Inst) -> Vec<Reg> {
    let lanes = 1 + u8::from(p.wave64);
    effects(&inst.machine, &inst.fields, inst.name(), lanes, p.vgprs)
        .ok()
        .flatten()
        .map_or_else(Vec::new, |e| e.0)
}

/// Registers an instruction reads and writes, one per dword, or `None` if
/// the IR drops it.
fn effects(
    inst: &Instruction,
    fields: &Fields,
    name: &'static str,
    lanes: u8,
    vgprs: u16,
) -> Result<Option<Effects>, ShaderError> {
    if DROPPED.contains(&name) {
        return Ok(None);
    }
    let unsupported = || ShaderError::Unsupported {
        offset: inst.offset,
        what: name,
    };
    let mask = |code: u8| (0..lanes).map(move |i| Reg::S(code + i));
    let (mut r, mut w) = (Vec::new(), Vec::new());
    let regs = |op: &Operand, out: &mut Vec<Reg>| {
        add_regs(op, lanes, out).map_err(|_| ShaderError::Isa {
            offset: inst.offset,
            reason: "register range out of bounds",
        })
    };
    let vector = !matches!(
        fields,
        Fields::Sop { .. } | Fields::Sopp { .. } | Fields::Smem { .. }
    );
    match fields {
        Fields::Sop { dst, src, .. } => {
            if name.starts_with("s_movrel")
                || name.starts_with("s_subvector_loop")
                || matches!(name, "s_call_b64" | "s_rfe_b64")
            {
                return Err(unsupported());
            }
            for s in src {
                regs(s, &mut r)?;
            }
            if let Some(d) = dst {
                regs(d, &mut w)?;
                // These change (or may keep) the destination's old value.
                if name.starts_with("s_bitset")
                    || name.starts_with("s_cmov")
                    || matches!(name, "s_addk_i32" | "s_mulk_i32")
                {
                    regs(d, &mut r)?;
                }
            }
            if matches!(name, "s_addc_u32" | "s_subb_u32" | "s_cmovk_i32")
                || name.starts_with("s_cselect_")
                || name.starts_with("s_cmov_")
            {
                r.push(Reg::Scc);
            }
            if name.contains("saveexec") || name.contains("wrexec") {
                r.extend(mask(EXEC));
                w.extend(mask(EXEC));
            }
            if writes_scc(inst.encoding, name) {
                w.push(Reg::Scc);
            }
        }
        Fields::Sopp { .. } => match name {
            "s_cbranch_scc0" | "s_cbranch_scc1" => r.push(Reg::Scc),
            "s_cbranch_vccz" | "s_cbranch_vccnz" => r.extend(mask(VCC)),
            "s_cbranch_execz" | "s_cbranch_execnz" => r.extend(mask(EXEC)),
            "s_sendmsg" | "s_sendmsghalt" => r.push(Reg::S(M0)),
            "s_endpgm"
            | "s_endpgm_saved"
            | "s_endpgm_ordered_ps_done"
            | "s_code_end"
            | "s_branch"
            | "s_barrier"
            | "s_icache_inv"
            | "s_wakeup"
            | "s_round_mode"
            | "s_denorm_mode" => {}
            _ => return Err(unsupported()),
        },
        Fields::Smem {
            data,
            base,
            soffset,
            ..
        } => match name {
            "s_memtime" | "s_memrealtime" | "s_get_waveid_in_workgroup" => {
                regs(data.as_ref().ok_or_else(unsupported)?, &mut w)?;
            }
            "s_gl1_inv" | "s_dcache_inv" | "s_dcache_wb" => {}
            _ if name.contains("_load_") || name.contains("_store_") => {
                regs(base, &mut r)?;
                regs(soffset, &mut r)?;
                let data = data.as_ref().ok_or_else(unsupported)?;
                regs(
                    data,
                    if name.contains("_load_") {
                        &mut w
                    } else {
                        &mut r
                    },
                )?;
            }
            _ => return Err(unsupported()),
        },
        Fields::Valu(v) => {
            let (src_indexed, dst_indexed) = match name {
                "v_movrels_b32" => (true, false),
                "v_movreld_b32" => (false, true),
                "v_movrelsd_b32" | "v_movrelsd_2_b32" | "v_swaprel_b32" => (true, true),
                _ => (false, false),
            };
            // v[reg + M0] for any M0 that stays inside the program's VGPRs.
            let indexed = |op: &Operand, out: &mut Vec<Reg>| match *op {
                Operand::Vector { reg, .. } if u16::from(reg) < vgprs => {
                    out.extend((u16::from(reg)..vgprs).map(|r| Reg::V(r as u8)));
                    Ok(())
                }
                _ => Err(ShaderError::Isa {
                    offset: inst.offset,
                    reason: "relative VGPR base outside the program's VGPRs",
                }),
            };
            for (i, s) in v.src.iter().enumerate() {
                if i == 0 && src_indexed {
                    indexed(s, &mut r)?;
                } else {
                    regs(s, &mut r)?;
                }
            }
            // GFX10 compares into EXEC write nothing else.
            if name.starts_with("v_cmpx") {
                w.extend(mask(EXEC));
            } else if let Some(d) = &v.dst {
                if dst_indexed {
                    indexed(d, &mut w)?;
                } else {
                    regs(d, &mut w)?;
                }
            }
            if let Some(c) = &v.carry {
                regs(c, &mut w)?;
            }
            match name {
                "v_swap_b32" => {
                    regs(&v.src[0], &mut w)?;
                    if let Some(d) = &v.dst {
                        regs(d, &mut r)?;
                    }
                }
                "v_swaprel_b32" => {
                    indexed(&v.src[0], &mut w)?;
                    if let Some(d) = &v.dst {
                        indexed(d, &mut r)?;
                    }
                }
                _ => {}
            }
            if name.starts_with("v_interp_") || src_indexed || dst_indexed {
                r.push(Reg::S(M0));
            }
        }
        Fields::Vintrp { dst, src, .. } => {
            regs(src, &mut r)?;
            r.push(Reg::S(M0));
            regs(dst, &mut w)?;
        }
        Fields::Ds {
            dst,
            addr,
            data,
            gds,
            ..
        } => {
            for o in addr.iter().chain(data) {
                regs(o, &mut r)?;
            }
            if let Some(d) = dst {
                regs(d, &mut w)?;
            }
            if *gds
                || name.starts_with("ds_gws")
                || name.contains("addtid")
                || matches!(name, "ds_ordered_count" | "ds_append" | "ds_consume")
            {
                r.push(Reg::S(M0));
            }
        }
        Fields::Buffer {
            data,
            addr,
            rsrc,
            soffset,
            offen,
            idxen,
            lds,
            cache,
            ..
        } => {
            if *offen || *idxen {
                regs(addr, &mut r)?;
            }
            regs(rsrc, &mut r)?;
            regs(soffset, &mut r)?;
            if *lds {
                r.push(Reg::S(M0));
            }
            if let Some(d) = data {
                if name.contains("_load_") {
                    if !lds {
                        regs(d, &mut w)?;
                    }
                } else if name.contains("_store_") {
                    regs(d, &mut r)?;
                } else if name.contains("_atomic_") {
                    regs(d, &mut r)?;
                    if cache.glc {
                        regs(&atomic_result(d, name), &mut w)?;
                    }
                } else {
                    return Err(unsupported());
                }
            }
        }
        Fields::Image {
            data,
            addr,
            rsrc,
            sampler,
            cache,
            ..
        } => {
            for a in addr.iter().chain([rsrc]).chain(sampler) {
                regs(a, &mut r)?;
            }
            if name.contains("_store") {
                regs(data, &mut r)?;
            } else if name.contains("_atomic_") {
                regs(data, &mut r)?;
                if cache.glc {
                    regs(&atomic_result(data, name), &mut w)?;
                }
            } else {
                regs(data, &mut w)?;
            }
        }
        Fields::Flat {
            dst,
            addr,
            data,
            saddr,
            ..
        } => {
            for o in [addr, data, saddr].into_iter().flatten() {
                regs(o, &mut r)?;
            }
            if let Some(d) = dst {
                regs(d, &mut w)?;
            }
        }
        Fields::Exp { src, .. } => {
            for s in src.iter().flatten() {
                regs(s, &mut r)?;
            }
        }
    }
    if vector {
        r.extend(mask(EXEC));
    }
    let mut seen = Vec::with_capacity(r.len());
    r.retain(|reg| {
        let new = !seen.contains(reg);
        seen.push(*reg);
        new
    });
    Ok(Some((r, w)))
}

/// Scalar ALU opcodes that set SCC (carry, overflow, comparison, or "result
/// is not zero").
fn writes_scc(encoding: gfx10::Encoding, name: &str) -> bool {
    use gfx10::Encoding::*;
    match encoding {
        Sopc => true,
        Sopk => name.starts_with("s_cmpk_") || name == "s_addk_i32",
        _ => {
            let stem = name.split('_').nth(1).unwrap_or("");
            name.contains("saveexec")
                || name.contains("wrexec")
                || matches!(
                    stem,
                    "add"
                        | "sub"
                        | "addc"
                        | "subb"
                        | "min"
                        | "max"
                        | "and"
                        | "or"
                        | "xor"
                        | "andn2"
                        | "orn2"
                        | "nand"
                        | "nor"
                        | "xnor"
                        | "lshl"
                        | "lshr"
                        | "ashr"
                        | "bfe"
                        | "absdiff"
                        | "lshl1"
                        | "lshl2"
                        | "lshl3"
                        | "lshl4"
                        | "not"
                        | "wqm"
                        | "bcnt0"
                        | "bcnt1"
                        | "quadmask"
                        | "abs"
                )
        }
    }
}

/// What an atomic with `glc` returns: the old value (for compare-and-swap,
/// half the data operand).
fn atomic_result(data: &Operand, name: &str) -> Operand {
    match *data {
        Operand::Vector { reg, dwords } if name.contains("cmpswap") => Operand::Vector {
            reg,
            dwords: dwords / 2,
        },
        op => op,
    }
}

/// Appends the register dwords an operand names; reading `vccz`, `execz`,
/// `scc` or `lds_direct` reads the register behind it.
fn add_regs(op: &Operand, lanes: u8, out: &mut Vec<Reg>) -> Result<(), ()> {
    match *op {
        Operand::Scalar { code, .. } if code == gfx10::NULL => {}
        Operand::Scalar { code, dwords } => {
            let end = u16::from(code) + u16::from(dwords);
            // A machine operand ends within the machine's codes, a parked
            // scalar (`unspill`) within those past them.
            let limit = if code < PARKED { 128 } else { 256 };
            if end > limit {
                return Err(());
            }
            out.extend((u16::from(code)..end).map(|c| Reg::S(c as u8)));
        }
        Operand::Vector { reg, dwords } => {
            if u16::from(reg) + u16::from(dwords) > 256 {
                return Err(());
            }
            out.extend((0..dwords).map(|i| Reg::V(reg + i)));
        }
        Operand::Const(251) => out.extend((0..lanes).map(|i| Reg::S(VCC + i))),
        Operand::Const(252) => out.extend((0..lanes).map(|i| Reg::S(EXEC + i))),
        Operand::Const(253) => out.push(Reg::Scc),
        Operand::Const(254) => out.push(Reg::S(M0)),
        Operand::Const(_) | Operand::Literal(_) | Operand::Attr { .. } => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_postorder(blocks: &[Block]) -> Vec<usize> {
        let mut seen = vec![false; blocks.len()];
        let mut post = Vec::with_capacity(blocks.len());
        let mut stack = vec![(0, 0)];
        seen[0] = true;
        while let Some(top) = stack.last_mut() {
            let (b, i) = *top;
            top.1 += 1;
            match blocks[b].exit.successors().get(i) {
                Some(&s) if !seen[s] => {
                    seen[s] = true;
                    stack.push((s, 0));
                }
                Some(_) => {}
                None => {
                    post.push(b);
                    stack.pop();
                }
            }
        }
        post.reverse();
        post
    }

    #[test]
    fn cfg_successors_preserve_order() {
        for exit in [
            Exit::End,
            Exit::Indirect,
            Exit::Jump(3),
            Exit::Branch {
                taken: 7,
                not_taken: 2,
            },
            Exit::Branch {
                taken: 4,
                not_taken: 4,
            },
        ] {
            assert_eq!(exit.successor_iter().collect::<Vec<_>>(), exit.successors());
        }
    }

    fn synthetic_cfg(diamonds: usize) -> Vec<Block> {
        let block = |exit| Block {
            offset: 0,
            preds: Vec::new(),
            phis: Vec::new(),
            insts: Vec::new(),
            exit,
        };

        let mut blocks = Vec::new();

        for i in 0..diamonds {
            let base = i * 3;

            blocks.push(block(Exit::Branch {
                taken: base + 1,
                not_taken: base + 2,
            }));

            blocks.push(block(Exit::Jump(base + 3)));

            let exit = if i % 5 == 0 {
                Exit::Branch {
                    taken: base,
                    not_taken: base + 3,
                }
            } else {
                Exit::Jump(base + 3)
            };

            blocks.push(block(exit));
        }

        blocks.push(block(Exit::End));

        // Unreachable block.
        blocks.push(block(Exit::Jump(diamonds * 3 + 1)));

        blocks
    }

    #[test]
    fn cfg_postorder_matches_reference() {
        for count in [0, 1, 2, 16, 128, 2048] {
            let blocks = synthetic_cfg(count);

            let expected = reference_postorder(&blocks);
            let actual = reverse_postorder(&blocks);

            assert_eq!(actual, expected);
            assert_eq!(actual.len(), count * 3 + 1);
        }
    }

    #[test]
    #[ignore = "synthetic release microbenchmark"]
    fn cfg_postorder_benchmark() {
        use std::hint::black_box;
        use std::time::Instant;

        fn measure(blocks: &[Block], run: impl Fn(&[Block]) -> Vec<usize>) -> f64 {
            let start = Instant::now();

            for _ in 0..128 {
                black_box(run(black_box(blocks)));
            }

            start.elapsed().as_secs_f64() * 1000.0
        }

        for count in [256, 2048, 8192] {
            let blocks = synthetic_cfg(count);

            assert_eq!(reference_postorder(&blocks), reverse_postorder(&blocks));

            black_box(measure(&blocks, reference_postorder));
            black_box(measure(&blocks, reverse_postorder));

            let mut original = Vec::new();
            let mut candidate = Vec::new();

            for round in 0..5 {
                let (a, b) = if round % 2 == 0 {
                    let a = measure(&blocks, reference_postorder);
                    let b = measure(&blocks, reverse_postorder);
                    (a, b)
                } else {
                    let b = measure(&blocks, reverse_postorder);
                    let a = measure(&blocks, reference_postorder);
                    (a, b)
                };

                original.push(a);
                candidate.push(b);

                println!(
                    "blocks={} round={round}: \
                    original={a:.3}ms candidate={b:.3}ms",
                    blocks.len()
                );
            }

            original.sort_by(f64::total_cmp);
            candidate.sort_by(f64::total_cmp);

            println!(
                "MEDIAN blocks={}: \
                 original={:.3}ms candidate={:.3}ms \
                 speedup={:.3}x",
                blocks.len(),
                original[2],
                candidate[2],
                original[2] / candidate[2]
            );
        }
    }

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn build_words(words: &[u32], wave64: bool) -> Program {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        build(&code, wave64, 256).unwrap()
    }

    fn input(p: &Program, reg: Reg) -> Value {
        p.inputs.iter().find(|i| i.0 == reg).expect("input").1
    }

    /// A compute program whose root arguments are in the spill table `s[0:1]`
    /// points to: the loads become copies of the user SGPRs the dwords stand for.
    #[test]
    fn spill_table_loads_read_user_sgprs() {
        let code: Vec<u8> = [
            0xBE82_0300u32, // s_mov_b32 s2, s0
            0x8703_FF01,    // s_and_b32 s3, s1, lit
            0x0000_00FF,    //   0xff
            0xF408_0201,    // s_load_dwordx4 s[8:11], s[2:3], 0x10
            0xFA00_0010,
            0xBE86_0308, // s_mov_b32 s6, s8
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let plain = build_with(&code, false, 256, false).unwrap();
        assert!(
            plain
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| i.name() == "s_load_dwordx4")
        );
        let p = build_with(&code, false, 256, true).unwrap();
        let insts: Vec<_> = p.blocks.iter().flat_map(|b| &b.insts).collect();
        assert!(!insts.iter().any(|i| i.name().starts_with("s_load")));
        // Table dwords 4..8 are user SGPRs 20..24.
        for k in 0..4 {
            assert!(
                p.inputs.iter().any(|i| i.0 == Reg::S(20 + k)),
                "s{} read",
                20 + k
            );
        }
        assert_eq!(
            insts.iter().filter(|i| i.name() == "s_mov_b32").count(),
            1 + 4 + 1
        );
    }

    /// Loads from a pointer that is not the spill table stay loads.
    #[test]
    fn spill_table_leaves_other_loads() {
        let code: Vec<u8> = [
            0xF408_0A00u32, // s_load_dwordx4 s[8:11], s[0:1], 0 (s0:1 not yet copied: tagged)
            0xFA00_0000,
            0xBE82_0380, // s_mov_b32 s2, 0
            0xF408_0201, // s_load_dwordx4 s[8:11], s[2:3], 0 (s2 overwritten)
            0xFA00_0000,
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let p = build_with(&code, false, 256, true).unwrap();
        let loads = p
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| i.name().starts_with("s_load"))
            .count();
        // The first load is from s[0:1] itself, so it is rewritten; the second
        // reads a pointer the program overwrote, so it stays.
        assert_eq!(loads, 1);
    }

    fn synth_inst(offset: usize, reads: Vec<Reg>, writes: Vec<Reg>) -> Inst {
        let mut machine = gfx10::decode(&S_ENDPGM.to_le_bytes())
            .unwrap()
            .pop()
            .unwrap();
        machine.offset = offset;
        let fields = machine.fields(false);
        Inst {
            machine,
            fields,
            reads: reads.into_iter().map(|r| (r, NONE)).collect(),
            writes: writes.into_iter().map(|w| (w, NONE)).collect(),
        }
    }

    fn synth_block(offset: usize, insts: Vec<Inst>, exit: Exit) -> Block {
        Block {
            offset,
            preds: Vec::new(),
            phis: Vec::new(),
            insts,
            exit,
        }
    }

    /// Unreachable predecessors must not leave `NONE` in phi.args (VIC-BT-002).
    #[test]
    fn prunes_unreachable_predecessor_before_ssa() {
        let mut blocks = vec![
            synth_block(
                0,
                vec![synth_inst(0, vec![], vec![Reg::S(0)])],
                Exit::Jump(2),
            ),
            synth_block(4, vec![], Exit::Jump(2)),
            synth_block(8, vec![synth_inst(8, vec![Reg::S(0)], vec![])], Exit::End),
        ];

        prune_unreachable(&mut blocks);
        link(&mut blocks);
        let p = rename(blocks, false, 256);

        assert!(p.inputs.is_empty());
        assert_eq!(p.blocks.len(), 2);
        assert_eq!(p.blocks[0].exit, Exit::Jump(1));
        assert_eq!(p.blocks[1].preds, [0]);
        assert!(p.blocks[1].phis.is_empty(), "{:?}", p.blocks[1].phis);
        assert_eq!(
            p.blocks[1].insts[0].read(Reg::S(0)),
            p.blocks[0].insts[0].write(Reg::S(0))
        );
    }

    #[test]
    fn prunes_unreachable_edge_into_entry_and_merge() {
        let mut blocks = vec![
            synth_block(
                0,
                vec![],
                Exit::Branch {
                    taken: 1,
                    not_taken: 2,
                },
            ),
            synth_block(
                4,
                vec![synth_inst(4, vec![], vec![Reg::S(0)])],
                Exit::Jump(3),
            ),
            synth_block(
                8,
                vec![synth_inst(8, vec![], vec![Reg::S(0)])],
                Exit::Jump(3),
            ),
            synth_block(12, vec![synth_inst(12, vec![Reg::S(0)], vec![])], Exit::End),
            synth_block(
                16,
                vec![],
                Exit::Branch {
                    taken: 0,
                    not_taken: 3,
                },
            ),
        ];

        prune_unreachable(&mut blocks);
        link(&mut blocks);
        assert!(blocks[0].preds.is_empty());
        let p = rename(blocks, false, 256);

        assert!(p.inputs.is_empty());
        assert_eq!(p.blocks.len(), 4);
        assert_eq!(p.blocks[3].preds, [1, 2]);
        let [phi] = &p.blocks[3].phis[..] else {
            panic!("expected 1 phi, got {:?}", p.blocks[3].phis);
        };
        assert_eq!(phi.reg, Reg::S(0));
        assert_eq!(
            phi.args,
            [
                p.blocks[1].insts[0].write(Reg::S(0)).unwrap(),
                p.blocks[2].insts[0].write(Reg::S(0)).unwrap(),
            ]
        );
        assert!(!phi.args.contains(&NONE));
        assert_eq!(p.blocks[3].insts[0].read(Reg::S(0)), Some(phi.value));
    }

    #[test]
    fn joins_get_phis_only_where_values_differ() {
        let code = [
            0xBF06_0100, // s_cmp_eq_u32 s0, s1
            0xBF85_0001, // s_cbranch_scc1 -> 12
            0xBE82_0380, // s_mov_b32 s2, 0
            0x7E00_0202, // 12: v_mov_b32 v0, s2
            0x7E02_0200, // v_mov_b32 v1, s0
            S_ENDPGM,
        ];
        for wave64 in [false, true] {
            let p = build_words(&code, wave64);
            let [b0, b1, b2] = &p.blocks[..] else {
                panic!("{:?}", p.blocks)
            };
            assert_eq!(
                b0.exit,
                Exit::Branch {
                    taken: 2,
                    not_taken: 1
                }
            );
            assert_eq!(b1.exit, Exit::Jump(2));
            assert_eq!(b2.preds, [0, 1]);
            assert_eq!(b0.insts[1].read(Reg::Scc), b0.insts[0].write(Reg::Scc));

            // s2 differs per path; s0, EXEC, v0 and v1 do not.
            let [phi] = &b2.phis[..] else {
                panic!("{:?}", b2.phis)
            };
            assert_eq!(phi.reg, Reg::S(2));
            assert_eq!(
                phi.args,
                [input(&p, Reg::S(2)), b1.insts[0].write(Reg::S(2)).unwrap()]
            );
            let [mov0, mov1, end] = &b2.insts[..] else {
                panic!()
            };
            assert_eq!(mov0.read(Reg::S(2)), Some(phi.value));
            assert_eq!(mov1.read(Reg::S(0)), Some(input(&p, Reg::S(0))));
            // The old v1 is read: lanes outside EXEC keep it.
            assert_eq!(mov1.read(Reg::V(1)), Some(input(&p, Reg::V(1))));
            assert_eq!(mov1.read(Reg::S(EXEC)), Some(input(&p, Reg::S(EXEC))));
            assert_eq!(mov1.read(Reg::S(EXEC + 1)).is_some(), wave64);
            assert_eq!((end.name(), b2.exit), ("s_endpgm", Exit::End));
        }
    }

    #[test]
    fn a_conditional_branch_to_its_fallthrough_has_one_edge() {
        let p = build_words(
            &[
                0xBF06_0100, // s_cmp_eq_u32 s0, s1
                0xBF85_0002, // s_cbranch_scc1 -> 16
                0xBE82_0380, // s_mov_b32 s2, 0
                0xBF82_0001, // s_branch -> 20
                0xBF85_0000, // 16: both arms go to 20
                0x7E00_0202, // 20: v_mov_b32 v0, s2
                S_ENDPGM,
            ],
            true,
        );
        assert_eq!(p.blocks[2].exit, Exit::Jump(3));
        let join = &p.blocks[3];
        assert_eq!(join.preds, [1, 2]);
        let [phi] = &join.phis[..] else {
            panic!("expected one live phi, got {:?}", join.phis);
        };
        assert_eq!(phi.reg, Reg::S(2));
        assert_eq!(
            phi.args,
            [
                p.blocks[1].insts[0].write(Reg::S(2)).unwrap(),
                input(&p, Reg::S(2)),
            ],
        );
        assert_eq!(join.insts[0].read(Reg::S(2)), Some(phi.value));
    }

    #[test]
    fn in_place_scalar_ops_read_their_destination() {
        let p = build_words(
            &[
                0xBE8D_1D94, // s_bitset1_b32 s13, 20
                0xB78E_0010, // s_addk_i32 s14, 0x10
                0xBE8F_0380, // s_mov_b32 s15, 0
                S_ENDPGM,
            ],
            false,
        );
        let [bitset, addk, mov, _] = &p.blocks[0].insts[..] else {
            panic!("{:?}", p.blocks[0].insts)
        };
        assert_eq!(bitset.read(Reg::S(13)), Some(input(&p, Reg::S(13))));
        assert_eq!(addk.read(Reg::S(14)), Some(input(&p, Reg::S(14))));
        assert_eq!(mov.read(Reg::S(15)), None);
    }

    #[test]
    fn loops_carry_values_through_phis() {
        let p = build_words(
            &[
                0xBE80_0380, // s_mov_b32 s0, 0
                0x8000_8100, // 4: s_add_u32 s0, s0, 1
                0xBF07_8400, // s_cmp_lg_u32 s0, 4
                0xBF85_FFFD, // s_cbranch_scc1 -> 4
                S_ENDPGM,
            ],
            true,
        );
        assert!(p.inputs.is_empty(), "{:?}", p.inputs);
        let [b0, b1, b2] = &p.blocks[..] else {
            panic!()
        };
        assert_eq!(b1.preds, [0, 1]);
        assert_eq!(
            b1.exit,
            Exit::Branch {
                taken: 1,
                not_taken: 2
            }
        );
        let [phi] = &b1.phis[..] else {
            panic!("{:?}", b1.phis)
        };
        let add = &b1.insts[0];
        assert_eq!(phi.reg, Reg::S(0));
        assert_eq!(
            phi.args,
            [
                b0.insts[0].write(Reg::S(0)).unwrap(),
                add.write(Reg::S(0)).unwrap()
            ]
        );
        assert_eq!(add.read(Reg::S(0)), Some(phi.value));
        assert_eq!(b1.insts[1].read(Reg::S(0)), add.write(Reg::S(0)));
        assert_eq!(b2.insts[0].name(), "s_endpgm");
    }

    #[test]
    fn loops_can_branch_to_offset_zero() {
        let p = build_words(
            &[
                0x8000_8100, // 0: s_add_u32 s0, s0, 1
                0xBF07_8400, // s_cmp_lg_u32 s0, 4
                0xBF85_FFFD, // s_cbranch_scc1 -> 0
                S_ENDPGM,
            ],
            false,
        );
        let [entry, header, end] = &p.blocks[..] else {
            panic!("{:?}", p.blocks);
        };
        assert!(entry.preds.is_empty());
        assert!(entry.insts.is_empty());
        assert_eq!(entry.exit, Exit::Jump(1));
        assert_eq!(header.offset, 0);
        assert_eq!(header.preds, [0, 1]);
        assert_eq!(
            header.exit,
            Exit::Branch {
                taken: 1,
                not_taken: 2,
            }
        );
        assert_eq!(end.exit, Exit::End);
        let [phi] = &header.phis[..] else {
            panic!("{:?}", header.phis);
        };
        let add = &header.insts[0];
        assert_eq!(phi.reg, Reg::S(0));
        assert_eq!(
            phi.args,
            [input(&p, Reg::S(0)), add.write(Reg::S(0)).unwrap()]
        );
        assert_eq!(add.read(Reg::S(0)), Some(phi.value));
        assert!(crate::structure::structure(&p).is_ok());
    }

    #[test]
    fn drops_scheduling_and_follows_far_jumps() {
        let p = build_words(
            &[
                0xBF80_0000, // s_nop 0
                0xBE88_1F00, // s_getpc_b64 s[8:9]
                0x8008_FF08,
                20,          // s_add_u32 s8, s8, 20  -> 8 + 20 = 28
                0x8209_8009, // s_addc_u32 s9, s9, 0
                0xBE80_2008, // s_setpc_b64 s[8:9]
                0xFFFF_FFFF, // not code
                S_ENDPGM,
            ],
            false,
        );
        let names: Vec<_> = p.blocks[0].insts.iter().map(Inst::name).collect();
        assert_eq!(
            names,
            ["s_getpc_b64", "s_add_u32", "s_addc_u32", "s_setpc_b64"]
        );
        assert_eq!(p.blocks[0].exit, Exit::Jump(1));
        assert_eq!(p.blocks[1].offset, 28);
        // s_addc_u32 reads the carry s_add_u32 left in SCC.
        let i = &p.blocks[0].insts;
        assert_eq!(i[2].read(Reg::Scc), i[1].write(Reg::Scc));
    }

    #[test]
    fn reports_what_it_cannot_express() {
        let code: Vec<u8> = [0xBE80_2E01_u32, S_ENDPGM] // s_movrels_b32 s0, s1
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        assert_eq!(
            build(&code, false, 256),
            Err(ShaderError::Unsupported {
                offset: 0,
                what: "s_movrels_b32"
            })
        );
    }

    #[test]
    fn relative_vgpr_access_spans_up_to_the_vgpr_count() {
        let code: Vec<u8> = [
            0x7E04_8501_u32, // v_movreld_b32 v2, v1: v[2 + m0] = v1
            0x7E00_8704,     // v_movrels_b32 v0, v4: v0 = v[4 + m0]
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let p = build(&code, false, 6).unwrap();
        let [movreld, movrels, _] = &p.blocks[0].insts[..] else {
            panic!()
        };
        let vgprs = |regs: &[(Reg, Value)]| -> Vec<u8> {
            regs.iter()
                .filter_map(|r| match r.0 {
                    Reg::V(v) => Some(v),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(vgprs(&movreld.writes), [2, 3, 4, 5]);
        // v1 is the source; v2-v5 are read for the lanes and indices not written.
        assert_eq!(vgprs(&movreld.reads), [1, 2, 3, 4, 5]);
        assert!(movreld.read(Reg::S(M0)).is_some());
        assert_eq!(vgprs(&movrels.writes), [0]);
        assert_eq!(vgprs(&movrels.reads), [4, 5, 0]);
        assert_eq!(movrels.read(Reg::V(4)), movreld.write(Reg::V(4)));
        assert!(build(&code, false, 4).is_err(), "v4 is past 4 VGPRs");
    }

    const fn s_branch(simm16: i16) -> u32 {
        0xBF82_0000 | simm16 as u16 as u32
    }

    const fn s_cbranch_scc1(simm16: i16) -> u32 {
        0xBF85_0000 | simm16 as u16 as u32
    }

    #[test]
    fn reverse_order_diamond_chain_eliminates_trivial_phis() {
        const LAYERS: usize = 4096;
        let mut reverse = vec![
            0xBE81_0381, // s_mov_b32 s1, 1
            0xBE82_0382, // s_mov_b32 s2, 2
            s_branch(((LAYERS - 1) * 4) as i16),
        ];
        for k in 0..LAYERS {
            reverse.extend([
                0xBF06_0201, // s_cmp_eq_u32 s1, s2
                s_cbranch_scc1(1),
                0xBF80_0000, // s_nop 0
                if k == 0 {
                    s_branch((LAYERS * 4 - 4) as i16)
                } else {
                    s_branch(-8)
                },
            ]);
        }
        reverse.extend([0x8000_0201, S_ENDPGM]); // s_add_u32 s0, s1, s2

        let mut forward = vec![
            0xBE81_0381, // s_mov_b32 s1, 1
            0xBE82_0382, // s_mov_b32 s2, 2
            s_branch(0),
        ];
        for _ in 0..LAYERS {
            forward.extend([
                0xBF06_0201, // s_cmp_eq_u32 s1, s2
                s_cbranch_scc1(1),
                0xBF80_0000, // s_nop 0
                s_branch(0),
            ]);
        }
        forward.extend([0x8000_0201, S_ENDPGM]); // s_add_u32 s0, s1, s2

        let rev = build_words(&reverse, false);
        let fwd = build_words(&forward, false);
        assert!(rev.inputs.is_empty(), "{:?}", rev.inputs);
        assert!(fwd.inputs.is_empty(), "{:?}", fwd.inputs);
        assert!(rev.blocks.iter().all(|b| b.phis.is_empty()));
        assert!(fwd.blocks.iter().all(|b| b.phis.is_empty()));
        assert_eq!(rev.blocks.len(), fwd.blocks.len());

        let rev_s1 = rev.blocks[0].insts[0].write(Reg::S(1)).unwrap();
        let rev_s2 = rev.blocks[0].insts[1].write(Reg::S(2)).unwrap();
        let rev_exit = rev.blocks.last().unwrap();
        assert_eq!(rev_exit.insts[0].name(), "s_add_u32");
        assert_eq!(rev_exit.insts[0].read(Reg::S(1)), Some(rev_s1));
        assert_eq!(rev_exit.insts[0].read(Reg::S(2)), Some(rev_s2));

        let fwd_s1 = fwd.blocks[0].insts[0].write(Reg::S(1)).unwrap();
        let fwd_s2 = fwd.blocks[0].insts[1].write(Reg::S(2)).unwrap();
        let fwd_exit = fwd.blocks.last().unwrap();
        assert_eq!(fwd_exit.insts[0].name(), "s_add_u32");
        assert_eq!(fwd_exit.insts[0].read(Reg::S(1)), Some(fwd_s1));
        assert_eq!(fwd_exit.insts[0].read(Reg::S(2)), Some(fwd_s2));
    }

    #[test]
    fn remove_trivial_phis_compresses_reverse_chain_in_bounded_steps() {
        const CHAIN: usize = 4096;
        let mut template = build_words(&[0xBE80_0380, S_ENDPGM], false).blocks[0].insts[0].clone();
        template.writes.clear();
        let mut blocks = Vec::with_capacity(CHAIN);
        for k in 0..CHAIN {
            let v1 = Value(2 + 2 * k as u32);
            let v2 = Value(3 + 2 * k as u32);
            let (n1, n2) = if k + 1 < CHAIN {
                (Value(2 + 2 * (k + 1) as u32), Value(3 + 2 * (k + 1) as u32))
            } else {
                (Value(0), Value(1))
            };
            let mut inst = template.clone();
            inst.reads = vec![(Reg::S(1), v1), (Reg::S(2), v2)];
            blocks.push(Block {
                offset: k * 4,
                preds: vec![],
                phis: vec![
                    Phi {
                        reg: Reg::S(1),
                        value: v1,
                        args: vec![n1, n1],
                    },
                    Phi {
                        reg: Reg::S(2),
                        value: v2,
                        args: vec![n2, n2],
                    },
                ],
                insts: vec![inst],
                exit: Exit::End,
            });
        }
        let steps = remove_trivial_phis_steps(&mut blocks, 2 + 2 * CHAIN as u32);
        assert!(blocks.iter().all(|b| b.phis.is_empty()));
        assert!(
            blocks
                .iter()
                .all(|b| b.insts[0].reads == [(Reg::S(1), Value(0)), (Reg::S(2), Value(1))])
        );
        assert!(steps < 100_000, "steps = {steps}");
    }

    #[test]
    fn non_trivial_phis_preserve_compressed_representatives() {
        const LAYERS: usize = 4;
        let mut code = vec![
            0xBE81_0381, // s_mov_b32 s1, 1
            0xBE82_0382, // s_mov_b32 s2, 2
            0xBF06_8000, // s_cmp_eq_u32 s0, 0
            s_cbranch_scc1((2 + (LAYERS - 1) * 4) as i16),
            0xBE81_0383, // s_mov_b32 s1, 3
            s_branch((LAYERS * 4) as i16),
        ];
        for k in 0..LAYERS {
            code.extend([
                0xBF06_0201, // s_cmp_eq_u32 s1, s2
                s_cbranch_scc1(1),
                0xBF80_0000, // s_nop 0
                if k == 0 {
                    s_branch((LAYERS * 4 - 4) as i16)
                } else {
                    s_branch(-8)
                },
            ]);
        }
        code.extend([0x8000_0201, S_ENDPGM]); // s_add_u32 s0, s1, s2

        let p = build_words(&code, false);
        let entry_s1 = p.blocks[0].insts[0].write(Reg::S(1)).unwrap();
        let entry_s2 = p.blocks[0].insts[1].write(Reg::S(2)).unwrap();
        let arm_s1 = p.blocks[1].insts[0].write(Reg::S(1)).unwrap();
        let merge = p.blocks.last().unwrap();
        for b in &p.blocks[..p.blocks.len() - 1] {
            assert!(b.phis.is_empty(), "{:?}", b.phis);
        }
        let [phi] = &merge.phis[..] else {
            panic!("{:?}", merge.phis)
        };
        assert_eq!(phi.reg, Reg::S(1));
        assert_eq!(phi.args, [arm_s1, entry_s1]);
        assert_eq!(merge.insts[0].read(Reg::S(1)), Some(phi.value));
        assert_eq!(merge.insts[0].read(Reg::S(2)), Some(entry_s2));

        let mut loop_code = vec![
            0xBE81_0381, // s_mov_b32 s1, 1
            0xBE82_0382, // s_mov_b32 s2, 2
            s_branch(((LAYERS - 1) * 4) as i16),
        ];
        for k in 0..LAYERS {
            loop_code.extend([
                0xBF06_0201, // s_cmp_eq_u32 s1, s2
                s_cbranch_scc1(1),
                0xBF80_0000, // s_nop 0
                if k == 0 {
                    s_branch((LAYERS * 4 - 4) as i16)
                } else {
                    s_branch(-8)
                },
            ]);
        }
        loop_code.extend([
            0x8001_0201, // s_add_u32 s1, s1, s2
            0xBF07_8401, // s_cmp_lg_u32 s1, 4
            s_cbranch_scc1(-3),
            0x8000_0201, // s_add_u32 s0, s1, s2
            S_ENDPGM,
        ]);

        let lp = build_words(&loop_code, false);
        let lp_s1 = lp.blocks[0].insts[0].write(Reg::S(1)).unwrap();
        let lp_s2 = lp.blocks[0].insts[1].write(Reg::S(2)).unwrap();
        let header = &lp.blocks[lp.blocks.len() - 2];
        let exit = lp.blocks.last().unwrap();
        for b in &lp.blocks[..lp.blocks.len() - 2] {
            assert!(b.phis.is_empty(), "{:?}", b.phis);
        }
        assert!(exit.phis.is_empty(), "{:?}", exit.phis);
        let [loop_phi] = &header.phis[..] else {
            panic!("{:?}", header.phis)
        };
        let next_s1 = header.insts[0].write(Reg::S(1)).unwrap();
        assert_eq!(loop_phi.reg, Reg::S(1));
        assert_eq!(loop_phi.args, [lp_s1, next_s1]);
        assert_eq!(header.insts[0].read(Reg::S(1)), Some(loop_phi.value));
        assert_eq!(header.insts[0].read(Reg::S(2)), Some(lp_s2));
        assert_eq!(exit.insts[0].read(Reg::S(1)), Some(next_s1));
        assert_eq!(exit.insts[0].read(Reg::S(2)), Some(lp_s2));
    }

    /// `v_writelane_b32 v{vgpr}, s{sgpr}, lane` (VOP3).
    fn writelane(vgpr: u32, sgpr: u32, lane: u32) -> [u32; 2] {
        [0xD761_0000 | vgpr, sgpr | (0x80 + lane) << 9]
    }

    /// `v_readlane_b32 s{sgpr}, v{vgpr}, lane` (VOP3).
    fn readlane(sgpr: u32, vgpr: u32, lane: u32) -> [u32; 2] {
        [0xD760_0000 | sgpr, (256 + vgpr) | (0x80 + lane) << 9]
    }

    /// `s_mov_b32 s{dst}, src` (`src`: a scalar or inline constant code).
    fn s_mov(dst: u32, src: u32) -> u32 {
        0xBE80_0300 | dst << 16 | src
    }

    fn insts(p: &Program) -> Vec<&Inst> {
        p.blocks.iter().flat_map(|b| &b.insts).collect()
    }

    fn lane_ops(p: &Program) -> usize {
        insts(p)
            .iter()
            .filter(|i| matches!(i.name(), "v_readlane_b32" | "v_writelane_b32"))
            .count()
    }

    fn unparked(words: &[u32], wave64: bool) -> Program {
        let p = build_words(words, wave64);
        assert_eq!(lane_ops(&p), 0, "parking left in {:#?}", insts(&p));
        p
    }

    /// The value `reg` holds after the instruction that writes it last.
    fn last_write(p: &Program, reg: Reg) -> Value {
        insts(p)
            .iter()
            .rev()
            .find_map(|i| i.write(reg))
            .expect("written")
    }

    /// A scalar saved in a VGPR lane and taken back into another register is
    /// a copy of the value saved.
    #[test]
    fn parked_scalar_is_a_copy() {
        let mut w = vec![s_mov(3, 0x85)]; // s3 = 5
        w.extend(writelane(5, 3, 2));
        w.push(s_mov(3, 0x87)); // s3 = 7
        w.extend(readlane(4, 5, 2));
        w.push(S_ENDPGM);
        let p = unparked(&w, false);
        let five = insts(&p)[0].write(Reg::S(3)).unwrap();
        let save = insts(&p)
            .into_iter()
            .find(|i| i.write(Reg::S(PARKED)).is_some())
            .unwrap();
        assert_eq!(save.read(Reg::S(3)), Some(five));
        let restore = insts(&p)
            .into_iter()
            .find(|i| i.write(Reg::S(4)).is_some())
            .unwrap();
        assert_eq!(restore.name(), "s_mov_b32");
        assert_eq!(restore.read(Reg::S(PARKED)), save.write(Reg::S(PARKED)));
    }

    /// A wave64 register pair parked in neighbouring lanes comes back as a
    /// pair, also when other work is scheduled between the two restores.
    #[test]
    fn parked_pair_restored_apart_is_a_pair() {
        let mut w = vec![s_mov(6, 0x81), s_mov(7, 0x82)];
        w.extend(writelane(5, 6, 0));
        w.push(s_mov(10, 0x83)); // other work between the saves
        w.extend(writelane(5, 7, 1));
        w.extend([s_mov(6, 0x80), s_mov(7, 0x80)]);
        w.extend(readlane(6, 5, 0));
        w.push(s_mov(9, 0x83)); // other work between the restores
        w.extend(readlane(7, 5, 1));
        w.push(S_ENDPGM);
        let p = unparked(&w, true);
        let wide: Vec<_> = insts(&p)
            .into_iter()
            .filter(|i| i.name() == "s_mov_b64")
            .collect();
        assert_eq!(wide.len(), 2, "{wide:#?}");
        let (save, restore) = (wide[0], wide[1]);
        assert!(save.write(Reg::S(PARKED)).is_some() && save.write(Reg::S(PARKED + 1)).is_some());
        for k in 0..2 {
            assert_eq!(
                restore.read(Reg::S(PARKED + k)),
                save.write(Reg::S(PARKED + k))
            );
        }
        // Both halves hold the values saved, 1 and 2.
        assert_eq!(save.read(Reg::S(6)), insts(&p)[0].write(Reg::S(6)));
        assert_eq!(save.read(Reg::S(7)), insts(&p)[1].write(Reg::S(7)));
        assert_eq!(last_write(&p, Reg::S(6)), restore.write(Reg::S(6)).unwrap());
        assert_eq!(last_write(&p, Reg::S(7)), restore.write(Reg::S(7)).unwrap());
    }

    /// A read of the second register between the two restores would see the
    /// restored value too early: the halves come back one by one.
    #[test]
    fn parked_pair_with_a_read_between_restores_is_two_scalars() {
        let mut w = vec![s_mov(6, 0x81), s_mov(7, 0x82)];
        w.extend(writelane(5, 6, 0));
        w.extend(writelane(5, 7, 1));
        w.push(s_mov(7, 0x80));
        w.extend(readlane(6, 5, 0));
        w.push(s_mov(9, 7)); // s9 = s7: still 0 here
        w.extend(readlane(7, 5, 1));
        w.push(S_ENDPGM);
        let p = unparked(&w, true);
        assert!(!insts(&p).iter().any(|i| i.name() == "s_mov_b64"));
        // s6 = 1, s7 = 2, two saves, s7 = 0, the restore of s6, then the copy.
        let zero = insts(&p)[4];
        assert_eq!(zero.name(), "s_mov_b32");
        let copy = insts(&p)
            .into_iter()
            .find(|i| i.write(Reg::S(9)).is_some())
            .unwrap();
        assert_eq!(copy.read(Reg::S(7)), zero.write(Reg::S(7)));
    }

    /// An M0-relative read names the parking VGPR without using it.
    #[test]
    fn relative_read_keeps_parking() {
        let mut w = vec![s_mov(3, 0x85)];
        w.extend(writelane(5, 3, 2));
        w.push(0x7E00_8704); // v_movrels_b32 v0, v4: v0 = v[4 + m0]
        w.extend(readlane(4, 5, 2));
        w.push(S_ENDPGM);
        unparked(&w, false);
    }

    /// A VGPR with other uses, or a lane read that nothing saves into, is
    /// left as it is.
    #[test]
    fn parking_needs_a_spare_vgpr_and_a_save() {
        let mut w = vec![s_mov(3, 0x85)];
        w.extend(writelane(5, 3, 2));
        w.push(0x7E0C_0305); // v_mov_b32 v6, v5
        w.extend(readlane(4, 5, 2));
        w.push(S_ENDPGM);
        assert_eq!(lane_ops(&build_words(&w, false)), 2);

        let mut w = vec![s_mov(3, 0x85)];
        w.extend(writelane(5, 3, 2));
        w.extend(readlane(4, 5, 3));
        w.push(S_ENDPGM);
        assert_eq!(lane_ops(&build_words(&w, false)), 2);
    }
}
