//! Instruction fields (ISA section 4).

use crate::isa_gen::{decode_op, Op};

#[derive(Clone, Copy, Debug)]
pub struct Decoded {
    pub op: Op,
    pub rd: usize,
    pub ra: usize,
    pub rb: usize,
    pub rc: usize,
    /// The scalar-broadcast bit: bit 10 in R format, bit 5 in R4 ([ENC-1]).
    pub sb: bool,
    pub imm16: u32,
    pub simm16: i32,
    pub soffset26: i32,
}

/// Decode `word`; None means an illegal instruction ([DEC-1]).
pub fn decode(word: u32) -> Option<Decoded> {
    let op = decode_op(word)?;
    let r4 = word >> 26 == 0x05;
    Some(Decoded {
        op,
        rd: ((word >> 21) & 31) as usize,
        ra: ((word >> 16) & 31) as usize,
        rb: ((word >> 11) & 31) as usize,
        rc: ((word >> 6) & 31) as usize,
        sb: if r4 {
            word >> 5 & 1 != 0
        } else {
            word >> 10 & 1 != 0
        },
        imm16: word & 0xFFFF,
        simm16: word as u16 as i16 as i32,
        soffset26: ((word << 6) as i32) >> 6,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conformance_vectors() {
        let d = decode(0x0862_1C00).unwrap();
        assert_eq!((d.op, d.rd, d.ra, d.rb, d.sb), (Op::VAdd, 3, 2, 3, true));
        let d = decode(0x14C5_11A0).unwrap();
        assert_eq!(
            (d.op, d.rd, d.ra, d.rb, d.rc, d.sb),
            (Op::VFfma, 6, 5, 2, 6, true)
        );
        let d = decode(0x8580_FFFB).unwrap();
        assert_eq!((d.op, d.rd, d.simm16), (Op::SBnz, 12, -5));
        assert_eq!(decode(0x11410C02).unwrap().op, Op::VCmpLt);
        assert!(decode(0x0200_0011).is_none()); // V_POPC
        assert!(decode(0x3800_0000).is_none()); // reserved opcode
    }
}
