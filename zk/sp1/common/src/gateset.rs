//! The gate sets the checker knows, with the semantics pinned in
//! `veritor/core/gates.py` and the identity digest that names them.
//!
//! A gate's semantics are part of its definition: the checker refuses a
//! statement whose gate-set digest is not the digest of the table it is
//! about to apply.  The digest is `tagged_sha256("veritor/gate-set/v1",
//! canonical_json(manifest))`, where the manifest lists the gates sorted by
//! name with their `arity`, `name`, `proof_cost`, `replay_cost`, `source`
//! and `width`, then the set's `name` and `version`.

use crate::frame::tagged_sha256;

pub const GATE_SET_IDENTITY_TAG: &[u8] = b"veritor/gate-set/v1";

/// One scalar gate: how the checker evaluates it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Lt,
    Eq,
    Shr,
    /// A source gate: no relation, its value is the public input of its rank.
    In,
    /// A source gate: no relation, its value is a leaf of `kappa_W`.
    Weight,
}

impl Op {
    pub fn is_source(self) -> bool {
        matches!(self, Op::In | Op::Weight)
    }

    pub fn arity(self) -> usize {
        if self.is_source() {
            0
        } else {
            2
        }
    }

    fn manifest_name(self) -> &'static str {
        match self {
            Op::Add => "add",
            Op::Sub => "sub",
            Op::Mul => "mul",
            Op::Lt => "lt",
            Op::Eq => "eq",
            Op::Shr => "shr",
            Op::In => "in",
            Op::Weight => "weight",
        }
    }

    fn costs(self) -> (u32, u32) {
        // (replay_cost, proof_cost) as declared by the Python constructors.
        match self {
            Op::Mul => (2, 2),
            Op::In | Op::Weight => (0, 1),
            _ => (1, 1),
        }
    }

    fn source(self) -> Option<&'static str> {
        match self {
            Op::In => Some("input"),
            Op::Weight => Some("weight"),
            _ => None,
        }
    }
}

/// A gate set family: name, version and the gates it declares (sorted by name).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GateSet {
    pub name: &'static str,
    pub version: &'static str,
    pub width: u32,
    gates: &'static [Op],
}

const TOY_ISA_GATES: &[Op] = &[Op::Add, Op::Eq, Op::In, Op::Lt, Op::Mul, Op::Shr, Op::Sub, Op::Weight];
const WORD_GATES: &[Op] = &[Op::Add, Op::In, Op::Mul, Op::Weight];

impl GateSet {
    /// The family named `id` (`name@version`) at `width` bits, if known.
    pub fn resolve(id: &str, width: u32) -> Option<GateSet> {
        let (name, version, gates) = match id {
            "veritor.toy-isa@1" => ("veritor.toy-isa", "1", TOY_ISA_GATES),
            "veritor.word-arithmetic@2" => ("veritor.word-arithmetic", "2", WORD_GATES),
            _ => return None,
        };
        if width == 0 || width > 64 {
            return None;
        }
        Some(GateSet { name, version, width, gates })
    }

    /// The gate named `op` in this set.
    pub fn op(&self, op: &str) -> Option<Op> {
        self.gates.iter().copied().find(|gate| gate.manifest_name() == op)
    }

    /// The canonical JSON manifest `veritor.core.GateSet.manifest`.
    pub fn manifest(&self) -> String {
        let mut out = String::from("{\"gates\":[");
        for (index, gate) in self.gates.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            let (replay, proof) = gate.costs();
            let source = match gate.source() {
                Some(source) => format!("\"{source}\""),
                None => "null".to_owned(),
            };
            out.push_str(&format!(
                "{{\"arity\":{},\"name\":\"{}\",\"proof_cost\":{},\"replay_cost\":{},\"source\":{},\"width\":{}}}",
                gate.arity(),
                gate.manifest_name(),
                proof,
                replay,
                source,
                self.width
            ));
        }
        out.push_str(&format!("],\"name\":\"{}\",\"version\":\"{}\"}}", self.name, self.version));
        out
    }

    /// `GateSet.digest` as raw bytes.
    pub fn digest(&self) -> [u8; 32] {
        tagged_sha256(GATE_SET_IDENTITY_TAG, self.manifest().as_bytes())
    }

    /// The mask of a `width`-bit word.
    pub fn mask(&self) -> u64 {
        if self.width == 64 {
            u64::MAX
        } else {
            (1u64 << self.width) - 1
        }
    }

    /// Evaluate a two-argument gate with the pinned modular semantics.
    pub fn evaluate(&self, op: Op, a: u64, b: u64) -> u64 {
        let mask = self.mask();
        match op {
            Op::Add => a.wrapping_add(b) & mask,
            Op::Sub => a.wrapping_sub(b) & mask,
            Op::Mul => a.wrapping_mul(b) & mask,
            Op::Lt => (a < b) as u64,
            Op::Eq => (a == b) as u64,
            Op::Shr => {
                if b < self.width as u64 {
                    a >> b
                } else {
                    0
                }
            }
            Op::In | Op::Weight => unreachable!("source gates have no relation"),
        }
    }

    /// Decode a canonical fixed-width big-endian value (`veritor.core.gates.decode_value`).
    pub fn decode(&self, payload: &[u8]) -> Option<u64> {
        let length = (self.width as usize).div_ceil(8);
        if payload.len() != length {
            return None;
        }
        let mut value = 0u64;
        for &byte in payload {
            value = (value << 8) | byte as u64;
        }
        if value > self.mask() {
            return None;
        }
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_matches_the_python_layout() {
        let set = GateSet::resolve("veritor.word-arithmetic@2", 8).unwrap();
        assert_eq!(
            set.manifest(),
            "{\"gates\":[\
             {\"arity\":2,\"name\":\"add\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":8},\
             {\"arity\":0,\"name\":\"in\",\"proof_cost\":1,\"replay_cost\":0,\"source\":\"input\",\"width\":8},\
             {\"arity\":2,\"name\":\"mul\",\"proof_cost\":2,\"replay_cost\":2,\"source\":null,\"width\":8},\
             {\"arity\":0,\"name\":\"weight\",\"proof_cost\":1,\"replay_cost\":0,\"source\":\"weight\",\"width\":8}\
             ],\"name\":\"veritor.word-arithmetic\",\"version\":\"2\"}"
        );
    }

    #[test]
    fn semantics_are_modular() {
        let set = GateSet::resolve("veritor.toy-isa@1", 16).unwrap();
        assert_eq!(set.evaluate(Op::Add, 0xFFFF, 1), 0);
        assert_eq!(set.evaluate(Op::Sub, 0, 1), 0xFFFF);
        assert_eq!(set.evaluate(Op::Mul, 0x8000, 2), 0);
        assert_eq!(set.evaluate(Op::Lt, 3, 4), 1);
        assert_eq!(set.evaluate(Op::Eq, 3, 4), 0);
        assert_eq!(set.evaluate(Op::Shr, 0x8000, 15), 1);
        assert_eq!(set.evaluate(Op::Shr, 0x8000, 16), 0);
        assert_eq!(set.decode(&[0x12, 0x34]), Some(0x1234));
        assert_eq!(set.decode(&[0x12]), None);
        let narrow = GateSet::resolve("veritor.word-arithmetic@2", 4).unwrap();
        assert_eq!(narrow.decode(&[0x1f]), None);
        assert_eq!(narrow.decode(&[0x0f]), Some(15));
    }
}
