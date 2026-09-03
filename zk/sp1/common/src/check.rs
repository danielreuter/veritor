//! The generic checker: decide one batch of obligations against a witness.
//!
//! Mirrors `TransparentBackend.verify` in `veritor/protocol/proofs/transparent.py`:
//! every opening is authenticated under its commitment with the exact leaf
//! and node framing, decoded canonically at the gate set's width, compared
//! with the public input where the statement pins one, and then every
//! obligation's gates are recomputed from its opened inputs under the pinned
//! semantics of the statement's gate set and compared with its opened
//! outputs.  Only a copy's inputs and outputs are opened; the gates between
//! are never committed, so the recomputation is what stands for them.  The two
//! passes are separate so the guest can attribute cycles to each.

use core::fmt;

use crate::codec::{Arg, Statement, Witness};
use crate::frame::{fold_path, leaf, merkle_depth};
use crate::gateset::GateSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckError {
    UnknownGateSet { id: String, width: u32 },
    GateSetDigest,
    WitnessShape(String),
    Opening { obligation: usize, position: usize, reason: &'static str },
    PublicInput { obligation: usize, position: usize },
    Value { obligation: usize, position: usize },
    UnknownOp { kind: [u8; 32], op: String },
    Arity { kind: [u8; 32], offset: usize },
    SourceNotOpened { kind: [u8; 32], offset: usize },
    Relation { obligation: usize, offset: usize, op: String },
}

impl fmt::Display for CheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CheckError::UnknownGateSet { id, width } => {
                write!(f, "unknown gate set {id} at width {width}")
            }
            CheckError::GateSetDigest => {
                f.write_str("the statement's gate set digest is not that of the pinned semantics")
            }
            CheckError::WitnessShape(detail) => write!(f, "witness shape: {detail}"),
            CheckError::Opening { obligation, position, reason } => {
                write!(f, "obligation {obligation} position {position}: {reason}")
            }
            CheckError::PublicInput { obligation, position } => {
                write!(f, "obligation {obligation} position {position} differs from the public input")
            }
            CheckError::Value { obligation, position } => {
                write!(f, "obligation {obligation} position {position} is not a canonical value")
            }
            CheckError::UnknownOp { op, .. } => write!(f, "unknown gate {op}"),
            CheckError::Arity { offset, .. } => write!(f, "gate {offset} has the wrong arity"),
            CheckError::SourceNotOpened { offset, .. } => {
                write!(f, "source gate {offset} is not opened")
            }
            CheckError::Relation { obligation, offset, op } => {
                write!(f, "obligation {obligation} gate {offset} violates {op}")
            }
        }
    }
}

impl std::error::Error for CheckError {}

/// Resolve and pin the statement's gate set.
pub fn gate_set(statement: &Statement) -> Result<GateSet, CheckError> {
    let set = GateSet::resolve(&statement.gate_set_id, statement.width).ok_or_else(|| {
        CheckError::UnknownGateSet { id: statement.gate_set_id.clone(), width: statement.width }
    })?;
    if set.digest() != statement.gate_set_digest {
        return Err(CheckError::GateSetDigest);
    }
    Ok(set)
}

/// Pass 1: authenticate and decode every opening.  Returns the decoded values
/// per obligation, indexed like `Obligation::positions`.
pub fn check_openings(
    statement: &Statement,
    witness: &Witness,
    set: &GateSet,
) -> Result<Vec<Vec<u64>>, CheckError> {
    if witness.obligations.len() != statement.obligations.len() {
        return Err(CheckError::WitnessShape(format!(
            "{} obligations in the witness, {} in the statement",
            witness.obligations.len(),
            statement.obligations.len()
        )));
    }
    let mut decoded = Vec::with_capacity(statement.obligations.len());
    for (index, (obligation, openings)) in
        statement.obligations.iter().zip(&witness.obligations).enumerate()
    {
        if openings.len() != obligation.positions.len() {
            return Err(CheckError::WitnessShape(format!(
                "obligation {index} has {} openings for {} positions",
                openings.len(),
                obligation.positions.len()
            )));
        }
        let mut values = Vec::with_capacity(openings.len());
        for (slot, (position, opening)) in obligation.positions.iter().zip(openings).enumerate() {
            let commitment = &obligation.commitments[position.commitment as usize];
            let fail = |reason| CheckError::Opening { obligation: index, position: slot, reason };
            if position.rank >= commitment.count {
                return Err(fail("rank outside the commitment"));
            }
            if opening.path.len() != merkle_depth(commitment.count) {
                return Err(fail("path length differs from the tree depth"));
            }
            let digest = leaf(
                &commitment.domain_id,
                position.rank,
                position.position,
                position.schema.as_bytes(),
                &opening.value,
            );
            if fold_path(&commitment.domain_id, position.rank, digest, &opening.path) != commitment.root
            {
                return Err(fail("authentication path does not reach the root"));
            }
            if let Some(expected) = &position.expected {
                if *expected != opening.value {
                    return Err(CheckError::PublicInput { obligation: index, position: slot });
                }
            }
            let value = set
                .decode(&opening.value)
                .ok_or(CheckError::Value { obligation: index, position: slot })?;
            values.push(value);
        }
        decoded.push(values);
    }
    Ok(decoded)
}

/// Pass 2: recompute every obligation's gates from its opened inputs; every
/// opened gate must agree with what was recomputed.
///
/// A source gate has no relation: its opened value *is* its value (the
/// boundary pins an `in` gate to the public input, `kappa_W` a weight), so
/// the kind must open it.  Every other gate is evaluated from what it reads
/// -- an opened input or an earlier gate of the copy -- and, when the kind
/// opens it, compared with the opened value.  Byte-identical in outcome to
/// `TransparentBackend._check_relations`.
pub fn check_relations(
    statement: &Statement,
    decoded: &[Vec<u64>],
    set: &GateSet,
) -> Result<(), CheckError> {
    for (index, (obligation, values)) in statement.obligations.iter().zip(decoded).enumerate() {
        let program = statement
            .program(&obligation.kind)
            .expect("the parser checked every obligation's kind");
        let mut local: Vec<u64> = Vec::with_capacity(program.gates.len());
        let mut next_output = 0usize;
        for (offset, gate) in program.gates.iter().enumerate() {
            let op = set
                .op(&gate.op)
                .ok_or_else(|| CheckError::UnknownOp { kind: program.kind, op: gate.op.clone() })?;
            if gate.args.len() != op.arity() {
                return Err(CheckError::Arity { kind: program.kind, offset });
            }
            // `program.outputs` is strictly increasing, so one cursor finds the opened slot.
            let slot = if next_output < program.outputs.len()
                && program.outputs[next_output] as usize == offset
            {
                let slot = obligation.outputs[next_output] as usize;
                next_output += 1;
                Some(slot)
            } else {
                None
            };
            if op.is_source() {
                let slot = slot.ok_or(CheckError::SourceNotOpened { kind: program.kind, offset })?;
                local.push(values[slot]);
                continue;
            }
            let read = |arg: Arg| -> u64 {
                match arg {
                    Arg::Port(k) => values[obligation.inputs[k as usize] as usize],
                    Arg::Local(j) => local[j as usize],
                }
            };
            let (a, b) = (read(gate.args[0]), read(gate.args[1]));
            let computed = set.evaluate(op, a, b);
            match slot {
                None => local.push(computed),
                Some(slot) => {
                    if values[slot] != computed {
                        return Err(CheckError::Relation {
                            obligation: index,
                            offset,
                            op: gate.op.clone(),
                        });
                    }
                    local.push(values[slot]);
                }
            }
        }
    }
    Ok(())
}

/// Both passes: the verdict for one batch.
pub fn check_batch(statement: &Statement, witness: &Witness) -> Result<(), CheckError> {
    let set = gate_set(statement)?;
    let decoded = check_openings(statement, witness, &set)?;
    check_relations(statement, &decoded, &set)
}
