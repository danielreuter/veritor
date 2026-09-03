//! The canonical binary encoding of a batch statement and its witness
//! (`veritor/protocol/proofs/wire.py`), parsed strictly: every index is
//! range-checked, kinds must be sorted and unique, and no byte may be left
//! over.  Integers are big-endian; a `bytes`/`str` is a `u32` length prefix
//! followed by the payload; a list is a `u32` count followed by its items;
//! digests are 32 raw bytes.
//!
//! ```text
//! Statement   = MAGIC "veritor/proofs/statement/v2\0"
//!               str gate_set_id  digest gate_set_digest  u32 width
//!               list<KindProgram> kinds (strictly increasing by kind digest)
//!               list<Obligation>  obligations
//! KindProgram = digest kind  u32 size  list<u32> ports  list<GateOp> gates (len == size)
//!               list<u32> outputs (gate offsets the copy opens, strictly increasing)
//! GateOp      = str op  list<Arg> args
//! Arg         = u8 space (0 = port index, 1 = local gate offset)  u32 value
//! Obligation  = digest session  digest compiled  u64 unit  u64 replay_unit  digest kind
//!               list<CommitmentRef> commitments  list<PositionRef> positions
//!               list<u32> inputs (one per port)  list<u32> outputs (one per opened offset)
//! CommitmentRef = u64 owner_plus_two  digest domain_id  digest root  u64 count
//! PositionRef   = u32 commitment  u64 rank  u64 position  str schema
//!                 u8 has_expected  [bytes expected]
//!
//! Witness     = MAGIC "veritor/proofs/witness/v2\0"
//!               list< list< Opening > >    (per obligation, per position)
//! Opening     = bytes value  list<digest> path
//! ```

use core::fmt;

pub const STATEMENT_MAGIC: &[u8] = b"veritor/proofs/statement/v2\0";
pub const WITNESS_MAGIC: &[u8] = b"veritor/proofs/witness/v2\0";

/// A malformed statement or witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodecError(pub String);

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "malformed proof input: {}", self.0)
    }
}

impl std::error::Error for CodecError {}

type Result<T> = core::result::Result<T, CodecError>;

fn err<T>(message: impl Into<String>) -> Result<T> {
    Err(CodecError(message.into()))
}

/// An argument of a gate: a port of the verification unit or an earlier gate of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arg {
    /// Index into `KindProgram::ports` (and so into `Obligation::inputs`).
    Port(u32),
    /// Offset of an earlier gate of the same copy (recomputed, or opened if the
    /// kind lists it in `KindProgram::outputs`).
    Local(u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GateOp {
    pub op: String,
    pub args: Vec<Arg>,
}

/// The relation of one kind of verification unit, in relative coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KindProgram {
    pub kind: [u8; 32],
    pub size: u32,
    /// The port ordinals the copy reads, ascending.
    pub ports: Vec<u32>,
    pub gates: Vec<GateOp>,
    /// The gate offsets a copy opens (its declared outputs and its source
    /// gates), ascending; every other gate is recomputed by the checker.
    pub outputs: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentRef {
    pub owner_plus_two: u64,
    pub domain_id: [u8; 32],
    pub root: [u8; 32],
    pub count: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PositionRef {
    pub commitment: u32,
    pub rank: u64,
    pub position: u64,
    pub schema: String,
    /// The public input this position must hold (an `in` gate), if any.
    pub expected: Option<Vec<u8>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Obligation {
    pub session: [u8; 32],
    pub compiled: [u8; 32],
    pub unit: u64,
    pub replay_unit: u64,
    pub kind: [u8; 32],
    pub commitments: Vec<CommitmentRef>,
    pub positions: Vec<PositionRef>,
    /// `inputs[k]`: the slot of the kind's `k`-th read port.
    pub inputs: Vec<u32>,
    /// `outputs[m]`: the slot of the gate at the kind's `m`-th opened offset.
    pub outputs: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statement {
    pub gate_set_id: String,
    pub gate_set_digest: [u8; 32],
    pub width: u32,
    pub kinds: Vec<KindProgram>,
    pub obligations: Vec<Obligation>,
}

impl Statement {
    /// The program of `kind`, by binary search over the sorted kinds.
    pub fn program(&self, kind: &[u8; 32]) -> Option<&KindProgram> {
        self.kinds
            .binary_search_by(|candidate| candidate.kind.cmp(kind))
            .ok()
            .map(|index| &self.kinds[index])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Opening {
    pub value: Vec<u8>,
    pub path: Vec<[u8; 32]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Witness {
    pub obligations: Vec<Vec<Opening>>,
}

struct Reader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Reader { data, offset: 0 }
    }

    fn take(&mut self, count: usize, what: &str) -> Result<&'a [u8]> {
        let end = self.offset.checked_add(count).ok_or_else(|| CodecError("overflow".into()))?;
        if end > self.data.len() {
            return err(format!("truncated while reading {what}"));
        }
        let slice = &self.data[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn magic(&mut self, expected: &[u8]) -> Result<()> {
        if self.take(expected.len(), "magic")? != expected {
            return err("wrong magic");
        }
        Ok(())
    }

    fn u8(&mut self, what: &str) -> Result<u8> {
        Ok(self.take(1, what)?[0])
    }

    fn u32(&mut self, what: &str) -> Result<u32> {
        let bytes = self.take(4, what)?;
        Ok(u32::from_be_bytes(bytes.try_into().expect("four bytes")))
    }

    fn u64(&mut self, what: &str) -> Result<u64> {
        let bytes = self.take(8, what)?;
        Ok(u64::from_be_bytes(bytes.try_into().expect("eight bytes")))
    }

    fn digest(&mut self, what: &str) -> Result<[u8; 32]> {
        let bytes = self.take(32, what)?;
        Ok(bytes.try_into().expect("32 bytes"))
    }

    fn bytes(&mut self, what: &str) -> Result<&'a [u8]> {
        let length = self.u32(what)? as usize;
        self.take(length, what)
    }

    fn string(&mut self, what: &str) -> Result<String> {
        let bytes = self.bytes(what)?;
        core::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| CodecError(format!("{what} is not UTF-8")))
    }

    fn count(&mut self, what: &str) -> Result<usize> {
        let count = self.u32(what)? as usize;
        // Every item is at least one byte; refuse counts the buffer cannot hold.
        if count > self.data.len() - self.offset {
            return err(format!("{what} count {count} exceeds the remaining bytes"));
        }
        Ok(count)
    }

    fn finish(self, what: &str) -> Result<()> {
        if self.offset != self.data.len() {
            return err(format!("{} trailing bytes after the {what}", self.data.len() - self.offset));
        }
        Ok(())
    }
}

fn parse_gate(reader: &mut Reader<'_>, offset: u32, ports: usize) -> Result<GateOp> {
    let op = reader.string("gate op")?;
    let count = reader.count("gate args")?;
    let mut args = Vec::with_capacity(count);
    for _ in 0..count {
        let space = reader.u8("arg space")?;
        let value = reader.u32("arg value")?;
        let arg = match space {
            0 => {
                if value as usize >= ports {
                    return err(format!("gate {offset} reads port index {value} of {ports}"));
                }
                Arg::Port(value)
            }
            1 => {
                if value >= offset {
                    return err(format!("gate {offset} reads local offset {value}, not earlier"));
                }
                Arg::Local(value)
            }
            other => return err(format!("unknown argument space {other}")),
        };
        args.push(arg);
    }
    Ok(GateOp { op, args })
}

fn parse_program(reader: &mut Reader<'_>) -> Result<KindProgram> {
    let kind = reader.digest("kind digest")?;
    let size = reader.u32("kind size")?;
    let port_count = reader.count("ports")?;
    let mut ports = Vec::with_capacity(port_count);
    for index in 0..port_count {
        let ordinal = reader.u32("port ordinal")?;
        if index > 0 && ports[index - 1] >= ordinal {
            return err("ports must be strictly increasing");
        }
        ports.push(ordinal);
    }
    let gate_count = reader.count("gates")?;
    if gate_count != size as usize {
        return err(format!("kind declares {size} gates but lists {gate_count}"));
    }
    let mut gates = Vec::with_capacity(gate_count);
    for offset in 0..gate_count {
        gates.push(parse_gate(reader, offset as u32, port_count)?);
    }
    let output_count = reader.count("outputs")?;
    let mut outputs = Vec::with_capacity(output_count);
    for index in 0..output_count {
        let offset = reader.u32("output offset")?;
        if offset >= size {
            return err(format!("kind opens gate offset {offset} of {size}"));
        }
        if index > 0 && outputs[index - 1] >= offset {
            return err("kind outputs must be strictly increasing");
        }
        outputs.push(offset);
    }
    Ok(KindProgram { kind, size, ports, gates, outputs })
}

fn parse_indices(reader: &mut Reader<'_>, what: &str, bound: usize) -> Result<Vec<u32>> {
    let count = reader.count(what)?;
    let mut indices = Vec::with_capacity(count);
    for _ in 0..count {
        let index = reader.u32(what)?;
        if index as usize >= bound {
            return err(format!("{what} index {index} is out of range ({bound})"));
        }
        indices.push(index);
    }
    Ok(indices)
}

fn parse_obligation(reader: &mut Reader<'_>, statement_kinds: &[KindProgram]) -> Result<Obligation> {
    let session = reader.digest("session")?;
    let compiled = reader.digest("compiled digest")?;
    let unit = reader.u64("unit")?;
    let replay_unit = reader.u64("replay unit")?;
    let kind = reader.digest("obligation kind")?;
    let program = statement_kinds
        .binary_search_by(|candidate| candidate.kind.cmp(&kind))
        .ok()
        .map(|index| &statement_kinds[index])
        .ok_or_else(|| CodecError("obligation names a kind the statement lacks".into()))?;
    let commitment_count = reader.count("commitments")?;
    let mut commitments = Vec::with_capacity(commitment_count);
    for _ in 0..commitment_count {
        commitments.push(CommitmentRef {
            owner_plus_two: reader.u64("owner")?,
            domain_id: reader.digest("domain id")?,
            root: reader.digest("root")?,
            count: reader.u64("commitment count")?,
        });
    }
    let position_count = reader.count("positions")?;
    let mut positions = Vec::with_capacity(position_count);
    for _ in 0..position_count {
        let commitment = reader.u32("position commitment")?;
        if commitment as usize >= commitment_count {
            return err("position names a commitment the obligation lacks");
        }
        let rank = reader.u64("rank")?;
        let position = reader.u64("position")?;
        let schema = reader.string("schema")?;
        let expected = match reader.u8("expected flag")? {
            0 => None,
            1 => Some(reader.bytes("expected value")?.to_vec()),
            other => return err(format!("bad expected flag {other}")),
        };
        positions.push(PositionRef { commitment, rank, position, schema, expected });
    }
    let inputs = parse_indices(reader, "inputs", position_count)?;
    if inputs.len() != program.ports.len() {
        return err(format!(
            "obligation binds {} inputs but its kind reads {} ports",
            inputs.len(),
            program.ports.len()
        ));
    }
    let outputs = parse_indices(reader, "outputs", position_count)?;
    if outputs.len() != program.outputs.len() {
        return err(format!(
            "obligation binds {} outputs but its kind opens {}",
            outputs.len(),
            program.outputs.len()
        ));
    }
    let mut distinct = outputs.clone();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() != outputs.len() {
        return err("obligation outputs must be distinct positions");
    }
    Ok(Obligation { session, compiled, unit, replay_unit, kind, commitments, positions, inputs, outputs })
}

/// Parse a canonical statement.
pub fn parse_statement(data: &[u8]) -> Result<Statement> {
    let mut reader = Reader::new(data);
    reader.magic(STATEMENT_MAGIC)?;
    let gate_set_id = reader.string("gate set id")?;
    let gate_set_digest = reader.digest("gate set digest")?;
    let width = reader.u32("width")?;
    let kind_count = reader.count("kinds")?;
    let mut kinds: Vec<KindProgram> = Vec::with_capacity(kind_count);
    for index in 0..kind_count {
        let program = parse_program(&mut reader)?;
        if index > 0 && kinds[index - 1].kind >= program.kind {
            return err("kinds must be strictly increasing by digest");
        }
        kinds.push(program);
    }
    let obligation_count = reader.count("obligations")?;
    let mut obligations = Vec::with_capacity(obligation_count);
    for _ in 0..obligation_count {
        obligations.push(parse_obligation(&mut reader, &kinds)?);
    }
    reader.finish("statement")?;
    Ok(Statement { gate_set_id, gate_set_digest, width, kinds, obligations })
}

/// Parse a canonical witness.
pub fn parse_witness(data: &[u8]) -> Result<Witness> {
    let mut reader = Reader::new(data);
    reader.magic(WITNESS_MAGIC)?;
    let obligation_count = reader.count("witness obligations")?;
    let mut obligations = Vec::with_capacity(obligation_count);
    for _ in 0..obligation_count {
        let opening_count = reader.count("openings")?;
        let mut openings = Vec::with_capacity(opening_count);
        for _ in 0..opening_count {
            let value = reader.bytes("opening value")?.to_vec();
            let depth = reader.count("path")?;
            let mut path = Vec::with_capacity(depth);
            for _ in 0..depth {
                path.push(reader.digest("path digest")?);
            }
            openings.push(Opening { value, path });
        }
        obligations.push(openings);
    }
    reader.finish("witness")?;
    Ok(Witness { obligations })
}
