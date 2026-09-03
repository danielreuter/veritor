//! The SP1 guest: a generic checker for one batch of veritor proof obligations.
//!
//! Input (two `read_vec` buffers): the canonical statement bytes and the
//! canonical witness bytes.  Output (public values): `sha256(statement) ||
//! verdict`, where the verdict is `1` iff every opening authenticates under
//! its commitment root with the exact `merkle.py` framing, every public input
//! matches, and every gate of every obligation satisfies the pinned semantics
//! of the statement's gate set.
//!
//! `cycle-tracker-report-*` prints let the host executor attribute cycles to
//! parsing, hashing the statement, Merkle authentication and relation checks.

#![no_main]
sp1_zkvm::entrypoint!(main);

use veritor_zk_common::{
    check_openings, check_relations, gate_set, parse_statement, parse_witness, public_values,
    sha256,
};

fn main() {
    println!("cycle-tracker-report-start: io");
    let statement_bytes = sp1_zkvm::io::read_vec();
    let witness_bytes = sp1_zkvm::io::read_vec();
    println!("cycle-tracker-report-end: io");

    println!("cycle-tracker-report-start: digest");
    let statement_digest = sha256(&statement_bytes);
    println!("cycle-tracker-report-end: digest");

    println!("cycle-tracker-report-start: parse");
    let parsed = parse_statement(&statement_bytes)
        .and_then(|statement| parse_witness(&witness_bytes).map(|witness| (statement, witness)));
    println!("cycle-tracker-report-end: parse");

    let verdict = match parsed {
        Err(error) => {
            println!("reject: {error}");
            false
        }
        Ok((statement, witness)) => match gate_set(&statement) {
            Err(error) => {
                println!("reject: {error}");
                false
            }
            Ok(set) => {
                println!("cycle-tracker-report-start: merkle");
                let decoded = check_openings(&statement, &witness, &set);
                println!("cycle-tracker-report-end: merkle");
                match decoded {
                    Err(error) => {
                        println!("reject: {error}");
                        false
                    }
                    Ok(decoded) => {
                        println!("cycle-tracker-report-start: gates");
                        let result = check_relations(&statement, &decoded, &set);
                        println!("cycle-tracker-report-end: gates");
                        match result {
                            Ok(()) => true,
                            Err(error) => {
                                println!("reject: {error}");
                                false
                            }
                        }
                    }
                }
            }
        },
    };

    sp1_zkvm::io::commit_slice(&public_values(&statement_digest, verdict));
}
