//! Host driver for the veritor SP1 checker guest.  Every subcommand prints one
//! JSON object to stdout (SP1's own logging and the guest's prints go to
//! stderr), which is the subprocess protocol `veritor.protocol.proofs.sp1`
//! speaks.
//!
//!   veritor-zk-host info
//!       -> {"backend", "sp1_version", "elf_sha256", "vk_hash"}
//!   veritor-zk-host execute --statement S --witness W
//!       -> {"public_values", "statement_digest", "verdict", "total_cycles",
//!           "cycle_tracker": {...}, "gas", "syscalls": {...}}
//!   veritor-zk-host prove --statement S --witness W --out P [--mode core|compressed]
//!       -> execute's fields plus {"proof_bytes", "shards", "setup_seconds",
//!           "prove_seconds", "verify_seconds", "vk_hash", "mode"}
//!   veritor-zk-host verify --proof P [--statement S]
//!       -> {"ok", "public_values", "statement_digest", "verdict", "vk_hash",
//!           "statement_match"}
//!
//! `execute` never proves: it is the exact cycle meter used by the Python
//! tests and by the cost table.

use std::{collections::BTreeMap, fs, path::PathBuf, time::Instant};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;
use sp1_sdk::prelude::*;
use sp1_sdk::{ProverClient, SP1Proof};
use veritor_zk_common::{sha256, PUBLIC_VALUES_LEN};

const ELF: Elf = include_elf!("veritor-zk-guest");
const BACKEND: &str = "sp1";
/// The pinned `sp1-sdk` (and guest `sp1-zkvm`) version; keep in step with Cargo.toml.
const SP1_VERSION: &str = "6.4.0";

#[derive(Parser)]
#[command(name = "veritor-zk-host")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Mode {
    Core,
    Compressed,
}

#[derive(Subcommand)]
enum Command {
    /// Identify the guest: ELF digest and verifying-key hash.
    Info,
    /// Run the guest in the executor (no proof) and report exact cycles.
    Execute {
        #[arg(long)]
        statement: PathBuf,
        #[arg(long)]
        witness: PathBuf,
    },
    /// Produce one proof of one batch and verify it locally.
    Prove {
        #[arg(long)]
        statement: PathBuf,
        #[arg(long)]
        witness: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_enum, default_value_t = Mode::Core)]
        mode: Mode,
    },
    /// Verify a saved proof against the guest's verifying key.
    Verify {
        #[arg(long)]
        proof: PathBuf,
        /// If given, also check the public statement digest against these bytes.
        #[arg(long)]
        statement: Option<PathBuf>,
    },
}

#[derive(Serialize)]
struct PublicOutcome {
    public_values: String,
    statement_digest: String,
    verdict: bool,
}

impl PublicOutcome {
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != PUBLIC_VALUES_LEN {
            bail!("guest committed {} public bytes, expected {PUBLIC_VALUES_LEN}", bytes.len());
        }
        if bytes[32] > 1 {
            bail!("guest committed a non-boolean verdict byte {}", bytes[32]);
        }
        Ok(PublicOutcome {
            public_values: hex::encode(bytes),
            statement_digest: hex::encode(&bytes[..32]),
            verdict: bytes[32] == 1,
        })
    }
}

#[derive(Serialize)]
struct ExecuteOutput {
    backend: &'static str,
    #[serde(flatten)]
    outcome: PublicOutcome,
    total_cycles: u64,
    cycle_tracker: BTreeMap<String, u64>,
    gas: Option<u64>,
    syscalls: BTreeMap<String, u64>,
    execute_seconds: f64,
}

#[derive(Serialize)]
struct ProveOutput {
    #[serde(flatten)]
    execute: ExecuteOutput,
    mode: &'static str,
    proof_path: String,
    proof_bytes: u64,
    shards: usize,
    vk_hash: String,
    setup_seconds: f64,
    prove_seconds: f64,
    verify_seconds: f64,
}

#[derive(Serialize)]
struct VerifyOutput {
    backend: &'static str,
    ok: bool,
    #[serde(flatten)]
    outcome: PublicOutcome,
    vk_hash: String,
    statement_match: Option<bool>,
    verify_seconds: f64,
}

#[derive(Serialize)]
struct InfoOutput {
    backend: &'static str,
    sp1_version: &'static str,
    elf_sha256: String,
    vk_hash: String,
    public_values_len: usize,
}

fn stdin_for(statement: &PathBuf, witness: &PathBuf) -> Result<(SP1Stdin, Vec<u8>)> {
    let statement_bytes =
        fs::read(statement).with_context(|| format!("reading {}", statement.display()))?;
    let witness_bytes = fs::read(witness).with_context(|| format!("reading {}", witness.display()))?;
    let mut stdin = SP1Stdin::new();
    stdin.write_vec(statement_bytes.clone());
    stdin.write_vec(witness_bytes);
    Ok((stdin, statement_bytes))
}

async fn execute<P: Prover>(client: &P, stdin: SP1Stdin) -> Result<ExecuteOutput> {
    let started = Instant::now();
    let (public_values, report) =
        client.execute(ELF, stdin).await.map_err(|error| anyhow!("execution failed: {error}"))?;
    let execute_seconds = started.elapsed().as_secs_f64();
    if report.exit_code != 0 {
        bail!("guest exited with status {}", report.exit_code);
    }
    let outcome = PublicOutcome::from_bytes(public_values.as_slice())?;
    let cycle_tracker = report.cycle_tracker.iter().map(|(k, v)| (k.clone(), *v)).collect();
    let syscalls = report
        .syscall_counts
        .iter()
        .filter(|(_, count)| **count > 0)
        .map(|(code, count)| (format!("{code:?}"), *count))
        .collect();
    Ok(ExecuteOutput {
        backend: BACKEND,
        outcome,
        total_cycles: report.total_instruction_count(),
        cycle_tracker,
        gas: report.gas(),
        syscalls,
        execute_seconds,
    })
}

fn emit<T: Serialize>(value: &T) -> Result<()> {
    println!("{}", serde_json::to_string(value)?);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Info => {
            let client = ProverClient::from_env().await;
            let pk = client.setup(ELF).await.map_err(|error| anyhow!("setup failed: {error}"))?;
            emit(&InfoOutput {
                backend: BACKEND,
                sp1_version: SP1_VERSION,
                elf_sha256: hex::encode(sha256(&ELF)),
                vk_hash: pk.verifying_key().bytes32(),
                public_values_len: PUBLIC_VALUES_LEN,
            })
        }
        Command::Execute { statement, witness } => {
            // The mock prover is the light node: an executor without the proving
            // machinery, so `execute` starts in well under a second.
            let client = ProverClient::builder().mock().build().await;
            let (stdin, statement_bytes) = stdin_for(&statement, &witness)?;
            let output = execute(&client, stdin).await?;
            if output.outcome.statement_digest != hex::encode(sha256(&statement_bytes)) {
                bail!("guest committed a digest of some other statement");
            }
            emit(&output)
        }
        Command::Prove { statement, witness, out, mode } => {
            let client = ProverClient::from_env().await;
            let (stdin, statement_bytes) = stdin_for(&statement, &witness)?;
            let execute_output = execute(&client, stdin.clone()).await?;
            if execute_output.outcome.statement_digest != hex::encode(sha256(&statement_bytes)) {
                bail!("guest committed a digest of some other statement");
            }

            let started = Instant::now();
            let pk = client.setup(ELF).await.map_err(|error| anyhow!("setup failed: {error}"))?;
            let setup_seconds = started.elapsed().as_secs_f64();

            let started = Instant::now();
            let request = client.prove(&pk, stdin);
            let proof = match mode {
                Mode::Core => request.core().await,
                Mode::Compressed => request.compressed().await,
            }
            .map_err(|error| anyhow!("prove failed: {error}"))?;
            let prove_seconds = started.elapsed().as_secs_f64();

            let started = Instant::now();
            client
                .verify(&proof, pk.verifying_key(), None)
                .map_err(|error| anyhow!("verify failed: {error}"))?;
            let verify_seconds = started.elapsed().as_secs_f64();

            let shards = match &proof.proof {
                SP1Proof::Core(shards) => shards.len(),
                _ => 1,
            };
            if let Some(parent) = out.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)?;
                }
            }
            proof.save(&out)?;
            let proof_bytes = fs::metadata(&out)?.len();
            emit(&ProveOutput {
                execute: execute_output,
                mode: match mode {
                    Mode::Core => "core",
                    Mode::Compressed => "compressed",
                },
                proof_path: out.display().to_string(),
                proof_bytes,
                shards,
                vk_hash: pk.verifying_key().bytes32(),
                setup_seconds,
                prove_seconds,
                verify_seconds,
            })
        }
        Command::Verify { proof, statement } => {
            let client = ProverClient::from_env().await;
            let pk = client.setup(ELF).await.map_err(|error| anyhow!("setup failed: {error}"))?;
            let proof = SP1ProofWithPublicValues::load(&proof)
                .with_context(|| format!("loading {}", proof.display()))?;
            let started = Instant::now();
            let ok = client.verify(&proof, pk.verifying_key(), None).is_ok();
            let verify_seconds = started.elapsed().as_secs_f64();
            let outcome = PublicOutcome::from_bytes(proof.public_values.as_slice())?;
            let statement_match = match statement {
                None => None,
                Some(path) => {
                    let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
                    Some(outcome.statement_digest == hex::encode(sha256(&bytes)))
                }
            };
            emit(&VerifyOutput {
                backend: BACKEND,
                ok,
                outcome,
                vk_hash: pk.verifying_key().bytes32(),
                statement_match,
                verify_seconds,
            })
        }
    }
}
