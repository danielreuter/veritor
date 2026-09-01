"""Command-line interface for reproducible circuit-cut reports."""

from __future__ import annotations

import argparse
import hashlib
import importlib
from collections.abc import Sequence
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

from circuit_cut_analysis.minimax_study import MinimaxStudyReport
from circuit_cut_analysis.models.gpt2 import analyze_gpt2_execution
from circuit_cut_analysis.profiles import PROFILES
from circuit_cut_analysis.report import write_reports
from circuit_cut_analysis.sampling_study import SamplingStudyReport

SAMPLING_STUDY_MODELS = ("gpt2", "kimi-k3", "deepseek-v4-pro", "inkling")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _probability(value: str) -> Fraction:
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError("must be a decimal or fraction") from error
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must lie in (0, 1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="circuit-cut",
        description="Analyze width-weighted cuts in finite circuit DAGs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    gpt2 = subparsers.add_parser(
        "gpt2",
        help="generate the GPT-2 Small indexed-circuit bottleneck report",
    )
    gpt2.add_argument(
        "--prompt-tokens",
        type=_positive_int,
        metavar="N",
        default=100,
        help="fixed prompt length (default: 100)",
    )
    gpt2.add_argument(
        "--generated-tokens",
        type=_positive_int,
        metavar="N",
        default=100,
        help="number of greedy output token IDs (default: 100)",
    )
    gpt2.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="vllm-fp16-reference",
        help="declared serving-width profile",
    )
    gpt2.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="report destination (default: reports)",
    )

    study = subparsers.add_parser(
        "sampling-study",
        help="run certified verification-sampling studies per model",
    )
    study.add_argument(
        "--model",
        dest="models",
        action="append",
        choices=(*SAMPLING_STUDY_MODELS, "all"),
        help="model to study; repeatable (default: all)",
    )
    study.add_argument(
        "--prompt-tokens",
        type=_positive_int,
        metavar="N",
        default=100,
        help="fixed prompt length (default: 100)",
    )
    study.add_argument(
        "--generated-tokens",
        type=_positive_int,
        metavar="N",
        default=100,
        help="number of greedy output token IDs (default: 100)",
    )
    study.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="report destination (default: reports)",
    )

    minimax = subparsers.add_parser(
        "minimax-study",
        help="solve the exact-budget scalar-gate minimax game",
    )
    minimax.add_argument(
        "--model",
        dest="models",
        action="append",
        choices=(*SAMPLING_STUDY_MODELS, "all"),
        help="model to study; repeatable (default: all)",
    )
    minimax.add_argument(
        "--prompt-tokens",
        type=_positive_int,
        metavar="N",
        default=100,
        help="fixed prompt length (default: 100)",
    )
    minimax.add_argument(
        "--generated-tokens",
        type=_positive_int,
        metavar="N",
        default=100,
        help="number of greedy output token IDs (default: 100)",
    )
    budget_group = minimax.add_mutually_exclusive_group()
    budget_group.add_argument(
        "--budget-fraction",
        dest="budget_fractions",
        action="append",
        type=_probability,
        metavar="Q",
        help="requested fraction q; uses exact B=floor(qN); repeatable",
    )
    budget_group.add_argument(
        "--checked-gates",
        dest="checked_gate_budgets",
        action="append",
        type=_nonnegative_int,
        metavar="B",
        help="exact number of scalar gates checked per run; repeatable",
    )
    minimax.add_argument(
        "--detection-threshold",
        type=_probability,
        default=Fraction(99, 100),
        metavar="P",
        help="required attack detection probability (default: 99/100)",
    )
    minimax.add_argument(
        "--max-quota-evaluations",
        type=_positive_int,
        default=6,
        metavar="N",
        help="deterministic quota candidates to replay (default: 6)",
    )
    minimax.add_argument(
        "--gpt2-class-granularity",
        choices=("row", "row-layer", "row-layer-band"),
        default="row",
        help="initial GPT-2 shared-probability classes (default: row)",
    )
    minimax.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="report destination (default: reports)",
    )
    return parser


def _run_one_sampling_study(
    model: str,
    *,
    prompt_tokens: int,
    generated_tokens: int,
) -> SamplingStudyReport:
    """Build and study one model; import adapters lazily to keep startup fast."""

    from circuit_cut_analysis.cross_model_study import run_profile_sampling_study

    if model == "gpt2":
        from circuit_cut_analysis.models.gpt2 import GPT2_SMALL
        from circuit_cut_analysis.models.gpt2_circuit import (
            build_gpt2_indexed_circuit,
        )
        from circuit_cut_analysis.models.gpt2_sampling_study import (
            run_gpt2_sampling_study,
        )

        indexed = build_gpt2_indexed_circuit(
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            config=GPT2_SMALL,
        )
        return run_gpt2_sampling_study(indexed)

    builders = {
        "kimi-k3": (
            "circuit_cut_analysis.models.kimi_k3",
            "build_kimi_k3_capacity_profile",
        ),
        "deepseek-v4-pro": (
            "circuit_cut_analysis.models.deepseek_v4_pro",
            "build_deepseek_v4_pro_capacity_profile",
        ),
        "inkling": (
            "circuit_cut_analysis.models.inkling",
            "build_inkling_capacity_profile",
        ),
    }
    if model not in builders:
        raise ValueError(f"unknown sampling-study model {model!r}")
    module_name, function_name = builders[model]
    module = importlib.import_module(module_name)
    profile = getattr(module, function_name)(prompt_tokens, generated_tokens)
    return run_profile_sampling_study(profile)


def _main_sampling_study(args: argparse.Namespace) -> int:
    from circuit_cut_analysis.cross_model_study import write_cross_model_summary
    from circuit_cut_analysis.sampling_study import write_study_report

    selected = args.models or ["all"]
    models = (
        list(SAMPLING_STUDY_MODELS)
        if "all" in selected
        else list(dict.fromkeys(selected))
    )

    studies: list[SamplingStudyReport] = []
    for model in models:
        study = _run_one_sampling_study(
            model,
            prompt_tokens=args.prompt_tokens,
            generated_tokens=args.generated_tokens,
        )
        studies.append(study)
        if model == "gpt2":
            from circuit_cut_analysis.models.gpt2_sampling_study import (
                write_gpt2_sampling_study,
            )

            json_path, markdown_path = write_gpt2_sampling_study(study, args.output_dir)
        else:
            stem = f"{model.replace('-', '_')}_sampling_study"
            json_path, markdown_path = write_study_report(
                study, args.output_dir, stem=stem
            )
        print(
            f"{study.model_id}: {study.total_checked_gate_count:,} gates, "
            f"frontier {study.output_frontier_bits:,.1f} bits "
            f"({study.elapsed_seconds:.1f}s)"
        )
        print(f"wrote {json_path}")
        print(f"wrote {markdown_path}")
    if len(studies) > 1:
        summary_path = write_cross_model_summary(studies, args.output_dir)
        print(f"wrote {summary_path}")
    return 0


def _run_one_minimax_study(
    model: str,
    *,
    prompt_tokens: int,
    generated_tokens: int,
    budget_fractions: Sequence[Fraction] | None,
    checked_gate_budgets: Sequence[int] | None,
    detection_threshold: Fraction,
    max_quota_evaluations: int,
    gpt2_class_granularity: str,
) -> MinimaxStudyReport:
    """Build one symbolic class partition and run its minimax study."""

    from circuit_cut_analysis.minimax_study import run_weighted_minimax_study

    if model == "gpt2":
        from circuit_cut_analysis.models.gpt2 import GPT2_SMALL
        from circuit_cut_analysis.models.gpt2_circuit import (
            build_gpt2_indexed_circuit,
        )
        from circuit_cut_analysis.models.gpt2_gate_classes import (
            build_gpt2_gate_class_catalog,
        )

        indexed = build_gpt2_indexed_circuit(
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            config=GPT2_SMALL,
        )
        catalog = build_gpt2_gate_class_catalog(
            indexed,
            granularity=gpt2_class_granularity,
        )
        return run_weighted_minimax_study(
            catalog.partition,
            budget_fractions=budget_fractions,
            checked_gate_budgets=checked_gate_budgets,
            detection_threshold=detection_threshold,
            max_quota_evaluations=max_quota_evaluations,
            assumptions=(
                "GPT-2 probability classes use certified lifted canonical cuts.",
                "The capped-linear structural envelope is treated as attained.",
            ),
            run_configuration={
                "model": model,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "numerical_profile_id": indexed.profile.id,
                "gpt2_class_granularity": gpt2_class_granularity,
                "certificate_kind": "lifted-canonical-cut",
                "architecture_config": asdict(GPT2_SMALL),
                "serving_profile": asdict(indexed.profile),
            },
        )

    builders = {
        "kimi-k3": (
            "circuit_cut_analysis.models.kimi_k3",
            "build_kimi_k3_capacity_profile",
            "KIMI_K3",
        ),
        "deepseek-v4-pro": (
            "circuit_cut_analysis.models.deepseek_v4_pro",
            "build_deepseek_v4_pro_capacity_profile",
            "DEEPSEEK_V4_PRO",
        ),
        "inkling": (
            "circuit_cut_analysis.models.inkling",
            "build_inkling_capacity_profile",
            "INKLING",
        ),
    }
    if model not in builders:
        raise ValueError(f"unknown minimax-study model {model!r}")
    module_name, function_name, config_name = builders[model]
    module = importlib.import_module(module_name)
    config = getattr(module, config_name)
    profile = getattr(module, function_name)(
        prompt_tokens,
        generated_tokens,
        config=config,
    )
    from circuit_cut_analysis.weighted_sampling import (
        weighted_partition_from_capacity_profile,
    )

    return run_weighted_minimax_study(
        weighted_partition_from_capacity_profile(profile),
        budget_fractions=budget_fractions,
        checked_gate_budgets=checked_gate_budgets,
        detection_threshold=detection_threshold,
        max_quota_evaluations=max_quota_evaluations,
        assumptions=profile.assumptions,
        run_configuration={
            "model": model,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "numerical_profile_id": profile.numerical_profile_id,
            "certificate_kind": "self-cut",
            "architecture_config": asdict(config),
        },
    )


def _main_minimax_study(args: argparse.Namespace) -> int:
    from circuit_cut_analysis.minimax_study import (
        DEFAULT_MINIMAX_BUDGET_FRACTIONS,
        write_cross_model_minimax_summary,
        write_minimax_study_report,
    )

    selected = args.models or ["all"]
    models = (
        list(SAMPLING_STUDY_MODELS)
        if "all" in selected
        else list(dict.fromkeys(selected))
    )
    budget_fractions = (
        tuple(args.budget_fractions)
        if args.budget_fractions
        else (None if args.checked_gate_budgets else DEFAULT_MINIMAX_BUDGET_FRACTIONS)
    )
    checked_gate_budgets = (
        tuple(args.checked_gate_budgets) if args.checked_gate_budgets else None
    )

    studies: list[MinimaxStudyReport] = []
    for model in models:
        study = _run_one_minimax_study(
            model,
            prompt_tokens=args.prompt_tokens,
            generated_tokens=args.generated_tokens,
            budget_fractions=budget_fractions,
            checked_gate_budgets=checked_gate_budgets,
            detection_threshold=args.detection_threshold,
            max_quota_evaluations=args.max_quota_evaluations,
            gpt2_class_granularity=args.gpt2_class_granularity,
        )
        studies.append(study)
        paths = write_minimax_study_report(study, args.output_dir)
        print(
            f"{study.model_id}: solved {study.solved_budget_count}/"
            f"{len(study.results)} budgets globally "
            f"({study.elapsed_seconds:.1f}s)"
        )
        for path in (paths.json, paths.csv, paths.markdown):
            print(f"wrote {path}")
    if len(studies) > 1:
        summary_digest = hashlib.sha256(
            "|".join(study.configuration_fingerprint for study in studies).encode()
        ).hexdigest()[:12]
        summary_path = write_cross_model_minimax_summary(
            studies,
            args.output_dir,
            stem=(
                f"cross_model_minimax_summary_p{args.prompt_tokens}_"
                f"g{args.generated_tokens}_{summary_digest}"
            ),
        )
        print(f"wrote {summary_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {"sampling-study", "minimax-study"}:
        try:
            if args.command == "sampling-study":
                return _main_sampling_study(args)
            return _main_minimax_study(args)
        except ValueError as error:
            parser.error(str(error))
    if args.command != "gpt2":
        parser.error(f"unsupported command: {args.command}")

    profile = PROFILES[args.profile]
    try:
        analysis = analyze_gpt2_execution(
            prompt_tokens=args.prompt_tokens,
            generated_tokens=args.generated_tokens,
            profile=profile,
        )
    except ValueError as error:
        parser.error(str(error))

    paths = write_reports(analysis, args.output_dir)
    print(
        f"prompt={analysis.prompt_tokens}, generated={analysis.generated_tokens}: "
        f"{analysis.total_unit_gates:,} primitive gates; "
        f"{analysis.contraction_flops:,} contraction FLOPs"
    )
    print(f"global canonical partition: {analysis.partition_status.value}")
    support = analysis.metadata["exact_partition_support"]
    print(
        "partition descriptors: "
        f"{support['evaluated_index_regions']:,} evaluated, "
        f"{support['retained_region_descriptors']:,} retained; "
        f"{support['materialized_scalar_nodes']:,} scalar nodes materialized"
    )
    for path in (paths.json, paths.csv, paths.markdown):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
