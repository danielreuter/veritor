from __future__ import annotations

from fractions import Fraction

import pytest

from circuit_cut_analysis.cli import build_parser, main


def test_minimax_parser_preserves_exact_probability_arguments() -> None:
    args = build_parser().parse_args(
        (
            "minimax-study",
            "--model",
            "gpt2",
            "--budget-fraction",
            "1/20",
            "--detection-threshold",
            "0.99",
        )
    )

    assert args.command == "minimax-study"
    assert args.models == ["gpt2"]
    assert args.budget_fractions == [Fraction(1, 20)]
    assert args.detection_threshold == Fraction(99, 100)


def test_minimax_parser_rejects_out_of_range_probability() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(("minimax-study", "--budget-fraction", "1.01"))


def test_minimax_parser_accepts_zero_exact_checked_gate_budget() -> None:
    args = build_parser().parse_args(("minimax-study", "--checked-gates", "0"))

    assert args.checked_gate_budgets == [0]
    assert args.budget_fractions is None


def test_minimax_parser_rejects_zero_denominator_cleanly() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(("minimax-study", "--budget-fraction", "1/0"))


def test_minimax_parser_rejects_two_budget_modes() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            (
                "minimax-study",
                "--budget-fraction",
                "1/10",
                "--checked-gates",
                "10",
            )
        )


def test_minimax_context_error_is_reported_without_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(
            (
                "minimax-study",
                "--model",
                "kimi-k3",
                "--prompt-tokens",
                "1048576",
                "--generated-tokens",
                "2",
                "--checked-gates",
                "0",
            )
        )

    stderr = capsys.readouterr().err
    assert "context window" in stderr
    assert "Traceback" not in stderr
