"""Step 23: the statistical report, tested on synthetic Step 21 artefacts.

    "Report mean and standard deviation across folds or random seeds. Use 95% confidence
     intervals for main metrics. Use McNemar test for paired comparison ... Report p-values
     only when the experimental design supports paired testing. Do not overstate minor
     improvements."

This stage does no modelling at all - it reads Step 21's saved predictions and turns them
into claims. Which means every way it can be wrong is a way of overstating evidence, and
none of them looks like a crash:

* **Uncorrected p-values.** Four primary comparisons at alpha=0.05 give a ~19% chance of at
  least one false positive if each is judged on its raw p. Holm is what makes the family
  claim honest, so the adjustment is asserted numerically rather than assumed.
* **Broken pairing.** A bootstrap that resamples the two models independently discards the
  correlation between models that saw identical images, and reports a wider interval that
  looks like an honest one. Worse, rows A6 and A3 reach their predictions through different
  datamodules - features versus images - so their sample order has to be *verified* aligned,
  not trusted.
* **An underpowered test reported anyway.** Three seeds cannot support a Wilcoxon; the
  two-sided floor at n=3 is 0.25. The refusal must survive.
* **A promoted comparison.** A7-vs-P is scientifically interesting and P has one seed, so it
  is descriptive. If it silently acquired a p-value the report would claim more than the
  design supports.

Nothing here trains or loads a checkpoint: Step 21's artefacts are synthesised on disk.
"""

import json
import zlib

import numpy as np
import pytest

from src.analysis.statistical_report import (
    PRIMARY_FAMILY,
    REPORT_COLUMNS,
    StatisticalReport,
)
from src.utils.statistics import MIN_WILCOXON_PAIRS, holm_bonferroni

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No-tumor"]

SEEDS = (42, 123, 7)

#: Resamples used in fixtures. The correction, pairing and determinism logic is independent
#: of this; the configured default of 2000 is asserted separately against the real config.
FAST_RESAMPLES = 200

#: Rows Step 21 saves predictions for.
TRAINED_ROWS = ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7")


def _stable_seed(row_id: str, seed: int) -> int:
    """A per-row, per-seed generator seed that is identical in every process.

    Python's built-in ``hash`` is randomised per interpreter by PYTHONHASHSEED, so using it
    here made the synthetic predictions differ between runs - which showed up as a test
    that passed in one invocation and failed in the next. A file asserting that the report
    is deterministic must not itself be non-deterministic.

    :param row_id: Row identifier.
    :param seed: Training seed.
    :return: A stable 32-bit seed.
    """
    return zlib.crc32(f"{row_id}:{seed}".encode()) % 2**32


def synth_predictions(rng, y_true, skill):
    """Predictions of a given skill over a fixed label vector.

    Sharing ``y_true`` across rows is what makes the comparison paired, exactly as a shared
    test split does in the real run.

    :param rng: Random generator.
    :param y_true: The shared label vector.
    :param skill: Higher means more accurate.
    :return: ``(y_pred, y_prob)``.
    """
    logits = rng.normal(size=(len(y_true), len(CLASS_NAMES)))
    logits[np.arange(len(y_true)), y_true] += skill
    y_prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return y_prob.argmax(axis=1), y_prob


#: Skill per row. A2 > A1 and A5 > A4 give H1/H2 a real effect; A7 == A6 in skill gives H3
#: a null, so the report has to be able to say "no difference" as well as "difference".
ROW_SKILL = {
    "A0": 0.6,
    "A1": 0.9,
    "A2": 1.6,
    "A3": 1.5,
    "A4": 1.0,
    "A5": 1.7,
    "A6": 1.5,
    "A7": 1.5,
}


@pytest.fixture
def ablation_dir(tmp_path):
    """A directory shaped like Step 21's output.

    :param tmp_path: Per-test directory.
    :return: The directory.
    """
    directory = tmp_path / "step21"
    directory.mkdir()

    rng = np.random.default_rng(0)
    y_true = rng.integers(0, len(CLASS_NAMES), 400)

    rows = {}
    for row_id in TRAINED_ROWS:
        per_seed = {}
        for seed in SEEDS:
            draw = np.random.default_rng(_stable_seed(row_id, seed))
            y_pred, y_prob = synth_predictions(draw, y_true, ROW_SKILL[row_id])
            np.savez_compressed(
                directory / f"step21_predictions_{row_id}_seed{seed}.npz",
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
            )
            from sklearn.metrics import f1_score

            per_seed[str(seed)] = {
                "n_samples": len(y_true),
                "overall": {
                    "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))
                },
                "per_class": [
                    {"class_name": name, "recall_sensitivity": 0.5} for name in CLASS_NAMES
                ],
                "calibration": {},
                "checkpoint": f"runs/{row_id}/seed_{seed}/checkpoints/epoch_003.ckpt",
                "provenance": "computed by step21 from the validation-selected checkpoint",
            }
        rows[row_id] = {"row_id": row_id, "per_seed": per_seed, "evaluated": True,
                        "missing_seeds": [], "trains": True}

    rows["A8"] = {
        "row_id": "A8",
        "per_seed": {
            s: dict(m, provenance="identical to A7 by construction; explanations change no weights")
            for s, m in rows["A7"]["per_seed"].items()
        },
        "evaluated": True,
        "missing_seeds": [],
        "mirrors": "A7",
        "trains": False,
    }
    rows["P"] = {
        "row_id": "P",
        "per_seed": {"42": dict(rows["A7"]["per_seed"]["42"],
                                provenance="reused from Step 16 (step16_internal_summary.json)")},
        "evaluated": True,
        "missing_seeds": [123, 7],
        "reused_from": "step16_internal",
        "trains": False,
    }

    (directory / "step21_ablation_summary.json").write_text(
        json.dumps({"seeds": list(SEEDS), "rows": rows,
                    "coverage": {"complete": True, "incomplete_rows": {}}}),
        encoding="utf-8",
    )
    return directory


@pytest.fixture
def report(ablation_dir, tmp_path):
    """:param ablation_dir: Synthetic Step 21 output.

    :param tmp_path: Per-test directory.
    :return: A configured report, not yet run.
    """
    analysis = StatisticalReport(ablation_dir=str(ablation_dir), n_resamples=FAST_RESAMPLES)
    analysis._output_dir = tmp_path / "out"
    analysis._output_dir.mkdir()
    return analysis


@pytest.fixture
def result(report):
    """:param report: The configured report.

    :return: The computed summary.
    """
    return report.compute(datamodule=None)


# ------------------------------------------------------------- Holm-Bonferroni


def test_holm_matches_the_textbook_step_down():
    """Sorted ascending, each p scaled by the number of remaining hypotheses.

    For p = .01, .02, .03, .04 with n=4: .04, .06, .06, .06 after the monotonicity
    constraint. Getting this wrong in the lenient direction manufactures significance.
    """
    adjusted = holm_bonferroni({"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04}, alpha=0.05)

    assert adjusted["a"]["adjusted_p_value"] == pytest.approx(0.04)
    assert adjusted["b"]["adjusted_p_value"] == pytest.approx(0.06)
    assert adjusted["c"]["adjusted_p_value"] == pytest.approx(0.06)
    assert adjusted["d"]["adjusted_p_value"] == pytest.approx(0.06)

    assert adjusted["a"]["significant"] is True
    assert all(not adjusted[k]["significant"] for k in "bcd")


def test_holm_is_monotone_non_decreasing():
    """A later hypothesis can never end up with a smaller adjusted p than an earlier one."""
    adjusted = holm_bonferroni({"h1": 0.001, "h2": 0.049, "h3": 0.05, "h4": 0.9})
    ordered = [adjusted[k]["adjusted_p_value"] for k in ("h1", "h2", "h3", "h4")]

    assert ordered == sorted(ordered)


def test_holm_never_exceeds_one():
    """Scaling can overshoot; a probability may not."""
    adjusted = holm_bonferroni({"a": 0.4, "b": 0.6, "c": 0.9, "d": 0.95})

    assert all(entry["adjusted_p_value"] <= 1.0 for entry in adjusted.values())


def test_holm_is_stricter_than_no_correction():
    """The whole point: a raw p below alpha may not survive the family."""
    raw = {"h1": 0.02, "h2": 0.03, "h3": 0.04, "h4": 0.9}
    adjusted = holm_bonferroni(raw, alpha=0.05)

    assert all(p < 0.05 for p in list(raw.values())[:3])
    assert not any(adjusted[k]["significant"] for k in ("h1", "h2", "h3"))


def test_holm_preserves_comparison_ids_through_the_sort():
    """The sort is internal; a caller must get its own labels back, correctly paired."""
    adjusted = holm_bonferroni({"H4": 0.001, "H1": 0.9, "H3": 0.5, "H2": 0.02})

    assert set(adjusted) == {"H1", "H2", "H3", "H4"}
    assert adjusted["H4"]["rank"] == 1
    assert adjusted["H1"]["rank"] == 4
    assert adjusted["H4"]["raw_p_value"] == 0.001
    assert adjusted["H1"]["raw_p_value"] == 0.9


def test_holm_respects_the_configured_alpha():
    """alpha is a parameter, not a constant baked into the decision."""
    lenient = holm_bonferroni({"a": 0.03, "b": 0.9}, alpha=0.10)
    strict = holm_bonferroni({"a": 0.03, "b": 0.9}, alpha=0.05)

    assert lenient["a"]["significant"] is True
    assert strict["a"]["significant"] is False


def test_holm_handles_a_missing_p_value():
    """McNemar returns None when the two models predicted identically."""
    adjusted = holm_bonferroni({"a": 0.01, "b": None})

    assert adjusted["b"]["adjusted_p_value"] is None
    assert adjusted["b"]["significant"] is False
    # And the family size shrinks to the testable members rather than silently staying 2.
    assert adjusted["a"]["adjusted_p_value"] == pytest.approx(0.01)


# ------------------------------------------------------- the primary family


def test_the_primary_family_is_exactly_the_four_declared_hypotheses():
    """Not "every row against A6" - only comparisons that isolate one factor."""
    assert [h["id"] for h in PRIMARY_FAMILY] == ["H1", "H2", "H3", "H4"]
    assert [(h["row_a"], h["row_b"]) for h in PRIMARY_FAMILY] == [
        ("A2", "A1"),
        ("A5", "A4"),
        ("A7", "A6"),
        ("A6", "A3"),
    ]
    assert [h["rq"] for h in PRIMARY_FAMILY] == ["RQ2", "RQ4", "RQ6", "RQ8"]


def test_only_the_declared_family_receives_significance_testing(result):
    """A significance claim outside the family would not be covered by the correction."""
    tested = {c["comparison"] for c in result["primary"]}
    assert tested == {"H1", "H2", "H3", "H4"}

    for comparison in result["secondary"]:
        assert comparison["adjusted_p_value"] is None
        assert comparison["significant"] is None


def test_an_undeclared_comparison_cannot_be_smuggled_into_the_family(report):
    """The family size drives the correction, so adding a member changes every verdict."""
    report.primary_comparisons = [
        *PRIMARY_FAMILY,
        {"id": "H5", "row_a": "A0", "row_b": "A1", "rq": "RQ2", "question": "extra"},
    ]

    with pytest.raises(ValueError, match="pre-registered"):
        report.compute(datamodule=None)


def test_every_primary_comparison_reports_raw_and_adjusted_p(result):
    """"Never report p-values without identifying whether they are raw or adjusted"."""
    for comparison in result["primary"]:
        assert "raw_p_value" in comparison
        assert "adjusted_p_value" in comparison
        assert comparison["correction"] == "holm-bonferroni"
        assert comparison["family_size"] == 4


def test_significance_follows_the_adjusted_p_not_the_raw_one(result):
    """The verdict must come from the corrected value."""
    for comparison in result["primary"]:
        if comparison["adjusted_p_value"] is None:
            assert comparison["significant"] is False
            continue
        assert comparison["significant"] == (
            comparison["adjusted_p_value"] <= result["parameters"]["alpha"]
        )


# ------------------------------------------------------------ paired testing


def test_delta_macro_f1_is_the_difference_of_the_two_rows(result, ablation_dir):
    """The effect metric is macro-F1, not accuracy."""
    from sklearn.metrics import f1_score

    h1 = next(c for c in result["primary"] if c["comparison"] == "H1")
    a2 = np.load(ablation_dir / "step21_predictions_A2_seed42.npz")
    a1 = np.load(ablation_dir / "step21_predictions_A1_seed42.npz")

    expected = f1_score(a2["y_true"], a2["y_pred"], average="macro", zero_division=0) - f1_score(
        a1["y_true"], a1["y_pred"], average="macro", zero_division=0
    )

    assert h1["metric"] == "macro_f1"
    assert h1["observed_delta"] == pytest.approx(expected)


def test_the_bootstrap_preserves_pairing(report, ablation_dir):
    """Independent resampling would discard the correlation and widen the interval.

    Two rows that made *identical* predictions must give a delta of exactly zero in every
    resample - an interval of zero width. Independent draws could not produce that.
    """
    payload = np.load(ablation_dir / "step21_predictions_A2_seed42.npz")
    np.savez_compressed(
        ablation_dir / "step21_predictions_A1_seed42.npz",
        y_true=payload["y_true"],
        y_pred=payload["y_pred"],
        y_prob=payload["y_prob"],
    )

    h1 = next(c for c in report.compute(datamodule=None)["primary"] if c["comparison"] == "H1")

    assert h1["observed_delta"] == pytest.approx(0.0)
    assert h1["ci_low"] == pytest.approx(0.0)
    assert h1["ci_high"] == pytest.approx(0.0)


def test_the_bootstrap_is_deterministic_for_a_fixed_seed(ablation_dir, tmp_path):
    """A report that changes between runs cannot be quoted in a paper."""
    intervals = []
    for run in range(2):
        analysis = StatisticalReport(ablation_dir=str(ablation_dir), random_seed=7, n_resamples=FAST_RESAMPLES)
        analysis._output_dir = tmp_path / f"out{run}"
        analysis._output_dir.mkdir()
        intervals.append(
            [(c["ci_low"], c["ci_high"]) for c in analysis.compute(datamodule=None)["primary"]]
        )

    assert intervals[0] == intervals[1]


def test_a_different_seed_gives_a_different_interval(ablation_dir, tmp_path):
    """Confirms the seed is actually reaching the resampler."""
    def run(seed, name):
        analysis = StatisticalReport(ablation_dir=str(ablation_dir), random_seed=seed, n_resamples=FAST_RESAMPLES)
        analysis._output_dir = tmp_path / name
        analysis._output_dir.mkdir()
        return [(c["ci_low"], c["ci_high"]) for c in analysis.compute(datamodule=None)["primary"]]

    assert run(1, "a") != run(999, "b")


def test_misaligned_predictions_are_refused(report, ablation_dir):
    """H4 pairs a feature-space row with an image-space row.

    A6 reaches its predictions through the feature cache and A3 through the image loader.
    Both preserve test order today, so the pairing is valid - but a paired test on
    misaligned samples produces a confident, wrong p-value rather than an error, so the
    alignment is verified rather than assumed.
    """
    payload = np.load(ablation_dir / "step21_predictions_A3_seed42.npz")
    np.savez_compressed(
        ablation_dir / "step21_predictions_A3_seed42.npz",
        y_true=payload["y_true"][::-1],
        y_pred=payload["y_pred"],
        y_prob=payload["y_prob"],
    )

    with pytest.raises(ValueError, match="not aligned"):
        report.compute(datamodule=None)


def test_mcnemar_uses_the_exact_test_when_discordance_is_small(report, ablation_dir):
    """Below 25 discordant pairs the chi-square approximation is unreliable."""
    payload = np.load(ablation_dir / "step21_predictions_A2_seed42.npz")
    y_true, y_pred = payload["y_true"], payload["y_pred"].copy()
    near = y_pred.copy()
    near[:6] = (near[:6] + 1) % len(CLASS_NAMES)  # a handful of disagreements
    np.savez_compressed(
        ablation_dir / "step21_predictions_A1_seed42.npz",
        y_true=y_true, y_pred=near, y_prob=payload["y_prob"],
    )

    h1 = next(c for c in report.compute(datamodule=None)["primary"] if c["comparison"] == "H1")

    assert h1["mcnemar"]["n_discordant"] < 25
    assert h1["mcnemar_method"] == "exact binomial"


def test_mcnemar_uses_chi_square_when_discordance_is_large(result):
    """With hundreds of samples and a real effect, the approximation is the right one."""
    h1 = next(c for c in result["primary"] if c["comparison"] == "H1")

    assert h1["mcnemar"]["n_discordant"] >= 25
    assert h1["mcnemar_method"] == "chi-square with continuity correction"


def test_the_p_value_source_is_named(result):
    """McNemar tests error discordance; the delta is macro-F1. Conflating them misreads both."""
    for comparison in result["primary"]:
        assert comparison["p_value_source"] == "mcnemar"
        assert "macro-F1" in comparison["interpretation"]


# ------------------------------------------------------------------- seeds


def test_seed_spread_is_descriptive_only(result):
    """Three seeds cannot support a significance claim; they describe variability."""
    for comparison in result["primary"]:
        seeds = comparison["seeds"]
        assert seeds["role"] == "descriptive"
        assert seeds["wilcoxon"]["p_value"] is None
        assert str(MIN_WILCOXON_PAIRS) in seeds["wilcoxon"]["note"]


def test_wilcoxon_is_refused_below_six_pairs():
    """The reference notebook's exact mistake, at n=4."""
    from src.utils.statistics import wilcoxon_paired

    result = wilcoxon_paired([0.1, 0.2, 0.3], [0.2, 0.3, 0.4], label="seeds")

    assert result["p_value"] is None
    assert "at least 6" in result["note"]


def test_wilcoxon_is_permitted_at_six_pairs():
    """The refusal is a power floor, not a blanket ban."""
    from src.utils.statistics import wilcoxon_paired

    result = wilcoxon_paired([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 8])

    assert result["n_pairs"] == 6
    assert result["p_value"] is not None


def test_per_seed_deltas_are_reported_with_mean_and_std(result):
    """Step 23: "Report mean and standard deviation across folds or random seeds"."""
    for comparison in result["primary"]:
        spread = comparison["seeds"]["delta_across_seeds"]
        assert spread["n"] == 3
        assert spread["std"] is not None


def test_a_missing_seed_is_reported_not_dropped(report, ablation_dir):
    """A two-seed spread presented as a three-seed one overstates the evidence."""
    (ablation_dir / "step21_predictions_A2_seed7.npz").unlink()
    summary = report.compute(datamodule=None)

    h1 = next(c for c in summary["primary"] if c["comparison"] == "H1")

    assert 7 in h1["seeds"]["missing_seeds"]
    assert h1["seeds"]["delta_across_seeds"]["n"] == 2
    assert summary["integrity"]["complete"] is False


# ------------------------------------------------------------- rows P and A8


def test_row_p_is_recorded_as_single_seed(result):
    """One evaluated checkpoint, by Step 16's design - not a variance estimate."""
    p = result["rows"]["P"]

    assert p["n_seeds"] == 1
    assert p["single_seed"] is True
    assert p["seed_spread"] is None
    assert "not directly comparable" in p["note"]


def test_a7_vs_p_is_descriptive_and_carries_no_significance_claim(result):
    """P has one seed, so a seed-level claim is unsupported by the design."""
    comparison = next(c for c in result["secondary"] if c["comparison"] == "A7_vs_P")

    assert comparison["adjusted_p_value"] is None
    assert comparison["significant"] is None
    assert comparison["raw_p_value"] is None
    assert "single-seed" in comparison["note"]


def test_a7_vs_p_degrades_to_metrics_only_without_fabricating_an_interval(result):
    """Step 21 saves no predictions for P, so there is nothing to resample.

    P is reported from Step 16's summary rather than re-evaluated - that is what keeps the
    once-only test budget intact - so a paired interval is genuinely unavailable. Reporting
    a difference of point estimates with a null interval is the honest answer; imputing one
    would be the tempting one.
    """
    comparison = next(c for c in result["secondary"] if c["comparison"] == "A7_vs_P")

    assert comparison["estimation"] == "metrics-only"
    assert comparison["observed_delta"] is not None
    assert comparison["ci_low"] is None and comparison["ci_high"] is None
    assert "step16_predictions" in comparison["note"]
    assert comparison["provenance"]["row_b_predictions"].endswith("rows.P")


def test_supplying_step16_predictions_restores_a_paired_interval(ablation_dir, tmp_path):
    """The opt-in upgrade: Step 16 already saved its predictions, so no new evaluation.

    Off by default because it consumes an artefact from outside Step 21, which is a scope
    decision rather than a technical one.
    """
    source = np.load(ablation_dir / "step21_predictions_A6_seed42.npz")
    shipped = tmp_path / "test_predictions.npz"
    np.savez_compressed(shipped, y_true=source["y_true"], y_pred=source["y_pred"],
                        y_prob=source["y_prob"])

    analysis = StatisticalReport(
        ablation_dir=str(ablation_dir),
        step16_predictions=str(shipped),
        n_resamples=FAST_RESAMPLES,
    )
    analysis._output_dir = tmp_path / "out"
    analysis._output_dir.mkdir()

    comparison = next(
        c for c in analysis.compute(datamodule=None)["secondary"] if c["comparison"] == "A7_vs_P"
    )

    assert comparison["ci_low"] is not None
    assert comparison["significant"] is None, "still descriptive - P remains single-seed"
    assert comparison["raw_p_value"] is None
    assert "once-only test budget is untouched" in comparison["provenance"]["note"]


def test_a_missing_primary_row_is_fatal_even_though_a_missing_secondary_row_is_not(report, ablation_dir):
    """The asymmetry is deliberate: a short family would be corrected as if it were whole."""
    for seed in SEEDS:
        (ablation_dir / f"step21_predictions_A6_seed{seed}.npz").unlink()

    with pytest.raises(FileNotFoundError, match="A6"):
        report.compute(datamodule=None)


def test_no_variance_is_invented_for_p(result):
    """Fabricating a spread would make P look comparable when it is not."""
    p = result["rows"]["P"]

    assert p["seed_spread"] is None
    assert p["missing_seeds"] == [123, 7]


def test_a8_is_not_treated_as_an_independent_model(result):
    """Its metrics are A7's; a delta against A7 would be exactly zero and meaningless."""
    assert not any("A8" in c["comparison"] for c in result["primary"])

    a8 = result["rows"]["A8"]
    assert a8["mirrors"] == "A7"
    assert a8["excluded_from_testing"] is True
    assert "explainability" in a8["note"].lower()


# ------------------------------------------------------------- class-wise recall


def test_class_wise_recall_is_preserved_as_a_primary_metric(result):
    """Step 21 names it primary; Step 23 must not quietly reduce everything to macro-F1."""
    for comparison in result["primary"]:
        recall = comparison["per_class_recall_delta"]
        assert set(recall) == set(CLASS_NAMES)
        assert all(v is not None for v in recall.values())


def test_macro_f1_is_the_effect_metric_not_accuracy(result):
    """:param result: The computed summary."""
    assert result["parameters"]["primary_metric"] == "macro_f1"
    assert all(c["metric"] == "macro_f1" for c in result["primary"])


# ------------------------------------------------------------------ hygiene


def test_the_report_reads_no_training_artefacts():
    """Step 21's saved predictions are the only input; metrics.csv is never opened."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path("src/analysis/statistical_report.py").read_text(encoding="utf-8"))
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }

    assert "read_csv" not in called
    assert not {"load_module", "find_checkpoint", "Trainer", "fit"} & called


def test_every_comparison_carries_provenance(result):
    """A number without a source cannot be checked."""
    for comparison in [*result["primary"], *result["secondary"]]:
        provenance = comparison["provenance"]
        assert provenance["source"] == "step21_ablation"
        assert provenance["row_a_predictions"].endswith(".npz")
        # Metrics-only comparisons point at the summary entry instead of a prediction file.
        assert provenance["row_b_predictions"].endswith(".npz") or (
            "step21_ablation_summary.json" in provenance["row_b_predictions"]
        )


def test_a_missing_ablation_directory_fails_clearly(tmp_path):
    """:param tmp_path: Per-test directory."""
    analysis = StatisticalReport(ablation_dir=str(tmp_path / "nope"))

    with pytest.raises(FileNotFoundError, match="Step 21"):
        analysis.compute(datamodule=None)


def test_a_missing_prediction_file_fails_clearly(report, ablation_dir):
    """Silently skipping H2 would leave a three-member family corrected as if it were four."""
    for seed in SEEDS:
        (ablation_dir / f"step21_predictions_A5_seed{seed}.npz").unlink()

    with pytest.raises(FileNotFoundError, match="A5"):
        report.compute(datamodule=None)


def test_a_malformed_summary_fails_clearly(report, ablation_dir):
    """:param report: The configured report.

    :param ablation_dir: Synthetic Step 21 output.
    """
    (ablation_dir / "step21_ablation_summary.json").write_text("{ not json", encoding="utf-8")

    with pytest.raises(ValueError, match="Step 21"):
        report.compute(datamodule=None)


def test_an_empty_ablation_summary_fails_clearly(report, ablation_dir):
    """:param report: The configured report.

    :param ablation_dir: Synthetic Step 21 output.
    """
    (ablation_dir / "step21_ablation_summary.json").write_text(
        json.dumps({"rows": {}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="no rows"):
        report.compute(datamodule=None)


# ------------------------------------------------------------------- schema


def test_the_report_schema_is_stable(result, report):
    """Downstream (Step 22, the write-up) keys on these fields."""
    required = {
        "comparison", "row_a", "row_b", "metric", "observed_delta", "ci_low", "ci_high",
        "raw_p_value", "adjusted_p_value", "significant", "effect_direction",
        "mcnemar_method", "mcnemar_p_value", "n_test_samples", "seeds", "provenance",
    }

    for comparison in result["primary"]:
        assert required <= set(comparison)


def test_the_flat_table_columns_are_pinned(report, result):
    """:param report: The configured report.

    :param result: The computed summary.
    """
    import pandas as pd

    frame = pd.read_csv(report.output_dir / "step23_comparisons.csv")

    assert list(frame.columns) == list(REPORT_COLUMNS)
    assert set(frame["comparison"]) >= {"H1", "H2", "H3", "H4"}


def test_an_empty_family_still_produces_the_full_schema(report):
    """:param report: The configured report."""
    frame = report._flat_table([], [])

    assert len(frame) == 0
    assert list(frame.columns) == list(REPORT_COLUMNS)


def test_the_parameters_are_recorded_in_the_output(result):
    """A report that does not state its own alpha cannot be audited."""
    parameters = result["parameters"]

    assert parameters["alpha"] == 0.05
    assert parameters["confidence_level"] == 0.95
    assert parameters["random_seed"] == 42
    assert parameters["minimum_wilcoxon_pairs"] == MIN_WILCOXON_PAIRS
    assert parameters["correction"] == "holm-bonferroni"
    # The fixture thins the bootstrap for speed; whatever it is, it is reported.
    assert parameters["n_resamples"] == FAST_RESAMPLES


def test_the_default_resample_count_follows_the_project_convention():
    """2000, matching Step 20 and src/utils/statistics.py - not a new number."""
    from src.utils.statistics import paired_bootstrap as reference

    assert StatisticalReport().n_resamples == 2000
    assert reference.__defaults__[0] == 2000


def test_a_configured_alpha_changes_the_verdicts(ablation_dir, tmp_path):
    """alpha must come from config, not from a literal in the code."""
    def run(alpha, name):
        analysis = StatisticalReport(ablation_dir=str(ablation_dir), alpha=alpha, n_resamples=FAST_RESAMPLES)
        analysis._output_dir = tmp_path / name
        analysis._output_dir.mkdir()
        summary = analysis.compute(datamodule=None)
        return summary["parameters"]["alpha"], [
            (c["adjusted_p_value"], c["significant"]) for c in summary["primary"]
        ]

    strict_alpha, strict = run(1e-12, "strict")
    lenient_alpha, lenient = run(1.0, "lenient")

    assert strict_alpha == 1e-12
    assert lenient_alpha == 1.0

    # Two caveats the bounds have to respect. A comparison with no p-value - McNemar is
    # undefined when two models predict identically - is never significant at any alpha.
    # And Holm caps adjusted values at 1.0, so only alpha=1.0 passes a capped one.
    testable = [significant for adjusted, significant in lenient if adjusted is not None]
    assert testable and all(testable), "alpha=1.0 should pass every testable comparison"
    assert not any(significant for _, significant in strict), "alpha=1e-12 should pass none"


# ----------------------------------------------------------- Hydra composition


def test_the_step23_stage_composes():
    """:return: None."""
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="analyze.yaml", overrides=["analysis=step23_statistics"])
    GlobalHydra.instance().clear()

    assert cfg.analysis._target_ == "src.analysis.statistical_report.StatisticalReport"
    assert cfg.analysis.alpha == 0.05
    assert cfg.analysis.confidence_level == 0.95
    assert cfg.analysis.n_resamples == 2000
    assert cfg.analysis.minimum_wilcoxon_pairs == 6


def test_the_config_declares_the_primary_family_explicitly():
    """The family is a pre-registration, so it belongs in the config, readable."""
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="analyze.yaml", overrides=["analysis=step23_statistics"])
    GlobalHydra.instance().clear()

    declared = [(h.id, h.row_a, h.row_b) for h in cfg.analysis.primary_comparisons]

    assert declared == [
        ("H1", "A2", "A1"),
        ("H2", "A5", "A4"),
        ("H3", "A7", "A6"),
        ("H4", "A6", "A3"),
    ]


def test_no_validated_artefact_was_touched():
    """:return: None."""
    import subprocess

    changed = subprocess.run(
        ["git", "status", "--short", "--", "configs/protocol",
         "configs/experiment/step15_final_protocol.yaml", "data/splits"],
        capture_output=True, text=True,
    ).stdout.strip()

    assert changed == "", f"validated artefacts were modified:\n{changed}"


# --------------------------------------------- the correction must change verdicts

# The fixture's synthetic effects are large enough that Holm never flips a verdict, which
# makes every assertion about the correction pass whether or not it is applied. These tests
# drive _apply_holm with p-values chosen so the corrected and uncorrected answers DIFFER,
# which is the only arrangement that can detect the correction being skipped.


def _record(comparison_id, raw_p):
    """A minimal primary record carrying a chosen raw p-value.

    :param comparison_id: Comparison identifier.
    :param raw_p: The raw p-value to correct.
    :return: The record.
    """
    return {"comparison": comparison_id, "raw_p_value": raw_p, "significant": False,
            "adjusted_p_value": None}


def test_holm_actually_changes_the_verdicts_it_is_applied_to(report):
    """Four raw p-values all below alpha; none survives the correction.

    p = .02/.03/.04/.045 scaled by 4/3/2/1 gives .08/.09/.09/.09 - every one above alpha,
    where every raw value was below it. If the adjustment were skipped, or the verdict read
    from the raw p, all four would come back significant.
    """
    records = [_record("H1", 0.02), _record("H2", 0.03), _record("H3", 0.04),
               _record("H4", 0.045)]
    report._apply_holm(records)

    assert all(r["raw_p_value"] < 0.05 for r in records), "premise: raw p all significant"
    assert all(r["adjusted_p_value"] > 0.05 for r in records)
    assert not any(r["significant"] for r in records)


def test_the_smallest_p_can_still_survive_the_correction(report):
    """Holm is not a blanket veto: a strong enough result passes."""
    records = [_record("H1", 0.001), _record("H2", 0.30), _record("H3", 0.40),
               _record("H4", 0.90)]
    report._apply_holm(records)

    by_id = {r["comparison"]: r for r in records}

    assert by_id["H1"]["adjusted_p_value"] == pytest.approx(0.004)
    assert by_id["H1"]["significant"] is True
    assert not any(by_id[k]["significant"] for k in ("H2", "H3", "H4"))


def test_the_adjusted_p_is_never_the_raw_p_when_the_family_has_members(report):
    """Directly catches "adjusted = raw", which no verdict assertion alone can see."""
    records = [_record("H1", 0.01), _record("H2", 0.02), _record("H3", 0.03),
               _record("H4", 0.04)]
    report._apply_holm(records)

    assert all(r["adjusted_p_value"] > r["raw_p_value"] for r in records)


def test_the_verdict_ignores_the_raw_p_even_when_they_disagree(report):
    """A record whose raw p is significant and whose adjusted p is not must read False."""
    records = [_record("H1", 0.02), _record("H2", 0.03), _record("H3", 0.04),
               _record("H4", 0.045)]
    report._apply_holm(records)

    borderline = next(r for r in records if r["comparison"] == "H1")

    assert borderline["raw_p_value"] <= report.alpha
    assert borderline["adjusted_p_value"] > report.alpha
    assert borderline["significant"] is False


def test_holm_ranks_are_recorded_for_audit(report):
    """The step-down order is part of the procedure and belongs in the output."""
    records = [_record("H1", 0.40), _record("H2", 0.01), _record("H3", 0.90),
               _record("H4", 0.20)]
    report._apply_holm(records)

    by_id = {r["comparison"]: r["holm_rank"] for r in records}

    assert by_id == {"H2": 1, "H4": 2, "H1": 3, "H3": 4}
