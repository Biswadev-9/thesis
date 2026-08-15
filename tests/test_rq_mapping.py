"""Step 22: the research-question evidence map, tested on synthetic artefacts.

Step 22 answers "for each RQ, what existing experiment supplies evidence?" - and nothing
else. It computes no metric, runs no test, and reaches no verdict Steps 21 and 23 have not
already reached. Which makes its failure modes quiet ones:

* **Promotion.** A7-vs-P and the A0-A7 ladder are descriptive by design. A mapping that
  presents them beside H1-H4 without distinction turns four corrected hypotheses into
  eight uncorrected ones, in a table a reader will quote directly.
* **Invention.** An RQ with no usable artefact is the tempting one to fill in from a
  neighbouring experiment. RQ5 is the live case: the dataset ships no tumour-size
  annotations, so subgroup evidence does not exist and must be reported as absent.
* **Silent omission.** A dropped RQ looks like a shorter table, not like a gap.
* **Drift from the specification.** The RQ text belongs to the specification; retyped from
  memory it slowly stops meaning what was asked. The config's text is checked against the
  specification file itself.

Nothing here trains, evaluates, or reads a checkpoint.
"""

import ast
import json
import re
from pathlib import Path

import pytest

from src.analysis.rq_mapping import (
    EVIDENCE_COLUMNS,
    EVIDENCE_ITEM_FIELDS,
    RQ_IDS,
    RQMapping,
)

SPEC = Path("docs/Instruction BY asif vai.md")


def spec_rq_table():
    """Parse the RQ table straight out of the specification.

    :return: ``{rq_id: question}`` as the specification words it.
    """
    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("Step 22:"):text.index("Step 23:")]

    table = {}
    for match in re.finditer(r"\|(RQ(\d+)):\s*([^|]+?)\|", section):
        table[match.group(1)] = match.group(3).replace("<br>", " ").strip()
    return table


def compose_step22(*overrides):
    """Compose the Step 22 analysis config.

    :param overrides: Extra Hydra overrides.
    :return: The composed DictConfig.
    """
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="analyze.yaml", overrides=["analysis=step22_rq_mapping", *overrides]
        )
    GlobalHydra.instance().clear()
    return cfg


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def artefacts(tmp_path):
    """A tree shaped like the study's completed analysis outputs.

    :param tmp_path: Per-test directory.
    :return: The directory holding one subdirectory per stage.
    """
    root = tmp_path / "analyze" / "runs"

    def write(stage, payload):
        directory = root / stage
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{stage}_summary.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = {
        rid: {
            "row_id": rid,
            "label": f"label {rid}",
            "recipe": "diffusion_i10_k15" if rid in ("A2", "A3", "A4", "A5", "A6", "A7", "A8") else "raw",
            "loss": "plain_ce",
            "per_seed": {"42": {"overall": {"macro_f1": 0.5 + index / 100}}},
            "across_seeds": {
                "macro_f1": {"mean": 0.5 + index / 100, "std": 0.01, "n": 3},
                "per_class_recall": {"Glioma": {"mean": 0.5, "std": 0.01, "n": 3}},
            },
            "evaluated": True,
            "missing_seeds": [],
        }
        for index, rid in enumerate(["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "P"])
    }
    rows["A8"]["mirrors"] = "A7"
    rows["P"]["missing_seeds"] = [123, 7]
    rows["P"]["reused_from"] = "step16_internal"
    write("step21_ablation", {"seeds": [42, 123, 7], "rows": rows,
                              "coverage": {"complete": True}})

    write("step23_statistics", {
        "parameters": {"alpha": 0.05, "correction": "holm-bonferroni", "family_size": 4,
                       "primary_metric": "macro_f1"},
        "primary": [
            {"comparison": "H1", "family": "primary", "row_a": "A2", "row_b": "A1",
             "metric": "macro_f1", "observed_delta": 0.08, "ci_low": 0.03, "ci_high": 0.13,
             "raw_p_value": 0.001, "adjusted_p_value": 0.004, "significant": True,
             "effect_direction": "favours A2", "mcnemar_method": "exact binomial"},
            {"comparison": "H2", "family": "primary", "row_a": "A5", "row_b": "A4",
             "metric": "macro_f1", "observed_delta": 0.04, "ci_low": -0.01, "ci_high": 0.09,
             "raw_p_value": 0.20, "adjusted_p_value": 0.60, "significant": False,
             "effect_direction": "favours A5", "mcnemar_method": "chi-square with continuity correction"},
            {"comparison": "H3", "family": "primary", "row_a": "A7", "row_b": "A6",
             "metric": "macro_f1", "observed_delta": 0.02, "ci_low": -0.02, "ci_high": 0.06,
             "raw_p_value": 0.30, "adjusted_p_value": 0.60, "significant": False,
             "effect_direction": "favours A7", "mcnemar_method": "exact binomial"},
            {"comparison": "H4", "family": "primary", "row_a": "A6", "row_b": "A3",
             "metric": "macro_f1", "observed_delta": -0.01, "ci_low": -0.05, "ci_high": 0.03,
             "raw_p_value": 0.70, "adjusted_p_value": 0.70, "significant": False,
             "effect_direction": "favours A3", "mcnemar_method": "exact binomial"},
        ],
        "secondary": [
            {"comparison": "A1_vs_A0", "family": "secondary", "row_a": "A1", "row_b": "A0",
             "metric": "macro_f1", "observed_delta": 0.03, "ci_low": 0.00, "ci_high": 0.06,
             "raw_p_value": None, "adjusted_p_value": None, "significant": None,
             "effect_direction": "favours A1", "mcnemar_method": None},
            {"comparison": "A3_vs_A2", "family": "secondary", "row_a": "A3", "row_b": "A2",
             "metric": "macro_f1", "observed_delta": 0.01, "ci_low": -0.03, "ci_high": 0.05,
             "raw_p_value": None, "adjusted_p_value": None, "significant": None,
             "effect_direction": "favours A3", "mcnemar_method": None},
            {"comparison": "A7_vs_P", "family": "secondary", "row_a": "A7", "row_b": "P",
             "metric": "macro_f1", "observed_delta": -0.02, "ci_low": None, "ci_high": None,
             "raw_p_value": None, "adjusted_p_value": None, "significant": None,
             "effect_direction": "favours P", "mcnemar_method": None,
             "estimation": "metrics-only"},
        ],
        "rows": {"P": {"single_seed": True, "n_seeds": 1, "seed_spread": None},
                 "A8": {"mirrors": "A7", "excluded_from_testing": True}},
    })

    write("step16_internal", {"overall": {"macro_f1": 0.91, "balanced_accuracy": 0.90},
                              "per_class": [{"class_name": "Glioma", "recall_sensitivity": 0.88}]})
    write("step17_external", {"restricted": {"macro_f1": 0.70},
                              "unrestricted": {"macro_f1": 0.62},
                              "restriction_effect_macro_f1": 0.08})
    write("step18_robustness", {"clean_scores": {"proposed": 0.91, "efficientnet_b0": 0.88,
                                                 "vit": 0.85},
                                "mean_drop": {"proposed": 0.05}})
    write("step19_explainability", {"grad_cam": {"mean_deletion_drop": 0.22,
                                                 "mean_insertion_recovery": 0.61},
                                    "mc_dropout": {"separation": 3.1},
                                    "interpretation": "concentrated"})
    write("step20_quantum_advantage", {"performance": {"delta_macro_f1": -0.004},
                                       "verdict": {"headline": "no measurable advantage",
                                                   "statistically_significant": False}})
    write("step08_imbalance", {"selected_strategy": "baseline", "selected_macro_f1": 0.62})
    write("step11_gate_morphology", {"scale_weight_correlation": 0.41})
    write("step06_preprocessing", {"selected_recipe": "clahe",
                                   "ranking": [{"recipe": "clahe", "edge_preservation": 0.94}]})
    write("step14_loss_selection", {"selected_loss": "weighted_ce"})
    return root


@pytest.fixture
def mapping(artefacts, tmp_path):
    """:param artefacts: The synthetic artefact tree.

    :param tmp_path: Per-test directory.
    :return: A configured mapping, not yet run.
    """
    cfg = compose_step22()
    from omegaconf import OmegaConf

    analysis = RQMapping(
        analyze_root=str(artefacts),
        research_questions=OmegaConf.to_container(cfg.analysis.research_questions, resolve=True),
    )
    analysis._output_dir = tmp_path / "out"
    analysis._output_dir.mkdir()
    return analysis


@pytest.fixture
def result(mapping):
    """:param mapping: The configured mapping.

    :return: The computed RQ table.
    """
    return mapping.compute(datamodule=None)


# --------------------------------------------------------------- completeness


def test_all_ten_research_questions_appear_exactly_once(result):
    """A dropped RQ looks like a shorter table, not like a gap."""
    ids = [entry["rq_id"] for entry in result["research_questions"]]

    assert ids == list(RQ_IDS)
    assert len(ids) == len(set(ids)) == 10


def test_the_ordering_is_deterministic(mapping, artefacts, tmp_path):
    """Same artefacts, same table - including order."""
    from omegaconf import OmegaConf

    cfg = compose_step22()
    orders = []
    for run in range(3):
        analysis = RQMapping(
            analyze_root=str(artefacts),
            research_questions=OmegaConf.to_container(cfg.analysis.research_questions, resolve=True),
        )
        analysis._output_dir = tmp_path / f"det{run}"
        analysis._output_dir.mkdir()
        orders.append([e["rq_id"] for e in analysis.compute(datamodule=None)["research_questions"]])

    assert orders[0] == orders[1] == orders[2] == list(RQ_IDS)


def test_rq_ids_are_rq1_through_rq10_in_numeric_order():
    """RQ10 must not sort between RQ1 and RQ2, which a lexical sort would do."""
    assert RQ_IDS == tuple(f"RQ{n}" for n in range(1, 11))


# ------------------------------------------------------ fidelity to the specification


def test_the_rq_questions_come_from_the_specification():
    """Retyped from memory, an RQ slowly stops meaning what was asked.

    The config's text is compared against the specification file itself, so a drifting
    paraphrase fails here rather than in the write-up.
    """
    spec = spec_rq_table()
    configured = {rq.id: rq.question for rq in compose_step22().analysis.research_questions}

    assert set(configured) == set(spec) == set(RQ_IDS)
    for rq_id, question in configured.items():
        assert question == spec[rq_id], f"{rq_id} drifted from the specification"


def test_the_specification_experiment_and_evidence_columns_are_carried(result):
    """Step 22 is a mapping; the thing being mapped from belongs in the output."""
    for entry in result["research_questions"]:
        assert entry["spec_experiment"]
        assert entry["spec_evidence"]


# --------------------------------------------------------- the statistical boundary


def test_only_the_four_primary_hypotheses_carry_formal_evidence(result):
    """Anything else presented as formal would be an uncorrected claim in a quoted table."""
    formal = {
        item["comparison_or_analysis"]
        for entry in result["research_questions"]
        for item in entry["evidence"]
        if item["statistical_status"] == "formal"
    }

    assert formal <= {"H1", "H2", "H3", "H4"}


def test_formal_evidence_references_step23(result):
    """The correction lives in Step 23; evidence claiming formality must come from it."""
    for entry in result["research_questions"]:
        for item in entry["evidence"]:
            if item["statistical_status"] == "formal":
                assert item["source_stage"] == "step23_statistics"
                assert item["provenance"]["comparison"] in ("H1", "H2", "H3", "H4")
                assert "adjusted_p_value" in item["value"]


def test_secondary_comparisons_stay_descriptive(result):
    """A1-vs-A0 and A3-vs-A2 are not hypotheses and must not read as though they were."""
    for entry in result["research_questions"]:
        for item in entry["evidence"]:
            if item["comparison_or_analysis"] in ("A1_vs_A0", "A3_vs_A2", "A7_vs_P"):
                assert item["statistical_status"] == "descriptive"
                assert item["value"].get("significant") is None


def test_a7_vs_p_is_descriptive_and_flags_the_single_seed(result):
    """P is single-seed by Step 16's design, so no seed-level claim is supported."""
    items = [
        item
        for entry in result["research_questions"]
        for item in entry["evidence"]
        if item["comparison_or_analysis"] == "A7_vs_P"
    ]

    assert items, "A7-vs-P should appear as RQ2 end-to-end evidence"
    for item in items:
        assert item["statistical_status"] == "descriptive"
        assert "single-seed" in item["interpretation"].lower()


def test_p_single_seed_status_is_carried_into_the_map(result):
    """:param result: The computed RQ table."""
    assert result["rows"]["P"]["single_seed"] is True
    assert result["rows"]["P"]["seed_spread"] is None


def test_a_binding_that_claims_formality_for_a_secondary_comparison_is_refused(
    mapping, artefacts
):
    """The guard that stops a descriptive comparison being promoted by editing a config."""
    for entry in mapping.research_questions:
        if entry["id"] == "RQ2":
            entry["evidence"].append(
                {"kind": "step23_comparison", "ref": "A7_vs_P", "role": "primary"}
            )

    with pytest.raises(ValueError, match="not a primary"):
        mapping.compute(datamodule=None)


def test_an_unknown_comparison_is_refused(mapping):
    """A typo must not silently become an absent evidence row."""
    for entry in mapping.research_questions:
        if entry["id"] == "RQ2":
            entry["evidence"].append(
                {"kind": "step23_comparison", "ref": "H9", "role": "secondary"}
            )

    with pytest.raises(ValueError, match="H9"):
        mapping.compute(datamodule=None)


# --------------------------------------------------------------------- row A8


def test_a8_is_not_given_an_independent_classification_delta(result):
    """Its metrics are A7's by construction; a delta would be exactly zero and meaningless."""
    rq10 = next(e for e in result["research_questions"] if e["rq_id"] == "RQ10")
    a8 = next(item for item in rq10["evidence"] if item["comparison_or_analysis"] == "A8")

    assert a8["statistical_status"] == "identical_by_construction"
    assert "A7" in a8["interpretation"]
    assert a8["value"].get("mirrors") == "A7"


def test_no_comparison_treats_a8_as_a_separate_model(result):
    """:param result: The computed RQ table."""
    formal = [
        item
        for entry in result["research_questions"]
        for item in entry["evidence"]
        if item["statistical_status"] == "formal" and "A8" in str(item["comparison_or_analysis"])
    ]

    assert formal == []


# ---------------------------------------------------------------------- RQ5


def test_rq5_is_metadata_limited_when_subgroup_annotations_are_absent(result):
    """The dataset ships no tumour-size annotations, so that evidence does not exist."""
    rq5 = next(e for e in result["research_questions"] if e["rq_id"] == "RQ5")

    assert rq5["status"] == "partially_supported"
    assert any("metadata" in limitation.lower() for limitation in rq5["limitations"])
    assert any("tumour size" in limitation.lower() or "tumor size" in limitation.lower()
               for limitation in rq5["limitations"])


def test_rq5_reports_the_class_wise_evidence_it_does_have(result):
    """Per-class metrics exist even though subgroup metadata does not; both are stated."""
    rq5 = next(e for e in result["research_questions"] if e["rq_id"] == "RQ5")
    sources = {item["source_stage"] for item in rq5["evidence"]}

    assert "step21_ablation" in sources or "step16_internal" in sources


def test_rq5_does_not_claim_subgroup_evidence_it_lacks(result):
    """:param result: The computed RQ table."""
    rq5 = next(e for e in result["research_questions"] if e["rq_id"] == "RQ5")
    subgroup = [
        item for item in rq5["evidence"]
        if item["comparison_or_analysis"] == "tumour_size_subgroups"
    ]

    assert subgroup, "the absent evidence should be listed, not omitted"
    assert subgroup[0]["statistical_status"] == "evidence_missing"
    assert subgroup[0]["value"] is None


# ------------------------------------------------- the per-RQ evidence bindings


@pytest.mark.parametrize(
    "rq_id,expected_stage",
    [
        ("RQ1", "step17_external"),
        ("RQ2", "step18_robustness"),
        ("RQ3", "step19_explainability"),
        ("RQ6", "step08_imbalance"),
        ("RQ7", "step17_external"),
        ("RQ8", "step20_quantum_advantage"),
        ("RQ9", "step19_explainability"),
    ],
)
def test_each_rq_references_the_stage_the_mapping_assigns_it(result, rq_id, expected_stage):
    """:param result: The computed RQ table.

    :param rq_id: Research question.
    :param expected_stage: Stage its evidence must include.
    """
    entry = next(e for e in result["research_questions"] if e["rq_id"] == rq_id)
    stages = {item["source_stage"] for item in entry["evidence"]}

    assert expected_stage in stages


def test_rq1_distinguishes_shipped_model_evidence_from_ablation_evidence(result):
    """P is the study's claim; the A-rows are its anatomy. Conflating them misreads both."""
    rq1 = next(e for e in result["research_questions"] if e["rq_id"] == "RQ1")
    stages = {item["source_stage"] for item in rq1["evidence"]}

    assert "step16_internal" in stages
    assert "step17_external" in stages
    assert any("shipped" in item["interpretation"].lower() for item in rq1["evidence"])


def test_rq10_covers_the_whole_ladder(result):
    """"Full ablation study" - every row, not a selection of them."""
    rq10 = next(e for e in result["research_questions"] if e["rq_id"] == "RQ10")
    rows = {item["comparison_or_analysis"] for item in rq10["evidence"]}

    assert {"A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"} <= rows


def test_rq4_labels_which_of_its_comparisons_is_formal(result):
    """A5-vs-A4 is H2 and corrected; A3-vs-A1 is not a registered hypothesis."""
    rq4 = next(e for e in result["research_questions"] if e["rq_id"] == "RQ4")
    statuses = {item["comparison_or_analysis"]: item["statistical_status"]
                for item in rq4["evidence"]}

    assert statuses.get("H2") == "formal"
    assert all(v != "formal" for k, v in statuses.items() if k != "H2")


def test_rq6_and_rq8_carry_their_holm_family_members(result):
    """H3 belongs to RQ6 and H4 to RQ8; both are corrected members."""
    rq6 = next(e for e in result["research_questions"] if e["rq_id"] == "RQ6")
    rq8 = next(e for e in result["research_questions"] if e["rq_id"] == "RQ8")

    assert "H3" in {i["comparison_or_analysis"] for i in rq6["evidence"]}
    assert "H4" in {i["comparison_or_analysis"] for i in rq8["evidence"]}


# ------------------------------------------------------------------- statuses


def test_a_significant_primary_hypothesis_makes_its_rq_supported(result):
    """H1 is significant in the fixture, so RQ2 is supported."""
    rq2 = next(e for e in result["research_questions"] if e["rq_id"] == "RQ2")

    assert rq2["status"] == "supported"


def test_an_rq_whose_formal_test_is_not_significant_is_not_called_supported(result):
    """H4 is not significant, so RQ8 must not read as though the question were settled."""
    rq8 = next(e for e in result["research_questions"] if e["rq_id"] == "RQ8")

    assert rq8["status"] in ("descriptive_only", "partially_supported")
    assert rq8["status"] != "supported"


def test_every_status_is_from_the_declared_vocabulary(result):
    """:param result: The computed RQ table."""
    allowed = {"supported", "partially_supported", "descriptive_only", "not_assessed",
               "evidence_missing"}

    assert {e["status"] for e in result["research_questions"]} <= allowed


def test_step22_reaches_no_conclusion_step23_has_not_reached(result):
    """It maps evidence; it does not adjudicate.

    A conclusion may only restate Step 23's own verdict on a formal comparison, and only
    for comparisons that appear as formal evidence for that question.
    """
    for entry in result["research_questions"]:
        conclusion = entry["conclusion"]
        formal = [i for i in entry["evidence"] if i["statistical_status"] == "formal"]

        if not formal:
            assert conclusion is None, f"{entry['rq_id']} concluded without a formal test"
            continue

        assert isinstance(conclusion, str)
        for item in formal:
            assert item["comparison_or_analysis"] in conclusion
            significant = (item["value"] or {}).get("significant")
            expected = "not significant after Holm correction" if not significant else (
                "significant after Holm correction"
            )
            assert expected in conclusion


# ------------------------------------------------------- missing artefacts


def test_a_missing_stage_summary_is_reported_not_invented(artefacts, tmp_path):
    """:param artefacts: The synthetic artefact tree.

    :param tmp_path: Per-test directory.
    """
    from omegaconf import OmegaConf

    (artefacts / "step20_quantum_advantage" / "step20_quantum_advantage_summary.json").unlink()

    cfg = compose_step22()
    analysis = RQMapping(
        analyze_root=str(artefacts),
        research_questions=OmegaConf.to_container(cfg.analysis.research_questions, resolve=True),
    )
    analysis._output_dir = tmp_path / "missing"
    analysis._output_dir.mkdir()
    summary = analysis.compute(datamodule=None)

    rq8 = next(e for e in summary["research_questions"] if e["rq_id"] == "RQ8")
    absent = [i for i in rq8["evidence"] if i["source_stage"] == "step20_quantum_advantage"]

    assert absent
    assert absent[0]["statistical_status"] == "evidence_missing"
    assert absent[0]["value"] is None
    assert "step20_quantum_advantage" in summary["missing_sources"]


def test_a_missing_source_is_never_substituted(artefacts, tmp_path):
    """A neighbouring experiment is not the same experiment."""
    from omegaconf import OmegaConf

    (artefacts / "step17_external" / "step17_external_summary.json").unlink()

    cfg = compose_step22()
    analysis = RQMapping(
        analyze_root=str(artefacts),
        research_questions=OmegaConf.to_container(cfg.analysis.research_questions, resolve=True),
    )
    analysis._output_dir = tmp_path / "sub"
    analysis._output_dir.mkdir()
    summary = analysis.compute(datamodule=None)

    rq7 = next(e for e in summary["research_questions"] if e["rq_id"] == "RQ7")

    assert rq7["status"] in ("evidence_missing", "not_assessed")
    assert all(i["value"] is None for i in rq7["evidence"])


def test_missing_step21_and_step23_fail_clearly(tmp_path):
    """Those two are what Step 22 exists to map; without them there is nothing to do."""
    from omegaconf import OmegaConf

    cfg = compose_step22()
    analysis = RQMapping(
        analyze_root=str(tmp_path / "empty"),
        research_questions=OmegaConf.to_container(cfg.analysis.research_questions, resolve=True),
    )
    analysis._output_dir = tmp_path / "out"
    analysis._output_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Step 21"):
        analysis.compute(datamodule=None)


# ---------------------------------------------------------------- provenance


def test_every_evidence_item_carries_provenance(result):
    """A number without a traceable source cannot be checked by a reader."""
    for entry in result["research_questions"]:
        for item in entry["evidence"]:
            provenance = item["provenance"]
            assert provenance["source_stage"] == item["source_stage"]
            assert provenance["rq"] == entry["rq_id"]
            assert "source_path" in provenance
            assert item["source_path"] is not None or item["statistical_status"] == "evidence_missing"


def test_the_evidence_schema_is_stable(result):
    """:param result: The computed RQ table."""
    for entry in result["research_questions"]:
        for item in entry["evidence"]:
            assert set(EVIDENCE_ITEM_FIELDS) <= set(item)


def test_the_flat_table_is_written_with_pinned_columns(mapping, result):
    """:param mapping: The configured mapping.

    :param result: The computed RQ table.
    """
    import pandas as pd

    frame = pd.read_csv(mapping.output_dir / "step22_rq_evidence.csv")

    assert list(frame.columns) == list(EVIDENCE_COLUMNS)
    assert set(frame["rq_id"]) == set(RQ_IDS)


# ------------------------------------------------------------------ hygiene


def test_no_numeric_metric_is_hard_coded_in_the_mapping():
    """Every number must come from an artefact, not from the module.

    A hard-coded value would survive the artefacts changing underneath it and would be
    invisible in the output - the exact failure Step 22 exists to prevent.
    """
    tree = ast.parse(Path("src/analysis/rq_mapping.py").read_text(encoding="utf-8"))
    numbers = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ]

    # Only structural constants are permissible: indices, counts, and the RQ range bound.
    assert all(float(value).is_integer() and 0 <= value <= 11 for value in numbers), (
        f"non-structural numeric literals in rq_mapping.py: "
        f"{[v for v in numbers if not (float(v).is_integer() and 0 <= v <= 11)]}"
    )


def test_the_mapping_reads_no_training_artefacts():
    """Step 21 and 23 summaries are the source; metrics.csv never is."""
    tree = ast.parse(Path("src/analysis/rq_mapping.py").read_text(encoding="utf-8"))
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }

    assert "read_csv" not in called
    assert not {"load_module", "find_checkpoint", "Trainer", "fit", "predict"} & called


def test_the_mapping_writes_only_its_own_outputs(mapping, result, artefacts):
    """Read-only with respect to every scientific artefact it consumes."""
    before = {
        path: path.read_bytes()
        for path in artefacts.rglob("*.json")
    }
    mapping.compute(datamodule=None)

    for path, payload in before.items():
        assert path.read_bytes() == payload, f"{path} was modified"


def test_no_validated_artefact_was_touched():
    """:return: None."""
    import subprocess

    changed = subprocess.run(
        ["git", "status", "--short", "--", "configs/protocol",
         "configs/experiment/step15_final_protocol.yaml", "data/splits"],
        capture_output=True, text=True,
    ).stdout.strip()

    assert changed == ""


# ------------------------------------------------------------ Hydra composition


def test_the_step22_stage_composes():
    """:return: None."""
    cfg = compose_step22()

    assert cfg.analysis._target_ == "src.analysis.rq_mapping.RQMapping"
    assert cfg.analysis.name == "step22_rq_mapping"
    assert len(cfg.analysis.research_questions) == 10


def test_the_config_declares_every_rq_with_its_bindings():
    """The mapping is a declaration; it belongs in a file a reader can check."""
    questions = compose_step22().analysis.research_questions

    assert [rq.id for rq in questions] == list(RQ_IDS)
    for rq in questions:
        assert rq.question
        assert rq.spec_experiment
        assert rq.spec_evidence
        assert len(rq.evidence) >= 1
