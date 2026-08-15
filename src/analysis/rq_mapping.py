"""Step 22: research question to experiment mapping.

    "|RQ1: Improve multiclass classification|Compare proposed model with all baselines on
     internal and external test sets.|Macro-F1, balanced accuracy, class-wise recall| ..."

This stage answers one question - *for each research question, what existing experiment
supplies evidence?* - and deliberately nothing else. It computes no metric, runs no test,
and reaches no verdict Steps 21 and 23 have not already reached. Every value in its table
is read from an artefact on disk and carries the path it came from, which is what makes the
table auditable rather than merely tidy.

Three rules follow from that, and each is enforced rather than documented.

**Descriptive evidence stays descriptive.** Step 23 corrects a family of exactly four
hypotheses. A mapping that presents A7-vs-P or the A0-A7 ladder beside them without
distinction turns four corrected tests into eight uncorrected ones - in a table a reader
will quote directly. A binding may claim ``role: primary`` only for a comparison Step 23
itself classifies as a primary family member; anything else is refused.

**Absent evidence is declared, not filled in.** RQ5 asks for performance by tumour size
"if metadata or masks are available". The dataset ships neither, so the subgroup half of
that question cannot be answered - and the honest output says so, with the alternative
being a proxy nobody asked for. Absent evidence is a row in the table with a null value and
a stated reason, never a silent omission.

**The specification's words are the specification's.** The RQ text lives in
``configs/analysis/step22_rq_mapping.yaml``, copied from the Step 22 table, and a test
parses that table out of the specification file and compares the two. A paraphrase that
drifts fails the suite rather than the write-up.

No number is written in this module. The one thing it must never do is carry a value of its
own, because such a value would survive the artefacts changing underneath it and would be
invisible in the output.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.analysis.base import Analysis
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

#: The specification's research questions, in the order it lists them. Numeric rather than
#: lexical: a lexical sort would place RQ10 between RQ1 and RQ2.
RQ_IDS: Tuple[str, ...] = tuple(f"RQ{number}" for number in range(1, 11))

#: The two artefacts Step 22 exists to map. Without them there is nothing to do.
REQUIRED_STAGES: Tuple[str, ...] = ("step21_ablation", "step23_statistics")

#: Status vocabulary. Deliberately small - a status is a claim about evidence, and inventing
#: new ones is how a table stops being comparable across questions.
STATUSES: Tuple[str, ...] = (
    "supported",
    "partially_supported",
    "descriptive_only",
    "not_assessed",
    "evidence_missing",
)

#: How strong a piece of evidence is, in the only sense Step 22 is entitled to assert.
FORMAL = "formal"
DESCRIPTIVE = "descriptive"
MISSING = "evidence_missing"
BY_CONSTRUCTION = "identical_by_construction"

#: Fields every evidence item carries. Distinct from the flat table's columns, which also
#: carry the question-level fields each item is nested under.
EVIDENCE_ITEM_FIELDS: Tuple[str, ...] = (
    "source_stage",
    "source_path",
    "comparison_or_analysis",
    "metric",
    "value",
    "statistical_status",
    "role",
    "interpretation",
    "provenance",
)

#: Flat-table columns, pinned so the schema does not follow dict iteration order.
EVIDENCE_COLUMNS: Tuple[str, ...] = (
    "rq_id",
    "rq_question",
    "status",
    "source_stage",
    "source_path",
    "comparison_or_analysis",
    "metric",
    "statistical_status",
    "role",
    "interpretation",
)


class RQMapping(Analysis):
    """Map each research question onto the artefacts that answer it.

    :param name: Analysis identifier.
    :param analyze_root: Directory holding one subdirectory per analysis stage, each with
        its ``<stage>_summary.json``.
    :param research_questions: The RQ declarations, including their evidence bindings.
        Supplied from config so the mapping is a readable declaration rather than code.
    :param strict: Fail when any bound artefact is absent, rather than recording it as
        missing evidence and continuing.
    """

    def __init__(
        self,
        name: str = "step22_rq_mapping",
        analyze_root: Optional[str] = None,
        research_questions: Optional[Sequence[Any]] = None,
        strict: bool = False,
    ) -> None:
        super().__init__(name=name)
        self.analyze_root = analyze_root
        self.research_questions = [dict(entry) for entry in (research_questions or [])]
        self.strict = strict
        self._summaries: Dict[str, Optional[Dict[str, Any]]] = {}
        self._missing: List[str] = []

    # -------------------------------------------------------------------- inputs

    def _summary_path(self, stage: str) -> Path:
        """:param stage: Stage identifier.

        :return: Where that stage's summary lives.
        """
        root = Path(self.analyze_root or "logs/analyze/runs")
        return root / stage / f"{stage}_summary.json"

    def _summary(self, stage: str) -> Optional[Dict[str, Any]]:
        """Read one stage's summary, recording it as missing if absent.

        :param stage: Stage identifier.
        :return: The parsed summary, or ``None``.
        """
        if stage in self._summaries:
            return self._summaries[stage]

        path = self._summary_path(stage)
        if not path.is_file():
            log.warning(f"{stage}: no summary at {path}; evidence from it will be marked missing.")
            if stage not in self._missing:
                self._missing.append(stage)
            self._summaries[stage] = None
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{stage}: summary at {path} is not valid JSON: {error}") from error

        self._summaries[stage] = payload
        return payload

    def _require(self, stage: str) -> Dict[str, Any]:
        """:param stage: Stage identifier.

        :return: That stage's summary.
        :raises FileNotFoundError: If it is absent.
        """
        payload = self._summary(stage)
        if payload is None:
            label = "Step 21 ablation" if stage.startswith("step21") else "Step 23 statistics"
            raise FileNotFoundError(
                f"Step 22 maps the {label} results; {self._summary_path(stage)} does not "
                f"exist. Run analysis={stage} first."
            )
        return payload

    # ------------------------------------------------------------------- compute

    def compute(self, datamodule: Any = None) -> Dict[str, Any]:
        """Build the RQ evidence table.

        :param datamodule: Unused; this stage reads artefacts, not data.
        :return: The mapping.
        :raises FileNotFoundError: If Step 21 or Step 23 has not run.
        :raises ValueError: If a binding is malformed or claims formality it cannot have.
        """
        self._summaries.clear()
        self._missing.clear()

        for stage in REQUIRED_STAGES:
            self._require(stage)

        declared = {entry["id"]: entry for entry in self.research_questions}
        absent = [rq for rq in RQ_IDS if rq not in declared]
        if absent:
            raise ValueError(
                f"Step 22 must map every research question; {absent} are not declared. "
                "An omitted RQ looks like a shorter table rather than like a gap."
            )

        # Iterated over RQ_IDS, not over the config's order, so the table cannot be
        # reordered by editing the config and cannot follow dict iteration.
        questions = [self._map_question(declared[rq_id]) for rq_id in RQ_IDS]

        table = self._flat_table(questions)
        self.save_table(table, "step22_rq_evidence.csv")

        summary = {
            "research_questions": questions,
            "rows": self._row_notes(),
            "missing_sources": list(self._missing),
            "statuses": {entry["rq_id"]: entry["status"] for entry in questions},
            "notes": self._notes(),
        }
        self._log(summary)
        return summary

    # -------------------------------------------------------------- one question

    def _map_question(self, declaration: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve one research question's bindings.

        :param declaration: The RQ's config entry.
        :return: The mapped record.
        """
        rq_id = declaration["id"]
        evidence: List[Dict[str, Any]] = []
        for binding in declaration.get("evidence", []):
            resolved = self._resolve(dict(binding), rq_id)
            # step21_ladder expands to one item per rung; every other kind yields one item.
            evidence.extend(resolved if isinstance(resolved, list) else [resolved])

        primary = [item for item in evidence if item["role"] == "primary"]
        secondary = [item for item in evidence if item["role"] != "primary"]

        return {
            "rq_id": rq_id,
            "rq_question": declaration["question"],
            "spec_experiment": declaration.get("spec_experiment"),
            "spec_evidence": declaration.get("spec_evidence"),
            "status": self._status(evidence),
            "evidence": evidence,
            "primary_evidence": [item["comparison_or_analysis"] for item in primary],
            "secondary_evidence": [item["comparison_or_analysis"] for item in secondary],
            "conclusion": self._conclusion(evidence),
            "limitations": list(declaration.get("limitations", [])),
            "provenance": {
                "source_stages": sorted({item["source_stage"] for item in evidence}),
                "analyze_root": str(self.analyze_root),
                "note": (
                    "Every value is read from the named artefact. Step 22 computes no "
                    "metric and reaches no conclusion Steps 21 and 23 have not reached."
                ),
            },
        }

    def _resolve(self, binding: Dict[str, Any], rq_id: str) -> Dict[str, Any]:
        """Turn one binding into an evidence item.

        :param binding: The binding declaration.
        :param rq_id: The research question it belongs to.
        :return: The evidence item.
        :raises ValueError: If the binding is malformed or claims unearned formality.
        """
        kind = binding.get("kind")
        resolver = {
            "step23_comparison": self._from_step23,
            "step21_row": self._from_step21_row,
            "step21_ladder": self._from_step21_ladder,
            "stage_summary": self._from_stage_summary,
            "absent": self._from_absent,
        }.get(kind)

        if resolver is None:
            raise ValueError(
                f"{rq_id}: unknown evidence kind {kind!r}. Expected one of "
                "step23_comparison, step21_row, step21_ladder, stage_summary, absent."
            )
        return resolver(binding, rq_id)

    # ------------------------------------------------------------- the resolvers

    def _from_step23(self, binding: Dict[str, Any], rq_id: str) -> Dict[str, Any]:
        """Evidence from the Step 23 statistical report.

        Formality is taken from Step 23's own classification, never from the binding: a
        config that asked for ``role: primary`` on a descriptive comparison would otherwise
        smuggle an uncorrected claim into the family.

        :param binding: The binding declaration.
        :param rq_id: The research question.
        :return: The evidence item.
        :raises ValueError: If the comparison is unknown, or is claimed as primary when
            Step 23 classifies it otherwise.
        """
        stage = "step23_statistics"
        report = self._summary(stage)
        reference = binding.get("ref")

        if report is None:
            return self._item(binding, rq_id, stage, reference, None, MISSING,
                              metric=None, path=None)

        records = {
            record["comparison"]: record
            for record in [*report.get("primary", []), *report.get("secondary", [])]
        }
        if reference not in records:
            raise ValueError(
                f"{rq_id}: Step 23 has no comparison {reference!r}. Known comparisons: "
                f"{sorted(records)}."
            )

        record = records[reference]
        is_primary = record.get("family") == "primary"

        if binding.get("role") == "primary" and not is_primary:
            raise ValueError(
                f"{rq_id}: {reference!r} is not a primary hypothesis. The binding declares it as "
                f"primary evidence, but Step 23 classifies it as {record.get('family')!r}. "
                "Only the pre-registered "
                "Holm-corrected family may carry formal significance evidence; presenting a "
                "descriptive comparison as a hypothesis would leave it uncorrected."
            )

        value = {
            key: record.get(key)
            for key in (
                "observed_delta", "ci_low", "ci_high", "raw_p_value", "adjusted_p_value",
                "significant", "effect_direction", "mcnemar_method", "row_a", "row_b",
            )
        }
        if not is_primary:
            # Descriptive comparisons carry no verdict; None means "no claim", not "no
            # effect", and it must survive into the mapping unchanged.
            value["significant"] = record.get("significant")

        interpretation = binding.get("interpretation", "")
        if reference == "A7_vs_P":
            interpretation = (
                f"{interpretation} Row P is single-seed, so this carries no seed-level "
                "claim."
            ).strip()

        return self._item(
            binding, rq_id, stage, reference, value,
            FORMAL if is_primary else DESCRIPTIVE,
            metric=record.get("metric"),
            path=self._summary_path(stage),
            interpretation=interpretation,
        )

    def _from_step21_row(self, binding: Dict[str, Any], rq_id: str) -> Dict[str, Any]:
        """Evidence from one ablation row's metrics.

        :param binding: The binding declaration.
        :param rq_id: The research question.
        :return: The evidence item.
        :raises ValueError: If the row is unknown.
        """
        stage = "step21_ablation"
        ablation = self._summary(stage)
        reference = binding.get("ref")

        if ablation is None:
            return self._item(binding, rq_id, stage, reference, None, MISSING,
                              metric=None, path=None)

        rows = ablation.get("rows", {})
        if reference not in rows:
            raise ValueError(
                f"{rq_id}: Step 21 has no row {reference!r}. Known rows: {sorted(rows)}."
            )

        record = rows[reference]
        value = {
            "across_seeds": record.get("across_seeds"),
            "recipe": record.get("recipe"),
            "loss": record.get("loss"),
            "missing_seeds": record.get("missing_seeds"),
        }
        status = DESCRIPTIVE
        interpretation = binding.get("interpretation", "")

        if record.get("mirrors"):
            # A8: explanations change no weights, so there is no delta of its own to report.
            value["mirrors"] = record["mirrors"]
            status = BY_CONSTRUCTION
            interpretation = (
                f"{interpretation} Classification metrics are identical to "
                f"{record['mirrors']}'s by construction."
            ).strip()

        return self._item(
            binding, rq_id, stage, reference, value, status,
            metric=None, path=self._summary_path(stage), interpretation=interpretation,
        )

    def _from_step21_ladder(self, binding: Dict[str, Any], rq_id: str) -> Dict[str, Any]:
        """Every ablation row at once, for the component-contribution question.

        Returned as one item per row rather than a single opaque blob, so the flat table
        shows the ladder and a missing rung is visible.

        :param binding: The binding declaration.
        :param rq_id: The research question.
        :return: One item per ablation row, in the order Step 21 recorded them.
        """
        stage = "step21_ablation"
        ablation = self._summary(stage)

        if ablation is None:
            return self._item(binding, rq_id, stage, "ladder", None, MISSING,
                              metric=None, path=None)

        items = []
        for row_id, record in ablation.get("rows", {}).items():
            status = BY_CONSTRUCTION if record.get("mirrors") else DESCRIPTIVE
            interpretation = binding.get("interpretation", "")
            if record.get("mirrors"):
                interpretation = (
                    f"{interpretation} Classification metrics are identical to "
                    f"{record['mirrors']}'s by construction, so this rung contributes no "
                    "delta of its own."
                ).strip()
            items.append(
                self._item(
                    binding, rq_id, stage, row_id,
                    {
                        "across_seeds": record.get("across_seeds"),
                        "mirrors": record.get("mirrors"),
                        "missing_seeds": record.get("missing_seeds"),
                    },
                    status, metric=None, path=self._summary_path(stage),
                    interpretation=interpretation,
                )
            )
        return items

    def _from_stage_summary(self, binding: Dict[str, Any], rq_id: str) -> Dict[str, Any]:
        """Evidence from an earlier analysis stage's summary.

        :param binding: The binding declaration.
        :param rq_id: The research question.
        :return: The evidence item.
        :raises FileNotFoundError: In strict mode, when the summary is absent.
        """
        stage = binding.get("ref")
        payload = self._summary(stage)

        if payload is None:
            if self.strict:
                raise FileNotFoundError(
                    f"{rq_id}: {stage} has no summary at {self._summary_path(stage)}."
                )
            return self._item(binding, rq_id, stage, stage, None, MISSING,
                              metric=None, path=None)

        key = binding.get("key")
        value = payload.get(key) if key else payload

        if key and value is None:
            return self._item(
                binding, rq_id, stage, stage, None, MISSING, metric=key,
                path=self._summary_path(stage),
                interpretation=(
                    f"{binding.get('interpretation', '')} The summary exists but carries no "
                    f"{key!r} field."
                ).strip(),
            )

        return self._item(binding, rq_id, stage, stage, value, DESCRIPTIVE,
                          metric=key, path=self._summary_path(stage))

    def _from_absent(self, binding: Dict[str, Any], rq_id: str) -> Dict[str, Any]:
        """Evidence the study cannot supply, declared rather than omitted.

        RQ5's tumour-size subgroups and RQ3's segmentation masks are the live cases: the
        specification conditions both on data the dataset does not ship. Listing them with
        a null value is what stops a reader assuming they were simply overlooked - and
        what stops a proxy being substituted for them.

        :param binding: The binding declaration.
        :param rq_id: The research question.
        :return: The evidence item.
        """
        return self._item(
            binding, rq_id, "not_available", binding.get("ref"), None, MISSING,
            metric=None, path=None,
        )

    # ---------------------------------------------------------------- assembly

    def _item(
        self,
        binding: Dict[str, Any],
        rq_id: str,
        stage: str,
        reference: Optional[str],
        value: Any,
        statistical_status: str,
        metric: Optional[str],
        path: Optional[Path],
        interpretation: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assemble one evidence item with its provenance.

        :param binding: The binding declaration.
        :param rq_id: The research question.
        :param stage: Source stage.
        :param reference: Comparison, row or analysis identifier.
        :param value: The evidence itself, or ``None`` when absent.
        :param statistical_status: How strong the evidence is.
        :param metric: Metric name, where one applies.
        :param path: Source file.
        :param interpretation: Override for the binding's own text.
        :return: The evidence item.
        """
        # A binding may only *request* a role; formality is decided by the resolver, so a
        # missing or descriptive source cannot end up labelled primary.
        role = binding.get("role", "secondary")
        if statistical_status != FORMAL:
            role = "secondary" if role == "primary" else role

        return {
            "source_stage": stage,
            "source_path": str(path) if path is not None else None,
            "comparison_or_analysis": reference,
            "metric": metric,
            "value": value,
            "statistical_status": statistical_status,
            "role": role,
            "interpretation": interpretation
            if interpretation is not None
            else binding.get("interpretation", ""),
            "provenance": {
                "source_stage": stage,
                "source_path": str(path) if path is not None else None,
                "comparison": reference,
                "rq": rq_id,
                "kind": binding.get("kind"),
            },
        }

    def _status(self, evidence: Sequence[Dict[str, Any]]) -> str:
        """Derive the RQ's status from what actually resolved.

        Deliberately conservative. ``supported`` requires a Holm-corrected hypothesis that
        Step 23 marked significant - Step 22 does not decide anything on its own, and an
        RQ whose formal test came back null must not read as though the question were
        settled.

        :param evidence: The resolved evidence items.
        :return: One of :data:`STATUSES`.
        """
        if not evidence:
            return "not_assessed"

        resolved = [item for item in evidence if item["statistical_status"] != MISSING]
        missing = [item for item in evidence if item["statistical_status"] == MISSING]

        if not resolved:
            return "evidence_missing"

        formal = [item for item in resolved if item["statistical_status"] == FORMAL]
        significant = [item for item in formal if (item["value"] or {}).get("significant")]

        if significant and not missing:
            return "supported"
        if significant:
            return "partially_supported"
        if missing:
            return "partially_supported"
        return "descriptive_only"

    def _conclusion(self, evidence: Sequence[Dict[str, Any]]) -> Optional[str]:
        """State only what Step 23 already established.

        :param evidence: The resolved evidence items.
        :return: A sentence, or ``None`` when there is no formal finding to report.
        """
        formal = [item for item in evidence if item["statistical_status"] == FORMAL]
        if not formal:
            return None

        parts = []
        for item in formal:
            value = item["value"] or {}
            verdict = (
                "significant after Holm correction"
                if value.get("significant")
                else "not significant after Holm correction"
            )
            parts.append(f"{item['comparison_or_analysis']} ({value.get('effect_direction')}): {verdict}")
        return "; ".join(parts)

    def _row_notes(self) -> Dict[str, Any]:
        """Carry Step 23's per-row caveats into the mapping.

        :return: Row records, or an empty mapping if Step 23 recorded none.
        """
        report = self._summary("step23_statistics") or {}
        return report.get("rows", {})

    def _flat_table(self, questions: Sequence[Dict[str, Any]]) -> pd.DataFrame:
        """One line per evidence item, with the schema pinned.

        :param questions: The mapped research questions.
        :return: The table, columns in :data:`EVIDENCE_COLUMNS` order.
        """
        records = [
            {
                "rq_id": entry["rq_id"],
                "rq_question": entry["rq_question"],
                "status": entry["status"],
                "source_stage": item["source_stage"],
                "source_path": item["source_path"],
                "comparison_or_analysis": item["comparison_or_analysis"],
                "metric": item["metric"],
                "statistical_status": item["statistical_status"],
                "role": item["role"],
                "interpretation": item["interpretation"],
            }
            for entry in questions
            for item in entry["evidence"]
        ]
        return pd.DataFrame(records, columns=list(EVIDENCE_COLUMNS))

    def _notes(self) -> Dict[str, str]:
        """:return: Standing caveats a reader needs before quoting the table."""
        return {
            "scope": (
                "Step 22 maps evidence; it does not produce it. No metric is computed here, "
                "no test is run, and no conclusion is drawn that Steps 21 and 23 have not "
                "already drawn."
            ),
            "formality": (
                "Only the four pre-registered comparisons Step 23 Holm-corrects carry "
                "formal significance evidence. Everything else - including A7 vs P and the "
                "full ladder - is descriptive, and is labelled so."
            ),
            "absent_evidence": (
                "Evidence the dataset cannot supply is listed with a null value and a "
                "reason rather than omitted or approximated. RQ5's tumour-size subgroups "
                "and RQ3's segmentation masks are the cases."
            ),
            "row_a8": (
                "A8's classification metrics are A7's by construction. It contributes "
                "explainability and uncertainty evidence, not a performance delta."
            ),
        }

    @staticmethod
    def _log(summary: Dict[str, Any]) -> None:
        """:param summary: The computed mapping."""
        log.info("=== Step 22: research question mapping ===")
        for entry in summary["research_questions"]:
            formal = sum(
                1 for item in entry["evidence"] if item["statistical_status"] == FORMAL
            )
            missing = sum(
                1 for item in entry["evidence"] if item["statistical_status"] == MISSING
            )
            log.info(
                f"  {entry['rq_id']:<5} {entry['status']:<21} "
                f"{len(entry['evidence'])} evidence item(s), {formal} formal, "
                f"{missing} missing - {entry['rq_question']}"
            )

        if summary["missing_sources"]:
            log.warning(
                f"Missing source summaries: {summary['missing_sources']}. The affected "
                "evidence is marked missing rather than substituted."
            )
