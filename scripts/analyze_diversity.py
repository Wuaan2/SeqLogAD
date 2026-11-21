"""Run the P2.6 diversity analysis and write DIV-P2-001.json plus a readable DIV-P2-001.md.

Evidence layout: <evidence-root>/<fold_id>/<EXPERT_ID>.jsonl and <evidence-root>/<fold_id>/comparators.parquet.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seqlogad.diversity.analysis import analyze  # noqa: E402


def _num(value, digits=4):
    return "n/a" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def _ci(ci):
    return f"[{_num(ci['lower'])}, {_num(ci['upper'])}]"


def render(report: dict) -> str:
    g2 = report["g2_recommendation"]
    lines = [f"# {report['id']} — expert diversity (P2.6)", "",
             f"Recommended G2: **{g2['verdict']}** ({g2['status']}; signed by {g2['signed_by']}).",
             f"Experts analysed: {', '.join(report['experts_analyzed'])} (complete: {report['complete']}).",
             "Label scope: labeled source validation units only; target labels read: "
             f"{report['label_scope']['target_labels_read']}.", ""]
    for fold_id, fold in report["folds"].items():
        lines.append(f"## {fold_id}")
        for arch, block in fold["labeled"].items():
            lines += ["", f"### {arch} (labeled): {block['units']} units, {block['positive_units']} positive"
                          f"{' — INSUFFICIENT_POWER' if block['insufficient_power'] else ''}; best single "
                          f"`{block['best_single']}`", "",
                      "| Expert | AP | AUROC | FAR@R0.5 | FAR@R0.8 | Coverage | Verdict |", "|---|---|---|---|---|---|---|"]
            for expert, m in block["singles"].items():
                lines.append(f"| {expert} | {_num(m['average_precision'])} | {_num(m['auroc'])} | "
                             f"{_num(m['far_at_recall_0.5'])} | {_num(m['far_at_recall_0.8'])} | "
                             f"{_num(m['coverage'])} | {block['verdicts'].get(expert, '')} |")
            lines += ["", "| Combination | AP | ΔAP vs best | 95% CI | Verdict |", "|---|---|---|---|---|"]
            for name, m in block["combinations"].items():
                lines.append(f"| {name} | {_num(m['average_precision'])} | {_num(m['delta_ap_vs_best'])} | "
                             f"{_ci(m['delta_ci'])} | {m.get('verdict', '')} |")
            p = block["p2_3_minus_p2_2"]
            paired = f"{_num(p['delta_ap'])} {_ci(p['delta_ci'])}" if p else "n/a (P2.2 or P2.3 not analysed)"
            lines += ["", f"P2.3 − P2.2 paired ΔAP: {paired}", "",
                      "| Comparator | AP | AUROC | FAR@R0.5 | FAR@R0.8 |", "|---|---|---|---|---|"]
            for name, m in block["comparators"].items():
                lines.append(f"| {name} | {_num(m['average_precision'])} | {_num(m['auroc'])} | "
                             f"{_num(m['far_at_recall_0.5'])} | {_num(m['far_at_recall_0.8'])} |")
            lines += ["", "| Pair | Spearman | Double fault | Disagreement |", "|---|---|---|---|"]
            for pair, m in block["pairs"].items():
                lines.append(f"| {pair} | {_num(m['spearman'])} | {_num(m['double_fault_rate'])} | {_num(m['disagreement'])} |")
        lines += ["", "### Label-free diagnostics", "", "| Source | Expert | Scored windows | ABSTAIN | Unit coverage |",
                  "|---|---|---|---|---|"]
        for arch, block in fold["label_free"].items():
            for expert, c in block["coverage"].items():
                lines.append(f"| {arch} | {expert} | {c['scored_windows']} | {c['abstain_windows']} | {_num(c['unit_coverage'])} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--data-root", required=True, help="extracted P2.2 sequence bundle (phase2_sequence)")
    parser.add_argument("--sidecar-root", required=True, help="extracted phase2_source_development_v13r1")
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experts", default=None,
                        help="comma list for a partial check; the report then has no G2 recommendation")
    args = parser.parse_args()
    experts = args.experts.split(",") if args.experts else None
    report = analyze(args.repo_root, args.data_root, args.sidecar_root, args.evidence_root, experts)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "DIV-P2-001.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "DIV-P2-001.md").write_text(render(report), encoding="utf-8")
    print(f"G2 recommendation: {report['g2_recommendation']['verdict']}")


if __name__ == "__main__":
    main()
