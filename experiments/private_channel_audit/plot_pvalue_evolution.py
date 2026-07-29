"""Plot running paired-test p-values for a completed private-channel audit run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


def _completed_metrics_by_arm(root: Path) -> dict[bool, dict[int, dict[str, Any]]]:
    arms: dict[bool, dict[int, dict[str, Any]]] = {False: {}, True: {}}
    for config_path in root.glob("runs/**/run_config.json"):
        metrics_path = config_path.with_name("metrics.json")
        final_path = config_path.with_name("final_summary.json")
        if not (metrics_path.is_file() and final_path.is_file()):
            continue

        run_config = json.loads(config_path.read_text(encoding="utf-8"))
        final_summary = json.loads(final_path.read_text(encoding="utf-8"))
        if str(final_summary.get("status", "")).lower() != "complete":
            continue

        audit_enabled = bool(
            (run_config.get("private_channel_audit") or {}).get("enabled")
        )
        arms[audit_enabled][int(run_config["seed"])] = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )
    return arms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        required=True,
        help="Completed experiment root containing runs/.",
    )
    parser.add_argument(
        "--metric",
        default="coalition_advantage_mean",
        help="Paired metric to test (default: coalition_advantage_mean).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: <run-root>/pvalue_evolution.png).",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser()
    arms = _completed_metrics_by_arm(run_root)
    matched_seeds = sorted(set(arms[False]) & set(arms[True]))
    if len(matched_seeds) < 2:
        raise ValueError("Need at least two completed matched seed pairs.")

    rows: list[dict[str, float | int]] = []
    for count in range(2, len(matched_seeds) + 1):
        seeds = matched_seeds[:count]
        no_audit = [float(arms[False][seed][args.metric]) for seed in seeds]
        audit = [float(arms[True][seed][args.metric]) for seed in seeds]
        rows.append(
            {
                "num_seeds": count,
                "two_sided_p": float(ttest_rel(audit, no_audit).pvalue),
                "one_sided_p_audit_lower": float(
                    ttest_rel(audit, no_audit, alternative="less").pvalue
                ),
            }
        )

    output_path = Path(args.output) if args.output else run_root / "pvalue_evolution.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(8, 4.8))
    plt.plot(
        [row["num_seeds"] for row in rows],
        [row["two_sided_p"] for row in rows],
        marker="o",
        label="Two-sided paired p-value",
    )
    plt.plot(
        [row["num_seeds"] for row in rows],
        [row["one_sided_p_audit_lower"] for row in rows],
        marker="o",
        label="One-sided paired p-value (audit lowers advantage)",
    )
    plt.axhline(0.05, color="black", linestyle="--", linewidth=1, label="0.05")
    plt.xlabel("Completed matched seed pairs")
    plt.ylabel("Running p-value")
    plt.title("Coalition-advantage p-values as seeds accumulate")
    plt.xlim(2, len(matched_seeds))
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Wrote {output_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
