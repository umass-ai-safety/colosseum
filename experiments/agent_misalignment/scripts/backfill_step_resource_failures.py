from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_FAILED_RE = re.compile(r"^(?P<pid>[^:]+):\s*RESOURCE FAILED\s*\(Steps:\s*(?P<steps>[^)]+)\)")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_failed_patients(final_summary: Dict[str, Any]) -> List[Tuple[str, int]]:
    conv = final_summary.get("convergence_report", {}) or {}
    failed = conv.get("failed_patients") or []
    out: List[Tuple[str, int]] = []
    if not isinstance(failed, list):
        return out
    for row in failed:
        if not isinstance(row, str):
            continue
        m = _FAILED_RE.match(row.strip())
        if not m:
            continue
        pid = m.group("pid").strip()
        steps_s = m.group("steps")
        for s in [x.strip() for x in steps_s.split(",") if x.strip()]:
            try:
                out.append((pid, int(s)))
            except Exception:
                continue
    return out


def _phase_rank(phase: Optional[str]) -> int:
    p = str(phase or "").lower()
    if p == "planning":
        return 0
    if p == "execution":
        return 1
    return 2


def _find_scheduler_agent(
    *,
    tool_events: List[Dict[str, Any]],
    patient_id: str,
    step_index: int,
) -> Optional[str]:
    best = None
    best_key = None
    for e in tool_events:
        if str(e.get("tool_name") or "") != "schedule_patient":
            continue
        args = e.get("arguments") or {}
        if str(args.get("patient_id") or "") != patient_id:
            continue
        try:
            if int(args.get("step_index")) != int(step_index):
                continue
        except Exception:
            continue
        key = (
            int(e.get("iteration") or 0),
            _phase_rank(e.get("phase")),
            int(e.get("planning_round") or 0),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = e
    if not best:
        return None
    return str(best.get("agent_name") or "") or None


def _infer_missing_resources(convergence_report: Dict[str, Any]) -> Optional[List[str]]:
    rf = convergence_report.get("resource_failures") or {}
    if not isinstance(rf, dict):
        return None
    nz = [str(r) for r, v in rf.items() if int(v or 0) > 0]
    if len(nz) == 1:
        return nz
    return None


_CONSUMPTION_MAP: Dict[str, List[str]] = {
    # Mirrors HospitalEnvironment.consumption_map in envs/dcops/hospital/hospital_env.py
    "Surgery": ["Anesthetics", "Surgical_Packs", "PPE_Sets", "IV_Kits", "Pain_Killers"],
    "Radiology": ["Radio_Contrast", "PPE_Sets", "IV_Kits"],
    "Ward": ["Oxygen_Tanks", "IV_Kits", "Pain_Killers", "PPE_Sets"],
    "Triage": ["PPE_Sets", "IV_Kits"],
}


def _infer_missing_for_agent(
    *,
    agent_name: Optional[str],
    failing_resources: List[str],
) -> Optional[List[str]]:
    if not agent_name:
        return failing_resources or None
    name = str(agent_name)
    if name == "Resource_Provisioner":
        return failing_resources or None
    service = name.split("_")[-1]
    candidates = _CONSUMPTION_MAP.get(service)
    if not candidates:
        return failing_resources or None
    missing = [r for r in candidates if r in failing_resources]
    return missing or (failing_resources or None)


def _agent_to_hospital(agent_name: str, hospitals: List[str]) -> Optional[str]:
    name = str(agent_name or "")
    for h in sorted(hospitals, key=len, reverse=True):
        if name == h or name.startswith(h + "_"):
            return h
    return None


def _infer_hospitals(convergence_report: Dict[str, Any]) -> List[str]:
    inv = convergence_report.get("final_inventory") or {}
    if isinstance(inv, dict):
        return [str(k) for k in inv.keys() if str(k) != "Resource_Provisioner"]
    return []


def backfill_run(run_dir: Path, *, dry_run: bool) -> bool:
    fs_path = run_dir / "final_summary.json"
    te_path = run_dir / "tool_events.json"
    rc_path = run_dir / "run_config.json"
    if not fs_path.exists() or not te_path.exists() or not rc_path.exists():
        return False

    fs = _read_json(fs_path)
    te = _read_json(te_path)
    rc = _read_json(rc_path)
    if not isinstance(fs, dict) or not isinstance(te, list) or not isinstance(rc, dict):
        return False

    conv = fs.get("convergence_report", {}) or {}
    if not isinstance(conv, dict):
        conv = {}

    # Always ensure key exists (even if empty).
    if not isinstance(fs.get("step_resource_failures"), list):
        fs["step_resource_failures"] = []

    pairs = _parse_failed_patients(fs)
    hospitals = _infer_hospitals(conv)
    failing_resources = [
        str(r)
        for r, v in (conv.get("resource_failures") or {}).items()
        if isinstance((conv.get("resource_failures") or {}), dict) and int(v or 0) > 0
    ]
    missing_hint = _infer_missing_resources(conv)

    step_events: List[Dict[str, Any]] = []
    for pid, step in pairs:
        agent = _find_scheduler_agent(tool_events=te, patient_id=pid, step_index=step)
        missing = missing_hint
        if missing is None:
            missing = _infer_missing_for_agent(
                agent_name=agent, failing_resources=failing_resources
            )
        ev = {
            "patient_id": pid,
            "step_index": int(step),
            "missing": missing,  # best-effort for older logs
            "agent": agent,
            "backfilled": True,
        }
        step_events.append(ev)

    # Fill if empty; otherwise update missing fields where absent.
    if not fs["step_resource_failures"] and step_events:
        fs["step_resource_failures"] = step_events
    else:
        if isinstance(fs.get("step_resource_failures"), list):
            for ev in fs["step_resource_failures"]:
                if not isinstance(ev, dict):
                    continue
                if ev.get("missing") is None or ev.get("missing") == []:
                    agent = ev.get("agent")
                    ev["missing"] = _infer_missing_for_agent(
                        agent_name=str(agent) if agent else None,
                        failing_resources=failing_resources,
                    )

    # Best-effort hospital_failures if missing OR inconsistent with global counts.
    global_rf = conv.get("resource_failures") or {}
    global_counts: Dict[str, int] = (
        {str(r): int(v or 0) for r, v in global_rf.items()} if isinstance(global_rf, dict) else {}
    )
    hf_existing = fs.get("hospital_failures")

    def _hf_matches_global(hf: Any) -> bool:
        if not isinstance(hf, dict) or not isinstance(global_counts, dict):
            return False
        for r, cnt in global_counts.items():
            s = 0
            for d in hf.values():
                if isinstance(d, dict):
                    s += int(d.get(r) or 0)
            if s != int(cnt):
                return False
        return True

    need_hf = (
        hospitals
        and sum(global_counts.values()) > 0
        and (not isinstance(hf_existing, dict) or not _hf_matches_global(hf_existing))
    )

    if need_hf and hospitals:
        # Weight hospitals by which agents scheduled failed steps for each resource.
        weights: Dict[str, Dict[str, float]] = {h: {r: 0.0 for r in global_counts} for h in hospitals}
        for ev in fs.get("step_resource_failures") or []:
            if not isinstance(ev, dict):
                continue
            agent = ev.get("agent")
            h = _agent_to_hospital(str(agent or ""), hospitals) if agent else None
            if not h:
                continue
            miss = ev.get("missing")
            if not isinstance(miss, list) or not miss:
                miss = failing_resources
            for r in miss:
                if r in global_counts and int(global_counts[r]) > 0:
                    weights[h][r] = float(weights[h].get(r, 0.0) + 1.0)

        # Allocate integer counts per resource with rounding that preserves totals.
        hf: Dict[str, Dict[str, int]] = {h: {} for h in hospitals}
        for r, total in global_counts.items():
            total = int(total)
            if total <= 0:
                continue
            denom = sum(float(weights[h].get(r, 0.0)) for h in hospitals)
            if denom <= 0.0:
                # Even split
                base = total // len(hospitals)
                rem = total - base * len(hospitals)
                for i, h in enumerate(hospitals):
                    hf[h][r] = base + (1 if i < rem else 0)
                continue
            raw = {h: (float(total) * float(weights[h].get(r, 0.0)) / denom) for h in hospitals}
            floors = {h: int(raw[h]) for h in hospitals}
            used = sum(floors.values())
            rem = total - used
            # Distribute remainder by largest fractional parts.
            fracs = sorted(((raw[h] - floors[h], h) for h in hospitals), reverse=True)
            for i in range(rem):
                _, h = fracs[i % len(fracs)]
                floors[h] += 1
            for h in hospitals:
                if floors[h] > 0:
                    hf[h][r] = int(floors[h])
        fs["hospital_failures"] = hf

    # agent_outstanding_resource_failures (by hospital).
    # Fill if missing OR previously empty while failures exist.
    hf_now = fs.get("hospital_failures")
    existing_aorf = fs.get("agent_outstanding_resource_failures")
    should_fill_aorf = (
        not isinstance(existing_aorf, dict)
        or (not existing_aorf and isinstance(hf_now, dict) and any(int(v or 0) > 0 for d in hf_now.values() if isinstance(d, dict) for v in d.values()))
    )
    if should_fill_aorf:
        agent_out: Dict[str, Dict[str, int]] = {}
        roles = rc.get("roles") or {}
        if isinstance(hf_now, dict) and isinstance(roles, dict):
            for agent_name in roles.keys():
                h = _agent_to_hospital(str(agent_name), hospitals) if hospitals else None
                if not h or h not in hf_now:
                    continue
                counts = hf_now.get(h) or {}
                if not isinstance(counts, dict):
                    continue
                nonzero = {
                    str(r): int(v or 0) for r, v in counts.items() if int(v or 0) > 0
                }
                if nonzero:
                    agent_out[str(agent_name)] = nonzero
        fs["agent_outstanding_resource_failures"] = agent_out

    if not dry_run:
        _write_json(fs_path, fs)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = sorted({p.parent for p in args.sweep_dir.rglob("final_summary.json")})
    changed = 0
    for run_dir in runs:
        if backfill_run(run_dir, dry_run=args.dry_run):
            changed += 1
    print(f"runs_total={len(runs)} processed={changed} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
