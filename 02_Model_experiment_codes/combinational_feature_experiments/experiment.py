#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the LFP combinational-feature experiment matrix for `fai_irrev`-style
features.

1. SOC as y-axis, pulse width as x-axis, C-rate groups as the third axis.
2. SOC as y-axis, pulse amplitude as x-axis, width groups as the third axis.
3. Pulse amplitude as y-axis, pulse width as x-axis, SOC groups as the third axis.

SLURM array parallelization:
- Each array task runs ONE (material, part_idx).
- Let:
    P = PARTS_PER_MATERIAL
    M = number of materials
  Then:
    total_tasks = M * P

Outputs:
- Experiment 1 saved under ./Results_exp1_soc_width/<timestamp>__<material>__<case_name>/
- Experiment 2 saved under ./Results_exp2_soc_crate/<timestamp>__<material>__<case_name>/
- Experiment 3 saved under ./Results_exp3_crate_width/<timestamp>__<material>__<case_name>/
"""

import math
import os
import sys
import warnings
import csv
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import model as bench  # noqa: E402


DEFAULT_LFP_MATERIALS = ["20Ah LFP", "35Ah LFP", "68Ah LFP"]
VALID_SOCS_BY_MATERIAL = {
    "20Ah LFP": list(range(5, 80, 5)),
    "35Ah LFP": list(range(5, 85, 5)),
    "68Ah LFP": list(range(5, 80, 5)),
}
EXPERIMENT_RESULT_DIRS = {
    "soc_width": "Results_exp1_soc_width",
    "soc_crate": "Results_exp2_soc_crate",
    "crate_width": "Results_exp3_crate_width",
}

EXPERIMENT_TAGS = {
    "soc_width": "exp1",
    "soc_crate": "exp2",
    "crate_width": "exp3",
}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value.lstrip("-").isdigit() else default


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def _env_list(name: str) -> List[str]:
    value = os.environ.get(name, "").strip()
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_crate(value: Any) -> str:
    return bench.normalize_rate_label(value)


def _format_width(width_ms: int) -> str:
    return "W{}".format(int(width_ms))


def _format_soc(soc: int) -> str:
    return "SOC{:02d}".format(int(soc))


def _format_crate(crate: str) -> str:
    return _normalize_crate(crate).replace(".", "p")


def _get_widths() -> List[int]:
    width_env = _env_list("GRID_WIDTHS")
    if width_env:
        out = []
        for value in width_env:
            if value.lstrip("-").isdigit():
                out.append(int(value))
        if out:
            return out
    return [int(value) for value in getattr(bench, "WIDTHS_MS", [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000])]


def _get_full_socs() -> List[int]:
    soc_env = _env_list("GRID_SOCS")
    if soc_env:
        out = []
        for value in soc_env:
            if value.lstrip("-").isdigit():
                out.append(int(value))
        if out:
            return out
    return [int(value) for value in getattr(bench, "FULL_SOC_LEVELS", list(range(5, 95, 5)))]


def _get_soc_groups_axis() -> List[int]:
    soc_env = _env_list("GRID_SOCS_TO_70")
    if soc_env:
        out = []
        for value in soc_env:
            if value.lstrip("-").isdigit():
                out.append(int(value))
        if out:
            return out
    return [int(value) for value in getattr(bench, "SOC_LEVELS_TO_70", list(range(5, 75, 5)))]


def _get_crates() -> List[str]:
    crate_env = _env_list("GRID_CRATES")
    if crate_env:
        return [_normalize_crate(value) for value in crate_env]
    return [_normalize_crate(value) for value in getattr(bench, "C_RATES", ["0.5C", "1C", "1.5C", "2C", "2.5C"])]


def _get_experiments() -> List[str]:
    exp_env = _env_list("EXPERIMENTS")
    if exp_env:
        return exp_env
    return ["soc_width", "soc_crate", "crate_width"]


def _resolve_data_root() -> Path:
    data_root_env = _env_str("DATA_ROOT", "")
    if data_root_env:
        path = Path(data_root_env).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError("DATA_ROOT is set but not a directory: {}".format(path))
        return path

    cfg = dict(bench.CONFIG)
    cfg["data_root"] = None
    project_root = bench.find_project_root(THIS_DIR)
    return bench.resolve_data_root(cfg, project_root)


def _detect_materials(data_root: Path) -> List[str]:
    override = _env_list("MATERIALS")
    if override:
        return override

    exclude_pat = _env_str("MATERIAL_EXCLUDE", "").strip()
    materials = []
    missing = []
    for material in DEFAULT_LFP_MATERIALS:
        path = data_root / material
        if not path.is_dir():
            missing.append(material)
            continue
        if exclude_pat and exclude_pat in material:
            continue
        materials.append(material)

    if missing:
        raise RuntimeError("Expected LFP materials are missing under {}: {}".format(data_root, missing))

    if not materials:
        raise RuntimeError("No materials detected under {} (exclude='{}').".format(data_root, exclude_pat))
    return materials


def _base_cfg(material: str, result_root: Path) -> Dict[str, Any]:
    cfg = dict(bench.CONFIG)
    cfg["material"] = material
    cfg["out_dir"] = str(result_root.resolve())
    cfg["torch_device"] = "auto"

    data_root_env = _env_str("DATA_ROOT", "")
    if data_root_env:
        cfg["data_root"] = data_root_env

    seeds_env = _env_list("SEEDS")
    if seeds_env:
        cfg["seeds"] = [int(value) for value in seeds_env if value.lstrip("-").isdigit()]

    models_env = _env_list("MODELS")
    if models_env:
        cfg["models"] = models_env

    if _env_str("TORCH_EPOCHS", "").isdigit():
        cfg["torch_epochs"] = int(os.environ["TORCH_EPOCHS"])
    if _env_str("TORCH_BATCH", "").isdigit():
        cfg["torch_batch_size"] = int(os.environ["TORCH_BATCH"])
    if _env_str("TOPK_FEATIMP", "").isdigit():
        cfg["topk_featimp"] = int(os.environ["TOPK_FEATIMP"])

    cfg["show_progress"] = bool(_env_int("SHOW_PROGRESS", 0))
    return cfg


def _result_root_for_experiment(result_base_dir: Path, experiment_name: str) -> Path:
    env_key = "RESULT_DIR_{}".format(experiment_name.upper())
    dir_name = _env_str(env_key, EXPERIMENT_RESULT_DIRS[experiment_name])
    return (result_base_dir / dir_name).resolve()


def _build_grid_feature_spec(
    name: str,
    feature_family: str,
    family_aliases: Sequence[str],
    widths_ms: Sequence[int],
    socs: Sequence[int],
    c_rates: Sequence[str],
) -> Dict[str, Any]:
    spec = dict(
        mode="grid_family",
        name=name,
        feature_family=feature_family,
        widths_ms=[int(value) for value in widths_ms],
        socs=[int(value) for value in socs],
        c_rates=[_normalize_crate(value) for value in c_rates],
        merge_keys=["Qn", "Q"],
    )
    aliases = [str(alias).strip() for alias in family_aliases if str(alias).strip()]
    if aliases:
        spec["family_aliases"] = aliases
    return spec


def _exp1_crate_groups(all_crates: Sequence[str]) -> List[Tuple[str, List[str]]]:
    return [
        ("all_crates_05_10_15_20_25", list(all_crates)),
        ("low_crates_05_10", [_normalize_crate("0.5C"), _normalize_crate("1C")]),
        ("mid_crates_10_15", [_normalize_crate("1C"), _normalize_crate("1.5C")]),
        ("mid_crates_15_20", [_normalize_crate("1.5C"), _normalize_crate("2C")]),
        ("high_crates_20_25", [_normalize_crate("2C"), _normalize_crate("2.5C")]),
    ]


def _exp2_width_groups(all_widths: Sequence[int]) -> List[Tuple[str, List[int]]]:
    return [
        ("all_widths", list(all_widths)),
        ("low_widths_30_50_70_100", [30, 50, 70, 100]),
        ("low_widths_50_70_100", [50, 70, 100]),
        ("low_widths_30_50_70", [30, 50, 70]),
        ("mid_widths_300_500_700", [300, 500, 700]),
        ("mid_widths_100_300_500", [100, 300, 500]),
        ("high_widths_1000_3000_5000", [1000, 3000, 5000]),
        ("high_widths_3000_5000", [3000, 5000]),
        ("high_widths_1000_3000", [1000, 3000]),
    ]


def _exp3_soc_groups() -> List[Tuple[str, List[int]]]:
    return [
        ("all_socs", list(getattr(bench, "SOC_LEVELS_TO_70", list(range(5, 75, 5))))),
        ("low_socs_05_30", [5, 10, 15, 20, 25, 30]),
        ("low_socs_05_20", [5, 10, 15, 20]),
        ("low_socs_05_10", [5, 10]),
        ("mid_socs_35_40_45_50", [35, 40, 45, 50]),
        ("mid_socs_35_40", [35, 40]),
        ("mid_socs_45_50", [45, 50]),
        ("high_socs_55_60_65_70", [55, 60, 65, 70]),
        ("high_socs_65_70", [65, 70]),
        ("high_socs_55_60", [55, 60]),
    ]


def _filtered_group(values: Sequence[int], allowed: Sequence[int]) -> List[int]:
    allowed_set = set(int(value) for value in allowed)
    return [int(value) for value in values if int(value) in allowed_set]


def _experiment_case_counts(cases: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for case in cases:
        exp_name = str(case.get("experiment_name", "unknown"))
        counts[exp_name] = counts.get(exp_name, 0) + 1
    return counts


def build_cases(material: str) -> List[Dict[str, Any]]:
    result_base_dir = Path(_env_str("RESULT_BASE_DIR", str(THIS_DIR))).expanduser().resolve()

    feature_family = _env_str("FEATURE_FAMILY", "fai_irrev")
    family_aliases = _env_list("FEATURE_FAMILY_ALIASES")

    widths = _get_widths()
    full_socs = list(VALID_SOCS_BY_MATERIAL.get(material, _get_full_socs()))
    soc_axis_to_70 = _get_soc_groups_axis()
    crates = _get_crates()
    experiments = set(_get_experiments())

    cases: List[Dict[str, Any]] = []

    if "soc_width" in experiments:
        result_root = _result_root_for_experiment(result_base_dir, "soc_width")
        for soc in full_socs:
            for width in widths:
                for group_name, crate_values in _exp1_crate_groups(crates):
                    cfg = _base_cfg(material, result_root)
                    case_name = "exp1__{}_{}_{}".format(_format_soc(soc), _format_width(width), group_name)
                    cfg["experiment_name"] = "soc_width"
                    cfg["case_name"] = case_name
                    cfg["axis_x"] = "pulse_width"
                    cfg["axis_y"] = "soc"
                    cfg["selected_soc"] = int(soc)
                    cfg["selected_width_ms"] = int(width)
                    cfg["selected_c_rate_group"] = group_name
                    cfg["feature_spec"] = _build_grid_feature_spec(
                        name=case_name,
                        feature_family=feature_family,
                        family_aliases=family_aliases,
                        widths_ms=[width],
                        socs=[soc],
                        c_rates=crate_values,
                    )
                    cases.append(cfg)

    if "soc_crate" in experiments:
        result_root = _result_root_for_experiment(result_base_dir, "soc_crate")
        for soc in full_socs:
            for crate in crates:
                for group_name, width_values in _exp2_width_groups(widths):
                    filtered_widths = _filtered_group(width_values, widths)
                    if not filtered_widths:
                        continue
                    cfg = _base_cfg(material, result_root)
                    case_name = "exp2__{}_{}__{}".format(_format_soc(soc), _format_crate(crate), group_name)
                    cfg["experiment_name"] = "soc_crate"
                    cfg["case_name"] = case_name
                    cfg["axis_x"] = "pulse_amplitude"
                    cfg["axis_y"] = "soc"
                    cfg["selected_soc"] = int(soc)
                    cfg["selected_c_rate"] = _normalize_crate(crate)
                    cfg["selected_width_group"] = group_name
                    cfg["feature_spec"] = _build_grid_feature_spec(
                        name=case_name,
                        feature_family=feature_family,
                        family_aliases=family_aliases,
                        widths_ms=filtered_widths,
                        socs=[soc],
                        c_rates=[crate],
                    )
                    cases.append(cfg)

    if "crate_width" in experiments:
        result_root = _result_root_for_experiment(result_base_dir, "crate_width")
        filtered_soc_groups = []
        for group_name, soc_values in _exp3_soc_groups():
            group_values = _filtered_group(soc_values, soc_axis_to_70)
            if group_values:
                filtered_soc_groups.append((group_name, group_values))

        for crate in crates:
            for width in widths:
                for group_name, soc_values in filtered_soc_groups:
                    cfg = _base_cfg(material, result_root)
                    case_name = "exp3__{}__{}__{}".format(_format_crate(crate), _format_width(width), group_name)
                    cfg["experiment_name"] = "crate_width"
                    cfg["case_name"] = case_name
                    cfg["axis_x"] = "pulse_width"
                    cfg["axis_y"] = "pulse_amplitude"
                    cfg["selected_c_rate"] = _normalize_crate(crate)
                    cfg["selected_width_ms"] = int(width)
                    cfg["selected_soc_group"] = group_name
                    cfg["feature_spec"] = _build_grid_feature_spec(
                        name=case_name,
                        feature_family=feature_family,
                        family_aliases=family_aliases,
                        widths_ms=[width],
                        socs=soc_values,
                        c_rates=[crate],
                    )
                    cases.append(cfg)

    return cases


def main() -> None:
    data_root = _resolve_data_root()
    materials = _detect_materials(data_root)
    result_base_dir = Path(_env_str("RESULT_BASE_DIR", str(THIS_DIR))).expanduser().resolve()

    parts_per_material = _env_int("PARTS_PER_MATERIAL", _env_int("PARTS_PER_WIDTH", 32))
    task_id = _env_int("SLURM_ARRAY_TASK_ID", 0)
    total_tasks = len(materials) * parts_per_material

    if task_id < 0:
        raise SystemExit(
            "Invalid negative SLURM_ARRAY_TASK_ID={}. Expected 0..{} (materials={}, parts_per_material={}).".format(
                task_id, total_tasks - 1, len(materials), parts_per_material
            )
        )
    if task_id >= total_tasks:
        print(
            "[SKIP] SLURM_ARRAY_TASK_ID={} exceeds total_tasks={} for materials={} and parts_per_material={}.".format(
                task_id, total_tasks, materials, parts_per_material
            )
        )
        return

    mat_idx = task_id // parts_per_material
    part_idx = task_id % parts_per_material
    material = materials[mat_idx]

    cases = build_cases(material)
    total_cases = len(cases)
    counts = _experiment_case_counts(cases)

    chunk_size = int(math.ceil(float(total_cases) / float(parts_per_material))) if parts_per_material > 0 else total_cases
    start = part_idx * chunk_size
    end = min(total_cases, (part_idx + 1) * chunk_size)

    print("==========================================================")
    print("[INFO] PWD               : {}".format(THIS_DIR))
    print("[INFO] DATA_ROOT         : {}".format(data_root))
    print("[INFO] MATERIALS         : {} -> {} (exclude='{}')".format(len(materials), materials, _env_str("MATERIAL_EXCLUDE", "")))
    print("[INFO] EXPERIMENTS       : {}".format(_get_experiments()))
    print("[INFO] RESULT_DIRS       : {}".format(dict((name, str(_result_root_for_experiment(result_base_dir, name))) for name in _get_experiments())))
    print("[INFO] GRID_WIDTHS       : {}".format(_get_widths()))
    print("[INFO] GRID_SOCS         : {}".format(_get_full_socs()))
    print("[INFO] GRID_SOCS_TO_70   : {}".format(_get_soc_groups_axis()))
    print("[INFO] GRID_CRATES       : {}".format(_get_crates()))
    print("[INFO] FEATURE_FAMILY    : {}".format(_env_str("FEATURE_FAMILY", "fai_irrev")))
    print("[INFO] FEATURE_ALIASES   : {}".format(_env_list("FEATURE_FAMILY_ALIASES")))
    print("[INFO] TASK_ID           : {}/{}".format(task_id, total_tasks - 1))
    print("[INFO] SELECTED material : {} (mat_idx={})".format(material, mat_idx))
    print("[INFO] PART              : {}/{} | slice=({}:{}) / total_cases={}".format(part_idx, parts_per_material, start, end, total_cases))
    print("[INFO] CASE_COUNTS       : {}".format(counts))
    print("==========================================================")

    if start >= end:
        print("[SKIP] Empty slice for this part: ({}:{})".format(start, end))
        return

    os.chdir(THIS_DIR)

    case_failures: List[Dict[str, Any]] = []
    skipped_cases = 0
    done_cases = 0

    for case_idx in range(start, end):
        case_cfg = cases[case_idx]
        try:
            out_dir = bench.run(case_cfg)
            print("[DONE] case {}/{} -> {}".format(case_idx + 1, total_cases, out_dir))
            done_cases += 1
        except getattr(bench, "CaseSkippedError", RuntimeError) as exc:
            skipped_cases += 1
            run_dir = getattr(exc, "run_dir", None)
            status = getattr(exc, "status", "skipped")
            print(
                "[SKIP] case {}/{} {} -> {} ({})".format(
                    case_idx + 1,
                    total_cases,
                    case_cfg.get("case_name", "<unknown>"),
                    run_dir if run_dir is not None else "<no_run_dir>",
                    status,
                )
            )
        except Exception as exc:
            case_failures.append(
                dict(
                    case_index=case_idx + 1,
                    total_cases=total_cases,
                    case_name=str(case_cfg.get("case_name", "")),
                    experiment_name=str(case_cfg.get("experiment_name", "")),
                    material=str(case_cfg.get("material", "")),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    traceback=traceback.format_exc(),
                )
            )
            print("[FAIL] case {}/{} {} -> {}".format(case_idx + 1, total_cases, case_cfg.get("case_name", "<unknown>"), exc))
            print(case_failures[-1]["traceback"], flush=True)

    print("[INFO] Task summary      : done={} | skipped={} | failed={}".format(done_cases, skipped_cases, len(case_failures)))
    if case_failures:
        failure_log = THIS_DIR / "logs" / "task_failures_{}_{}.csv".format(os.environ.get("SLURM_JOB_ID", "local"), task_id)
        failure_log.parent.mkdir(parents=True, exist_ok=True)
        with failure_log.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["case_index", "total_cases", "case_name", "experiment_name", "material", "error_type", "error_message", "traceback"],
            )
            writer.writeheader()
            writer.writerows(case_failures)
        print("[WARN] Wrote task failure log: {}".format(failure_log))


if __name__ == "__main__":
    main()
