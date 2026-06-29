#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run single fai_irrev feature LFP experiments.

Default matrix:
- materials: 20Ah LFP, 35Ah LFP, 68Ah LFP
- widths: model.WIDTHS_MS
- SOC: 20Ah/68Ah -> 5..75, 35Ah -> 5..80
- features: Hyst_M3_*C raw columns, reported as fai_irrev_*C

SLURM array mapping:
    task_id = material_index * PARTS_PER_MATERIAL + part_index
Each task runs one chunk of that material's single-feature cases.
"""

import csv
import math
import os
import re
import sys
import traceback
import warnings
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

DEFAULT_RAW_FEATURE_REGEX = r"^Hyst_M3_"


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


def _safe_name(value: Any) -> str:
    return re.sub(r"[^\w\-.]+", "_", str(value)).strip("_")


def _public_feature_name(feature: str) -> str:
    return str(feature).replace("Hyst_M3", "fai_irrev").replace("hyst_m3", "fai_irrev")


def _format_width(width_ms: int) -> str:
    return "W{}".format(int(width_ms))


def _format_soc(soc: int) -> str:
    return "SOC{:02d}".format(int(soc))


def _get_widths() -> List[int]:
    width_env = _env_list("GRID_WIDTHS")
    if width_env:
        out = [int(value) for value in width_env if value.lstrip("-").isdigit()]
        if out:
            return out
    return [int(value) for value in getattr(bench, "WIDTHS_MS", [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000])]


def _valid_socs_for_material(material: str) -> List[int]:
    soc_env = _env_list("GRID_SOCS")
    if soc_env:
        out = [int(value) for value in soc_env if value.lstrip("-").isdigit()]
        if out:
            max_allowed = max(VALID_SOCS_BY_MATERIAL.get(material, out))
            return [soc for soc in out if soc <= max_allowed]
    return list(VALID_SOCS_BY_MATERIAL.get(material, list(range(5, 80, 5))))


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

    materials = []
    missing = []
    for material in DEFAULT_LFP_MATERIALS:
        if (data_root / material).is_dir():
            materials.append(material)
        else:
            missing.append(material)

    if missing:
        raise RuntimeError("Expected LFP materials are missing under {}: {}".format(data_root, missing))
    if not materials:
        raise RuntimeError("No LFP materials detected under {}.".format(data_root))
    return materials


def _choose_feature_probe(materials: Sequence[str], widths: Sequence[int]) -> Tuple[str, int, int]:
    material_env = _env_str("FEATURE_SOURCE_MATERIAL", "")
    material = material_env if material_env else str(materials[0])

    width_env = _env_str("FEATURE_SOURCE_WIDTH", "")
    width_ms = int(width_env) if width_env.lstrip("-").isdigit() else int(widths[0])

    soc_env = _env_str("FEATURE_SOURCE_SOC", "")
    if soc_env.lstrip("-").isdigit():
        soc = int(soc_env)
    else:
        soc = int(_valid_socs_for_material(material)[0])

    return material, width_ms, soc


def _discover_features(data_root: Path, materials: Sequence[str], widths: Sequence[int]) -> List[str]:
    explicit = _env_list("FEATURES")
    if explicit:
        return [feature.replace("fai_irrev", "Hyst_M3") for feature in explicit]

    material, width_ms, soc = _choose_feature_probe(materials, widths)
    xlsx_path = bench.find_xlsx(data_root, material, width_ms)
    sheet_name = bench.detect_soc_sheet(xlsx_path, soc)

    import pandas as pd  # imported lazily so syntax checks do not need pandas

    header = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl", nrows=0)
    raw_features = [str(col) for col in header.columns]
    default_pattern = re.compile(DEFAULT_RAW_FEATURE_REGEX)
    features = [feature for feature in raw_features if default_pattern.search(feature)]

    include_regex = _env_str("FEATURE_REGEX", "")
    if include_regex:
        pattern = re.compile(include_regex)
        features = [feature for feature in features if pattern.search(feature)]

    exclude_regex = _env_str("FEATURE_EXCLUDE_REGEX", "")
    if exclude_regex:
        pattern = re.compile(exclude_regex)
        features = [feature for feature in features if not pattern.search(feature)]

    if not features:
        raise RuntimeError(
            "No Hyst_M3/fai_irrev features discovered from {} sheet {}. Check FEATURES or FEATURE_REGEX.".format(xlsx_path, sheet_name)
        )
    return features


def _base_cfg(material: str, width_ms: int, soc: int, feature: str, result_root: Path) -> Dict[str, Any]:
    cfg = dict(bench.CONFIG)
    cfg["material"] = material
    cfg["width_ms"] = int(width_ms)
    cfg["soc"] = int(soc)
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
    if _env_str("PERM_REPEATS", "").isdigit():
        cfg["perm_repeats"] = int(os.environ["PERM_REPEATS"])

    public_feature = _public_feature_name(feature)
    case_name = "single__{}".format(_safe_name(public_feature))
    cfg["case_name"] = case_name
    cfg["experiment_name"] = "single_feature"
    cfg["selected_feature"] = feature
    cfg["selected_feature_public"] = public_feature
    cfg["feature_spec"] = dict(
        mode="legacy_combo",
        name=case_name,
        combo=None,
        include_groups=None,
        exclude_groups=None,
        add_features=[feature],
        drop_features=None,
        keep_regex=None,
        meta_columns=["Qn", "Q", "SOC"],
    )
    cfg["show_progress"] = bool(_env_int("SHOW_PROGRESS", 0))
    return cfg


def build_cases(material: str, widths: Sequence[int], features: Sequence[str]) -> List[Dict[str, Any]]:
    result_base_dir = Path(_env_str("RESULT_BASE_DIR", str(THIS_DIR))).expanduser().resolve()
    result_root = result_base_dir / _env_str("RESULT_DIR_SINGLE", "Results_single_feature")

    cases: List[Dict[str, Any]] = []
    for width_ms in widths:
        for soc in _valid_socs_for_material(material):
            for feature in features:
                cfg = _base_cfg(material, int(width_ms), int(soc), str(feature), result_root)
                cfg["case_matrix_tag"] = "{}__{}__{}".format(_format_width(width_ms), _format_soc(soc), _safe_name(_public_feature_name(feature)))
                cases.append(cfg)
    return cases


def _count_cases_by_material(materials: Iterable[str], widths: Sequence[int], features: Sequence[str]) -> Dict[str, int]:
    return {
        material: int(len(widths) * len(_valid_socs_for_material(material)) * len(features))
        for material in materials
    }


def main() -> None:
    data_root = _resolve_data_root()
    widths = _get_widths()
    materials = _detect_materials(data_root)
    features = _discover_features(data_root, materials, widths)
    result_base_dir = Path(_env_str("RESULT_BASE_DIR", str(THIS_DIR))).expanduser().resolve()

    parts_per_material = _env_int("PARTS_PER_MATERIAL", 96)
    task_id = _env_int("SLURM_ARRAY_TASK_ID", 0)
    total_tasks = len(materials) * parts_per_material

    counts_by_material = _count_cases_by_material(materials, widths, features)

    print("==========================================================")
    print("[INFO] PWD                : {}".format(THIS_DIR))
    print("[INFO] DATA_ROOT          : {}".format(data_root))
    print("[INFO] RESULT_BASE_DIR    : {}".format(result_base_dir))
    print("[INFO] MATERIALS          : {}".format(materials))
    print("[INFO] WIDTHS             : {}".format(widths))
    print("[INFO] SOC_LIMITS         : {}".format(dict((m, "{}..{}".format(min(_valid_socs_for_material(m)), max(_valid_socs_for_material(m)))) for m in materials)))
    print("[INFO] N_FEATURES         : {}".format(len(features)))
    print("[INFO] RAW_FEATURES       : {}".format(features))
    print("[INFO] PUBLIC_FEATURES    : {}".format([_public_feature_name(feature) for feature in features]))
    print("[INFO] CASES_BY_MATERIAL  : {}".format(counts_by_material))
    print("[INFO] PARTS_PER_MATERIAL : {}".format(parts_per_material))
    print("[INFO] TASK_ID            : {}/{}".format(task_id, total_tasks - 1))
    print("==========================================================")

    if bool(_env_int("LIST_ONLY", 0)):
        return

    if task_id < 0:
        raise SystemExit("Invalid negative SLURM_ARRAY_TASK_ID={}.".format(task_id))
    if task_id >= total_tasks:
        print("[SKIP] SLURM_ARRAY_TASK_ID={} exceeds total_tasks={}.".format(task_id, total_tasks))
        return

    mat_idx = task_id // parts_per_material
    part_idx = task_id % parts_per_material
    material = materials[mat_idx]

    cases = build_cases(material, widths, features)
    total_cases = len(cases)
    chunk_size = int(math.ceil(float(total_cases) / float(parts_per_material))) if parts_per_material > 0 else total_cases
    start = part_idx * chunk_size
    end = min(total_cases, (part_idx + 1) * chunk_size)

    print("[INFO] SELECTED material  : {} (mat_idx={})".format(material, mat_idx))
    print("[INFO] PART               : {}/{} | slice=({}:{}) / total_cases={}".format(part_idx, parts_per_material, start, end, total_cases))

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
                    case_cfg.get("case_matrix_tag", case_cfg.get("case_name", "<unknown>")),
                    run_dir if run_dir is not None else "<no_run_dir>",
                    status,
                )
            )
        except Exception as exc:
            case_failures.append(
                dict(
                    case_index=case_idx + 1,
                    total_cases=total_cases,
                    material=str(case_cfg.get("material", "")),
                    width_ms=str(case_cfg.get("width_ms", "")),
                    soc=str(case_cfg.get("soc", "")),
                    feature=str(case_cfg.get("selected_feature", "")),
                    case_name=str(case_cfg.get("case_name", "")),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    traceback=traceback.format_exc(),
                )
            )
            print(
                "[FAIL] case {}/{} {} -> {}".format(
                    case_idx + 1,
                    total_cases,
                    case_cfg.get("case_matrix_tag", case_cfg.get("case_name", "<unknown>")),
                    exc,
                )
            )
            print(case_failures[-1]["traceback"], flush=True)

    print("[INFO] Task summary       : done={} | skipped={} | failed={}".format(done_cases, skipped_cases, len(case_failures)))
    if case_failures:
        failure_log = THIS_DIR / "logs" / "task_failures_{}_{}.csv".format(os.environ.get("SLURM_JOB_ID", "local"), task_id)
        failure_log.parent.mkdir(parents=True, exist_ok=True)
        with failure_log.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "case_index",
                    "total_cases",
                    "material",
                    "width_ms",
                    "soc",
                    "feature",
                    "case_name",
                    "error_type",
                    "error_message",
                    "traceback",
                ],
            )
            writer.writeheader()
            writer.writerows(case_failures)
        print("[WARN] Wrote task failure log: {}".format(failure_log))


if __name__ == "__main__":
    main()
