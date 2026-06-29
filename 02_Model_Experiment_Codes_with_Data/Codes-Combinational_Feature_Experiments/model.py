# -*- coding: utf-8 -*-
"""
model.py

Flat-output benchmark runner for the combinational pulse-excitation
experiments.

Outputs:
<run_dir>/
  config.json
  Exp_summary.json
  predictions_all.csv
  featimp_all.csv
  featimp_summary_all_models.csv
  metrics_by_seed.csv
  metrics_summary.csv
"""

import json
import math
import os
import re
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def _get_cpu_njobs() -> int:
    """Best-effort: respect Slurm allocation to avoid oversubscription."""
    for key in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(key, "").strip()
        if value.isdigit():
            return max(1, int(value))
    return max(1, (os.cpu_count() or 1))


CPU_NJOBS = _get_cpu_njobs()


class CaseSkippedError(RuntimeError):
    def __init__(self, status: str, message: str, run_dir: Optional[Path] = None):
        super().__init__(message)
        self.status = str(status)
        self.run_dir = Path(run_dir) if run_dir is not None else None


_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


FEATURE_GROUPS = {
    "Hyst": [
        "Fai_0.5C", "Fai_1C", "Fai_1.5C", "Fai_2C", "Fai_2.5C",
    ],
    "R0": [
        "R0_p_0.5C", "R0_n_0.5C",
        "R0_p_1C", "R0_n_1C",
        "R0_p_1.5C", "R0_n_1.5C",
        "R0_p_2C", "R0_n_2C",
        "R0_p_2.5C", "R0_n_2.5C",
    ],
    "Vpol": [
        "Vpol_p_0.5C", "Vpol_n_0.5C",
        "Vpol_p_1C", "Vpol_n_1C",
        "Vpol_p_1.5C", "Vpol_n_1.5C",
        "Vpol_p_2C", "Vpol_n_2C",
        "Vpol_p_2.5C", "Vpol_n_2.5C",
    ],
    "Vrelax": [
        "Vrelax_p_0.5C", "Vrelax_n_0.5C",
        "Vrelax_p_1C", "Vrelax_n_1C",
        "Vrelax_p_1.5C", "Vrelax_n_1.5C",
        "Vrelax_p_2C", "Vrelax_n_2C",
        "Vrelax_p_2.5C", "Vrelax_n_2.5C",
    ],
    "Reff": [
        "Reff_p_0.5C", "Reff_n_0.5C",
        "Reff_p_1C", "Reff_n_1C",
        "Reff_p_1.5C", "Reff_n_1.5C",
        "Reff_p_2C", "Reff_n_2C",
        "Reff_p_2.5C", "Reff_n_2.5C",
    ],
    "Eloss": [
        "Eloss_proxy_p_0.5C", "Eloss_proxy_n_0.5C",
        "Eloss_proxy_p_1C", "Eloss_proxy_n_1C",
        "Eloss_proxy_p_1.5C", "Eloss_proxy_n_1.5C",
        "Eloss_proxy_p_2C", "Eloss_proxy_n_2C",
        "Eloss_proxy_p_2.5C", "Eloss_proxy_n_2.5C",
    ],
}

FEATURE_COMBOS = {
    "Hyst_Only": ["Hyst"],
    "R0_Only": ["R0"],
    "Vpol_Only": ["Vpol"],
    "Vrelax_Only": ["Vrelax"],
    "Basic_Physics": ["Hyst", "R0", "Vpol", "Vrelax"],
    "NoHyst_Physics": ["R0", "Vpol", "Vrelax"],
}

WIDTHS_MS = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]
SOC_LIST = [40, 50, 70]
FULL_SOC_LEVELS = list(range(5, 95, 5))
SOC_LEVELS_TO_70 = list(range(5, 75, 5))
C_RATES = ["0.5C", "1C", "1.5C", "2C", "2.5C"]

DEFAULT_GRID_FAMILY_ALIASES = {
    # The manuscript denotes this pulse feature as phi. The Excel tables use
    # the ASCII label Fai_*C, while older internal exports used Hyst_M3_*C.
    "fai_irrev": ["fai_irrev", "fai", "phi_irrev", "phi", "faiirrev", "phiirrev", "irrev", "hyst_m3", "hystm3"],
}


def _unique_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def build_feature_list(
    combo: Optional[str] = None,
    include_groups: Optional[List[str]] = None,
    exclude_groups: Optional[List[str]] = None,
    add_features: Optional[List[str]] = None,
    drop_features: Optional[List[str]] = None,
    keep_regex: Optional[str] = None,
) -> List[str]:
    feats = []

    if combo is not None:
        if combo not in FEATURE_COMBOS:
            raise KeyError("Unknown combo '{}'. Available combos: {}".format(combo, list(FEATURE_COMBOS.keys())))
        for group_name in FEATURE_COMBOS[combo]:
            if group_name not in FEATURE_GROUPS:
                raise KeyError("Combo '{}' references unknown group '{}'".format(combo, group_name))
            feats.extend(FEATURE_GROUPS[group_name])

    if include_groups:
        for group_name in include_groups:
            if group_name not in FEATURE_GROUPS:
                raise KeyError("Unknown group '{}'. Available groups: {}".format(group_name, list(FEATURE_GROUPS.keys())))
            feats.extend(FEATURE_GROUPS[group_name])

    feats = _unique_preserve_order(feats)

    if exclude_groups:
        exclude_set = set()
        for group_name in exclude_groups:
            if group_name not in FEATURE_GROUPS:
                raise KeyError("Unknown group '{}'. Available groups: {}".format(group_name, list(FEATURE_GROUPS.keys())))
            exclude_set.update(FEATURE_GROUPS[group_name])
        feats = [feat for feat in feats if feat not in exclude_set]

    if add_features:
        feats.extend(add_features)
        feats = _unique_preserve_order(feats)

    if drop_features:
        drop_set = set(drop_features)
        feats = [feat for feat in feats if feat not in drop_set]

    if keep_regex:
        pattern = re.compile(str(keep_regex))
        feats = [feat for feat in feats if pattern.search(feat)]

    return feats


def _now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _safe_name(value: Any) -> str:
    text = re.sub(r"[^\w\-\.\s]+", "_", str(value))
    text = text.strip().replace(" ", "_")
    return text


def _compact_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


RATE_TOKEN_RE = re.compile(r"(?<![0-9.])([0-9]+(?:\.[0-9]+)?)c(?![0-9.])", re.IGNORECASE)


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return ("{:.6g}".format(float(value))).rstrip("0").rstrip(".")


def normalize_rate_label(value: Any) -> str:
    text = str(value).strip().upper().replace(" ", "")
    if not text:
        raise ValueError("Empty C-rate value.")
    if text.endswith("C"):
        text = text[:-1]
    text = text.rstrip("%")
    number = float(text)
    return "{}C".format(_format_number(number))


def crate_match_tokens(rate_label: Any) -> List[str]:
    normalized = normalize_rate_label(rate_label)
    core = normalized[:-1]
    number = float(core)
    tokens = [
        "{}C".format(_format_number(number)),
        "{}C".format("{:.1f}".format(number)),
        "{}".format(_format_number(number)),
        "{}".format("{:.1f}".format(number)),
    ]
    return _unique_preserve_order([token.lower().replace(" ", "") for token in tokens if token])


def feature_tag_from_spec(spec: Dict[str, Any]) -> str:
    mode = str(spec.get("mode") or "legacy_combo").lower().strip()

    if mode == "grid_family":
        if spec.get("name"):
            return _safe_name(spec["name"])[:120]
        parts = [str(spec.get("feature_family") or "grid")]
        if spec.get("socs"):
            parts.append("SOCx{}".format(len(spec["socs"])))
        if spec.get("widths_ms"):
            parts.append("Wx{}".format(len(spec["widths_ms"])))
        if spec.get("c_rates"):
            parts.append("Cx{}".format(len(spec["c_rates"])))
        return _safe_name("__".join(parts))[:120]

    parts = []
    if spec.get("name"):
        parts.append(str(spec["name"]))
    if spec.get("combo"):
        parts.append("combo-{}".format(spec["combo"]))
    if spec.get("include_groups"):
        parts.append("inc-" + "_".join(spec["include_groups"]))
    if spec.get("exclude_groups"):
        parts.append("exc-" + "_".join(spec["exclude_groups"]))
    if spec.get("keep_regex"):
        parts.append("keep")
    if spec.get("add_features"):
        parts.append("add{}".format(len(spec["add_features"])))
    if spec.get("drop_features"):
        parts.append("drop{}".format(len(spec["drop_features"])))
    if not parts:
        return "features"
    return _safe_name("__".join(parts))[:120]


BASE_DIR = Path(__file__).resolve().parent
ENV_DATA_ROOT = os.environ.get("DATA_ROOT", "").strip()
ENV_OUT_DIR = os.environ.get("OUT_DIR", "").strip()

CONFIG = dict(
    data_root=ENV_DATA_ROOT if ENV_DATA_ROOT else None,
    material="20Ah LFP",
    width_ms=5000,
    soc=50,
    label_col="SOH",
    feature_spec=dict(
        mode="legacy_combo",
        name="Hyst_Only",
        combo="Hyst_Only",
        include_groups=None,
        exclude_groups=None,
        add_features=None,
        drop_features=None,
        keep_regex=None,
    ),
    test_size=0.2,
    seeds=list(range(100)),
    models=["linear", "ridge", "lasso", "en", "svm", "rf", "xgb", "gpr", "mlp", "transformer", "informer"],
    standardize=True,
    perm_repeats=10,
    topk_featimp=50,
    torch_device="auto",
    torch_epochs=200,
    torch_batch_size=64,
    torch_lr=1e-3,
    torch_weight_decay=1e-6,
    torch_patience=30,
    torch_seed_offset=12345,
    out_dir=ENV_OUT_DIR if ENV_OUT_DIR else str((BASE_DIR / "Results").resolve()),
    show_progress=True,
)


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for _ in range(30):
        if (current / "Fts").is_dir() and (current / "Model").is_dir():
            return current
        if (current / "Fts").is_dir():
            return current
        current = current.parent
    return start.resolve().parent


def resolve_data_root(cfg: Dict[str, Any], project_root: Path) -> Path:
    if cfg.get("data_root"):
        candidate = Path(cfg["data_root"]).expanduser()
        candidate = (project_root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if not candidate.is_dir():
            raise FileNotFoundError("CONFIG['data_root'] is not a directory: {}".format(candidate))
        return candidate

    candidates = [
        project_root / "Fts" / "Fts-For-Model",
        project_root / "Fts" / "Fts_For_Model",
        project_root / "Fts-For-Model",
        project_root / "Fts_For_Model",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    tried = "\n  - " + "\n  - ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(
        "Cannot auto-detect data_root.\n"
        "Please set CONFIG['data_root'] to the absolute path of your Fts-For-Model folder.\n"
        "Tried:{}".format(tried)
    )


def resolve_run_dir(cfg: Dict[str, Any], run_tag: str) -> Path:
    out_dir = Path(cfg["out_dir"]).expanduser()
    out_dir = (script_dir() / out_dir).resolve() if not out_dir.is_absolute() else out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def find_xlsx(data_root: Path, material: str, width_ms: int) -> Path:
    material_dir = data_root / material
    pattern = "*_W_{}.xlsx".format(int(width_ms))

    if material_dir.is_dir():
        hits = sorted(material_dir.glob(pattern))
        if hits:
            return sorted(hits, key=lambda path: (len(path.name), path.name))[0]

    hits = []
    for path in data_root.rglob(pattern):
        if material in path.parts:
            hits.append(path)
    if hits:
        return sorted(hits, key=lambda path: (len(path.as_posix()), path.as_posix()))[0]

    raise FileNotFoundError(
        "Cannot find any '{}' for material='{}' under data_root='{}'.\n"
        "Expected example: {}/{}/{}".format(pattern, material, data_root, data_root, material, pattern)
    )


def detect_soc_sheet(xlsx_path: Path, soc: int) -> str:
    import openpyxl  # noqa

    workbook = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    sheet_names = workbook.sheetnames
    workbook.close()

    targets = [
        "SOC{}".format(int(soc)),
        "SOC_{}".format(int(soc)),
        "soc{}".format(int(soc)),
        "soc_{}".format(int(soc)),
        str(int(soc)),
    ]
    for target in targets:
        if target in sheet_names:
            return target
    lower_map = dict((sheet_name.lower(), sheet_name) for sheet_name in sheet_names)
    for target in targets:
        if target.lower() in lower_map:
            return lower_map[target.lower()]
    raise KeyError(
        "Cannot find SOC sheet for soc={} in '{}'. Available: {}".format(soc, xlsx_path.name, sheet_names[:30])
    )


def detect_label_col(df: pd.DataFrame, hint: str = "SOH") -> str:
    columns = list(df.columns)
    if hint in columns:
        return hint
    lower_map = dict((str(col).lower(), col) for col in columns)
    if hint.lower() in lower_map:
        return lower_map[hint.lower()]
    for candidate in ["soh", "label", "y", "target"]:
        if candidate in lower_map:
            return lower_map[candidate]
    raise KeyError("Cannot detect label column. Columns: {}".format(columns[:30]))


def map_feature_list(df: pd.DataFrame, feature_list: Sequence[str]) -> List[str]:
    columns = list(df.columns)
    lower_map = dict((str(col).lower(), col) for col in columns)
    mapped = []
    missing = []
    for feature in feature_list:
        if feature in columns:
            mapped.append(feature)
        elif str(feature).lower() in lower_map:
            mapped.append(lower_map[str(feature).lower()])
        else:
            missing.append(feature)
    if missing:
        raise KeyError(
            "Some requested features are missing.\n"
            "Missing examples: {}\n"
            "Available preview: {}".format(
                missing[:20] if len(missing) <= 20 else missing[:20] + ["..."],
                columns[:40] if len(columns) <= 40 else columns[:40] + ["..."],
            )
        )
    return mapped


def to_numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df[list(columns)].copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def resolve_grid_family_aliases(spec: Dict[str, Any]) -> List[str]:
    aliases = spec.get("family_aliases") or []
    aliases = [str(alias).strip() for alias in aliases if str(alias).strip()]
    family = str(spec.get("feature_family") or "").strip()
    if family:
        aliases = [family] + aliases
    defaults = DEFAULT_GRID_FAMILY_ALIASES.get(family.lower(), [])
    aliases.extend(defaults)
    aliases = _unique_preserve_order(aliases)
    if not aliases:
        raise ValueError("Grid feature_spec requires 'feature_family' or 'family_aliases'.")
    return aliases


def column_matches_family(column_name: Any, aliases: Sequence[str]) -> bool:
    raw = str(column_name).lower().replace(" ", "")
    compact = _compact_token(column_name)
    for alias in aliases:
        alias_raw = str(alias).lower().replace(" ", "")
        alias_compact = _compact_token(alias)
        if alias_raw and alias_raw in raw:
            return True
        if alias_compact and alias_compact in compact:
            return True
    return False


def column_matches_rate(column_name: Any, rate_label: Any) -> bool:
    raw = str(column_name).lower().replace(" ", "")
    target = normalize_rate_label(rate_label).lower()
    for match in RATE_TOKEN_RE.finditer(raw):
        matched = normalize_rate_label(match.group(1)).lower()
        if matched == target:
            return True
    return False


def select_grid_feature_columns(
    columns: Sequence[Any],
    aliases: Sequence[str],
    c_rates: Sequence[str],
) -> List[str]:
    matched = []
    for column in columns:
        column_name = str(column)
        if not column_matches_family(column_name, aliases):
            continue
        if c_rates and not any(column_matches_rate(column_name, rate) for rate in c_rates):
            continue
        matched.append(column_name)
    return _unique_preserve_order(matched)


def canonicalize_grid_feature_name(column_name: Any, feature_family: str) -> str:
    name = str(column_name)
    family = str(feature_family or "").strip().lower()
    if family == "fai_irrev":
        name = re.sub(r"(?i)hyst[_ ]?m3", "fai_irrev", name)
        return re.sub(r"(?i)^fai(?=_)", "fai_irrev", name)
    return name


def sanitize_feature_spec_for_output(spec: Dict[str, Any]) -> Dict[str, Any]:
    public_spec = dict(spec)
    # Keep raw alias expansion as an internal runtime detail so exported config
    # and summaries consistently reflect the public experiment naming.
    public_spec.pop("family_aliases_resolved", None)
    return public_spec


def resolve_feature_spec(cfg: Dict[str, Any]) -> Dict[str, Any]:
    spec = cfg.get("feature_spec")
    if spec is None or not isinstance(spec, dict):
        raise ValueError("CONFIG['feature_spec'] must be a dict.")

    resolved = dict(spec)
    mode = str(spec.get("mode") or "").strip().lower()
    if not mode:
        if spec.get("feature_family") or spec.get("family_aliases") or spec.get("widths_ms") or spec.get("socs") or spec.get("c_rates"):
            mode = "grid_family"
        else:
            mode = "legacy_combo"
    resolved["mode"] = mode

    if mode == "grid_family":
        resolved["feature_family"] = str(spec.get("feature_family") or spec.get("family") or "").strip()
        resolved["family_aliases_resolved"] = resolve_grid_family_aliases(resolved)
        resolved["widths_ms"] = [int(value) for value in spec.get("widths_ms", [])]
        resolved["socs"] = [int(value) for value in spec.get("socs", [])]
        resolved["c_rates"] = [normalize_rate_label(value) for value in spec.get("c_rates", [])]
        resolved["merge_keys"] = [str(key) for key in spec.get("merge_keys", ["Qn", "Q"])]
        if not resolved["widths_ms"]:
            raise ValueError("Grid feature_spec requires non-empty 'widths_ms'.")
        if not resolved["socs"]:
            raise ValueError("Grid feature_spec requires non-empty 'socs'.")
        if not resolved["c_rates"]:
            raise ValueError("Grid feature_spec requires non-empty 'c_rates'.")
        resolved["n_widths"] = len(resolved["widths_ms"])
        resolved["n_socs"] = len(resolved["socs"])
        resolved["n_c_rates"] = len(resolved["c_rates"])
        return resolved

    feats = build_feature_list(
        combo=spec.get("combo"),
        include_groups=spec.get("include_groups"),
        exclude_groups=spec.get("exclude_groups"),
        add_features=spec.get("add_features"),
        drop_features=spec.get("drop_features"),
        keep_regex=spec.get("keep_regex"),
    )
    if not feats:
        raise ValueError("Feature spec resolved to an empty feature list.")
    resolved["feature_list_requested"] = feats
    resolved["n_features_requested"] = len(feats)
    resolved["meta_columns"] = [str(col) for col in spec.get("meta_columns", ["Qn", "Q", "SOC"])]
    return resolved


def load_xy_single(
    xlsx_path: Path,
    soc: int,
    label_hint: str,
    feature_list: Sequence[str],
    meta_columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.Series, List[str], str, str, pd.DataFrame]:
    sheet_name = detect_soc_sheet(xlsx_path, soc)
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")

    label_col = detect_label_col(df, label_hint)
    mapped_features = map_feature_list(df, feature_list)
    Xdf = to_numeric_frame(df, mapped_features)
    y = pd.to_numeric(df[label_col], errors="coerce")

    meta_cols = [col for col in meta_columns if col in df.columns and col not in mapped_features and col != label_col]
    meta_df = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

    mask = (~y.isna()) & (~Xdf.isna().any(axis=1))
    Xdf = Xdf.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    meta_df = meta_df.loc[mask].reset_index(drop=True)

    return Xdf, y, mapped_features, sheet_name, label_col, meta_df


def load_xy_grid(
    data_root: Path,
    material: str,
    label_hint: str,
    spec: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.Series, List[str], str, pd.DataFrame, List[Dict[str, Any]]]:
    aliases = list(spec["family_aliases_resolved"])
    feature_family = str(spec.get("feature_family") or "").strip()
    widths_ms = [int(value) for value in spec["widths_ms"]]
    socs = [int(value) for value in spec["socs"]]
    c_rates = [normalize_rate_label(value) for value in spec["c_rates"]]
    merge_keys = [str(key) for key in spec.get("merge_keys", ["Qn", "Q"])]

    merged = None
    label_col_ref = None
    feature_cols = []
    block_summaries = []

    for width_ms in widths_ms:
        xlsx_path = find_xlsx(data_root, material, width_ms)
        for soc in socs:
            sheet_name = detect_soc_sheet(xlsx_path, soc)
            df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")

            label_col = detect_label_col(df, label_hint)
            missing_keys = [key for key in merge_keys if key not in df.columns]
            if missing_keys:
                raise KeyError(
                    "Grid merge keys {} are missing in '{}' sheet '{}'.".format(missing_keys, xlsx_path.name, sheet_name)
                )

            matched = select_grid_feature_columns(df.columns, aliases=aliases, c_rates=c_rates)
            if not matched:
                raise KeyError(
                    "No grid-family features matched for family_aliases={} and c_rates={} in '{}' sheet '{}'.".format(
                        aliases, c_rates, xlsx_path.name, sheet_name
                    )
                )

            label_block = df[merge_keys + [label_col]].copy()
            label_block[label_col] = pd.to_numeric(label_block[label_col], errors="coerce")

            if merged is None:
                merged = label_block.copy()
                label_col_ref = label_col
            else:
                check_block = label_block.rename(columns={label_col: "__label_check__"})
                merged = merged.merge(check_block, on=merge_keys, how="inner")
                valid_mask = (~merged[label_col_ref].isna()) & (~merged["__label_check__"].isna())
                if valid_mask.any():
                    max_diff = float(np.max(np.abs(merged.loc[valid_mask, label_col_ref] - merged.loc[valid_mask, "__label_check__"])))
                    if max_diff > 1e-8:
                        warnings.warn(
                            "Label mismatch detected while merging width={} / soc={}; max abs diff={:.6g}".format(
                                width_ms, soc, max_diff
                            )
                        )
                merged = merged.drop(columns=["__label_check__"])

            rename_map = dict(
                (
                    column,
                    "{}__W{}__SOC{}".format(
                        canonicalize_grid_feature_name(column, feature_family),
                        int(width_ms),
                        int(soc),
                    ),
                )
                for column in matched
            )
            block_features = df[merge_keys + matched].copy().rename(columns=rename_map)
            for renamed in rename_map.values():
                block_features[renamed] = pd.to_numeric(block_features[renamed], errors="coerce")

            merged = merged.merge(block_features, on=merge_keys, how="inner")
            feature_cols.extend(list(rename_map.values()))
            block_summaries.append(
                dict(
                    width_ms=int(width_ms),
                    soc=int(soc),
                    xlsx=str(xlsx_path),
                    sheet=sheet_name,
                    n_features=int(len(matched)),
                    matched_feature_preview=[canonicalize_grid_feature_name(column, feature_family) for column in matched[:10]],
                )
            )

    if merged is None or label_col_ref is None:
        raise RuntimeError("Grid feature loading produced no merged data.")

    Xdf = merged[feature_cols].copy()
    y = pd.to_numeric(merged[label_col_ref], errors="coerce")
    meta_df = merged[merge_keys].copy()

    mask = (~y.isna()) & (~Xdf.isna().any(axis=1))
    Xdf = Xdf.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    meta_df = meta_df.loc[mask].reset_index(drop=True)

    return Xdf, y, feature_cols, label_col_ref, meta_df, block_summaries


def metric_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def metric_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def metric_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def summarize_metrics(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    for model_name, group in df.groupby("model"):
        row = {"model": model_name}
        for column in columns:
            row["{}_median".format(column)] = float(np.median(group[column].values))
            row["{}_std".format(column)] = float(np.std(group[column].values, ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def native_importance(estimator: Any, model_name: str, feature_names: Sequence[str]) -> Optional[pd.DataFrame]:
    model_key = model_name.lower()
    if model_key in ["linear", "ridge", "lasso", "en", "elasticnet"] and hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_).reshape(-1)
        df = pd.DataFrame({"feature": feature_names, "importance": np.abs(coef), "signed": coef})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
    if model_key == "rf" and hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_).reshape(-1)
        df = pd.DataFrame({"feature": feature_names, "importance": importance})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
    if model_key == "xgb":
        try:
            booster = estimator.get_booster()
            scores = booster.get_score(importance_type="gain")
            importance = np.zeros(len(feature_names), dtype=float)
            for key, value in scores.items():
                match = re.match(r"f(\d+)", key)
                if match:
                    index = int(match.group(1))
                    if 0 <= index < len(importance):
                        importance[index] = float(value)
            df = pd.DataFrame({"feature": feature_names, "importance": importance})
            return df.sort_values("importance", ascending=False).reset_index(drop=True)
        except Exception:
            return None
    return None


def permutation_importance_df(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:
    def _neg_mae(estimator: Any, X_values: np.ndarray, y_values: np.ndarray) -> float:
        pred = estimator.predict(X_values)
        return -metric_mae(y_values, pred)

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=_neg_mae,
        n_repeats=int(n_repeats),
        random_state=int(seed),
        n_jobs=CPU_NJOBS,
    )
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def summarize_featimp(all_seed_imp: Sequence[pd.DataFrame], topk: Optional[int]) -> pd.DataFrame:
    pieces = []
    for seed_index, frame in enumerate(all_seed_imp):
        block = frame[["feature", "importance"]].copy()
        block["seed_idx"] = seed_index
        pieces.append(block)
    all_df = pd.concat(pieces, axis=0, ignore_index=True)
    agg = all_df.groupby("feature")["importance"].agg(["median", "std", "mean"]).reset_index()
    agg = agg.sort_values("median", ascending=False).reset_index(drop=True)
    agg = agg.rename(
        columns={
            "median": "importance_median",
            "std": "importance_std",
            "mean": "importance_mean",
        }
    )
    if topk is not None:
        agg = agg.head(int(topk)).reset_index(drop=True)
    return agg


def _resolve_torch_device(cfg: Dict[str, Any]) -> str:
    mode = str(cfg.get("torch_device", "auto")).lower()
    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        return "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    return "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"


class _TabTransformerNet(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4, num_layers: int = 3, dim_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_features, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = x.unsqueeze(-1)
        hidden = self.value_proj(x) + self.pos_emb
        hidden = self.encoder(hidden)
        hidden = hidden.mean(dim=1)
        return self.head(hidden).squeeze(-1)


class _InformerLiteNet(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4, num_layers: int = 4, dim_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_features, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = x.unsqueeze(-1)
        hidden = self.value_proj(x) + self.pos_emb
        hidden = self.encoder(hidden)
        hidden = hidden.mean(dim=1)
        return self.head(hidden).squeeze(-1)


class TorchRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, net: nn.Module, cfg: Dict[str, Any], seed: int):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch not available.")

        self.cfg = cfg
        self.seed = int(seed)
        self.device = _resolve_torch_device(cfg)
        self.epochs = int(cfg.get("torch_epochs", 200))
        self.batch_size = int(cfg.get("torch_batch_size", 64))
        self.lr = float(cfg.get("torch_lr", 1e-3))
        self.weight_decay = float(cfg.get("torch_weight_decay", 1e-6))
        self.patience = int(cfg.get("torch_patience", 30))
        self.seed_offset = int(cfg.get("torch_seed_offset", 12345))
        self.net = net.to(self.device)
        self.is_fitted_ = False
        self.n_features_in_ = None
        try:
            torch.set_num_threads(int(CPU_NJOBS))
            torch.set_num_interop_threads(int(min(4, CPU_NJOBS)))
        except Exception:
            pass

    def _set_seed(self) -> None:
        seed_value = int(self.seed + self.seed_offset)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchRegressor":
        self._set_seed()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        self.n_features_in_ = int(X.shape[1])

        n = int(len(y))
        indices = np.arange(n)
        rng = np.random.RandomState(self.seed)
        rng.shuffle(indices)
        n_val = max(1, int(0.15 * n))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        ds_train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        ds_val = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        pin_memory = self.device == "cuda"
        dl_train = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=0, pin_memory=pin_memory)
        dl_val = DataLoader(ds_val, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=pin_memory)

        optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        bad_epochs = 0

        for _ in range(self.epochs):
            self.net.train()
            for xb, yb in dl_train:
                xb = xb.to(self.device, non_blocking=pin_memory)
                yb = yb.to(self.device, non_blocking=pin_memory).view(-1)
                optimizer.zero_grad(set_to_none=True)
                pred = self.net(xb).view(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

            self.net.eval()
            with torch.no_grad():
                losses = []
                for xb, yb in dl_val:
                    xb = xb.to(self.device, non_blocking=pin_memory)
                    yb = yb.to(self.device, non_blocking=pin_memory).view(-1)
                    pred = self.net(xb).view(-1)
                    losses.append(loss_fn(pred, yb).item())
                val_loss = float(np.mean(losses)) if losses else float("inf")

            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_state = dict((key, value.detach().cpu().clone()) for key, value in self.net.state_dict().items())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("TorchRegressor not fit yet.")

        X = np.asarray(X, dtype=np.float32)
        ds = TensorDataset(torch.from_numpy(X))
        pin_memory = self.device == "cuda"
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=pin_memory)

        self.net.eval()
        outputs = []
        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(self.device, non_blocking=pin_memory)
                pred = self.net(xb).view(-1).detach().cpu().numpy()
                outputs.append(pred)
        return np.concatenate(outputs, axis=0)


def build_model(name: str, random_state: int, n_features: int, cfg: Dict[str, Any]) -> Any:
    name = name.lower().strip()

    if name == "linear":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=1.0, random_state=random_state)
    if name == "lasso":
        return Lasso(alpha=1e-3, random_state=random_state, max_iter=20000)
    if name in ["en", "elasticnet"]:
        return ElasticNet(alpha=1e-3, l1_ratio=0.5, random_state=random_state, max_iter=20000)
    if name in ["svm", "svr"]:
        return SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.01)
    if name == "rf":
        return RandomForestRegressor(n_estimators=600, random_state=random_state, n_jobs=CPU_NJOBS, min_samples_leaf=1)
    if name == "gpr":
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-5)
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=random_state)
    if name == "mlp":
        validation_fraction = 0.15
        early_stopping = True
        train_size_hint = cfg.get("_train_size_hint")
        min_train_for_validation = int(math.ceil(2.0 / validation_fraction))
        if train_size_hint is not None and int(train_size_hint) < min_train_for_validation:
            early_stopping = False
        return MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-5,
            batch_size=64,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=5000,
            random_state=random_state,
            early_stopping=early_stopping,
            n_iter_no_change=50,
            validation_fraction=validation_fraction,
        )
    if name == "xgb":
        try:
            import xgboost as xgb  # noqa
        except Exception as exc:
            raise ImportError("xgboost not installed. Install xgboost or remove 'xgb' from models.") from exc
        return xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=CPU_NJOBS,
            objective="reg:squarederror",
        )
    if name == "transformer":
        if not _TORCH_AVAILABLE:
            raise ImportError("torch not installed; cannot run transformer.")
        net = _TabTransformerNet(n_features=n_features, d_model=64, nhead=4, num_layers=3, dim_ff=128, dropout=0.1)
        return TorchRegressor(net=net, cfg=cfg, seed=random_state)
    if name == "informer":
        if not _TORCH_AVAILABLE:
            raise ImportError("torch not installed; cannot run informer.")
        net = _InformerLiteNet(n_features=n_features, d_model=64, nhead=4, num_layers=4, dim_ff=256, dropout=0.1)
        return TorchRegressor(net=net, cfg=cfg, seed=random_state)
    raise ValueError("Unknown model: {}".format(name))


def prog(message: str, enabled: bool = True) -> None:
    if enabled:
        print(message, flush=True)


def load_dataset(
    cfg: Dict[str, Any],
    data_root: Path,
    feature_spec: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.Series, List[str], str, pd.DataFrame, Dict[str, Any]]:
    mode = str(feature_spec.get("mode") or "legacy_combo")
    material = str(cfg["material"])
    label_hint = str(cfg.get("label_col", "SOH"))

    if mode == "grid_family":
        Xdf, yser, feature_cols, label_col, meta_df, block_summaries = load_xy_grid(
            data_root=data_root,
            material=material,
            label_hint=label_hint,
            spec=feature_spec,
        )
        data_info = dict(
            mode="grid_family",
            block_count=len(block_summaries),
            blocks=block_summaries,
            widths_ms=[int(value) for value in feature_spec["widths_ms"]],
            socs=[int(value) for value in feature_spec["socs"]],
            c_rates=list(feature_spec["c_rates"]),
            merge_keys=list(feature_spec["merge_keys"]),
        )
        return Xdf, yser, feature_cols, label_col, meta_df, data_info

    width_ms = int(cfg["width_ms"])
    soc = int(cfg["soc"])
    xlsx_path = find_xlsx(data_root, material, width_ms)
    Xdf, yser, feature_cols, sheet_name, label_col, meta_df = load_xy_single(
        xlsx_path=xlsx_path,
        soc=soc,
        label_hint=label_hint,
        feature_list=feature_spec["feature_list_requested"],
        meta_columns=feature_spec.get("meta_columns", ["Qn", "Q", "SOC"]),
    )
    data_info = dict(
        mode="legacy_combo",
        block_count=1,
        blocks=[
            dict(
                width_ms=int(width_ms),
                soc=int(soc),
                xlsx=str(xlsx_path),
                sheet=sheet_name,
                n_features=int(len(feature_cols)),
            )
        ],
        widths_ms=[int(width_ms)],
        socs=[int(soc)],
        c_rates=[],
        merge_keys=[col for col in meta_df.columns],
    )
    return Xdf, yser, feature_cols, label_col, meta_df, data_info


def _case_name_from_cfg(cfg: Dict[str, Any], feature_spec: Dict[str, Any]) -> str:
    if cfg.get("case_name"):
        return str(cfg["case_name"])
    if feature_spec.get("name"):
        return str(feature_spec["name"])
    return feature_tag_from_spec(feature_spec)


def run(cfg: Dict[str, Any]) -> Path:
    this_dir = script_dir()
    project_root = find_project_root(this_dir)
    data_root = resolve_data_root(cfg, project_root)
    show_progress = bool(cfg.get("show_progress", True))
    material = str(cfg["material"])

    feature_spec_resolved = resolve_feature_spec(cfg)
    feature_spec_public = sanitize_feature_spec_for_output(feature_spec_resolved)
    feat_tag = feature_tag_from_spec(feature_spec_resolved)
    case_name = _case_name_from_cfg(cfg, feature_spec_resolved)
    mode = str(feature_spec_resolved.get("mode") or "legacy_combo")

    if mode == "grid_family":
        run_tag = "{}__{}__{}".format(_now_ts(), _safe_name(material), _safe_name(case_name))
    else:
        run_tag = "{}__{}__W{}__SOC{}__{}".format(
            _now_ts(),
            _safe_name(material),
            int(cfg["width_ms"]),
            int(cfg["soc"]),
            _safe_name(case_name),
        )

    run_dir = resolve_run_dir(cfg, run_tag)

    cfg_dump = dict(cfg)
    cfg_dump["project_root_resolved"] = str(project_root)
    cfg_dump["data_root_resolved"] = str(data_root)
    cfg_dump["feature_spec_resolved"] = feature_spec_public
    (run_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2, ensure_ascii=False), encoding="utf-8")

    Xdf, yser, feature_cols, label_col, meta_df, data_info = load_dataset(
        cfg=cfg,
        data_root=data_root,
        feature_spec=feature_spec_resolved,
    )

    X = Xdf.values.astype(float)
    y = yser.values.astype(float)
    sample_indices = np.arange(len(y))

    data_summary = dict(
        material=material,
        label_col=label_col,
        n_samples=int(len(yser)),
        n_features=int(len(feature_cols)),
        features=feature_cols,
        feature_spec=feature_spec_public,
        data_mode=data_info["mode"],
        block_count=int(data_info["block_count"]),
        blocks=data_info["blocks"],
        merge_keys=data_info["merge_keys"],
        y_min=float(np.min(y)) if len(y) else None,
        y_max=float(np.max(y)) if len(y) else None,
        y_mean=float(np.mean(y)) if len(y) else None,
        y_std=float(np.std(y)) if len(y) else None,
        torch_available=_TORCH_AVAILABLE,
    )
    (run_dir / "Exp_summary.json").write_text(json.dumps(data_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_case_status(status: str, reason: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = dict(
            case_name=case_name,
            feature_tag=feat_tag,
            feature_name=feature_spec_resolved.get("name", ""),
            material=material,
            status=str(status),
            reason=str(reason),
            run_dir=str(run_dir),
        )
        if extra:
            payload.update(extra)
        (run_dir / "case_status.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        summary_payload = dict(data_summary)
        summary_payload["case_status"] = str(status)
        summary_payload["case_status_reason"] = str(reason)
        if extra:
            summary_payload.update(extra)
        (run_dir / "Exp_summary.json").write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if len(y) < 2:
        reason = "Insufficient usable samples after filtering and merge: n_samples={}".format(len(y))
        _write_case_status(
            "unsupported_insufficient_samples",
            reason,
            extra=dict(n_samples=int(len(y)), n_features=int(X.shape[1])),
        )
        prog("[SKIP] {}".format(reason), show_progress)
        raise CaseSkippedError("unsupported_insufficient_samples", reason, run_dir=run_dir)

    seeds = list(cfg["seeds"])
    models = [str(model).lower().strip() for model in cfg["models"]]
    test_size = float(cfg["test_size"])
    standardize = bool(cfg.get("standardize", True))
    perm_repeats = int(cfg.get("perm_repeats", 10))
    topk_featimp = cfg.get("topk_featimp", 50)

    prog("[INFO] Data mode: {} | material={} | N={} | d={}".format(data_info["mode"], material, len(y), X.shape[1]), show_progress)
    prog("[INFO] FeatureSpec: {}".format(feature_spec_resolved), show_progress)
    prog("[INFO] Output dir: {}".format(run_dir), show_progress)
    if ("transformer" in models or "informer" in models) and (not _TORCH_AVAILABLE):
        prog("[WARN] torch not available: transformer/informer will be skipped.", show_progress)

    metrics_rows = []
    prediction_rows = []
    featimp_rows = []
    featimp_summary_rows = []
    model_failure_rows = []

    total_jobs = len(models) * len(seeds)
    done_jobs = 0

    for model_name in models:
        try:
            _ = build_model(model_name, random_state=0, n_features=X.shape[1], cfg=cfg)
        except Exception as exc:
            msg = "{}".format(exc)
            warnings.warn("[SKIP] {}: {}".format(model_name, msg))
            prog("[SKIP] {}: {}".format(model_name, msg), show_progress)
            model_failure_rows.append(
                dict(
                    case_name=case_name,
                    feature_tag=feat_tag,
                    feature_name=feature_spec_resolved.get("name", ""),
                    model=model_name,
                    seed="",
                    stage="build_model_precheck",
                    error_type=type(exc).__name__,
                    error_message=msg,
                    traceback="",
                )
            )
            continue

        featimp_by_seed = []

        for seed in seeds:
            done_jobs += 1
            prog("[{}/{}] Train/Test -> model={} | seed={}".format(done_jobs, total_jobs, model_name, seed), show_progress)
            try:
                idx_train, idx_test = train_test_split(sample_indices, test_size=test_size, random_state=seed, shuffle=True)
                X_train = X[idx_train]
                X_test = X[idx_test]
                y_train = y[idx_train]
                y_test = y[idx_test]

                cfg_for_model = dict(cfg)
                cfg_for_model["_train_size_hint"] = int(len(y_train))
                estimator = build_model(model_name, random_state=seed, n_features=X.shape[1], cfg=cfg_for_model)
                need_scaler = standardize and (model_name in ["linear", "ridge", "lasso", "en", "elasticnet", "svm", "svr", "gpr", "mlp", "transformer", "informer"])
                model = Pipeline([("scaler", StandardScaler()), ("est", estimator)]) if need_scaler and model_name not in ["rf", "xgb"] else estimator

                t0 = time.perf_counter()
                model.fit(X_train, y_train)
                t1 = time.perf_counter()

                y_pred_train = model.predict(X_train)

                t2 = time.perf_counter()
                y_pred_test = model.predict(X_test)
                t3 = time.perf_counter()

                fit_time_s = float(t1 - t0)
                pred_time_test_s = float(t3 - t2)
                fit_ms_per_train_sample = fit_time_s / max(len(y_train), 1) * 1000.0
                pred_us_per_test_sample = pred_time_test_s / max(len(y_test), 1) * 1e6

                metrics_rows.append(
                    dict(
                        case_name=case_name,
                        feature_tag=feat_tag,
                        feature_name=feature_spec_resolved.get("name", ""),
                        model=model_name,
                        seed=seed,
                        n_train=len(y_train),
                        n_test=len(y_test),
                        mae_train=metric_mae(y_train, y_pred_train),
                        rmse_train=metric_rmse(y_train, y_pred_train),
                        mape_train=metric_mape(y_train, y_pred_train),
                        mae_test=metric_mae(y_test, y_pred_test),
                        rmse_test=metric_rmse(y_test, y_pred_test),
                        mape_test=metric_mape(y_test, y_pred_test),
                        fit_time_s=fit_time_s,
                        pred_time_test_s=pred_time_test_s,
                        fit_ms_per_train_sample=fit_ms_per_train_sample,
                        pred_us_per_test_sample=pred_us_per_test_sample,
                    )
                )

                def _pred_df(split: str, idx_values: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
                    base = meta_df.iloc[idx_values].reset_index(drop=True).copy()
                    pred_df = pd.DataFrame(
                        {
                            "sample_idx": idx_values,
                            "case_name": case_name,
                            "feature_tag": feat_tag,
                            "feature_name": feature_spec_resolved.get("name", ""),
                            "model": model_name,
                            "seed": seed,
                            "split": split,
                            "y_true": np.asarray(y_true).reshape(-1),
                            "y_pred": np.asarray(y_pred).reshape(-1),
                        }
                    )
                    pred_df["err"] = pred_df["y_pred"] - pred_df["y_true"]
                    pred_df["abs_err"] = np.abs(pred_df["err"])
                    pred_df["ape_%"] = pred_df["abs_err"] / np.maximum(np.abs(pred_df["y_true"]), 1e-9) * 100.0
                    if not base.empty:
                        pred_df = pd.concat([base, pred_df], axis=1)
                    return pred_df

                prediction_rows.append(_pred_df("train", idx_train, y_train, y_pred_train))
                prediction_rows.append(_pred_df("test", idx_test, y_test, y_pred_test))

                core_estimator = model.named_steps["est"] if hasattr(model, "named_steps") else model
                featimp_df = native_importance(core_estimator, model_name, feature_cols)
                if featimp_df is None:
                    featimp_df = permutation_importance_df(model, X_test, y_test, feature_cols, perm_repeats, seed)

                featimp_df["case_name"] = case_name
                featimp_df["feature_tag"] = feat_tag
                featimp_df["feature_name"] = feature_spec_resolved.get("name", "")
                featimp_df["model"] = model_name
                featimp_df["seed"] = seed
                featimp_rows.append(featimp_df.copy())
                featimp_by_seed.append(featimp_df.copy())

                prog(
                    "        test: MAE={:.6g} | RMSE={:.6g} | MAPE={:.4g}% | fit={:.3f} ms/sample | pred={:.1f} us/sample".format(
                        metrics_rows[-1]["mae_test"],
                        metrics_rows[-1]["rmse_test"],
                        metrics_rows[-1]["mape_test"],
                        metrics_rows[-1]["fit_ms_per_train_sample"],
                        metrics_rows[-1]["pred_us_per_test_sample"],
                    ),
                    show_progress,
                )
            except Exception as exc:
                msg = "{}".format(exc)
                model_failure_rows.append(
                    dict(
                        case_name=case_name,
                        feature_tag=feat_tag,
                        feature_name=feature_spec_resolved.get("name", ""),
                        model=model_name,
                        seed=seed,
                        stage="fit_or_predict",
                        error_type=type(exc).__name__,
                        error_message=msg,
                        traceback=traceback.format_exc(),
                    )
                )
                prog("        [MODEL-FAIL] {} seed={} -> {}".format(model_name, seed, msg), show_progress)
                continue

        if featimp_by_seed:
            summary_df = summarize_featimp(featimp_by_seed, topk=topk_featimp)
            summary_df.insert(0, "case_name", case_name)
            summary_df.insert(1, "feature_tag", feat_tag)
            summary_df.insert(2, "feature_name", feature_spec_resolved.get("name", ""))
            summary_df.insert(3, "model", model_name)
            featimp_summary_rows.append(summary_df)

    if model_failure_rows:
        pd.DataFrame(model_failure_rows).to_csv(run_dir / "model_failures.csv", index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        reason = "No model completed successfully for this case."
        _write_case_status(
            "all_models_failed",
            reason,
            extra=dict(model_failure_count=int(len(model_failure_rows)), n_samples=int(len(y)), n_features=int(X.shape[1])),
        )
        raise CaseSkippedError("all_models_failed", reason, run_dir=run_dir)

    metrics_df.to_csv(run_dir / "metrics_by_seed.csv", index=False)

    summary_columns = [
        "mae_test", "rmse_test", "mape_test",
        "mae_train", "rmse_train", "mape_train",
        "fit_time_s", "pred_time_test_s",
        "fit_ms_per_train_sample", "pred_us_per_test_sample",
    ]
    metrics_summary = summarize_metrics(metrics_df, summary_columns)
    metrics_summary.insert(0, "case_name", case_name)
    metrics_summary.insert(1, "feature_tag", feat_tag)
    metrics_summary.insert(2, "feature_name", feature_spec_resolved.get("name", ""))
    metrics_summary.to_csv(run_dir / "metrics_summary.csv", index=False)

    predictions_all = pd.concat(prediction_rows, ignore_index=True)
    predictions_all.to_csv(run_dir / "predictions_all.csv", index=False)

    featimp_all = pd.concat(featimp_rows, ignore_index=True)
    featimp_all.to_csv(run_dir / "featimp_all.csv", index=False)

    if featimp_summary_rows:
        featimp_summary_all = pd.concat(featimp_summary_rows, ignore_index=True)
    else:
        featimp_summary_all = pd.DataFrame(columns=["case_name", "feature_tag", "feature_name", "model"])
    featimp_summary_all.to_csv(run_dir / "featimp_summary_all_models.csv", index=False)

    if model_failure_rows:
        _write_case_status(
            "completed_with_model_failures",
            "At least one model/seed failed; see model_failures.csv",
            extra=dict(model_failure_count=int(len(model_failure_rows)), successful_metric_rows=int(len(metrics_rows))),
        )
    else:
        _write_case_status(
            "completed",
            "All requested model/seed runs completed.",
            extra=dict(successful_metric_rows=int(len(metrics_rows))),
        )

    prog("\n=== DONE ===", show_progress)
    prog("Run dir: {}".format(run_dir), show_progress)
    prog(
        "Saved: config.json, Exp_summary.json, predictions_all.csv, featimp_all.csv, featimp_summary_all_models.csv, metrics_by_seed.csv, metrics_summary.csv",
        show_progress,
    )

    return run_dir


def main() -> None:
    run(CONFIG)


if __name__ == "__main__":
    main()
