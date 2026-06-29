#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared helpers for PulseBat-combo Figure 5.

The plotting data are stored in an xlsx workbook. This module reads the xlsx
with Python standard-library OpenXML parsing so the panel scripts do not need
pandas or openpyxl.
"""

from __future__ import annotations

import csv
import math
import posixpath
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple
from xml.etree import ElementTree as ET
from zipfile import ZipFile


FIGURE5_ROOT = Path(__file__).resolve().parent
DATA_XLSX = FIGURE5_ROOT / "Figure5_plotting_data.xlsx"
DPI = 600

MATERIAL_ORDER: Sequence[str] = ("20Ah LFP", "35Ah LFP", "68Ah LFP")
MATERIAL_SHORT: Mapping[str, str] = {
    "20Ah LFP": "20Ah",
    "35Ah LFP": "35Ah",
    "68Ah LFP": "68Ah",
}
FEATURE_ORDER: Sequence[str] = ("Comb.-C", "Comb.-W", "Comb.-SOC")
FEATURE_COLORS: Mapping[str, str] = {
    "Single": "#9DB6CC",
    "Comb.-C": "#6BB7B2",
    "Comb.-W": "#F28E2B",
    "Comb.-SOC": "#73B66B",
}
MODEL_ORDER: Sequence[str] = ("linear", "rf", "xgb", "gpr", "mlp", "transformer", "informer")
SCORE_COLUMNS: Sequence[Tuple[str, str]] = (
    ("Accuracy_score", "Accuracy"),
    ("Stability_score", "Stability"),
    ("Inference_score", "Inference"),
    ("Fit_score", "Fit"),
)

NUMERIC_COLUMNS = {
    "mae_test_median",
    "mae_test_std",
    "combo_mae",
    "combo_mae_std",
    "single_baseline_mae",
    "mae_improvement_pct",
    "mae",
    "mae_std",
    "fit_time_s",
    "pred_time_us",
    "pred_time_s",
    "total_time_s",
    "rank",
    "rank_within_type",
    "pred_us",
    "fit_s",
    "Accuracy_score",
    "Stability_score",
    "Inference_score",
    "Fit_score",
    "Composite_score",
    "Composite_equal_weight_score",
}

_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def cm_to_in(cm: float) -> float:
    return cm / 2.54


def setup_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.75,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.unicode_minus": False,
    })


def zero_trim(value: float, _pos: int | None = None) -> str:
    if abs(value) < 1e-12:
        return "0"
    if abs(value) >= 1:
        return f"{value:.0f}"
    return f"{value:.2f}"


def _col_to_idx(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref or "")
    if match is None:
        return 0
    value = 0
    for ch in match.group(1):
        value = value * 26 + ord(ch) - 64
    return value


def _resolve_part(base_dir: str, target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join(base_dir, target))


def read_workbook_rows(path: Path = DATA_XLSX) -> Dict[str, List[List[str]]]:
    with ZipFile(path) as zf:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("main:si", _NS):
                shared_strings.append("".join(t.text or "" for t in item.findall(".//main:t", _NS)))

        workbook_root = ET.fromstring(zf.read("xl/workbook.xml"))
        rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {
            rel.attrib["Id"]: _resolve_part("xl", rel.attrib["Target"])
            for rel in rels_root
        }

        out: Dict[str, List[List[str]]] = {}
        for sheet in workbook_root.findall("main:sheets/main:sheet", _NS):
            sheet_name = sheet.attrib["name"]
            rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            sheet_root = ET.fromstring(zf.read(rid_to_target[rid]))
            rows: List[List[str]] = []
            for row in sheet_root.findall("main:sheetData/main:row", _NS):
                values: List[str] = []
                last_idx = 0
                for cell in row.findall("main:c", _NS):
                    idx = _col_to_idx(cell.attrib.get("r", ""))
                    while last_idx + 1 < idx:
                        values.append("")
                        last_idx += 1
                    cell_type = cell.attrib.get("t", "")
                    value_node = cell.find("main:v", _NS)
                    inline_node = cell.find("main:is", _NS)
                    if cell_type == "s" and value_node is not None:
                        text = shared_strings[int(value_node.text or 0)]
                    elif cell_type == "inlineStr" and inline_node is not None:
                        text = "".join(t.text or "" for t in inline_node.findall(".//main:t", _NS))
                    elif value_node is not None:
                        text = value_node.text or ""
                    else:
                        text = ""
                    values.append(text)
                    last_idx = idx
                rows.append(values)
            out[sheet_name] = rows
        return out


def _to_number(value: str) -> str | float:
    if value == "":
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    if math.isfinite(number) and abs(number - round(number)) < 1e-12:
        return int(round(number))
    return number


def _coerce_row(row: MutableMapping[str, str | float]) -> Dict[str, str | float]:
    out: Dict[str, str | float] = {}
    for key, value in row.items():
        out[key] = _to_number(str(value)) if key in NUMERIC_COLUMNS else value
    return out


def rows_to_dicts(rows: Sequence[Sequence[str]], header_index: int) -> List[Dict[str, str | float]]:
    header = [str(v).strip() for v in rows[header_index]]
    records: List[Dict[str, str | float]] = []
    for raw in rows[header_index + 1:]:
        if not any(str(v).strip() for v in raw):
            continue
        row: Dict[str, str | float] = {}
        for i, key in enumerate(header):
            if not key:
                continue
            row[key] = raw[i] if i < len(raw) else ""
        records.append(_coerce_row(row))
    return records


def find_header(rows: Sequence[Sequence[str]], required_columns: Iterable[str]) -> int:
    required = set(required_columns)
    for i, row in enumerate(rows):
        values = {str(v).strip() for v in row}
        if required.issubset(values):
            return i
    raise RuntimeError(f"Could not find header with columns: {sorted(required)}")


def load_sheet_table(sheet_name: str, path: Path = DATA_XLSX) -> List[Dict[str, str | float]]:
    rows = read_workbook_rows(path)[sheet_name]
    if sheet_name in {"5d", "5e"}:
        header_idx = find_header(rows, ["rank", "model"])
    elif sheet_name == "5c":
        header_idx = find_header(rows, ["scenario_group_type", "scenario_group_label", "mae"])
    else:
        header_idx = find_header(rows, ["plot_component", "model"])
    records = rows_to_dicts(rows, header_idx)
    return [r for r in records if str(r.get(next(iter(r.keys())), "")).strip()]


def load_figure5a_tables(path: Path = DATA_XLSX) -> Tuple[List[Dict[str, str | float]], List[Dict[str, str | float]]]:
    rows = read_workbook_rows(path)["5a"]
    violin_header = find_header(rows, ["plot_component", "mae_test_median", "mae_test_std"])
    bubble_header = find_header(rows, ["plot_component", "combo_mae", "mae_improvement_pct"])
    violin = [
        r for r in rows_to_dicts(rows, violin_header)
        if r.get("plot_component") == "violin_raw_points"
    ]
    bubble = [
        r for r in rows_to_dicts(rows, bubble_header)
        if r.get("plot_component") == "bubble_heatmap_summary"
    ]
    return violin, bubble


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        columns = keys
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
