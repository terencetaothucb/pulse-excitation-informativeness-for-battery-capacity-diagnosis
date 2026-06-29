#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from xml.etree import ElementTree as ET


ROOT = Path(r"E:\Datasets\PulseBat_combo\Figure1\Figure1c")
SELECTED_CSV = ROOT / "Figure1c_selected_samples.csv"
OUT_DIR = ROOT / "ProcessData"
TARGET_SHEET_NAME = "记录层-2"

NS_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
NS_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
NS_PACKAGE_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"

ROW_RANGES = {
    "BEGT215HKBV200162": (653011, 732412),
    "BEGT213HKSU200214": (658183, 738202),
    "BEGT201HKBJ200005": (658510, 738529),
    "1号": (655861, 735880),
    "32号": (657275, 737294),
    "56号": (656950, 736969),
    "02HCB68111211Y88D0003997": (655645, 737664),
    "02HCB68111211Y8700000791": (656629, 738648),
    "02HCB68111211Y88D0003027": (656737, 738756),
}


def load_selected_samples(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sample_key(raw_path: Path) -> str:
    stem = raw_path.stem
    if "_ID_" in stem:
        return stem.split("_ID_", 1)[1].replace("(1)", "")
    return stem.replace("(1)", "")


def safe_label(text: str) -> str:
    text = text.replace(" ", "")
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", text).strip("_")


def col_to_index(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def load_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    values: List[str] = []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    for si in root.findall(NS_MAIN + "si"):
        parts = [t.text or "" for t in si.iter(NS_MAIN + "t")]
        values.append("".join(parts))
    return values


def worksheet_path(zf: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rid_to_target = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall(NS_PACKAGE_REL + "Relationship")
    }
    sheets = workbook.find(NS_MAIN + "sheets")
    if sheets is None:
        raise RuntimeError("No sheets found in workbook")
    for sheet in sheets.findall(NS_MAIN + "sheet"):
        if sheet.attrib.get("name") == sheet_name:
            rid = sheet.attrib[NS_REL + "id"]
            target = rid_to_target[rid].lstrip("/")
            return target if target.startswith("xl/") else f"xl/{target}"
    names = [sheet.attrib.get("name", "") for sheet in sheets.findall(NS_MAIN + "sheet")]
    raise RuntimeError(f"Sheet '{sheet_name}' not found. Available: {names}")


def cell_value(cell: ET.Element, shared_strings: List[str]) -> str:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.iter(NS_MAIN + "t"))
    v = cell.find(NS_MAIN + "v")
    if v is None or v.text is None:
        return ""
    text = v.text
    if cell_type == "s":
        try:
            return shared_strings[int(text)]
        except Exception:
            return text
    return text


def row_values(row: ET.Element, width: int, shared_strings: List[str]) -> List[str]:
    values = [""] * width
    for fallback_idx, cell in enumerate(row.findall(NS_MAIN + "c")):
        ref = cell.attrib.get("r", "")
        idx = col_to_index(ref) if ref else fallback_idx
        if 0 <= idx < width:
            values[idx] = cell_value(cell, shared_strings)
    return values


def iter_rows(
    zf: zipfile.ZipFile,
    sheet_xml: str,
    wanted_rows: Iterable[int],
) -> Iterable[ET.Element]:
    wanted = set(wanted_rows)
    with zf.open(sheet_xml) as f:
        for _, row in ET.iterparse(f, events=("end",)):
            if row.tag != NS_MAIN + "row":
                continue
            r = int(row.attrib.get("r", "0"))
            if r in wanted:
                yield row
            if r > max(wanted):
                row.clear()
                break
            row.clear()


def extract_one(sample: Dict[str, str]) -> Tuple[Path, int]:
    raw_path = Path(sample["raw_path"])
    key = sample_key(raw_path)
    if key not in ROW_RANGES:
        raise RuntimeError(f"No row range configured for sample key: {key}")
    start_row, end_row = ROW_RANGES[key]
    if end_row < start_row:
        raise RuntimeError(f"Invalid row range for {key}: {start_row}-{end_row}")

    group = safe_label(sample["capacity_label"])
    level = safe_label(sample["soh_level"])
    out_path = OUT_DIR / f"{group}_{level}_{key}_rows_{start_row}-{end_row}.csv"

    with zipfile.ZipFile(raw_path) as zf:
        shared_strings = load_shared_strings(zf)
        sheet_xml = worksheet_path(zf, TARGET_SHEET_NAME)

        header_row = None
        with zf.open(sheet_xml) as f:
            for _, row in ET.iterparse(f, events=("end",)):
                if row.tag == NS_MAIN + "row" and int(row.attrib.get("r", "0")) == 1:
                    header_row = row
                    break
                if row.tag == NS_MAIN + "row":
                    row.clear()
        if header_row is None:
            raise RuntimeError(f"Header row not found: {raw_path}")

        header_cells = header_row.findall(NS_MAIN + "c")
        header_refs = [cell.attrib.get("r", "") for cell in header_cells]
        if any(header_refs):
            header_width = max(col_to_index(ref) + 1 for ref in header_refs if ref)
        else:
            header_width = len(header_cells)
        headers = row_values(header_row, header_width, shared_strings)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        count = 0
        with out_path.open("w", encoding="utf-8-sig", newline="") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(["source_excel_row", *headers])

            with zf.open(sheet_xml) as f_in:
                context = ET.iterparse(f_in, events=("end",))
                for _, row in context:
                    if row.tag != NS_MAIN + "row":
                        continue
                    row_idx = int(row.attrib.get("r", "0"))
                    if row_idx < start_row:
                        row.clear()
                        continue
                    if row_idx > end_row:
                        row.clear()
                        break
                    writer.writerow([row_idx, *row_values(row, header_width, shared_strings)])
                    count += 1
                    row.clear()

    expected = end_row - start_row + 1
    if count != expected:
        raise RuntimeError(f"{key}: expected {expected} rows, extracted {count}")
    return out_path, count


def main() -> int:
    samples = load_selected_samples(SELECTED_CSV)
    seen = set()
    for sample in samples:
        key = sample_key(Path(sample["raw_path"]))
        seen.add(key)
        out_path, count = extract_one(sample)
        print(f"[OK] {key}: {count} rows -> {out_path.name}")
    missing = set(ROW_RANGES) - seen
    if missing:
        raise RuntimeError(f"Configured row ranges not present in selected samples: {sorted(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
