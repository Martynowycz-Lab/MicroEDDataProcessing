#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gold-standard MicroED MRC/XML/MDOC to miniCBF conversion.

v8 keeps the raw MRC stack plus SerialEM/Velox sidecars as the source of truth,
then makes the XDS offset policy explicit. The production default for current
XDS is radial-reduced conditioning: target images are software-binned to at most
1024 px, edge frames are trimmed aggressively, a robust radial/plane background
is removed, a modest positive CBF storage pedestal is added, and XDS ``OFFSET``
is set to the conditioned high-resolution median-minus-sigma value.

Do not promote an offset just because aggregate I/sigma gets prettier. In May
2026 tests, lowered quantile offsets made new XDS report fantasy high-shell
signal at roughly 0.61 A. The safest reference remains XDS 20230630 run under
faketime with the real offset; new-XDS settings should be judged by agreement
with that shell behavior, not by larger high-resolution I/sigma.

The v7 storage-pedestal path, quantile mode, and older radial offset/buffer
modes remain available for direct comparison. Do not promote an option unless
real XDS/model validation shows that it improves final structures and preserves
sensible shell behavior rather than just improving pixel diagnostics or headline
XDS statistics.

Important current limitation: XDS and the generated CBF scan metadata use one
constant oscillation width for every written frame. XML fraction timing is used
to validate that assumption when present, but v8 does not yet emit per-frame
variable oscillation widths.
"""

import os
import time
import json
import logging
import warnings
import argparse
import math
import re
import shlex
import hashlib
import xml.etree.ElementTree as ET
from functools import partial
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union

import numpy as np
import mrcfile
import fabio
from fabio.cbfimage import CbfImage
from scipy.ndimage import binary_dilation, gaussian_filter, gaussian_filter1d, median_filter, shift as ndimage_shift
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.signal import savgol_filter
from numpy.typing import NDArray

from adaptive_gain_estimator import (
    estimate_gain_from_binned_stack,
    parse_grid_specs,
    parse_int_csv,
)

# --- Physical Constants ---
H_PLANCK = 6.62607015e-34  # J·s
M_ELECTRON = 9.1093837015e-31 # kg
E_CHARGE = 1.602176634e-19  # C
C_LIGHT = 2.99792458e8      # m/s

# --- Configuration ---
warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


@dataclass(frozen=True)
class DetectorProfile:
    name: str
    detector_model: str
    sensor_kind: str
    pixel_size_mm_unbinned: Optional[float] = None
    sensor_material: Optional[str] = None
    sensor_thickness_mm: Optional[float] = None
    default_gain: Optional[float] = None
    native_size_px: Optional[int] = None


DETECTOR_PROFILES: Dict[str, DetectorProfile] = {
    "ceta_falcon": DetectorProfile(
        name="ceta_falcon",
        detector_model="GENERIC_MICROED",
        sensor_kind="ccd",
        pixel_size_mm_unbinned=0.014,
        native_size_px=4096,
    ),
}


CONVERTER_VERSION = "mrc2cbf_pipeline_v8"
CONVERTER_CONTRACTS: Dict[str, Dict[str, str]] = {
    "v8_radial_reduced": {
        "default": "true",
        "conditioning_mode": "radial_reduced",
        "storage_pedestal_policy": "fixed modest nonnegative CBF pedestal after radial/plane background subtraction",
        "xds_offset_policy": "conditioned high-resolution median minus one robust sigma",
        "trust_boundary": "candidate current-XDS default; requires XDS smoke/model validation for final trust",
    },
    "v7_real_offset": {
        "default": "false",
        "conditioning_mode": "v7",
        "storage_pedestal_policy": "minimum signed ADU pedestal for nonnegative CBF storage",
        "xds_offset_policy": "XDS OFFSET equals the storage pedestal",
        "trust_boundary": "old reference and fallback; can trigger new-XDS truncation pathology",
    },
}

MDOC_GAIN_KEYS = (
    "counts_per_electron",
    "countsperelectron",
    "camera_gain",
    "cameragain",
    "gain",
)


def _normalize_metadata_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")


def _coerce_metadata_value(value: str) -> Union[int, float, str]:
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def _is_finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and np.isfinite(value) and value > 0


def _write_text_atomic(path: Union[str, Path], text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding=encoding)
    os.replace(tmp_path, path)


def _write_bytes_atomic(path: Union[str, Path], payload: bytes) -> None:
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(payload)
    os.replace(tmp_path, path)


def _current_file_sha256() -> Optional[str]:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception:
        return None


def resolve_converter_contract(conditioning_mode: str) -> Tuple[str, Dict[str, str]]:
    mode = str(conditioning_mode or "v7").strip().lower()
    if mode == "radial_reduced":
        name = "v8_radial_reduced"
    elif mode in {"v7", "none", "off", "storage_pedestal"}:
        name = "v7_real_offset"
    else:
        name = f"experimental_{mode}"
    contract = dict(CONVERTER_CONTRACTS.get(name, {}))
    contract.setdefault("default", "false")
    contract.setdefault("conditioning_mode", mode)
    contract.setdefault("storage_pedestal_policy", "experimental")
    contract.setdefault("xds_offset_policy", "experimental")
    contract.setdefault("trust_boundary", "diagnostic probe; do not treat as production without validation")
    return name, contract


def _fmt_report_value(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "NA"
        return f"{float(value):.{digits}g}"
    return str(value)


def _sample_stack_values(data: NDArray, max_pixels: int = 2_000_000, subtract: float = 0.0) -> NDArray[np.float64]:
    flat = np.asarray(data).ravel()
    idx = _sample_indices(flat.size, max_pixels)
    if idx.size == 0:
        return np.asarray([], dtype=np.float64)
    values = np.asarray(flat[idx], dtype=np.float64)
    if subtract:
        values = values - float(subtract)
    return values[np.isfinite(values)]


def summarize_sample_distribution(values: NDArray) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    median_value = float(np.median(arr))
    mad_value = float(np.median(np.abs(arr - median_value)))
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": median_value,
        "mad": mad_value,
        "robust_sigma": float(1.4826 * mad_value),
        "q001": float(np.quantile(arr, 0.001)),
        "q01": float(np.quantile(arr, 0.01)),
        "q05": float(np.quantile(arr, 0.05)),
        "q95": float(np.quantile(arr, 0.95)),
        "q99": float(np.quantile(arr, 0.99)),
        "q999": float(np.quantile(arr, 0.999)),
        "negative_fraction": float(np.count_nonzero(arr < 0) / arr.size),
    }


def get_plt():
    import matplotlib.pyplot as plt

    return plt

# --- Logging Setup ---
def setup_logging(log_file_path: str, console_level=logging.INFO, file_level=logging.DEBUG):
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(file_level)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(console_level)
    logger = logging.getLogger()
    logger.setLevel(min(console_level, file_level)) 
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# --- MDoc Parsing ---
def parse_mdoc(mdoc_path: str) -> Dict[str, Any]:
    metadata = {}
    current_frameset_data = {}
    in_first_frameset = False
    try:
        with open(mdoc_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('[FrameSet'):
                    if line.strip() == '[FrameSet = 0]':
                        in_first_frameset = True
                    else:
                        in_first_frameset = False
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key_lower = key.strip().lower()
                    key_normalized = _normalize_metadata_key(key)
                    value = value.strip()
                    target_dict = current_frameset_data if in_first_frameset else metadata
                    coerced_value = _coerce_metadata_value(value)
                    target_dict[key_lower] = coerced_value
                    if key_normalized and key_normalized != key_lower:
                        target_dict[key_normalized] = coerced_value
        combined_metadata = {k.lower(): v for k, v in metadata.items()}
        combined_metadata.update({k.lower(): v for k, v in current_frameset_data.items()})
        return combined_metadata
    except FileNotFoundError:
        logging.error(f"Mdoc file not found: {mdoc_path}")
        return {}
    except Exception as e:
        logging.error(f"Error parsing mdoc file {mdoc_path}: {e}", exc_info=True)
        return {}


def resolve_sidecar_path(
    data_path: Union[str, Path],
    explicit_path: Optional[str],
    candidates: List[Union[str, Path]],
    label: str,
) -> Optional[str]:
    if explicit_path:
        sidecar = Path(explicit_path)
        if sidecar.exists():
            return str(sidecar)
        logging.warning(f"Requested {label} sidecar does not exist: {sidecar}")
        return None

    for candidate in candidates:
        sidecar = Path(candidate)
        if sidecar.exists():
            logging.info(f"Auto-detected {label} sidecar: {sidecar}")
            return str(sidecar)
    return None


def _xml_local_name(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _xml_child_text(parent: ET.Element, child_name: str) -> Optional[str]:
    for child in list(parent):
        if _xml_local_name(child.tag) != child_name:
            continue
        text = (child.text or "").strip()
        return text or None
    return None


def _xml_children_text(parent: ET.Element) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for child in list(parent):
        text = (child.text or "").strip()
        if text:
            values[_xml_local_name(child.tag)] = text
    return values


def _coerce_optional_metadata_value(values: Dict[str, str], key: str) -> Any:
    value = values.get(key)
    return _coerce_metadata_value(value) if value is not None else None


def _finite_float_list(values: List[Any]) -> List[float]:
    result: List[float] = []
    for value in values:
        if isinstance(value, (int, float)) and np.isfinite(value):
            result.append(float(value))
    return result


def summarize_numeric_values(values: List[Any]) -> Dict[str, Any]:
    finite = _finite_float_list(values)
    if not finite:
        return {"count": 0}
    arr = np.asarray(finite, dtype=np.float64)
    unique_values = sorted({round(float(value), 9) for value in arr.tolist()})
    summary: Dict[str, Any] = {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "spread": float(np.max(arr) - np.min(arr)),
    }
    if len(unique_values) <= 16:
        summary["unique_values"] = unique_values
    return summary


def _parse_iso_datetime(timestamp_text: Optional[str]) -> Tuple[Optional[str], Optional[datetime], Optional[float]]:
    if not timestamp_text:
        return None, None, None

    normalized = timestamp_text.strip()
    if not normalized:
        return None, None, None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return normalized, None, None

    if parsed.tzinfo is not None:
        cbf_timestamp_dt = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    else:
        cbf_timestamp_dt = parsed.replace(tzinfo=None)
    iso_text = cbf_timestamp_dt.isoformat(timespec="milliseconds")
    try:
        epoch_s = parsed.timestamp() if parsed.tzinfo is not None else None
    except Exception:
        epoch_s = None
    return iso_text, parsed, epoch_s


def parse_acquisition_xml(xml_path: str) -> Dict[str, Any]:
    try:
        root = ET.parse(xml_path).getroot()
    except FileNotFoundError:
        logging.error(f"XML sidecar not found: {xml_path}")
        return {}
    except Exception as e:
        logging.error(f"Error parsing XML sidecar {xml_path}: {e}", exc_info=True)
        return {}

    info_element = None
    fractions_container = None
    for element in root.iter():
        local_name = _xml_local_name(element.tag)
        if local_name == "Info" and info_element is None:
            info_element = element
        elif local_name == "Fractions" and fractions_container is None:
            fractions_container = element

    metadata: Dict[str, Any] = {"path": xml_path, "fractions": []}
    if info_element is not None:
        metadata["commercial_name"] = _xml_child_text(info_element, "CommercialName")
        metadata["camera_name"] = _xml_child_text(info_element, "CameraName")
        metadata["serial_number"] = _xml_child_text(info_element, "SerialNumber")
        metadata["format"] = _xml_child_text(info_element, "Format")
        metadata["expected_number_of_fractions"] = _coerce_metadata_value(
            _xml_child_text(info_element, "ExpectedNumberOfFractions") or ""
        ) if _xml_child_text(info_element, "ExpectedNumberOfFractions") else None
        metadata["recorded_number_of_fractions"] = _coerce_metadata_value(
            _xml_child_text(info_element, "RecordedNumberOfFractions") or ""
        ) if _xml_child_text(info_element, "RecordedNumberOfFractions") else None
        metadata["binning"] = _coerce_metadata_value(
            _xml_child_text(info_element, "Binning") or ""
        ) if _xml_child_text(info_element, "Binning") else None
        info_timestamp_iso, _, info_epoch_s = _parse_iso_datetime(_xml_child_text(info_element, "DateTimeWithTimeZone"))
        metadata["info_timestamp_iso"] = info_timestamp_iso
        metadata["info_epoch_s"] = info_epoch_s

    fraction_rows: List[Dict[str, Any]] = []
    if fractions_container is not None:
        for fraction_element in list(fractions_container):
            if _xml_local_name(fraction_element.tag) != "Fraction":
                continue
            fraction_values = _xml_children_text(fraction_element)
            timestamp_iso, _, epoch_s = _parse_iso_datetime(fraction_values.get("DateTimeWithTimeZone"))
            fraction_row = {
                "index": _coerce_optional_metadata_value(fraction_values, "Index"),
                "start_frame": _coerce_optional_metadata_value(fraction_values, "StartFrame"),
                "number_of_frames": _coerce_optional_metadata_value(fraction_values, "NumberOfFrames"),
                "sample_type": fraction_values.get("SampleType"),
                "timestamp_iso": timestamp_iso,
                "epoch_s": epoch_s,
                "pixel_value_to_camera_counts": _coerce_optional_metadata_value(fraction_values, "PixelValueToCameraCounts"),
                "exposure_time_s": _coerce_optional_metadata_value(fraction_values, "ExposureTime"),
                "counts_to_electrons": _coerce_optional_metadata_value(fraction_values, "CountsToElectrons"),
                "mean_pixel_value": _coerce_optional_metadata_value(fraction_values, "MeanPixelValue"),
                "total_dose": _coerce_optional_metadata_value(fraction_values, "TotalDose"),
                "dose_rate": _coerce_optional_metadata_value(fraction_values, "DoseRate"),
                "checksum": fraction_values.get("Checksum"),
            }
            fraction_rows.append(fraction_row)

    positive_epoch_deltas = [
        float(curr["epoch_s"] - prev["epoch_s"])
        for prev, curr in zip(fraction_rows, fraction_rows[1:])
        if prev.get("epoch_s") is not None and curr.get("epoch_s") is not None and curr["epoch_s"] > prev["epoch_s"]
    ]
    median_period_s = float(np.median(positive_epoch_deltas)) if positive_epoch_deltas else None
    for idx, fraction_row in enumerate(fraction_rows):
        direct_period = None
        if idx < len(fraction_rows) - 1:
            prev_epoch = fraction_row.get("epoch_s")
            next_epoch = fraction_rows[idx + 1].get("epoch_s")
            if prev_epoch is not None and next_epoch is not None and next_epoch > prev_epoch:
                direct_period = float(next_epoch - prev_epoch)
        fraction_row["exposure_period_s"] = (
            direct_period
            if _is_finite_positive(direct_period)
            else (median_period_s if _is_finite_positive(median_period_s) else fraction_row.get("exposure_time_s"))
        )

    metadata["fractions"] = fraction_rows
    metadata["fraction_count"] = len(fraction_rows)
    metadata["fraction_period_s"] = median_period_s
    metadata["fraction_timing_summary"] = {
        "exposure_time_s": summarize_numeric_values([row.get("exposure_time_s") for row in fraction_rows]),
        "exposure_period_s": summarize_numeric_values([row.get("exposure_period_s") for row in fraction_rows]),
        "number_of_frames": summarize_numeric_values([row.get("number_of_frames") for row in fraction_rows]),
        "start_frame_delta": summarize_numeric_values([
            curr.get("start_frame") - prev.get("start_frame")
            for prev, curr in zip(fraction_rows, fraction_rows[1:])
            if isinstance(prev.get("start_frame"), (int, float)) and isinstance(curr.get("start_frame"), (int, float))
        ]),
    }

    if fraction_rows:
        first_fraction = fraction_rows[0]
        exposure_time_s = first_fraction.get("exposure_time_s")
        exposure_period_s = first_fraction.get("exposure_period_s")
        pixel_value_to_camera_counts = first_fraction.get("pixel_value_to_camera_counts")
        counts_to_electrons = first_fraction.get("counts_to_electrons")
        metadata["fraction_exposure_time_s"] = float(exposure_time_s) if _is_finite_positive(exposure_time_s) else None
        metadata["fraction_exposure_period_s"] = float(exposure_period_s) if _is_finite_positive(exposure_period_s) else None
        metadata["pixel_value_to_camera_counts"] = (
            float(pixel_value_to_camera_counts) if _is_finite_positive(pixel_value_to_camera_counts) else None
        )
        metadata["counts_to_electrons"] = float(counts_to_electrons) if _is_finite_positive(counts_to_electrons) else None
        if _is_finite_positive(pixel_value_to_camera_counts) and _is_finite_positive(counts_to_electrons):
            camera_counts_per_electron = 1.0 / float(counts_to_electrons)
            saved_counts_per_electron = 1.0 / (
                float(counts_to_electrons) * float(pixel_value_to_camera_counts)
            )
            metadata["camera_counts_per_electron"] = camera_counts_per_electron
            metadata["saved_counts_per_electron"] = saved_counts_per_electron

    return metadata


def build_raw_frame_metadata(
    num_input_frames: int,
    xml_metadata: Optional[Dict[str, Any]],
    fallback_exposure_time_s: Optional[float],
    fallback_exposure_period_s: Optional[float],
    fallback_timestamp_iso: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[str]]]:
    raw_frame_metadata: List[Dict[str, Any]] = []
    metadata_sources = {
        "exposure_time_source": None,
        "exposure_period_source": None,
        "timestamp_source": None,
    }

    fractions = list((xml_metadata or {}).get("fractions") or [])
    if len(fractions) == num_input_frames:
        for fraction in fractions:
            raw_frame_metadata.append(
                {
                    "exposure_time_s": float(fraction["exposure_time_s"]) if _is_finite_positive(fraction.get("exposure_time_s")) else None,
                    "exposure_period_s": float(fraction["exposure_period_s"]) if _is_finite_positive(fraction.get("exposure_period_s")) else None,
                    "timestamp_iso": fraction.get("timestamp_iso"),
                    "epoch_s": fraction.get("epoch_s"),
                }
            )
        metadata_sources["exposure_time_source"] = "xml:Fraction ExposureTime"
        metadata_sources["exposure_period_source"] = "xml:Fraction DateTimeWithTimeZone deltas"
        metadata_sources["timestamp_source"] = "xml:Fraction DateTimeWithTimeZone"
        return raw_frame_metadata, metadata_sources

    if fractions:
        logging.warning(
            "XML fraction count (%d) does not match input frame count (%d); falling back to synthesized per-frame metadata",
            len(fractions),
            num_input_frames,
        )

    effective_exposure_time_s = (
        float(fallback_exposure_time_s)
        if _is_finite_positive(fallback_exposure_time_s)
        else (
            float((xml_metadata or {}).get("fraction_exposure_time_s"))
            if _is_finite_positive((xml_metadata or {}).get("fraction_exposure_time_s"))
            else None
        )
    )
    effective_exposure_period_s = (
        float(fallback_exposure_period_s)
        if _is_finite_positive(fallback_exposure_period_s)
        else (
            float((xml_metadata or {}).get("fraction_exposure_period_s"))
            if _is_finite_positive((xml_metadata or {}).get("fraction_exposure_period_s"))
            else effective_exposure_time_s
        )
    )

    start_timestamp_text = (
        fallback_timestamp_iso
        or (xml_metadata or {}).get("info_timestamp_iso")
        or datetime.now().astimezone().isoformat(timespec="milliseconds")
    )
    parsed_timestamp_iso, parsed_dt, parsed_epoch_s = _parse_iso_datetime(start_timestamp_text)
    if parsed_timestamp_iso is None:
        parsed_timestamp_iso = datetime.now().astimezone().isoformat(timespec="milliseconds")
        _, parsed_dt, parsed_epoch_s = _parse_iso_datetime(parsed_timestamp_iso)

    for idx in range(num_input_frames):
        timestamp_iso = parsed_timestamp_iso
        epoch_s = parsed_epoch_s
        if parsed_dt is not None and _is_finite_positive(effective_exposure_period_s):
            current_dt = parsed_dt + timedelta(seconds=float(effective_exposure_period_s) * idx)
            timestamp_iso, _, _ = _parse_iso_datetime(current_dt.isoformat())
            try:
                epoch_s = current_dt.timestamp() if current_dt.tzinfo is not None else None
            except Exception:
                epoch_s = None
        raw_frame_metadata.append(
            {
                "exposure_time_s": effective_exposure_time_s,
                "exposure_period_s": effective_exposure_period_s,
                "timestamp_iso": timestamp_iso,
                "epoch_s": epoch_s,
            }
        )

    metadata_sources["exposure_time_source"] = "CLI" if _is_finite_positive(fallback_exposure_time_s) else (
        "xml average/fallback" if _is_finite_positive((xml_metadata or {}).get("fraction_exposure_time_s")) else "fallback"
    )
    metadata_sources["exposure_period_source"] = "CLI" if _is_finite_positive(fallback_exposure_period_s) else (
        "xml average/fallback" if _is_finite_positive((xml_metadata or {}).get("fraction_exposure_period_s")) else metadata_sources["exposure_time_source"]
    )
    metadata_sources["timestamp_source"] = (
        "CLI"
        if fallback_timestamp_iso
        else ("xml:Info DateTimeWithTimeZone" if (xml_metadata or {}).get("info_timestamp_iso") else "generated-now")
    )
    return raw_frame_metadata, metadata_sources


def aggregate_output_frame_metadata(
    raw_frame_metadata: List[Dict[str, Any]],
    bin_z: int,
    num_output_frames: int,
) -> List[Dict[str, Any]]:
    output_frame_metadata: List[Dict[str, Any]] = []
    usable_input_frames = min(len(raw_frame_metadata), int(bin_z) * int(num_output_frames))
    trimmed_metadata = raw_frame_metadata[:usable_input_frames]

    for output_idx in range(num_output_frames):
        start_idx = output_idx * int(bin_z)
        end_idx = start_idx + int(bin_z)
        chunk = trimmed_metadata[start_idx:end_idx]
        if not chunk:
            output_frame_metadata.append({})
            continue

        exposure_time_values = [float(row["exposure_time_s"]) for row in chunk if _is_finite_positive(row.get("exposure_time_s"))]
        exposure_period_values = [
            float(row["exposure_period_s"])
            for row in chunk
            if _is_finite_positive(row.get("exposure_period_s"))
        ]
        timestamp_iso = next((row.get("timestamp_iso") for row in chunk if row.get("timestamp_iso")), None)
        epoch_s = next((row.get("epoch_s") for row in chunk if row.get("epoch_s") is not None), None)

        output_frame_metadata.append(
            {
                "exposure_time_s": float(sum(exposure_time_values)) if exposure_time_values else None,
                "exposure_period_s": float(sum(exposure_period_values)) if exposure_period_values else None,
                "timestamp_iso": timestamp_iso,
                "epoch_s": epoch_s,
            }
        )

    return output_frame_metadata


def validate_constant_wedge_from_metadata(
    *,
    angle_increment_deg_per_input_frame: Optional[float],
    bin_z: int,
    mdoc_data: Optional[Dict[str, Any]],
    xml_metadata: Optional[Dict[str, Any]],
    input_frame_count: Optional[int],
    tolerance_fraction: float = 1.0e-3,
    tolerance_seconds: float = 1.0e-4,
) -> Dict[str, Any]:
    """Check whether sidecar timing supports v7's constant oscillation model.

    XDS and the miniCBF headers written here use one oscillation width for every
    output frame. SerialEM supplies the rotation speed in MDOC; Velox XML
    supplies per-fraction exposure/timing. When both are present, the XML is a
    useful consistency check but not a source for variable-width frame metadata.
    """

    report: Dict[str, Any] = {
        "constant_wedge_required": True,
        "status": "unchecked",
        "ok_for_constant_wedge": None,
        "warnings": [],
        "angle_increment_deg_per_input_frame": (
            float(angle_increment_deg_per_input_frame)
            if angle_increment_deg_per_input_frame is not None and np.isfinite(angle_increment_deg_per_input_frame)
            else None
        ),
        "angle_increment_deg_per_output_frame": (
            float(angle_increment_deg_per_input_frame) * int(bin_z)
            if angle_increment_deg_per_input_frame is not None and np.isfinite(angle_increment_deg_per_input_frame)
            else None
        ),
        "bin_z": int(bin_z),
        "input_frame_count": int(input_frame_count) if input_frame_count is not None else None,
    }

    warnings_out: List[str] = report["warnings"]
    fractions = list((xml_metadata or {}).get("fractions") or [])
    report["xml_fraction_count"] = int(len(fractions))
    if not fractions:
        report["status"] = "no_xml_fraction_timing"
        report["ok_for_constant_wedge"] = True
        warnings_out.append("No XML fraction timing was available; constant wedge relies on MDOC/CLI scan metadata only.")
        return report

    if input_frame_count is not None and len(fractions) != int(input_frame_count):
        warnings_out.append(
            f"XML fraction count {len(fractions)} does not match MRC frame count {int(input_frame_count)}."
        )

    exposure_values = _finite_float_list([row.get("exposure_time_s") for row in fractions])
    period_values = _finite_float_list([row.get("exposure_period_s") for row in fractions])
    number_of_frames_values = _finite_float_list([row.get("number_of_frames") for row in fractions])
    start_frame_values = _finite_float_list([row.get("start_frame") for row in fractions])
    start_deltas = [
        float(curr - prev)
        for prev, curr in zip(start_frame_values, start_frame_values[1:])
        if np.isfinite(prev) and np.isfinite(curr)
    ]

    report["xml_timing_summary"] = {
        "exposure_time_s": summarize_numeric_values(exposure_values),
        "exposure_period_s": summarize_numeric_values(period_values),
        "number_of_frames": summarize_numeric_values(number_of_frames_values),
        "start_frame_delta": summarize_numeric_values(start_deltas),
    }

    exposure_uniform = None
    exposure_median = None
    if exposure_values:
        exposure_summary = report["xml_timing_summary"]["exposure_time_s"]
        exposure_median = float(exposure_summary["median"])
        exposure_tolerance = max(float(tolerance_seconds), abs(exposure_median) * float(tolerance_fraction))
        exposure_spread = float(exposure_summary["spread"])
        exposure_uniform = exposure_spread <= exposure_tolerance
        report["xml_exposure_uniform"] = bool(exposure_uniform)
        report["xml_exposure_uniform_tolerance_s"] = float(exposure_tolerance)
        if not exposure_uniform:
            warnings_out.append(
                "XML per-fraction ExposureTime is not constant enough for a physically exact constant wedge "
                f"(spread {exposure_spread:.6g} s > tolerance {exposure_tolerance:.6g} s)."
            )
    else:
        warnings_out.append("XML fractions did not contain usable ExposureTime values.")

    period_uniform = None
    if period_values:
        period_summary = report["xml_timing_summary"]["exposure_period_s"]
        period_median = float(period_summary["median"])
        period_tolerance = max(float(tolerance_seconds), abs(period_median) * float(tolerance_fraction))
        period_spread = float(period_summary["spread"])
        period_uniform = period_spread <= period_tolerance
        report["xml_period_uniform"] = bool(period_uniform)
        report["xml_period_uniform_tolerance_s"] = float(period_tolerance)
        if not period_uniform:
            warnings_out.append(
                "XML frame timestamp periods are not constant within tolerance "
                f"(spread {period_spread:.6g} s > tolerance {period_tolerance:.6g} s)."
            )

    mdoc = mdoc_data or {}
    degrees_per_second = mdoc.get("degreespersecond")
    if _is_finite_positive(degrees_per_second):
        report["rotation_rate_deg_per_s"] = float(degrees_per_second)
    elif _is_finite_positive(mdoc.get("rotationrate")):
        report["rotation_rate_deg_per_s"] = float(math.degrees(float(mdoc["rotationrate"])))
    else:
        report["rotation_rate_deg_per_s"] = None

    if _is_finite_positive(mdoc.get("exposuretime")) and _is_finite_positive(mdoc.get("numsubframes")):
        mdoc_exposure_per_frame = float(mdoc["exposuretime"]) / float(mdoc["numsubframes"])
        report["mdoc_exposure_time_s_per_input_frame"] = mdoc_exposure_per_frame
        if exposure_median is not None:
            diff = abs(mdoc_exposure_per_frame - exposure_median)
            tolerance = max(float(tolerance_seconds), abs(mdoc_exposure_per_frame) * float(tolerance_fraction))
            report["mdoc_vs_xml_exposure_diff_s"] = float(diff)
            report["mdoc_vs_xml_exposure_tolerance_s"] = float(tolerance)
            if diff > tolerance:
                warnings_out.append(
                    "MDOC total ExposureTime/NumSubFrames and XML median ExposureTime differ "
                    f"({diff:.6g} s > tolerance {tolerance:.6g} s)."
                )

    if report.get("rotation_rate_deg_per_s") is not None and exposure_median is not None:
        xml_angle = float(report["rotation_rate_deg_per_s"]) * float(exposure_median)
        report["xml_median_angle_increment_deg_per_input_frame"] = float(xml_angle)
        if report["angle_increment_deg_per_input_frame"] is not None:
            chosen_angle = float(report["angle_increment_deg_per_input_frame"])
            angle_diff = abs(chosen_angle - xml_angle)
            angle_tolerance = max(1.0e-6, abs(chosen_angle) * float(tolerance_fraction))
            report["chosen_vs_xml_angle_diff_deg"] = float(angle_diff)
            report["chosen_vs_xml_angle_tolerance_deg"] = float(angle_tolerance)
            if angle_diff > angle_tolerance:
                warnings_out.append(
                    "Chosen oscillation width and XML/MDOC timing-derived width differ "
                    f"({angle_diff:.6g} deg > tolerance {angle_tolerance:.6g} deg)."
                )

    report["ok_for_constant_wedge"] = not warnings_out or all(
        warning.startswith("No XML fraction timing") for warning in warnings_out
    )
    report["status"] = "ok" if report["ok_for_constant_wedge"] else "warning"
    if exposure_uniform is False:
        report["recommended_action"] = (
            "Use the constant wedge only as an approximation, or add variable-width scan support before treating "
            "these frames as physically exact."
        )
    elif period_uniform is False:
        report["recommended_action"] = "Review XML timestamps; exposure times still carry the stronger wedge check."
    else:
        report["recommended_action"] = "Constant per-frame oscillation is supported by available sidecar timing."
    return report

# --- Wavelength Calculation ---
def calculate_wavelength_A(voltage_kv: float) -> float:
    if voltage_kv <= 0: raise ValueError("Voltage must be positive.")
    voltage_v = voltage_kv * 1000.0
    term1 = 2 * M_ELECTRON * E_CHARGE * voltage_v
    term2 = 1 + (E_CHARGE * voltage_v) / (2 * M_ELECTRON * C_LIGHT**2)
    wavelength_m = H_PLANCK / math.sqrt(term1 * term2)
    wavelength_A = wavelength_m * 1e10
    logging.debug(f"Calculated wavelength for {voltage_kv} kV: {wavelength_A:.6f} Å")
    return wavelength_A


def infer_detector_profile(
    profile_name: str,
    mdoc_data: Dict[str, Any],
    xml_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[DetectorProfile], Optional[str]]:
    if profile_name and profile_name != "auto":
        if profile_name == "none":
            return None, None
        profile = DETECTOR_PROFILES.get(profile_name)
        if profile is None:
            logging.warning(f"Unknown detector profile '{profile_name}'. Ignoring.")
            return None, None
        return profile, f"user selected '{profile_name}'"

    for key, value in mdoc_data.items():
        if not isinstance(value, str):
            continue
        text = value.lower()
        if "ceta" in text or "falcon" in text:
            return DETECTOR_PROFILES["ceta_falcon"], f"mdoc text field '{key}' contains Ceta/Falcon marker"

    for key in ("commercial_name", "camera_name", "format"):
        value = (xml_metadata or {}).get(key)
        if not isinstance(value, str):
            continue
        text = value.lower()
        if "ceta" in text or "falcon" in text:
            return DETECTOR_PROFILES["ceta_falcon"], f"xml field '{key}' contains Ceta/Falcon marker"
    return None, None


def infer_hardware_binning(
    mdoc_data: Dict[str, Any],
    image_shape: Optional[Tuple[int, int, int]],
    detector_profile: Optional[DetectorProfile],
) -> Tuple[int, int, str]:
    mdoc_binning = mdoc_data.get("binning")
    if _is_finite_positive(mdoc_binning):
        factor = max(1, int(round(float(mdoc_binning))))
        return factor, factor, "mdoc Binning"

    if detector_profile and detector_profile.native_size_px and image_shape is not None:
        height = int(image_shape[-2])
        width = int(image_shape[-1])
        if height == width and detector_profile.native_size_px % width == 0:
            factor = detector_profile.native_size_px // width
            if factor >= 1:
                reason = (
                    f"{detector_profile.name} profile: dimension-based ({factor}) "
                    f"from {width}x{height} with {detector_profile.native_size_px} native reference"
                )
                return factor, factor, reason

    return 1, 1, "default (no hardware binning metadata)"


def resolve_unbinned_pixel_size_mm(
    cli_pixel_size_mm: Optional[float],
    mdoc_data: Dict[str, Any],
    detector_profile: Optional[DetectorProfile],
    hardware_bin_x: int,
    hardware_bin_y: int,
) -> Tuple[Optional[float], str]:
    if cli_pixel_size_mm is not None:
        return float(cli_pixel_size_mm), "CLI"

    mdoc_pixel_size_mm = None
    pixel_source = ""
    if "camerapixelsize" in mdoc_data and _is_finite_positive(mdoc_data["camerapixelsize"]):
        mdoc_pixel_size_mm = float(mdoc_data["camerapixelsize"]) / 1000.0
        pixel_source = "mdoc CameraPixelSize"
    elif "pixelspacing" in mdoc_data and _is_finite_positive(mdoc_data["pixelspacing"]):
        mdoc_pixel_size_mm = float(mdoc_data["pixelspacing"]) / 1000.0
        pixel_source = "mdoc PixelSpacing"

    if detector_profile and detector_profile.pixel_size_mm_unbinned is not None:
        profile_pixel_size = detector_profile.pixel_size_mm_unbinned
        if mdoc_pixel_size_mm is not None:
            expected_binned_x = profile_pixel_size * hardware_bin_x
            expected_binned_y = profile_pixel_size * hardware_bin_y
            if not math.isclose(mdoc_pixel_size_mm, profile_pixel_size, rel_tol=0.05, abs_tol=1e-6):
                logging.warning(
                    "Detector profile pixel size %.6f mm overrides mdoc pixel size %.6f mm",
                    profile_pixel_size,
                    mdoc_pixel_size_mm,
                )
            if not (
                math.isclose(mdoc_pixel_size_mm, profile_pixel_size, rel_tol=0.05, abs_tol=1e-6)
                or math.isclose(mdoc_pixel_size_mm, expected_binned_x, rel_tol=0.05, abs_tol=1e-6)
                or math.isclose(mdoc_pixel_size_mm, expected_binned_y, rel_tol=0.05, abs_tol=1e-6)
            ):
                logging.warning(
                    "mdoc pixel size %.6f mm is not consistent with profile pixel size %.6f mm and hardware binning (%d, %d)",
                    mdoc_pixel_size_mm,
                    profile_pixel_size,
                    hardware_bin_x,
                    hardware_bin_y,
                )
        return profile_pixel_size, f"detector profile ({detector_profile.name})"

    return mdoc_pixel_size_mm, pixel_source or "unresolved"


def resolve_gain_value(
    cli_gain: Optional[float],
    mdoc_data: Dict[str, Any],
    xml_metadata: Optional[Dict[str, Any]],
    detector_profile: Optional[DetectorProfile],
) -> Tuple[Optional[float], Optional[str]]:
    if _is_finite_positive(cli_gain):
        return float(cli_gain), "CLI"

    xml_saved_gain = (xml_metadata or {}).get("saved_counts_per_electron")
    if _is_finite_positive(xml_saved_gain):
        return float(xml_saved_gain), "xml:saved_counts_per_electron"

    for key in MDOC_GAIN_KEYS:
        value = mdoc_data.get(key)
        if _is_finite_positive(value):
            return float(value), f"mdoc:{key}"

    if detector_profile and _is_finite_positive(detector_profile.default_gain):
        return float(detector_profile.default_gain), f"detector profile ({detector_profile.name})"

    return None, None


def derive_cbf_glob_from_template(filename_template: str) -> str:
    try:
        sample_name = filename_template.format(1)
    except Exception:
        return "*.cbf"
    derived_glob = re.sub(r"\d+", "*", sample_name, count=1)
    return derived_glob if "*" in derived_glob else "*.cbf"


def absolute_output_pattern(output_dir: Union[str, Path], frame_pattern: str) -> str:
    """Return a host-local absolute path for generated CBF inputs."""
    output_root = Path(os.path.abspath(os.fspath(output_dir)))
    return str(output_root / frame_pattern)


def synology_relative_suffix(output_dir: Union[str, Path]) -> str:
    """Return the path below a per-user synology mount when the output path has one."""
    output_root = Path(os.path.abspath(os.fspath(output_dir)))
    parts = output_root.parts
    if "synology" not in parts:
        return ""
    synology_index = parts.index("synology")
    suffix_parts = parts[synology_index + 1 :]
    if not suffix_parts:
        return ""
    return str(Path(*suffix_parts))


def filename_template_looks_dials_friendly(filename_template: str) -> bool:
    try:
        sample_name = filename_template.format(1)
    except Exception:
        return False
    return re.search(r"\d+", Path(sample_name).name) is not None


def _build_sensor_header_lines(template_params: Dict[str, Any]) -> List[str]:
    sensor_kind = template_params.get("sensor_kind", "ccd").lower()
    sensor_material = template_params.get("sensor_material")
    sensor_thickness_mm = template_params.get("sensor_thickness_mm")

    if sensor_kind == "ccd":
        return ["# CCD detector"]

    if sensor_material and sensor_thickness_mm is not None:
        material_map = {"si": "Silicon", "cdte": "CdTe"}
        material_name = material_map.get(sensor_material.lower(), sensor_material)
        return [f"# {material_name} sensor, thickness {sensor_thickness_mm / 1000.0:.6f} m"]

    return ["# CCD detector"]


def rewrite_fabio_cbf_as_generic_minicbf(
    cbf_path: Union[str, Path],
    header_contents: str,
    header_convention: str = "GENERIC_MINI",
) -> None:
    cbf_path = Path(cbf_path)
    raw_bytes = cbf_path.read_bytes()
    marker = b"--CIF-BINARY-FORMAT-SECTION--"
    marker_index = raw_bytes.find(marker)
    if marker_index < 0:
        raise ValueError(f"Could not locate CBF binary marker in {cbf_path}")

    header_prefix = raw_bytes[:marker_index].decode("latin1", "replace")
    newline = "\r\n" if "\r\n" in header_prefix else "\n"
    prefix_lines = header_prefix.splitlines()
    version_line = prefix_lines[0] if prefix_lines else "###CBF: VERSION 1.5"
    data_block_line = next((line for line in prefix_lines if line.startswith("data_")), f"data_{cbf_path.stem}")

    rebuilt_lines = [
        version_line,
        data_block_line,
        f"_array_data.header_convention        {header_convention}",
        "_array_data.header_contents",
        ";",
        *header_contents.splitlines(),
        ";",
        "",
        "_array_data.data",
        ";",
        "",
    ]
    rebuilt_bytes = newline.join(rebuilt_lines).encode("latin1") + raw_bytes[marker_index:]
    _write_bytes_atomic(cbf_path, rebuilt_bytes)


def write_dials_import_helper(
    output_dir: Union[str, Path],
    filename_template: str,
    probe: str,
    gain: Optional[float],
    pedestal: Optional[float],
    goniometer_axis: Optional[str] = None,
) -> Optional[str]:
    output_dir = Path(output_dir)
    input_glob = derive_cbf_glob_from_template(filename_template)
    absolute_input_glob = absolute_output_pattern(output_dir, input_glob)
    absolute_source_dir = str(Path(os.path.abspath(os.fspath(output_dir))))
    source_suffix = synology_relative_suffix(output_dir)
    xds_frame_template = derive_xds_name_template_from_template(filename_template)
    helper_path = output_dir / "dials_import_helper.sh"
    default_pedestal = (
        str(int(round(float(pedestal))))
        if pedestal is not None and np.isfinite(pedestal) and pedestal >= 0
        else ""
    )
    gain_note = (
        f"# Metadata/header gain candidate, not applied automatically: {float(gain):.6f}"
        if gain is not None and np.isfinite(gain) and gain > 0
        else "# No metadata/header gain candidate was resolved."
    )

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "usage() {",
        "  cat <<'USAGE'",
        "Usage: ./dials_import_helper.sh [options] [-- dials.import options...]",
        "",
        "Imports this dataset's CBF sweep from its absolute source path.",
        "The helper is intentionally copyable: run it from a writable work directory.",
        "",
        "Options:",
        "  --output PATH          Output experiments file (default: imported.expt)",
        "  --source-dir PATH      Override the CBF source directory",
        "  --gain VALUE           Add/override DIALS panel.gain=VALUE",
        "  --offset VALUE         Add/override DIALS panel.pedestal=VALUE",
        "  --pedestal VALUE       Alias for --offset",
        "  --no-offset            Do not pass panel.pedestal",
        "  --lookup-gain PATH     Add lookup.gain=PATH",
        "  --probe NAME           Override probe=... (default from conversion)",
        "  --axis X,Y,Z           Override geometry.goniometer.axis=...",
        "  --no-prepare-xds       Do not update local copied XDS.INP path(s)",
        "  --prepare-xds PATH     Update this copied XDS input instead of the defaults",
        "  --prepare-xds-only     Update copied XDS input path(s), then exit",
        "  --dry-run              Print the command without running it",
        "  -h, --help             Show this help",
        "",
        "Examples:",
        "  ./dials_import_helper.sh",
        "  ./dials_import_helper.sh --gain 38.8873 --offset 360",
        "  ./dials_import_helper.sh --no-offset -- spotfinder.threshold.dispersion.gain=38.8873",
        "USAGE",
        "}",
        "",
        f'SOURCE_DIR_DEFAULT={shlex.quote(absolute_source_dir)}',
        f'SOURCE_SUFFIX={shlex.quote(source_suffix)}',
        f'INPUT_PATTERN={shlex.quote(input_glob)}',
        f'XDS_FRAME_TEMPLATE={shlex.quote(xds_frame_template)}',
        'SOURCE_DIR="${CBF_SOURCE_DIR:-}"',
        'OUTPUT_EXPERIMENTS="${OUTPUT_EXPERIMENTS:-imported.expt}"',
        'LOOKUP_GAIN=""',
        'PANEL_GAIN=""',
        f'PANEL_PEDESTAL={shlex.quote(default_pedestal)}',
        f'PROBE_VALUE={shlex.quote(str(probe or ""))}',
        f'GONIOMETER_AXIS_VALUE={shlex.quote(str(goniometer_axis or ""))}',
        'PREPARE_XDS=1',
        'PREPARE_XDS_ONLY=0',
        'LOCAL_XDS_FILES=("XDS.INP" "XDS_20230630.INP")',
        'DRY_RUN=0',
        'EXTRA_ARGS=()',
        "",
        "# Source path resolution is account-aware for cc02-style per-user Synology mounts.",
        "# If $HOME/synology/<same dataset suffix> exists, that path is preferred;",
        "# otherwise the original conversion host path is used.",
        f"# Original absolute CBF glob: {absolute_input_glob}",
        "# Geometry, wavelength and scan come from the CBF headers.",
        "# The storage pedestal is offered as DIALS panel.pedestal by default;",
        "# override it with --offset/--pedestal, or suppress it with --no-offset.",
        "# Flat gain is intentionally manual: pass --gain or a DIALS option after --.",
        gain_note,
        "",
        'while [[ $# -gt 0 ]]; do',
        '  case "$1" in',
        '    --output|--output-experiments)',
        '      OUTPUT_EXPERIMENTS="${2:?missing value for $1}"; shift 2 ;;',
        '    --source-dir|--cbf-dir)',
        '      SOURCE_DIR="${2:?missing value for $1}"; shift 2 ;;',
        '    --gain|--panel-gain)',
        '      PANEL_GAIN="${2:?missing value for $1}"; shift 2 ;;',
        '    --offset|--pedestal|--panel-pedestal)',
        '      PANEL_PEDESTAL="${2:?missing value for $1}"; shift 2 ;;',
        '    --no-offset|--no-pedestal)',
        '      PANEL_PEDESTAL=""; shift ;;',
        '    --lookup-gain)',
        '      LOOKUP_GAIN="${2:?missing value for $1}"; shift 2 ;;',
        '    --probe)',
        '      PROBE_VALUE="${2:?missing value for $1}"; shift 2 ;;',
        '    --axis|--goniometer-axis)',
        '      GONIOMETER_AXIS_VALUE="${2:?missing value for $1}"; shift 2 ;;',
        '    --no-prepare-xds)',
        '      PREPARE_XDS=0; shift ;;',
        '    --prepare-xds)',
        '      LOCAL_XDS_FILES=("${2:?missing value for $1}"); PREPARE_XDS=1; shift 2 ;;',
        '    --prepare-xds-only)',
        '      PREPARE_XDS_ONLY=1; PREPARE_XDS=1; shift ;;',
        '    --dry-run)',
        '      DRY_RUN=1; shift ;;',
        '    -h|--help)',
        '      usage; exit 0 ;;',
        '    --)',
        '      shift; EXTRA_ARGS+=("$@"); break ;;',
        '    *)',
        '      EXTRA_ARGS+=("$1"); shift ;;',
        '  esac',
        'done',
        "",
        'if [[ -z "$SOURCE_DIR" ]]; then',
        '  if [[ -n "$SOURCE_SUFFIX" && -d "$HOME/synology/$SOURCE_SUFFIX" ]]; then',
        '    SOURCE_DIR="$HOME/synology/$SOURCE_SUFFIX"',
        "  else",
        '    SOURCE_DIR="$SOURCE_DIR_DEFAULT"',
        "  fi",
        "fi",
        'INPUT_GLOB="$SOURCE_DIR/$INPUT_PATTERN"',
        "",
        "prepare_xds_file() {",
        '  local xds_path="$1"',
        '  [[ -f "$xds_path" ]] || return 0',
        '  local frame_template="$SOURCE_DIR/$XDS_FRAME_TEMPLATE"',
        '  local tmp_path="${xds_path}.tmp.$$"',
        '  awk -v tmpl="$frame_template" \'/^NAME_TEMPLATE_OF_DATA_FRAMES=/ { print "NAME_TEMPLATE_OF_DATA_FRAMES= " tmpl; next } { print }\' "$xds_path" > "$tmp_path"',
        '  mv "$tmp_path" "$xds_path"',
        '  printf "Updated %s to use %s\\n" "$xds_path" "$frame_template" >&2',
        "}",
        "",
        'if [[ "$PREPARE_XDS" == "1" ]]; then',
        '  for xds_path in "${LOCAL_XDS_FILES[@]}"; do prepare_xds_file "$xds_path"; done',
        "fi",
        'if [[ "$PREPARE_XDS_ONLY" == "1" ]]; then exit 0; fi',
        "",
        "shopt -s nullglob",
        "INPUT_FILES=( $INPUT_GLOB )",
        "shopt -u nullglob",
        'if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then',
        '  printf "ERROR: no CBF files matched %s\\n" "$INPUT_GLOB" >&2',
        "  exit 2",
        "fi",
        'cmd=(dials.import "${INPUT_FILES[@]}" output.experiments="$OUTPUT_EXPERIMENTS")',
    ]

    lines.append('if [[ -n "$LOOKUP_GAIN" ]]; then cmd+=("lookup.gain=$LOOKUP_GAIN"); fi')
    lines.append('if [[ -n "$PANEL_GAIN" ]]; then cmd+=("panel.gain=$PANEL_GAIN"); fi')
    lines.append('if [[ -n "$PANEL_PEDESTAL" ]]; then cmd+=("panel.pedestal=$PANEL_PEDESTAL"); fi')
    lines.append('if [[ -n "$PROBE_VALUE" ]]; then cmd+=("probe=$PROBE_VALUE"); fi')
    lines.append('if [[ -n "$GONIOMETER_AXIS_VALUE" ]]; then cmd+=("geometry.goniometer.axis=$GONIOMETER_AXIS_VALUE"); fi')
    lines.append('if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then cmd+=("${EXTRA_ARGS[@]}"); fi')

    lines.extend([
        "",
        'printf "Running:" >&2',
        'printf " %q" "${cmd[@]}" >&2',
        'printf "\\n" >&2',
        'if [[ "$DRY_RUN" == "1" ]]; then exit 0; fi',
        'exec "${cmd[@]}"',
        "",
    ])

    _write_text_atomic(helper_path, "\n".join(lines))
    os.chmod(helper_path, 0o755)
    return str(helper_path)


def _parse_axis_triplet(axis_text: Optional[str]) -> Tuple[float, float, float]:
    if not axis_text:
        raise ValueError("Axis text is empty")
    parts = [part for part in re.split(r"[\s,]+", axis_text.strip()) if part]
    if len(parts) != 3:
        raise ValueError(f"Expected three axis components, got: {axis_text!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _imgcif_axis_to_xds(axis_triplet: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x_comp, y_comp, z_comp = axis_triplet
    return float(x_comp), float(-y_comp), float(z_comp)


def derive_xds_name_template_from_template(filename_template: str) -> str:
    try:
        sample_name = Path(filename_template.format(1)).name
    except Exception:
        return "image_?????.cbf"
    match = re.search(r"\d+", sample_name)
    if not match:
        return sample_name
    return sample_name[: match.start()] + ("?" * (match.end() - match.start())) + sample_name[match.end() :]


def write_xds_inp(
    output_dir: Union[str, Path],
    filename_template: str,
    num_frames: int,
    nrows: int,
    ncols: int,
    pixel_size_x_mm: float,
    pixel_size_y_mm: float,
    detector_distance_mm: float,
    beam_center_x_px_1based: float,
    beam_center_y_px_1based: float,
    start_angle_deg: float,
    oscillation_range_deg: float,
    wavelength_A: float,
    goniometer_axis_imgcif: str,
    overload_value: int,
    pedestal: int,
    xds_offset: int,
    legacy_offset: Optional[int] = None,
    gain: Optional[float] = None,
    detector_model: Optional[str] = None,
    xds_offset_details: Optional[Dict[str, Any]] = None,
    suggested_exclude_ranges: Optional[List[Tuple[int, int]]] = None,
    output_filename: str = "XDS.INP",
    offset_compatibility_note: str = "For XDS versions after 20230630",
    converter_contract: Optional[str] = None,
) -> Optional[str]:
    output_dir = Path(output_dir)
    xds_path = output_dir / output_filename
    frame_template = derive_xds_name_template_from_template(filename_template)
    name_template = absolute_output_pattern(output_dir, frame_template)
    rotation_axis_xds = _imgcif_axis_to_xds(_parse_axis_triplet(goniometer_axis_imgcif))
    data_end = max(1, int(num_frames))
    offset_note = f" ! {offset_compatibility_note}" if offset_compatibility_note else ""

    lines = [
        f"! Generated automatically from {CONVERTER_VERSION}.py",
        f"! Detector model: {detector_model or 'GENERIC_MICROED'}",
        "! Copyable XDS input: frame paths are absolute and outputs are written to the current directory.",
        "! If copied to another account with a per-user Synology mount, run ./dials_import_helper.sh --prepare-xds-only to rewrite the path.",
        "! Beam center and geometry correspond to the stored CBF images.",
        f"! CONVERTER_CONTRACT= {converter_contract or 'unspecified'}",
        "! GAIN is intentionally left unset; uncomment/edit the next line only for a manual XDS gain override.",
        "! GAIN= 1.0",
        f"! STORAGE_PEDESTAL= {int(pedestal)}",
        "JOB= XYCORR INIT COLSPOT IDXREF DEFPIX INTEGRATE CORRECT",
        "TEST_RESOLUTION_RANGE= 10 1",
        "! INCLUDE_RESOLUTION_RANGE= 30 0.95",
        "DETECTOR= PILATUS",
        f"NX= {int(ncols)}",
        f"NY= {int(nrows)}",
        f"QX= {float(pixel_size_x_mm):.5f}",
        f"QY= {float(pixel_size_y_mm):.5f}",
        f"OVERLOAD= {int(overload_value)}",
        "MINIMUM_VALID_PIXEL_VALUE= 0",
        f"OFFSET= {int(xds_offset)}{offset_note}",
        "TRUSTED_REGION= 0.0 1.4142",
        "DIRECTION_OF_DETECTOR_X-AXIS= 1.0 0.0 0.0",
        "DIRECTION_OF_DETECTOR_Y-AXIS= 0.0 1.0 0.0",
        f"ORGX= {float(beam_center_x_px_1based):.2f}",
        f"ORGY= {float(beam_center_y_px_1based):.2f}",
        f"DETECTOR_DISTANCE= {float(detector_distance_mm):.3f}",
        f"NAME_TEMPLATE_OF_DATA_FRAMES= {name_template}",
        f"DATA_RANGE= 1 {data_end}",
        f"SPOT_RANGE= 1 {data_end}",
        f"BACKGROUND_RANGE= 1 {data_end}",
        f"ROTATION_AXIS= {rotation_axis_xds[0]:.5f} {rotation_axis_xds[1]:.5f} {rotation_axis_xds[2]:.5f}",
        f"STARTING_ANGLE= {float(start_angle_deg):.6f}",
        f"OSCILLATION_RANGE= {float(oscillation_range_deg):.6f}",
        "STARTING_FRAME= 1",
        f"X-RAY_WAVELENGTH= {float(wavelength_A):.6f}",
        "INCIDENT_BEAM_DIRECTION= 0.0 0.0 1.0",
        "FRACTION_OF_POLARIZATION= 0.0",
        "POLARIZATION_PLANE_NORMAL= 0.0 1.0 0.0",
        "DELPHI= 99",
    ]

    if legacy_offset is not None and int(legacy_offset) != int(xds_offset):
        lines.append(
            f"! REDUCED_BACKGROUND_OFFSET= {int(legacy_offset)} "
            "! Previous background-only/non-total offset estimate; informational only for new XDS."
        )
    if xds_offset_details:
        reduced_offset_details = xds_offset_details.get("reduced_offset_estimation") or {}
        selected_offset_basis = (
            xds_offset_details.get("selected_candidate")
            or xds_offset_details.get("selected_mode")
            or xds_offset_details.get("mode_value")
        )
        robust_sigma = xds_offset_details.get("robust_sigma", reduced_offset_details.get("robust_sigma", 0.0))
        sample_size_pixels = xds_offset_details.get(
            "sample_size_pixels",
            reduced_offset_details.get("sample_size_pixels"),
        )
        median_value = xds_offset_details.get("median_value", reduced_offset_details.get("median_value"))
        lines.append(
            f"! XDS_OFFSET_METHOD= {xds_offset_details.get('method', 'unknown')} "
            f"basis={selected_offset_basis} "
            f"median={median_value} "
            f"robust_sigma={float(robust_sigma or 0.0):.3f} "
            f"samples={sample_size_pixels}"
        )
    if suggested_exclude_ranges:
        lines.append("! Suggested frame exclusions from frame-quality screening:")
        for start_frame, end_frame in suggested_exclude_ranges:
            lines.append(f"! EXCLUDE_DATA_RANGE= {int(start_frame)} {int(end_frame)}")

    lines.extend(
        [
            "",
            "! Optional crystal inputs:",
            "! SPACE_GROUP_NUMBER= 0",
            "! UNIT_CELL_CONSTANTS=",
            "",
            "REFINE(IDXREF)= BEAM AXIS ORIENTATION CELL ! POSITION intentionally left off",
            "REFINE(INTEGRATE)= BEAM ORIENTATION ! POSITION intentionally left off",
            "REFINE(CORRECT)= BEAM ORIENTATION CELL ! POSITION intentionally left off",
            "MINIMUM_I/SIGMA= 1.0                    ! Minimum I/sigma for refinement",
            "",
        ]
    )

    _write_text_atomic(xds_path, "\n".join(lines) + "\n")
    return str(xds_path)


def write_xds_offsets_report(
    output_dir: Union[str, Path],
    *,
    storage_pedestal: int,
    current_xds_offset: int,
    legacy_xds_offset: int,
    current_xds_inp: Optional[str],
    legacy_xds_inp: Optional[str],
    xds_offset_details: Optional[Dict[str, Any]],
    reduced_background_offset: Optional[int] = None,
    reduced_background_offset_details: Optional[Dict[str, Any]] = None,
    storage_pedestal_report: Optional[Dict[str, Any]] = None,
) -> str:
    report_path = Path(output_dir) / "xds_offsets.json"
    payload = {
        "storage_pedestal": int(storage_pedestal),
        "current_xds": {
            "applies_to": "XDS versions after 20230630",
            "xds_inp": current_xds_inp,
            "offset": int(current_xds_offset),
        },
        "legacy_xds_20230630": {
            "applies_to": "XDS 20230630 and before",
            "xds_inp": legacy_xds_inp,
            "offset": int(legacy_xds_offset),
        },
        "offset_estimation": xds_offset_details,
        "storage_pedestal_estimation": storage_pedestal_report,
        "reduced_background_offset": {
            "offset": int(reduced_background_offset) if reduced_background_offset is not None else None,
            "note": "Previous background-only/non-total offset estimate; informational only for new XDS.",
            "estimation": reduced_background_offset_details,
        },
    }
    _write_text_atomic(report_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return str(report_path)


def write_conversion_report(
    output_dir: Union[str, Path],
    report_payload: Dict[str, Any],
) -> Optional[str]:
    output_dir = Path(output_dir)
    report_path = output_dir / "conversion_report.json"
    _write_text_atomic(report_path, json.dumps(report_payload, indent=2, sort_keys=True) + "\n")
    return str(report_path)


def write_xds_inp_fragment(output_dir: Union[str, Path], xds_inp_path: Optional[str]) -> Optional[str]:
    if not xds_inp_path or not Path(xds_inp_path).exists():
        return None
    keep_prefixes = (
        "! CONVERTER_CONTRACT=",
        "! STORAGE_PEDESTAL=",
        "! XDS_OFFSET_METHOD=",
        "JOB=",
        "DETECTOR=",
        "NX=",
        "NY=",
        "QX=",
        "QY=",
        "OVERLOAD=",
        "MINIMUM_VALID_PIXEL_VALUE=",
        "OFFSET=",
        "ORGX=",
        "ORGY=",
        "DETECTOR_DISTANCE=",
        "NAME_TEMPLATE_OF_DATA_FRAMES=",
        "DATA_RANGE=",
        "SPOT_RANGE=",
        "BACKGROUND_RANGE=",
        "ROTATION_AXIS=",
        "STARTING_ANGLE=",
        "OSCILLATION_RANGE=",
        "STARTING_FRAME=",
        "X-RAY_WAVELENGTH=",
        "REFINE(",
    )
    lines = ["! XDS.INP fragment generated for audit/copy review"]
    for line in Path(xds_inp_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(keep_prefixes):
            lines.append(line)
    fragment_path = Path(output_dir) / "XDS.INP.fragment"
    _write_text_atomic(fragment_path, "\n".join(lines) + "\n")
    return str(fragment_path)


def write_pixel_histogram_before_after_csv(
    output_dir: Union[str, Path],
    raw_signed_sample: NDArray,
    conditioned_storage_sample: NDArray,
    *,
    bins: int = 200,
) -> Optional[str]:
    raw = np.asarray(raw_signed_sample, dtype=np.float64)
    conditioned = np.asarray(conditioned_storage_sample, dtype=np.float64)
    raw = raw[np.isfinite(raw)]
    conditioned = conditioned[np.isfinite(conditioned)]
    if raw.size == 0 or conditioned.size == 0:
        return None
    combined = np.concatenate([raw, conditioned])
    low = float(np.min(combined))
    high = float(np.max(combined))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low -= 0.5
        high += 0.5
    edges = np.linspace(low, high, max(2, int(bins)) + 1)
    raw_counts, _ = np.histogram(raw, bins=edges)
    conditioned_counts, _ = np.histogram(conditioned, bins=edges)
    rows = ["bin_left,bin_right,raw_signed_count,conditioned_storage_count"]
    for idx in range(len(edges) - 1):
        rows.append(
            f"{edges[idx]:.6g},{edges[idx + 1]:.6g},{int(raw_counts[idx])},{int(conditioned_counts[idx])}"
        )
    path = Path(output_dir) / "pixel_histogram_before_after.csv"
    _write_text_atomic(path, "\n".join(rows) + "\n")
    return str(path)


def assess_offset_confidence(
    *,
    storage_pedestal: int,
    xds_offset: int,
    xds_offset_details: Optional[Dict[str, Any]],
    pixel_diagnostics: Optional[Dict[str, Any]],
    raw_signed_summary: Dict[str, Any],
    conditioned_storage_summary: Dict[str, Any],
) -> Dict[str, Any]:
    unsafe: List[str] = []
    warnings_out: List[str] = []
    method = (xds_offset_details or {}).get("method", "unknown")
    highres_clip = None
    highres_mean = None
    highres_sigma = None
    if pixel_diagnostics:
        highres_clip = pixel_diagnostics.get("fraction_highres_ic_zero_after_xds")
        highres_mean = pixel_diagnostics.get("highres_ic_mean_after_xds")
        highres_sigma = pixel_diagnostics.get("highres_ic_sigma_after_xds")
    if highres_clip is None:
        warnings_out.append("No high-resolution post-offset clipping diagnostic was available.")
    elif float(highres_clip) > 0.35:
        unsafe.append(
            f"High-resolution IC=0 fraction is {float(highres_clip):.3f}, above the 0.35 unsafe threshold."
        )
    elif float(highres_clip) > 0.25:
        warnings_out.append(
            f"High-resolution IC=0 fraction is {float(highres_clip):.3f}, above the 0.25 warning threshold."
        )
    if highres_mean is not None and highres_sigma is not None and float(highres_sigma) > 0:
        if float(highres_mean) <= 0.0:
            unsafe.append("High-resolution post-offset mean is not positive.")
        elif float(highres_mean) < 0.25 * float(highres_sigma):
            warnings_out.append("High-resolution post-offset mean is small relative to sigma.")
    if int(storage_pedestal) == int(xds_offset) and method != "storage_pedestal":
        warnings_out.append("Storage pedestal and XDS OFFSET are numerically equal even though the method says they are distinct.")
    if "quantile" in str(method).lower():
        warnings_out.append("Quantile offset mode is a diagnostic probe; it previously produced biased high-shell signal.")
    status = "unsafe" if unsafe else ("conditional_pass" if warnings_out else "pass")
    return {
        "status": status,
        "unsafe_reasons": unsafe,
        "warnings": warnings_out,
        "storage_pedestal": int(storage_pedestal),
        "xds_offset": int(xds_offset),
        "offset_minus_storage_pedestal": int(xds_offset) - int(storage_pedestal),
        "xds_offset_method": method,
        "raw_signed_summary": raw_signed_summary,
        "conditioned_storage_summary": conditioned_storage_summary,
        "pixel_diagnostics": pixel_diagnostics,
        "requires_xds_response_sweep": True,
        "requires_model_validation": True,
    }


def write_offset_confidence_report(
    output_dir: Union[str, Path],
    *,
    contract_name: str,
    contract: Dict[str, str],
    storage_pedestal: int,
    xds_offset: int,
    xds_offset_details: Optional[Dict[str, Any]],
    storage_pedestal_report: Optional[Dict[str, Any]],
    raw_signed_summary: Dict[str, Any],
    conditioned_storage_summary: Dict[str, Any],
    pixel_diagnostics: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    confidence = assess_offset_confidence(
        storage_pedestal=storage_pedestal,
        xds_offset=xds_offset,
        xds_offset_details=xds_offset_details,
        pixel_diagnostics=pixel_diagnostics,
        raw_signed_summary=raw_signed_summary,
        conditioned_storage_summary=conditioned_storage_summary,
    )
    reduced = (xds_offset_details or {}).get("reduced_offset_estimation") or {}
    candidates = reduced.get("candidate_offsets") or (xds_offset_details or {}).get("candidate_offsets") or {}
    lines = [
        "# Offset Confidence Report",
        "",
        f"Contract: `{contract_name}`",
        f"Contract rule: {contract.get('xds_offset_policy', 'unknown')}",
        f"Status: **{confidence['status']}**",
        "",
        "## Chosen Numbers",
        "",
        f"- `STORAGE_PEDESTAL`: `{int(storage_pedestal)}`",
        f"- `XDS_OFFSET`: `{int(xds_offset)}`",
        f"- Difference (`XDS_OFFSET - STORAGE_PEDESTAL`): `{int(xds_offset) - int(storage_pedestal)}`",
        f"- Offset method: `{(xds_offset_details or {}).get('method', 'unknown')}`",
        "",
        "These are separate quantities. The storage pedestal only makes the CBF nonnegative; XDS OFFSET is the value XDS subtracts before integration.",
        "",
        "## Raw And Conditioned Pixel Statistics",
        "",
        "| quantity | raw signed retained sample | conditioned storage sample |",
        "| --- | ---: | ---: |",
        f"| median | {_fmt_report_value(raw_signed_summary.get('median'))} | {_fmt_report_value(conditioned_storage_summary.get('median'))} |",
        f"| MAD | {_fmt_report_value(raw_signed_summary.get('mad'))} | {_fmt_report_value(conditioned_storage_summary.get('mad'))} |",
        f"| robust sigma | {_fmt_report_value(raw_signed_summary.get('robust_sigma'))} | {_fmt_report_value(conditioned_storage_summary.get('robust_sigma'))} |",
        f"| min | {_fmt_report_value(raw_signed_summary.get('min'))} | {_fmt_report_value(conditioned_storage_summary.get('min'))} |",
        f"| q001 | {_fmt_report_value(raw_signed_summary.get('q001'))} | {_fmt_report_value(conditioned_storage_summary.get('q001'))} |",
        f"| q999 | {_fmt_report_value(raw_signed_summary.get('q999'))} | {_fmt_report_value(conditioned_storage_summary.get('q999'))} |",
        "",
        "## High-Resolution Offset Diagnostics",
        "",
        f"- Conditioned high-resolution median: `{_fmt_report_value(reduced.get('median_value'))}`",
        f"- Conditioned high-resolution robust sigma: `{_fmt_report_value(reduced.get('robust_sigma'))}`",
        f"- Conditioned high-resolution median-minus-sigma candidate: `{_fmt_report_value(candidates.get('highres_median_minus_sigma'))}`",
        f"- Conditioned high-resolution quantile candidate: `{_fmt_report_value(candidates.get('highres_q'))}`",
        f"- Fraction of high-resolution pixels at IC=0 after XDS offset: `{_fmt_report_value((pixel_diagnostics or {}).get('fraction_highres_ic_zero_after_xds'))}`",
        f"- High-resolution IC mean after XDS offset: `{_fmt_report_value((pixel_diagnostics or {}).get('highres_ic_mean_after_xds'))}`",
        f"- High-resolution IC sigma after XDS offset: `{_fmt_report_value((pixel_diagnostics or {}).get('highres_ic_sigma_after_xds'))}`",
        "",
        "## Health Gate",
        "",
    ]
    if confidence["unsafe_reasons"]:
        lines.extend([f"- UNSAFE: {item}" for item in confidence["unsafe_reasons"]])
    if confidence["warnings"]:
        lines.extend([f"- WARNING: {item}" for item in confidence["warnings"]])
    if not confidence["unsafe_reasons"] and not confidence["warnings"]:
        lines.append("- No pre-XDS offset health warnings were triggered.")
    lines.extend(
        [
            "- Required next check: run the short XDS response sweep around the chosen offset.",
            "- Required final check: compare model-vs-data correlation against CC* when a refined model is available.",
            "",
            "## Storage Pedestal Estimation",
            "",
            "```json",
            json.dumps(storage_pedestal_report or {}, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    path = Path(output_dir) / "offset_confidence_report.md"
    _write_text_atomic(path, "\n".join(lines))
    return str(path), confidence


def write_background_model_report(
    output_dir: Union[str, Path],
    *,
    contract_name: str,
    background_conditioning_report: Dict[str, Any],
) -> str:
    storage_report = background_conditioning_report.get("storage_pedestal") or {}
    xds_details = background_conditioning_report.get("xds_offset_details") or {}
    diagnostics = background_conditioning_report.get("pixel_diagnostics") or {}
    lines = [
        "# Background Model Report",
        "",
        f"Contract: `{contract_name}`",
        f"Conditioning mode: `{background_conditioning_report.get('mode', 'unknown')}`",
        "",
        "The background model is deliberately smooth. It is meant to remove slow radial/planar detector background without fitting Bragg spots or narrow diffraction features.",
        "",
        "## Storage And Offset",
        "",
        f"- Conditioned storage pedestal: `{_fmt_report_value(storage_report.get('suggested_pedestal'))}`",
        f"- Auto pedestal before manual override: `{_fmt_report_value(storage_report.get('auto_pedestal'))}`",
        f"- Residual values clipped before storage: `{_fmt_report_value(storage_report.get('negative_count_before_storage_clip'))}`",
        f"- Residual clipped fraction: `{_fmt_report_value(storage_report.get('negative_fraction_before_storage_clip'))}`",
        f"- XDS offset method: `{xds_details.get('method', 'unknown')}`",
        f"- XDS offset: `{_fmt_report_value(xds_details.get('suggested_offset'))}`",
        "",
        "## Pixel Diagnostics",
        "",
        f"- High-resolution IC=0 fraction after offset: `{_fmt_report_value(diagnostics.get('fraction_highres_ic_zero_after_xds'))}`",
        f"- High-resolution IC mean after offset: `{_fmt_report_value(diagnostics.get('highres_ic_mean_after_xds'))}`",
        f"- High-resolution IC sigma after offset: `{_fmt_report_value(diagnostics.get('highres_ic_sigma_after_xds'))}`",
        "",
        "## Sampled Background Fits",
        "",
        "```json",
        json.dumps(
            {
                "radial_parameters": xds_details.get("radial_parameters"),
                "background_sample_reports": background_conditioning_report.get("background_sample_reports"),
                "radial_residual_summary": background_conditioning_report.get("radial_residual_summary"),
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
    ]
    path = Path(output_dir) / "background_model_report.md"
    _write_text_atomic(path, "\n".join(lines))
    return str(path)


def write_frame_timing_audit(
    output_dir: Union[str, Path],
    *,
    mrc_path: str,
    filename_template: str,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    output_frame_origin_0_based: int,
    bin_z: int,
    start_angle_deg: float,
    oscillation_range_deg: float,
    frame_metadata_sources: Dict[str, Any],
    oscillation_validation_report: Optional[Dict[str, Any]],
    xml_metadata: Optional[Dict[str, Any]],
    mdoc_metadata: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    num_frames = int(output_shape[0])
    mrc_depth = int(input_shape[0]) if input_shape else None
    xml_fraction_count = (xml_metadata or {}).get("fraction_count")
    mdoc_num_subframes = (mdoc_metadata or {}).get("numsubframes") if mdoc_metadata else None
    sample_indices = sorted(
        {
            idx
            for idx in [0, 1, 2, num_frames // 2, num_frames - 3, num_frames - 2, num_frames - 1]
            if 0 <= idx < num_frames
        }
    )
    mapping_rows: List[Dict[str, Any]] = []
    for idx in sample_indices:
        original_output_idx = int(output_frame_origin_0_based) + int(idx)
        mrc_first = original_output_idx * int(bin_z) + 1
        mrc_last = mrc_first + int(bin_z) - 1
        angle_start = float(start_angle_deg) + float(idx) * float(oscillation_range_deg)
        angle_end = angle_start + float(oscillation_range_deg)
        mapping_rows.append(
            {
                "xds_frame_1_based": int(idx + 1),
                "mrc_input_frame_range_1_based": [int(mrc_first), int(mrc_last)],
                "angle_start_deg": angle_start,
                "angle_center_deg": 0.5 * (angle_start + angle_end),
                "angle_end_deg": angle_end,
                "filename": filename_template.format(idx + 1),
            }
        )
    report = {
        "mrc_path": str(mrc_path),
        "mrc_stack_depth": mrc_depth,
        "mdoc_num_subframes": int(mdoc_num_subframes) if isinstance(mdoc_num_subframes, (int, float)) and np.isfinite(mdoc_num_subframes) else None,
        "xml_fraction_count": int(xml_fraction_count) if isinstance(xml_fraction_count, (int, float)) and np.isfinite(xml_fraction_count) else None,
        "written_frames": num_frames,
        "output_frame_origin_0_based_after_trim": int(output_frame_origin_0_based),
        "software_bin_z": int(bin_z),
        "start_angle_deg": float(start_angle_deg),
        "oscillation_range_deg_per_written_frame": float(oscillation_range_deg),
        "total_angular_range_deg": float(num_frames) * float(oscillation_range_deg),
        "frame_metadata_sources": frame_metadata_sources,
        "oscillation_wedge_validation": oscillation_validation_report,
        "mapping_rows": mapping_rows,
    }
    status = (oscillation_validation_report or {}).get("status", "unchecked")
    ok = (oscillation_validation_report or {}).get("ok_for_constant_wedge")
    lines = [
        "# Frame Timing Audit",
        "",
        f"Status: **{status}**",
        f"OK for constant XDS wedge: `{_fmt_report_value(ok)}`",
        "",
        "## Counts And Wedge",
        "",
        f"- MRC stack depth: `{_fmt_report_value(mrc_depth)}`",
        f"- MDOC `NumSubFrames`: `{_fmt_report_value(report['mdoc_num_subframes'])}`",
        f"- XML fraction count: `{_fmt_report_value(report['xml_fraction_count'])}`",
        f"- Written CBF frames: `{num_frames}`",
        f"- Start angle: `{_fmt_report_value(start_angle_deg)}` deg",
        f"- Oscillation per written frame: `{_fmt_report_value(oscillation_range_deg)}` deg",
        f"- Total angular range: `{_fmt_report_value(report['total_angular_range_deg'])}` deg",
        "",
        "## Timing Summary",
        "",
        "```json",
        json.dumps((oscillation_validation_report or {}).get("xml_timing_summary", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## XDS Frame Mapping Spot Check",
        "",
        "| XDS frame | MRC input frame(s) | angle start | angle center | angle end | filename |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in mapping_rows:
        frame_range = row["mrc_input_frame_range_1_based"]
        lines.append(
            f"| {row['xds_frame_1_based']} | {frame_range[0]}-{frame_range[1]} | "
            f"{row['angle_start_deg']:.6f} | {row['angle_center_deg']:.6f} | {row['angle_end_deg']:.6f} | `{row['filename']}` |"
        )
    warnings_out = list((oscillation_validation_report or {}).get("warnings") or [])
    if warnings_out:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in warnings_out])
    path = Path(output_dir) / "frame_timing_audit.md"
    _write_text_atomic(path, "\n".join(lines) + "\n")
    return str(path), report


def write_xds_smoke_score_stub(
    output_dir: Union[str, Path],
    *,
    xds_inp_path: Optional[str],
    xds_offset: int,
) -> str:
    helper = Path(__file__).with_name("xds_offset_response_sweep.py")
    command = (
        f"/usr/bin/python3 {shlex.quote(str(helper))} --xds-inp {shlex.quote(str(xds_inp_path or 'XDS.INP'))} "
        f"--center-offset {int(xds_offset)} --deltas=-4,-2,0,2,4 "
        "--xds-par /home/mike/xds_new/xds_par --root xds_offset_response_sweep"
    )
    lines = [
        "# XDS Smoke Score",
        "",
        "Status: **not_run**",
        "",
        "This conversion is not fully trusted until a short XDS response sweep confirms that the chosen offset is not sitting on a cliff.",
        "",
        "Suggested command on cc02:",
        "",
        "```bash",
        command,
        "```",
        "",
        "The score should penalize both known failure modes: high-resolution I/sigma staying artificially high when CC1/2 is poor, and I/sigma collapsing to zero while CC1/2 or Rmeas remains nonphysical.",
        "",
    ]
    path = Path(output_dir) / "xds_smoke_score.md"
    _write_text_atomic(path, "\n".join(lines))
    return str(path)


def write_model_validation_hook_report(output_dir: Union[str, Path]) -> str:
    helper = Path(__file__).with_name("model_vs_data_validation.py")
    lines = [
        "# Model-vs-Data Validation Hook",
        "",
        "Status: **pending_model**",
        "",
        "This converter run is only a candidate until a refined model is checked against the processed data. The required comparison is shell-wise model-vs-data correlation against CC*; prettier XDS I/sigma alone is not an acceptance criterion.",
        "",
        "Suggested command after XDS has produced `XDS_ASCII.HKL` and `CORRECT.LP`:",
        "",
        "```bash",
        f"/usr/bin/env python3 {shlex.quote(str(helper))} --output-dir model_vs_data_validation --case 'dataset|XDS_ASCII.HKL|CORRECT.LP|refined_model.cif'",
        "```",
        "",
        "Minimum fields to fill after refinement:",
        "",
        "| shell | CC1/2 | CC* | model-vs-data CC | verdict |",
        "| --- | ---: | ---: | ---: | --- |",
        "| low |  |  |  |  |",
        "| mid |  |  |  |  |",
        "| high |  |  |  |  |",
        "",
        "A mode that improves XDS tables but worsens model correlation or refinement statistics should be rejected.",
        "",
    ]
    path = Path(output_dir) / "model_validation_hook.md"
    _write_text_atomic(path, "\n".join(lines))
    return str(path)


def write_conversion_manifest(
    output_dir: Union[str, Path],
    manifest_payload: Dict[str, Any],
) -> str:
    path = Path(output_dir) / "conversion_manifest.json"
    _write_text_atomic(path, json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n")
    return str(path)

# --- Gaussian Fitting ---
def gaussian_2d(coords: Tuple[NDArray[np.float64], NDArray[np.float64]],
                amplitude: float, center_x: float, center_y: float,
                sigma_x: float, sigma_y: float, offset: float) -> NDArray[np.float64]:
    x, y = coords; xo, yo = center_x, center_y
    exponent = -(((x - xo)**2) / (2 * sigma_x**2) + ((y - yo)**2) / (2 * sigma_y**2))
    return (offset + amplitude * np.exp(exponent)).ravel()

# --- Beam Center Finding ---
def find_beam_center_blurred_peak(
    image: NDArray[np.float64], roi_size: int = 128, sigma_blur: float = 5.0,
    max_initial_deviation: float = 80.0, fit_bounds: bool = True
) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    del fit_bounds
    if image.ndim != 2:
        raise ValueError("Input image must be 2D.")
    if roi_size <= 0 or sigma_blur < 0:
        raise ValueError("roi_size must be positive, sigma_blur must be non-negative.")

    img_h, img_w = image.shape
    center_y, center_x = img_h // 2, img_w // 2
    half_roi = roi_size // 2
    roi_y_start = max(0, center_y - half_roi)
    roi_y_end = min(img_h, center_y + half_roi + (roi_size % 2))
    roi_x_start = max(0, center_x - half_roi)
    roi_x_end = min(img_w, center_x + half_roi + (roi_size % 2))
    roi = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    roi_h, roi_w = roi.shape
    if roi_h == 0 or roi_w == 0:
        logging.warning(f"Beam centering: ROI size ({roi_size}) resulted in an empty array. Skipping peak find.")
        return None, None, {'status': 'Failed (Empty ROI)', 'method': 'Blurred Peak'}

    roi_center_y, roi_center_x = roi_h / 2.0, roi_w / 2.0
    blurred_roi = gaussian_filter(roi, sigma=sigma_blur)
    try:
        peak_idx_flat = np.argmax(blurred_roi)
        peak_y_roi, peak_x_roi = np.unravel_index(peak_idx_flat, blurred_roi.shape)
    except ValueError:
        logging.warning("Beam centering: Could not find peak in blurred ROI. Skipping peak find.")
        return None, None, {'status': 'Failed (argmax error)', 'method': 'Blurred Peak'}

    deviation = np.sqrt((peak_y_roi - roi_center_y) ** 2 + (peak_x_roi - roi_center_x) ** 2)
    if deviation > max_initial_deviation:
        logging.warning(
            f"Beam centering: Initial peak ({peak_x_roi:.1f}, {peak_y_roi:.1f}) in ROI is {deviation:.1f} px from ROI center. Skipping peak find."
        )
        return None, None, {'status': 'Failed (Initial Peak Too Far)', 'method': 'Blurred Peak'}

    beam_center_x = roi_x_start + float(peak_x_roi)
    beam_center_y = roi_y_start + float(peak_y_roi)
    if not (0 <= beam_center_x < img_w and 0 <= beam_center_y < img_h):
        logging.error("Beam centering: Calculated blurred peak outside image bounds.")
        return None, None, {'status': 'Failed (Outside Bounds)', 'method': 'Blurred Peak'}

    return beam_center_y, beam_center_x, {
        'status': 'Peak Success',
        'method': 'Blurred Peak',
        'roi_coords': (roi_x_start, roi_y_start),
        'roi_shape': roi.shape,
        'peak_xy_roi': (float(peak_x_roi), float(peak_y_roi)),
        'peak_xy_image': (float(beam_center_x), float(beam_center_y)),
    }


def find_beam_center_robust(
    image: NDArray[np.float64], roi_size: int = 128, sigma_blur: float = 5.0,
    max_initial_deviation: float = 80.0, fit_bounds: bool = True
) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    del fit_bounds
    peak_y, peak_x, peak_details = find_beam_center_blurred_peak(
        image,
        roi_size=roi_size,
        sigma_blur=sigma_blur,
        max_initial_deviation=max_initial_deviation,
        fit_bounds=False,
    )
    if peak_y is None or peak_x is None:
        return peak_y, peak_x, peak_details

    img_h, img_w = image.shape
    center_y, center_x = img_h // 2, img_w // 2
    half_roi = roi_size // 2
    roi_y_start = max(0, center_y - half_roi)
    roi_y_end = min(img_h, center_y + half_roi + (roi_size % 2))
    roi_x_start = max(0, center_x - half_roi)
    roi_x_end = min(img_w, center_x + half_roi + (roi_size % 2))
    roi = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    blurred_roi = gaussian_filter(roi, sigma=sigma_blur)

    peak_x_roi = float(peak_x - roi_x_start)
    peak_y_roi = float(peak_y - roi_y_start)
    yy, xx = np.indices(roi.shape, dtype=np.float64)
    rr = np.sqrt((yy - peak_y_roi) ** 2 + (xx - peak_x_roi) ** 2)

    background = float(np.percentile(blurred_roi, 20.0))
    high_clip = float(np.percentile(blurred_roi, 99.5))
    clipped_roi = np.clip(blurred_roi.astype(np.float64), background, high_clip)
    weights = np.clip(clipped_roi - background, 0.0, None)
    if not np.any(weights > 0):
        return peak_y, peak_x, {
            **peak_details,
            "status": "Robust fallback to peak (no positive halo weights)",
            "method": "Robust Halo COM Fallback",
        }

    core_exclusion_px = max(2.0, 1.5 * float(sigma_blur))
    halo_radius_px = min(max(core_exclusion_px + 3.0, 6.0 * float(sigma_blur)), min(roi.shape) / 2.0)
    robust_x_roi = peak_x_roi
    robust_y_roi = peak_y_roi
    halo_weights = weights
    weight_sum = 0.0
    iterations_used = 0
    for iteration in range(2):
        rr_iter = np.sqrt((yy - robust_y_roi) ** 2 + (xx - robust_x_roi) ** 2)
        halo_mask = (rr_iter >= core_exclusion_px) & (rr_iter <= halo_radius_px)
        halo_weights = np.where(halo_mask, weights, 0.0)
        if np.count_nonzero(halo_weights) < 16 or float(np.sum(halo_weights)) <= 0.0:
            halo_weights = weights

        weight_sum = float(np.sum(halo_weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            return peak_y, peak_x, {
                **peak_details,
                "status": "Robust fallback to peak (invalid halo weight sum)",
                "method": "Robust Halo COM Fallback",
            }

        next_x_roi = float(np.sum(xx * halo_weights) / weight_sum)
        next_y_roi = float(np.sum(yy * halo_weights) / weight_sum)
        if not np.isfinite(next_x_roi) or not np.isfinite(next_y_roi):
            return peak_y, peak_x, {
                **peak_details,
                "status": "Robust fallback to peak (non-finite COM)",
                "method": "Robust Halo COM Fallback",
            }
        robust_x_roi = next_x_roi
        robust_y_roi = next_y_roi
        iterations_used = iteration + 1

    robust_deviation = float(np.sqrt((robust_x_roi - peak_x_roi) ** 2 + (robust_y_roi - peak_y_roi) ** 2))
    if robust_deviation > max(3.0, halo_radius_px):
        return peak_y, peak_x, {
            **peak_details,
            "status": f"Robust fallback to peak (halo COM drift {robust_deviation:.2f} px)",
            "method": "Robust Halo COM Fallback",
        }

    beam_center_x = roi_x_start + robust_x_roi
    beam_center_y = roi_y_start + robust_y_roi
    if not (0 <= beam_center_x < img_w and 0 <= beam_center_y < img_h):
        return peak_y, peak_x, {
            **peak_details,
            "status": "Robust fallback to peak (outside bounds)",
            "method": "Robust Halo COM Fallback",
        }

    return beam_center_y, beam_center_x, {
        "status": "Robust Halo COM Success",
        "method": "Robust Halo COM",
        "roi_coords": (roi_x_start, roi_y_start),
        "roi_shape": roi.shape,
        "peak_xy_roi": (peak_x_roi, peak_y_roi),
        "peak_xy_image": (float(peak_x), float(peak_y)),
        "center_xy_roi": (robust_x_roi, robust_y_roi),
        "center_xy_image": (float(beam_center_x), float(beam_center_y)),
        "background_level": background,
        "high_clip_level": high_clip,
        "core_exclusion_px": core_exclusion_px,
        "halo_radius_px": halo_radius_px,
        "iterations_used": iterations_used,
        "used_halo_pixels": int(np.count_nonzero(halo_weights)),
        "weight_sum": weight_sum,
    }


def find_beam_center_gaussian_fit(
    image: NDArray[np.float64], roi_size: int = 100, sigma_blur: float = 3.0,
    max_initial_deviation: float = 50.0, fit_bounds: bool = True
) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    if image.ndim != 2: raise ValueError("Input image must be 2D.")
    if roi_size <= 0 or sigma_blur < 0: raise ValueError("roi_size must be positive, sigma_blur must be non-negative.")
    img_h, img_w = image.shape; center_y, center_x = img_h // 2, img_w // 2
    half_roi = roi_size // 2
    roi_y_start = max(0, center_y - half_roi); roi_y_end = min(img_h, center_y + half_roi + (roi_size % 2))
    roi_x_start = max(0, center_x - half_roi); roi_x_end = min(img_w, center_x + half_roi + (roi_size % 2))
    roi = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]; roi_h, roi_w = roi.shape
    if roi_h == 0 or roi_w == 0:
        logging.warning(f"Beam centering: ROI size ({roi_size}) resulted in an empty array. Skipping fit.")
        return None, None, {'status': 'Failed (Empty ROI)'}
    roi_center_y, roi_center_x = roi_h / 2.0, roi_w / 2.0
    blurred_roi = gaussian_filter(roi, sigma=sigma_blur)
    try: 
        peak_idx_flat = np.argmax(blurred_roi)
        peak_y_roi, peak_x_roi = np.unravel_index(peak_idx_flat, blurred_roi.shape)
    except ValueError: 
        logging.warning("Beam centering: Could not find peak in blurred ROI. Skipping fit.")
        return None, None, {'status': 'Failed (argmax error)'}
    deviation = np.sqrt((peak_y_roi - roi_center_y)**2 + (peak_x_roi - roi_center_x)**2)
    if deviation > max_initial_deviation:
        logging.warning(f"Beam centering: Initial peak ({peak_x_roi:.1f}, {peak_y_roi:.1f}) in ROI is {deviation:.1f} px from ROI center. Skipping fit.")
        return None, None, {'status': 'Failed (Initial Peak Too Far)'}
    y_roi_coords, x_roi_coords = np.indices(roi.shape); roi_data_flat = roi.ravel()
    initial_amplitude = blurred_roi[peak_y_roi, peak_x_roi] - np.percentile(blurred_roi, 10)
    if initial_amplitude <= 0: initial_amplitude = 1.0
    initial_guess = (initial_amplitude, peak_x_roi, peak_y_roi, max(1.0, roi_w / 10.0), max(1.0, roi_h / 10.0), np.percentile(roi, 10))
    bounds = (-np.inf, np.inf)
    if fit_bounds: bounds = ([0, 0, 0, 0.5, 0.5, -np.inf], [np.inf, roi_w, roi_h, roi_w, roi_h, np.inf])
    fit_details = {'initial_guess': initial_guess, 'bounds': bounds if fit_bounds else 'None', 'roi_coords': (roi_x_start, roi_y_start),
                   'roi_shape': roi.shape, 'popt': None, 'pcov': None, 'status': '', 'method': ''}
    try:
        popt, pcov = curve_fit(gaussian_2d, (x_roi_coords, y_roi_coords), roi_data_flat, p0=initial_guess, bounds=bounds if fit_bounds else (-np.inf, np.inf), maxfev=5000)
        if fit_bounds:
             if not (bounds[0][1] < popt[1] < bounds[1][1] and bounds[0][2] < popt[2] < bounds[1][2]): 
                 logging.warning(f"Beam centering: Fit center ({popt[1]:.2f}, {popt[2]:.2f}) hit ROI boundary.")
             if popt[3] > roi_w or popt[4] > roi_h: 
                 logging.warning(f"Beam centering: Fit sigmas ({popt[3]:.2f}, {popt[4]:.2f}) seem large for ROI.")
        fit_center_x_roi, fit_center_y_roi = popt[1], popt[2]
        fit_details.update({'popt': popt, 'pcov': pcov, 'status': 'Fit Success', 'method': 'Gaussian Fit'})
    except RuntimeError as e: 
        logging.warning(f"Beam centering: Gaussian fit failed (RuntimeError: {e}). Falling back.")
        fit_center_x_roi, fit_center_y_roi = peak_x_roi, peak_y_roi
        fit_details.update({'status': 'Fit Failed (RuntimeError)', 'method': 'Blurred Peak Fallback'})
    except ValueError as e: 
        logging.warning(f"Beam centering: Gaussian fit failed (ValueError: {e}). Falling back.")
        fit_center_x_roi, fit_center_y_roi = peak_x_roi, peak_y_roi
        fit_details.update({'status': 'Fit Failed (Bounds Error)', 'method': 'Blurred Peak Fallback'})
    except Exception as e: 
        logging.error(f"Beam centering: Unexpected error: {e}", exc_info=True)
        fit_center_x_roi, fit_center_y_roi = peak_x_roi, peak_y_roi
        fit_details.update({'status': 'Fit Failed (Unexpected)', 'method': 'Blurred Peak Fallback'})
    beam_center_x = roi_x_start + fit_center_x_roi
    beam_center_y = roi_y_start + fit_center_y_roi
    if not (0 <= beam_center_x < img_w and 0 <= beam_center_y < img_h): 
        logging.error(f"Beam centering: Calculated center outside image bounds.")
        return None, None, {**fit_details, 'status': 'Failed (Outside Bounds)'}
    return beam_center_y, beam_center_x, fit_details

# --- Image Shifting ---
def shift_image(image: NDArray, current_center_yx: Tuple[float, float], target_center_yx: Tuple[float, float],
                order: int = 1, mode: str = 'constant', cval: float = 0.0) -> Tuple[NDArray, Tuple[float, float]]:
    current_y, current_x = current_center_yx
    target_y, target_x = target_center_yx
    shift_vector_yx = (target_y - current_y, target_x - current_x)
    shifted_image = ndimage_shift(image, shift=shift_vector_yx, order=order, mode=mode, cval=cval)
    return shifted_image, shift_vector_yx

# --- Array Binning ---
def bin_array(data: NDArray, bin_z: int, bin_y: int, bin_x: int) -> Tuple[NDArray, Tuple[int, int, int]]:
    if not all(isinstance(b, int) and b >= 1 for b in [bin_z, bin_y, bin_x]): raise ValueError("Bin factors must be positive integers.")
    if data.ndim != 3: raise ValueError(f"Input data must be 3D, got {data.ndim}D.")
    nz, ny, nx = data.shape
    nz_new, ny_new, nx_new = nz // bin_z, ny // bin_y, nx // bin_x
    if nz % bin_z != 0 or ny % bin_y != 0 or nx % bin_x != 0: 
        logging.warning(f"Binning: Data shape not perfectly divisible. Trailing slices discarded.")
    trimmed_data = data[:nz_new * bin_z, :ny_new * bin_y, :nx_new * bin_x]
    shape = (nz_new, bin_z, ny_new, bin_y, nx_new, bin_x)
    binned_data = trimmed_data.reshape(shape).sum(axis=(5, 3, 1))
    return binned_data, (nz_new, ny_new, nx_new)


def auto_software_bin_for_target(pixels: Optional[int], target_size_px: int = 1024) -> int:
    if pixels is None or int(pixels) <= 0 or int(target_size_px) <= 0:
        return 1
    pixels_int = int(pixels)
    if pixels_int <= int(target_size_px):
        return 1
    return max(1, int(math.ceil(pixels_int / float(target_size_px))))


def estimate_storage_pedestal(
    data: NDArray,
    *,
    negative_quantile: float = 0.0,
) -> Tuple[int, Dict[str, Any]]:
    arr = np.asarray(data)
    min_value = float(np.min(arr))
    quantile = min(1.0, max(0.0, float(negative_quantile)))
    method = "minimum" if quantile <= 0.0 else "negative_quantile"
    report: Dict[str, Any] = {
        "method": method,
        "negative_quantile": float(quantile),
        "min_value": min_value,
        "negative_count": 0,
        "negative_fraction": 0.0,
        "selected_negative_value": None,
        "suggested_pedestal": 0,
        "residual_negative_count_after_pedestal": 0,
    }
    if min_value >= 0:
        report["method"] = "none_needed"
        return 0, report

    negative = arr[arr < 0].astype(np.float64, copy=False)
    if negative.size == 0:
        report["method"] = "none_needed"
        return 0, report

    selected_negative = float(np.quantile(negative, quantile))
    suggested_pedestal = int(max(0, math.ceil(-selected_negative)))
    residual_negative = int(np.count_nonzero(negative + float(suggested_pedestal) < 0))
    report.update(
        {
            "negative_count": int(negative.size),
            "negative_fraction": float(negative.size / arr.size),
            "selected_negative_value": selected_negative,
            "suggested_pedestal": int(suggested_pedestal),
            "residual_negative_count_after_pedestal": residual_negative,
            "residual_negative_fraction_of_negative_pixels": float(residual_negative / negative.size),
        }
    )
    return suggested_pedestal, report


def estimate_xds_offset_from_outer_shell(
    stored_data: NDArray,
    sample_frames: int = 5,
    inner_fraction: float = 0.85,
    outer_fraction: float = 0.95,
) -> Dict[str, Any]:
    if stored_data.ndim != 3:
        raise ValueError(f"Expected 3D stack for XDS offset estimate, got {stored_data.ndim}D")
    num_frames, nrows, ncols = stored_data.shape
    if num_frames <= 0:
        raise ValueError("Stored stack is empty")
    if not (0.0 <= inner_fraction < outer_fraction <= 1.0):
        raise ValueError("Outer-shell fractions must satisfy 0 <= inner < outer <= 1")

    sample_frames = max(1, min(int(sample_frames), num_frames))
    frame_indices = np.unique(np.linspace(0, num_frames - 1, sample_frames, dtype=int))

    center_y = (nrows - 1) / 2.0
    center_x = (ncols - 1) / 2.0
    y_idx, x_idx = np.indices((nrows, ncols))
    radii = np.sqrt((y_idx - center_y) ** 2 + (x_idx - center_x) ** 2)
    max_radius = float(np.max(radii))
    annulus_mask = (radii >= inner_fraction * max_radius) & (radii <= outer_fraction * max_radius)
    if not np.any(annulus_mask):
        raise ValueError("Outer-shell annulus mask is empty")

    sampled = [np.asarray(stored_data[idx][annulus_mask], dtype=np.int32).ravel() for idx in frame_indices]
    sampled = [arr for arr in sampled if arr.size > 0]
    if not sampled:
        raise ValueError("No outer-shell pixels collected for XDS offset estimate")

    values = np.concatenate(sampled)
    min_value = int(np.min(values))
    max_value = int(np.max(values))
    histogram_bins = np.arange(min_value, max_value + 2, dtype=np.int64)
    hist, edges = np.histogram(values, bins=histogram_bins)
    mode_value = int(edges[:-1][int(np.argmax(hist))])

    median_value = float(np.median(values))
    mad_value = float(np.median(np.abs(values.astype(np.float64) - median_value)))
    robust_sigma = float(mad_value * 1.4826)
    sample_std = float(np.std(values.astype(np.float64)))
    sigma_for_offset = robust_sigma if np.isfinite(robust_sigma) and robust_sigma > 0 else sample_std
    mode_minus_sigma_offset = max(min_value, int(np.floor(mode_value - sigma_for_offset)))
    mode_offset = int(mode_value)
    median_offset = int(round(median_value))

    return {
        "method": "outer_shell_mode_minus_sigma",
        "sampled_frames": [int(v) for v in frame_indices.tolist()],
        "sample_size_pixels": int(values.size),
        "inner_fraction": float(inner_fraction),
        "outer_fraction": float(outer_fraction),
        "min_value": min_value,
        "max_value": max_value,
        "mode_value": mode_value,
        "median_value": median_value,
        "mad_value": mad_value,
        "robust_sigma": robust_sigma,
        "std_value": sample_std,
        "suggested_offset": int(mode_minus_sigma_offset),
        "candidate_offsets": {
            "outer_shell_mode": mode_offset,
            "outer_shell_mode_minus_sigma": int(mode_minus_sigma_offset),
            "outer_shell_median": median_offset,
        },
    }


def _robust_median_sigma(values: NDArray) -> Tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    median_value = float(np.median(finite))
    mad_value = float(np.median(np.abs(finite - median_value)))
    sigma_value = float(1.4826 * mad_value)
    if not np.isfinite(sigma_value) or sigma_value <= 0:
        sigma_value = float(np.std(finite))
    if not np.isfinite(sigma_value) or sigma_value <= 0:
        sigma_value = 1.0
    return median_value, sigma_value


def _fill_profile_gaps(profile: NDArray) -> NDArray:
    out = np.asarray(profile, dtype=np.float64).copy()
    idx = np.arange(out.size, dtype=np.float64)
    valid = np.isfinite(out)
    if not np.any(valid):
        out.fill(0.0)
        return out
    if np.count_nonzero(valid) == 1:
        out.fill(float(out[valid][0]))
        return out
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


def _sample_indices(length: int, max_count: int) -> NDArray[np.int64]:
    length = int(length)
    max_count = int(max_count)
    if length <= 0:
        return np.asarray([], dtype=np.int64)
    if max_count <= 0 or length <= max_count:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, max_count, dtype=np.int64)


def _radial_index_for_center(
    nrows: int,
    ncols: int,
    center_yx: Tuple[float, float],
    bin_width_px: float,
) -> NDArray[np.int32]:
    yy, xx = np.indices((int(nrows), int(ncols)), dtype=np.float32)
    center_y, center_x = center_yx
    radii = np.sqrt((yy - float(center_y)) ** 2 + (xx - float(center_x)) ** 2)
    return np.floor(radii / max(float(bin_width_px), 1.0)).astype(np.int32)


def estimate_radial_plane_background(
    frame: NDArray,
    center_yx: Tuple[float, float],
    *,
    bin_width_px: float = 2.0,
    radial_quantile: float = 0.50,
    profile_smooth_sigma_bins: float = 4.0,
    mask_smooth_sigma_px: float = 18.0,
    clip_high_sigma: float = 5.0,
    clip_low_sigma: float = 8.0,
    mask_dilation_px: int = 2,
    center_mask_radius_px: float = 28.0,
    fit_plane: bool = True,
    plane_sample_pixels: int = 120_000,
) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
    """Estimate only the slow detector/scattering background for one frame.

    The mask is deliberately built from a broad blurred residual rather than a
    sharp radial model. That keeps Bragg spots, beamstop dips, and isolated
    detector defects out of the fit without letting the radial profile chase
    spot-scale features. The returned field should be smooth on scales much
    wider than a Bragg spot.
    """
    work = np.asarray(frame, dtype=np.float32)
    nrows, ncols = work.shape
    radius_idx = _radial_index_for_center(nrows, ncols, center_yx, bin_width_px)
    n_bins = int(radius_idx.max()) + 1

    smooth = gaussian_filter(work, sigma=float(mask_smooth_sigma_px), mode="nearest") if mask_smooth_sigma_px > 0 else work
    residual_for_mask = work - smooth
    valid = np.isfinite(work)
    if center_mask_radius_px > 0:
        valid &= (radius_idx.astype(np.float32) * float(bin_width_px)) >= float(center_mask_radius_px)

    median_value, sigma_value = _robust_median_sigma(residual_for_mask[valid])
    valid &= residual_for_mask <= median_value + float(clip_high_sigma) * sigma_value
    valid &= residual_for_mask >= median_value - float(clip_low_sigma) * sigma_value
    if mask_dilation_px > 0:
        rejected = ~valid
        rejected = binary_dilation(rejected, iterations=int(mask_dilation_px))
        valid = ~rejected
        if center_mask_radius_px > 0:
            valid &= (radius_idx.astype(np.float32) * float(bin_width_px)) >= float(center_mask_radius_px)

    profile = np.full(n_bins, np.nan, dtype=np.float64)
    valid_bins = radius_idx[valid]
    valid_values = work[valid]
    if valid_values.size:
        unique_bins = np.unique(valid_bins)
        quantile = min(1.0, max(0.0, float(radial_quantile)))
        for ridx in unique_bins:
            ring_values = valid_values[valid_bins == ridx]
            if ring_values.size:
                profile[int(ridx)] = float(np.quantile(ring_values, quantile))
    profile = _fill_profile_gaps(profile)
    if profile_smooth_sigma_bins > 0:
        profile = gaussian_filter1d(profile, sigma=float(profile_smooth_sigma_bins), mode="nearest")
    background = profile[radius_idx].astype(np.float32, copy=False)

    plane_coefficients = [0.0, 0.0, 0.0]
    if fit_plane and np.count_nonzero(valid) >= 1024:
        resid_after_radial = (work - background).astype(np.float64, copy=False)
        flat_valid = np.flatnonzero(valid.ravel())
        sample_flat = flat_valid[_sample_indices(flat_valid.size, int(plane_sample_pixels))]
        if sample_flat.size >= 256:
            yy, xx = np.indices((nrows, ncols), dtype=np.float64)
            x = (xx.ravel()[sample_flat] - (ncols - 1) / 2.0) / max(float(ncols), 1.0)
            y = (yy.ravel()[sample_flat] - (nrows - 1) / 2.0) / max(float(nrows), 1.0)
            z = resid_after_radial.ravel()[sample_flat]
            z_med, z_sig = _robust_median_sigma(z)
            keep = np.abs(z - z_med) <= 4.0 * z_sig
            if np.count_nonzero(keep) >= 64:
                design = np.column_stack([x[keep], y[keep], np.ones(np.count_nonzero(keep), dtype=np.float64)])
                coeff, *_ = np.linalg.lstsq(design, z[keep], rcond=None)
                plane_coefficients = [float(v) for v in coeff.tolist()]
                x_full = (xx - (ncols - 1) / 2.0) / max(float(ncols), 1.0)
                y_full = (yy - (nrows - 1) / 2.0) / max(float(nrows), 1.0)
                background = (background.astype(np.float32) + (coeff[0] * x_full + coeff[1] * y_full + coeff[2]).astype(np.float32))

    report = {
        "valid_fraction": float(np.count_nonzero(valid) / valid.size),
        "mask_residual_median": float(median_value),
        "mask_residual_sigma": float(sigma_value),
        "profile_median": float(np.median(profile)),
        "profile_min": float(np.min(profile)),
        "profile_max": float(np.max(profile)),
        "plane_coefficients": plane_coefficients,
    }
    return background.astype(np.float32, copy=False), report


def estimate_highres_xds_offset(
    stored_data: NDArray,
    beam_centers_yx: Optional[List[Optional[Tuple[float, float]]]] = None,
    *,
    quantile: float = 0.001,
    sample_frames: int = 9,
    inner_fraction: float = 0.82,
    outer_fraction: float = 0.98,
    clip_high_sigma: float = 5.0,
    clip_low_sigma: float = 8.0,
    max_pixels: int = 4_000_000,
) -> Dict[str, Any]:
    """Estimate a Kay-style lower XDS OFFSET from high-resolution background.

    The input is the nonnegative CBF-storage image stack, not the signed raw
    MRC stack. The returned offset is intentionally allowed to be lower than
    the storage pedestal so XDS sees a positive high-resolution background
    buffer instead of clipping a large fraction of weak pixels to zero.
    """
    if stored_data.ndim != 3:
        raise ValueError(f"Expected 3D stack for XDS offset estimate, got {stored_data.ndim}D")
    num_frames, nrows, ncols = stored_data.shape
    frame_indices = np.unique(np.linspace(0, num_frames - 1, max(1, min(int(sample_frames), num_frames)), dtype=int))
    values_list: List[NDArray] = []
    for frame_idx in frame_indices:
        center = None
        if beam_centers_yx and int(frame_idx) < len(beam_centers_yx):
            center = beam_centers_yx[int(frame_idx)]
        if center is None or not np.all(np.isfinite(center)):
            center = ((nrows - 1) / 2.0, (ncols - 1) / 2.0)
        center_y, center_x = float(center[0]), float(center[1])
        yy, xx = np.indices((nrows, ncols), dtype=np.float32)
        radii = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        max_radius = float(np.max(radii))
        mask = (radii >= float(inner_fraction) * max_radius) & (radii <= float(outer_fraction) * max_radius)
        frame_values = np.asarray(stored_data[int(frame_idx)][mask], dtype=np.float64)
        if frame_values.size == 0:
            continue
        med, sig = _robust_median_sigma(frame_values)
        keep = (frame_values >= med - float(clip_low_sigma) * sig) & (frame_values <= med + float(clip_high_sigma) * sig)
        frame_values = frame_values[keep]
        if frame_values.size:
            values_list.append(frame_values)
    if not values_list:
        raise ValueError("No high-resolution pixels available for XDS offset estimate")
    values = np.concatenate(values_list)
    sample_idx = _sample_indices(values.size, int(max_pixels))
    values = values[sample_idx]
    quantile = min(0.5, max(0.0, float(quantile)))
    median_value = float(np.median(values))
    mad_value = float(np.median(np.abs(values - median_value)))
    robust_sigma = float(1.4826 * mad_value)
    q_value = float(np.quantile(values, quantile))
    candidate_mode_minus_sigma = int(max(0, math.floor(median_value - robust_sigma)))
    candidate_mode_minus_2sigma = int(max(0, math.floor(median_value - 2.0 * robust_sigma)))
    candidate_quantile = int(max(0, math.floor(q_value)))
    return {
        "method": "highres_masked_quantile",
        "sampled_frames": [int(v) for v in frame_indices.tolist()],
        "sample_size_pixels": int(values.size),
        "inner_fraction": float(inner_fraction),
        "outer_fraction": float(outer_fraction),
        "quantile": float(quantile),
        "quantile_value": q_value,
        "median_value": median_value,
        "mad_value": mad_value,
        "robust_sigma": robust_sigma,
        "suggested_offset": int(candidate_quantile),
        "candidate_offsets": {
            "highres_q": int(candidate_quantile),
            "highres_median_minus_sigma": int(candidate_mode_minus_sigma),
            "highres_median_minus_2sigma": int(candidate_mode_minus_2sigma),
            "highres_median": int(round(median_value)),
        },
    }


def compute_xds_pixel_diagnostics(
    stored_data: NDArray,
    xds_offset: int,
    beam_centers_yx: Optional[List[Optional[Tuple[float, float]]]] = None,
    *,
    sample_frames: int = 9,
    inner_fraction: float = 0.82,
    outer_fraction: float = 0.98,
    max_pixels: int = 4_000_000,
) -> Dict[str, Any]:
    """Measure the pixel-level truncation risk for a proposed XDS OFFSET."""
    if stored_data.ndim != 3:
        raise ValueError(f"Expected 3D stack for diagnostics, got {stored_data.ndim}D")
    num_frames, nrows, ncols = stored_data.shape
    frame_indices = np.unique(np.linspace(0, num_frames - 1, max(1, min(int(sample_frames), num_frames)), dtype=int))
    all_values: List[NDArray] = []
    hi_values: List[NDArray] = []
    for frame_idx in frame_indices:
        frame = np.asarray(stored_data[int(frame_idx)], dtype=np.float64)
        flat = frame.ravel()
        all_values.append(flat[_sample_indices(flat.size, max(1, int(max_pixels // max(len(frame_indices), 1))))])

        center = None
        if beam_centers_yx and int(frame_idx) < len(beam_centers_yx):
            center = beam_centers_yx[int(frame_idx)]
        if center is None or not np.all(np.isfinite(center)):
            center = ((nrows - 1) / 2.0, (ncols - 1) / 2.0)
        center_y, center_x = float(center[0]), float(center[1])
        yy, xx = np.indices((nrows, ncols), dtype=np.float32)
        radii = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        max_radius = float(np.max(radii))
        mask = (radii >= float(inner_fraction) * max_radius) & (radii <= float(outer_fraction) * max_radius)
        vals = frame[mask]
        hi_values.append(vals[_sample_indices(vals.size, max(1, int(max_pixels // max(len(frame_indices), 1))))])

    values = np.concatenate([arr for arr in all_values if arr.size])
    highres = np.concatenate([arr for arr in hi_values if arr.size])
    offset = float(int(xds_offset))
    ic_highres = np.maximum(0.0, highres - offset) if highres.size else np.asarray([], dtype=np.float64)
    return {
        "xds_offset": int(xds_offset),
        "sampled_frames": [int(v) for v in frame_indices.tolist()],
        "sample_size_pixels": int(values.size),
        "highres_sample_size_pixels": int(highres.size),
        "fraction_pixels_below_xds_offset": float(np.count_nonzero(values < offset) / values.size) if values.size else None,
        "fraction_pixels_at_or_below_xds_offset": float(np.count_nonzero(values <= offset) / values.size) if values.size else None,
        "fraction_highres_pixels_below_xds_offset": float(np.count_nonzero(highres < offset) / highres.size) if highres.size else None,
        "fraction_highres_ic_zero_after_xds": float(np.count_nonzero(highres <= offset) / highres.size) if highres.size else None,
        "highres_ic_mean_after_xds": float(np.mean(ic_highres)) if ic_highres.size else None,
        "highres_ic_sigma_after_xds": float(np.std(ic_highres)) if ic_highres.size else None,
        "stored_min": float(np.min(values)) if values.size else None,
        "stored_median": float(np.median(values)) if values.size else None,
        "stored_highres_median": float(np.median(highres)) if highres.size else None,
    }


def apply_v8_conditioning_policy(
    stored_data: NDArray,
    beam_centers_yx: List[Optional[Tuple[float, float]]],
    *,
    mode: str,
    current_storage_pedestal: int,
    current_storage_pedestal_report: Dict[str, Any],
    xds_offset_quantile: float,
    xds_offset_inner_fraction: float,
    xds_offset_outer_fraction: float,
    radial_store_pedestal: Optional[int],
    radial_pedestal_quantile: float,
    radial_pedestal_margin: float,
    radial_buffer: int,
    radial_bin_width_px: float,
    radial_quantile: float,
    radial_profile_smooth_sigma_bins: float,
    radial_mask_smooth_sigma_px: float,
    radial_clip_high_sigma: float,
    radial_clip_low_sigma: float,
    radial_mask_dilation_px: int,
    radial_center_mask_radius_px: float,
    radial_fit_plane: bool,
    radial_plane_sample_pixels: int,
) -> Tuple[NDArray, int, int, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Apply the selected v8 storage/background/XDS-offset policy.

    ``quantile_offset`` leaves the image values exactly in the v7 storage
    convention and only lowers XDS ``OFFSET``. The radial modes are more
    invasive: they subtract a smooth per-frame background and then write a new
    nonnegative storage stack. ``radial_reduced`` sets XDS ``OFFSET`` to the
    conditioned high-resolution median-minus-sigma value; that is the tested
    current-XDS path that avoids both full-offset truncation and quantile
    high-shell bias. Radial modes report any residual values clipped at zero
    because that clipping is the cost of using a modest storage pedestal.
    """
    mode_norm = str(mode or "v7").strip().lower()
    if mode_norm in {"none", "off", "storage_pedestal"}:
        mode_norm = "v7"

    report: Dict[str, Any] = {
        "mode": mode_norm,
        "input_storage_pedestal": int(current_storage_pedestal),
        "input_storage_pedestal_report": current_storage_pedestal_report,
    }

    if mode_norm == "v7":
        xds_details = {
            "method": "storage_pedestal",
            "suggested_offset": int(current_storage_pedestal),
            "storage_pedestal_report": current_storage_pedestal_report,
        }
        report["pixel_diagnostics"] = compute_xds_pixel_diagnostics(
            stored_data,
            int(current_storage_pedestal),
            beam_centers_yx,
            inner_fraction=xds_offset_inner_fraction,
            outer_fraction=xds_offset_outer_fraction,
        )
        return stored_data, int(current_storage_pedestal), int(current_storage_pedestal), xds_details, current_storage_pedestal_report, report

    if mode_norm == "quantile_offset":
        xds_details = estimate_highres_xds_offset(
            stored_data,
            beam_centers_yx,
            quantile=xds_offset_quantile,
            inner_fraction=xds_offset_inner_fraction,
            outer_fraction=xds_offset_outer_fraction,
        )
        xds_offset_value = int(xds_details["suggested_offset"])
        report["pixel_diagnostics"] = compute_xds_pixel_diagnostics(
            stored_data,
            xds_offset_value,
            beam_centers_yx,
            inner_fraction=xds_offset_inner_fraction,
            outer_fraction=xds_offset_outer_fraction,
        )
        return stored_data, int(current_storage_pedestal), xds_offset_value, xds_details, current_storage_pedestal_report, report

    if mode_norm not in {"radial_reduced", "radial_offset0", "radial_buffer"}:
        raise ValueError(f"Unknown v8 conditioning mode: {mode}")

    num_frames, nrows, ncols = stored_data.shape
    logging.info("v8 radial conditioning: first pass estimating storage pedestal from sampled residuals...")
    sampled_residuals: List[NDArray] = []
    background_reports: List[Dict[str, Any]] = []
    for frame_idx in range(num_frames):
        center = beam_centers_yx[frame_idx]
        if center is None or not np.all(np.isfinite(center)):
            center = ((nrows - 1) / 2.0, (ncols - 1) / 2.0)
        background, bg_report = estimate_radial_plane_background(
            stored_data[frame_idx],
            center,
            bin_width_px=radial_bin_width_px,
            radial_quantile=radial_quantile,
            profile_smooth_sigma_bins=radial_profile_smooth_sigma_bins,
            mask_smooth_sigma_px=radial_mask_smooth_sigma_px,
            clip_high_sigma=radial_clip_high_sigma,
            clip_low_sigma=radial_clip_low_sigma,
            mask_dilation_px=radial_mask_dilation_px,
            center_mask_radius_px=radial_center_mask_radius_px,
            fit_plane=radial_fit_plane,
            plane_sample_pixels=radial_plane_sample_pixels,
        )
        corrected = np.asarray(stored_data[frame_idx], dtype=np.float32) - background
        sample_flat = corrected.ravel()[_sample_indices(corrected.size, 200_000)]
        sampled_residuals.append(sample_flat.astype(np.float32, copy=False))
        if frame_idx in set(np.linspace(0, num_frames - 1, min(9, num_frames), dtype=int).tolist()):
            bg_report.update(
                {
                    "frame_0_based": int(frame_idx),
                    "corrected_sample_median": float(np.median(sample_flat)),
                    "corrected_sample_q001": float(np.quantile(sample_flat, 0.001)),
                    "corrected_sample_q999": float(np.quantile(sample_flat, 0.999)),
                }
            )
            background_reports.append(bg_report)
        if (frame_idx + 1) % max(1, num_frames // 10) == 0 or frame_idx + 1 == num_frames:
            logging.info("  v8 radial conditioning pedestal pass: %d/%d", frame_idx + 1, num_frames)

    residual_sample = np.concatenate(sampled_residuals).astype(np.float64, copy=False)
    quantile = min(0.1, max(0.0, float(radial_pedestal_quantile)))
    auto_store_pedestal = int(max(0, math.ceil(-float(np.quantile(residual_sample, quantile)) + float(radial_pedestal_margin))))
    selected_store_pedestal = int(radial_store_pedestal) if radial_store_pedestal is not None else auto_store_pedestal
    selected_store_pedestal = max(0, selected_store_pedestal)
    logging.info(
        "v8 radial conditioning selected storage pedestal %d (auto=%d, q=%g, margin=%.2f)",
        selected_store_pedestal,
        auto_store_pedestal,
        quantile,
        float(radial_pedestal_margin),
    )

    conditioned = np.empty(stored_data.shape, dtype=np.int32)
    negative_before_clip = 0
    radial_residual_summary: List[Dict[str, Any]] = []
    logging.info("v8 radial conditioning: second pass writing conditioned stack...")
    for frame_idx in range(num_frames):
        center = beam_centers_yx[frame_idx]
        if center is None or not np.all(np.isfinite(center)):
            center = ((nrows - 1) / 2.0, (ncols - 1) / 2.0)
        background, _ = estimate_radial_plane_background(
            stored_data[frame_idx],
            center,
            bin_width_px=radial_bin_width_px,
            radial_quantile=radial_quantile,
            profile_smooth_sigma_bins=radial_profile_smooth_sigma_bins,
            mask_smooth_sigma_px=radial_mask_smooth_sigma_px,
            clip_high_sigma=radial_clip_high_sigma,
            clip_low_sigma=radial_clip_low_sigma,
            mask_dilation_px=radial_mask_dilation_px,
            center_mask_radius_px=radial_center_mask_radius_px,
            fit_plane=radial_fit_plane,
            plane_sample_pixels=radial_plane_sample_pixels,
        )
        corrected = np.asarray(stored_data[frame_idx], dtype=np.float32) - background
        stored_float = np.rint(corrected + float(selected_store_pedestal))
        negative_count = int(np.count_nonzero(stored_float < 0))
        negative_before_clip += negative_count
        if negative_count:
            stored_float = np.maximum(stored_float, 0.0)
        conditioned[frame_idx] = np.asarray(stored_float, dtype=np.int32)
        if frame_idx in set(np.linspace(0, num_frames - 1, min(9, num_frames), dtype=int).tolist()):
            residual_sample_frame = corrected.ravel()[_sample_indices(corrected.size, 200_000)]
            radial_residual_summary.append(
                {
                    "frame_0_based": int(frame_idx),
                    "corrected_median": float(np.median(residual_sample_frame)),
                    "corrected_sigma": float(np.std(residual_sample_frame)),
                    "corrected_q001": float(np.quantile(residual_sample_frame, 0.001)),
                    "corrected_q999": float(np.quantile(residual_sample_frame, 0.999)),
                }
            )
        if (frame_idx + 1) % max(1, num_frames // 10) == 0 or frame_idx + 1 == num_frames:
            logging.info("  v8 radial conditioning write pass: %d/%d", frame_idx + 1, num_frames)

    reduced_offset_details: Optional[Dict[str, Any]] = None
    if mode_norm == "radial_reduced":
        reduced_offset_details = estimate_highres_xds_offset(
            conditioned,
            beam_centers_yx,
            quantile=xds_offset_quantile,
            inner_fraction=xds_offset_inner_fraction,
            outer_fraction=xds_offset_outer_fraction,
        )
        xds_offset_value = int(
            reduced_offset_details.get("candidate_offsets", {}).get(
                "highres_median_minus_sigma",
                reduced_offset_details.get("suggested_offset", 0),
            )
        )
        xds_method = "radial_conditioned_highres_median_minus_sigma"
        selected_candidate = "highres_median_minus_sigma"
    elif mode_norm == "radial_offset0":
        xds_offset_value = 0
        xds_method = "radial_conditioned_offset0"
        selected_candidate = "offset_zero"
    else:
        xds_offset_value = max(0, int(selected_store_pedestal) - int(radial_buffer))
        xds_method = "radial_conditioned_positive_buffer"
        selected_candidate = "storage_pedestal_minus_buffer"

    storage_report = {
        "method": "radial_conditioned_storage_pedestal",
        "suggested_pedestal": int(selected_store_pedestal),
        "auto_pedestal": int(auto_store_pedestal),
        "manual_pedestal": int(radial_store_pedestal) if radial_store_pedestal is not None else None,
        "residual_sample_quantile": float(quantile),
        "residual_sample_quantile_value": float(np.quantile(residual_sample, quantile)),
        "pedestal_margin": float(radial_pedestal_margin),
        "negative_count_before_storage_clip": int(negative_before_clip),
        "negative_fraction_before_storage_clip": float(negative_before_clip / conditioned.size),
    }
    xds_details = {
        "method": xds_method,
        "suggested_offset": int(xds_offset_value),
        "storage_pedestal": int(selected_store_pedestal),
        "buffer": int(selected_store_pedestal - xds_offset_value),
        "selected_candidate": selected_candidate,
        "median_value": reduced_offset_details.get("median_value") if reduced_offset_details else None,
        "robust_sigma": reduced_offset_details.get("robust_sigma") if reduced_offset_details else None,
        "sample_size_pixels": reduced_offset_details.get("sample_size_pixels") if reduced_offset_details else None,
        "reduced_offset_estimation": reduced_offset_details,
        "radial_parameters": {
            "bin_width_px": float(radial_bin_width_px),
            "radial_quantile": float(radial_quantile),
            "profile_smooth_sigma_bins": float(radial_profile_smooth_sigma_bins),
            "mask_smooth_sigma_px": float(radial_mask_smooth_sigma_px),
            "clip_high_sigma": float(radial_clip_high_sigma),
            "clip_low_sigma": float(radial_clip_low_sigma),
            "mask_dilation_px": int(radial_mask_dilation_px),
            "center_mask_radius_px": float(radial_center_mask_radius_px),
            "fit_plane": bool(radial_fit_plane),
        },
    }
    report.update(
        {
            "storage_pedestal": storage_report,
            "xds_offset_details": xds_details,
            "background_sample_reports": background_reports,
            "radial_residual_summary": radial_residual_summary,
            "pixel_diagnostics": compute_xds_pixel_diagnostics(
                conditioned,
                xds_offset_value,
                beam_centers_yx,
                inner_fraction=xds_offset_inner_fraction,
                outer_fraction=xds_offset_outer_fraction,
            ),
        }
    )
    return conditioned, int(selected_store_pedestal), int(xds_offset_value), xds_details, storage_report, report


def _contiguous_ranges_from_indices(indices: List[int]) -> List[Tuple[int, int]]:
    if not indices:
        return []
    sorted_unique = sorted(set(int(idx) for idx in indices))
    ranges: List[Tuple[int, int]] = []
    start = prev = sorted_unique[0]
    for idx in sorted_unique[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev))
        start = prev = idx
    ranges.append((start, prev))
    return ranges


def summarize_frame_quality(
    stored_data: NDArray,
    beam_centers_yx: List[Optional[Tuple[float, float]]],
    *,
    window_radius_px: int = 48,
) -> Dict[str, Any]:
    if stored_data.ndim != 3:
        raise ValueError(f"Expected 3D stack for frame-quality screening, got {stored_data.ndim}D")
    num_frames, nrows, ncols = stored_data.shape
    if len(beam_centers_yx) != num_frames:
        raise ValueError("Beam center list length does not match stack length")

    def _robust_low_threshold(values: List[float], ratio_floor: float) -> float:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        median_value = float(np.median(arr))
        if median_value <= 0:
            return median_value
        mad_value = float(np.median(np.abs(arr - median_value)))
        robust_sigma = float(mad_value * 1.4826)
        sigma_candidate = median_value - 5.0 * robust_sigma
        return float(max(ratio_floor * median_value, min(0.5 * median_value, sigma_candidate)))

    half_window = max(12, int(window_radius_px))
    metrics: List[Dict[str, Any]] = []
    peak_values: List[float] = []
    signal_values: List[float] = []
    std_values: List[float] = []

    for frame_idx in range(num_frames):
        frame = np.asarray(stored_data[frame_idx], dtype=np.float32)
        frame_median = float(np.median(frame))
        frame_std = float(np.std(frame))
        center = beam_centers_yx[frame_idx]
        beam_peak = float("nan")
        beam_signal = float("nan")
        flags: List[str] = []

        if center is None or not np.all(np.isfinite(center)):
            flags.append("missing_center")
        else:
            cy = int(round(float(center[0])))
            cx = int(round(float(center[1])))
            y0 = max(0, cy - half_window)
            y1 = min(nrows, cy + half_window + 1)
            x0 = max(0, cx - half_window)
            x1 = min(ncols, cx + half_window + 1)
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                flags.append("empty_beam_window")
            else:
                beam_peak = float(np.percentile(crop, 99.5) - frame_median)
                beam_signal = float(np.sum(np.clip(crop - frame_median, 0.0, None)))
                if np.isfinite(beam_peak):
                    peak_values.append(beam_peak)
                if np.isfinite(beam_signal):
                    signal_values.append(beam_signal)
        if np.isfinite(frame_std):
            std_values.append(frame_std)

        metrics.append(
            {
                "frame_0_based": int(frame_idx),
                "frame_1_based": int(frame_idx + 1),
                "frame_median": frame_median,
                "frame_std": frame_std,
                "beam_peak_excess": beam_peak,
                "beam_signal_excess_sum": beam_signal,
                "flags": flags,
            }
        )

    peak_threshold = _robust_low_threshold(peak_values, ratio_floor=0.20)
    signal_threshold = _robust_low_threshold(signal_values, ratio_floor=0.15)
    std_threshold = _robust_low_threshold(std_values, ratio_floor=0.25)

    flagged_indices: List[int] = []
    reason_counts: Dict[str, int] = {}
    for entry in metrics:
        low_reasons: List[str] = []
        if "missing_center" in entry["flags"] or "empty_beam_window" in entry["flags"]:
            low_reasons.append("missing_center")
        if np.isfinite(entry["beam_peak_excess"]) and np.isfinite(peak_threshold) and entry["beam_peak_excess"] < peak_threshold:
            low_reasons.append("low_beam_peak")
        if np.isfinite(entry["beam_signal_excess_sum"]) and np.isfinite(signal_threshold) and entry["beam_signal_excess_sum"] < signal_threshold:
            low_reasons.append("low_beam_signal")
        if np.isfinite(entry["frame_std"]) and np.isfinite(std_threshold) and entry["frame_std"] < std_threshold:
            low_reasons.append("low_frame_std")

        combined_reasons = list(dict.fromkeys(entry["flags"] + low_reasons))
        entry["flags"] = combined_reasons
        if "missing_center" in combined_reasons or len(low_reasons) >= 2:
            flagged_indices.append(int(entry["frame_0_based"]))
            for reason in combined_reasons:
                reason_counts[reason] = int(reason_counts.get(reason, 0) + 1)

    flagged_ranges_0_based = _contiguous_ranges_from_indices(flagged_indices)
    flagged_ranges_1_based = [(start + 1, end + 1) for start, end in flagged_ranges_0_based]

    return {
        "window_radius_px": int(half_window),
        "frames_evaluated": int(num_frames),
        "thresholds": {
            "beam_peak_excess": peak_threshold,
            "beam_signal_excess_sum": signal_threshold,
            "frame_std": std_threshold,
        },
        "flagged_frames_0_based": [int(v) for v in flagged_indices],
        "flagged_frames_1_based": [int(v + 1) for v in flagged_indices],
        "flagged_ranges_0_based": [[int(start), int(end)] for start, end in flagged_ranges_0_based],
        "flagged_ranges_1_based": [[int(start), int(end)] for start, end in flagged_ranges_1_based],
        "flagged_fraction": float(len(flagged_indices) / num_frames) if num_frames else 0.0,
        "reason_counts": reason_counts,
        "metrics": metrics,
    }


def determine_edge_trim_window(flagged_indices: List[int], num_frames: int) -> Tuple[int, int]:
    flagged_set = {int(idx) for idx in flagged_indices}
    trim_start = 0
    while trim_start < num_frames and trim_start in flagged_set:
        trim_start += 1

    trim_end_exclusive = int(num_frames)
    while trim_end_exclusive > trim_start and (trim_end_exclusive - 1) in flagged_set:
        trim_end_exclusive -= 1

    return trim_start, trim_end_exclusive


def _central_finite_median(metrics: List[Dict[str, Any]], key: str) -> float:
    if not metrics:
        return float("nan")
    start = int(len(metrics) * 0.15)
    end = int(math.ceil(len(metrics) * 0.85))
    central_metrics = metrics[start:end] or metrics
    values = [
        float(entry.get(key))
        for entry in central_metrics
        if isinstance(entry.get(key), (int, float)) and np.isfinite(float(entry.get(key)))
    ]
    if not values:
        values = [
            float(entry.get(key))
            for entry in metrics
            if isinstance(entry.get(key), (int, float)) and np.isfinite(float(entry.get(key)))
        ]
    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def determine_trim_window_from_quality(
    frame_quality_summary: Dict[str, Any],
    num_frames: int,
    *,
    trim_policy: str = "aggressive",
    aggressive_min_strong_frames: int = 3,
    aggressive_signal_ratio: float = 0.55,
    aggressive_std_ratio: float = 0.45,
    aggressive_peak_ratio: float = 0.35,
) -> Tuple[int, int, Dict[str, Any]]:
    policy = str(trim_policy or "off").strip().lower()
    if policy in {"none", "false", "no"}:
        policy = "off"
    if policy in {"edge", "edge_quality", "trim_edge_bad_frames"}:
        policy = "quality"
    if policy not in {"off", "quality", "aggressive"}:
        logging.warning("Unknown trim policy '%s'; using aggressive.", trim_policy)
        policy = "aggressive"

    flagged_indices = frame_quality_summary.get("flagged_frames_0_based", [])
    quality_start, quality_end = determine_edge_trim_window(flagged_indices, num_frames)
    details: Dict[str, Any] = {
        "policy": policy,
        "quality_edge_window_0_based": [int(quality_start), int(quality_end)],
    }
    if policy == "off":
        return 0, int(num_frames), details
    if policy == "quality":
        return quality_start, quality_end, details

    metrics = list(frame_quality_summary.get("metrics", []))
    min_run = max(1, int(aggressive_min_strong_frames))
    ref_signal = _central_finite_median(metrics, "beam_signal_excess_sum")
    ref_std = _central_finite_median(metrics, "frame_std")
    ref_peak = _central_finite_median(metrics, "beam_peak_excess")
    thresholds = frame_quality_summary.get("thresholds", {}) or {}

    def _metric_threshold(key: str, reference: float, ratio: float) -> float:
        candidates: List[float] = []
        if isinstance(thresholds.get(key), (int, float)) and np.isfinite(float(thresholds.get(key))):
            candidates.append(float(thresholds.get(key)))
        if np.isfinite(reference) and reference > 0:
            candidates.append(float(reference) * float(ratio))
        return max(candidates) if candidates else float("nan")

    signal_threshold = _metric_threshold("beam_signal_excess_sum", ref_signal, aggressive_signal_ratio)
    std_threshold = _metric_threshold("frame_std", ref_std, aggressive_std_ratio)
    peak_threshold = _metric_threshold("beam_peak_excess", ref_peak, aggressive_peak_ratio)

    strong = [False] * int(num_frames)
    for entry in metrics:
        idx = int(entry.get("frame_0_based", -1))
        if idx < 0 or idx >= num_frames:
            continue
        flags = set(entry.get("flags", []))
        if "missing_center" in flags or "empty_beam_window" in flags:
            continue
        signal = entry.get("beam_signal_excess_sum")
        frame_std = entry.get("frame_std")
        peak = entry.get("beam_peak_excess")
        signal_ok = not np.isfinite(signal_threshold) or (
            isinstance(signal, (int, float)) and np.isfinite(float(signal)) and float(signal) >= signal_threshold
        )
        std_ok = not np.isfinite(std_threshold) or (
            isinstance(frame_std, (int, float)) and np.isfinite(float(frame_std)) and float(frame_std) >= std_threshold
        )
        peak_ok = not np.isfinite(peak_threshold) or (
            isinstance(peak, (int, float)) and np.isfinite(float(peak)) and float(peak) >= peak_threshold
        )
        strong[idx] = bool(signal_ok and std_ok and peak_ok)

    def _first_strong_run() -> Optional[int]:
        for idx in range(0, max(0, int(num_frames) - min_run + 1)):
            if all(strong[idx:idx + min_run]):
                return idx
        return None

    def _last_strong_run_end() -> Optional[int]:
        for idx in range(max(0, int(num_frames) - min_run), -1, -1):
            if all(strong[idx:idx + min_run]):
                return idx + min_run
        return None

    aggressive_start = _first_strong_run()
    aggressive_end = _last_strong_run_end()
    details.update(
        {
            "aggressive_min_strong_frames": int(min_run),
            "aggressive_reference": {
                "beam_signal_excess_sum": ref_signal,
                "frame_std": ref_std,
                "beam_peak_excess": ref_peak,
            },
            "aggressive_thresholds": {
                "beam_signal_excess_sum": signal_threshold,
                "frame_std": std_threshold,
                "beam_peak_excess": peak_threshold,
            },
            "aggressive_strong_frames": int(sum(1 for value in strong if value)),
            "aggressive_window_0_based": [
                int(aggressive_start) if aggressive_start is not None else None,
                int(aggressive_end) if aggressive_end is not None else None,
            ],
        }
    )
    if aggressive_start is None or aggressive_end is None or aggressive_start >= aggressive_end:
        details["fallback"] = "quality_edge_window"
        return quality_start, quality_end, details

    trim_start = max(int(quality_start), int(aggressive_start))
    trim_end_exclusive = min(int(quality_end), int(aggressive_end))
    if trim_start >= trim_end_exclusive:
        details["fallback"] = "quality_edge_window_after_empty_aggressive_intersection"
        return quality_start, quality_end, details
    return trim_start, trim_end_exclusive, details


def _prepare_binned_numeric_map(
    map_array: np.ndarray,
    *,
    raw_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
    bin_y: int,
    bin_x: int,
    reducer: str,
    label: str,
) -> np.ndarray:
    arr = np.asarray(map_array)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2D array, got shape {arr.shape}")
    if tuple(arr.shape) == tuple(target_shape):
        return np.asarray(arr, dtype=np.float32)
    if tuple(arr.shape) != tuple(raw_shape):
        raise ValueError(f"{label} shape {arr.shape} does not match raw {raw_shape} or binned {target_shape}")
    if raw_shape[0] % bin_y != 0 or raw_shape[1] % bin_x != 0:
        raise ValueError(f"{label} raw shape {raw_shape} is not divisible by software bins ({bin_y}, {bin_x})")

    ny2 = raw_shape[0] // bin_y
    nx2 = raw_shape[1] // bin_x
    reshaped = arr.reshape(ny2, bin_y, nx2, bin_x)
    if reducer == "sum":
        out = reshaped.sum(axis=(1, 3))
    elif reducer == "mean":
        out = reshaped.mean(axis=(1, 3))
    else:
        raise ValueError(f"Unsupported numeric reducer: {reducer}")
    return np.asarray(out, dtype=np.float32)


def _prepare_binned_mask(
    mask_array: np.ndarray,
    *,
    raw_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
    bin_y: int,
    bin_x: int,
    label: str,
) -> np.ndarray:
    arr = np.asarray(mask_array)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2D array, got shape {arr.shape}")
    if tuple(arr.shape) == tuple(target_shape):
        return np.asarray(arr != 0, dtype=bool)
    if tuple(arr.shape) != tuple(raw_shape):
        raise ValueError(f"{label} shape {arr.shape} does not match raw {raw_shape} or binned {target_shape}")
    if raw_shape[0] % bin_y != 0 or raw_shape[1] % bin_x != 0:
        raise ValueError(f"{label} raw shape {raw_shape} is not divisible by software bins ({bin_y}, {bin_x})")
    ny2 = raw_shape[0] // bin_y
    nx2 = raw_shape[1] // bin_x
    reshaped = arr.reshape(ny2, bin_y, nx2, bin_x)
    return np.any(reshaped != 0, axis=(1, 3))


def repair_bad_pixels_in_frame(frame: np.ndarray, bad_mask: np.ndarray, fill_mode: str = "median3") -> np.ndarray:
    if fill_mode == "none" or bad_mask is None or not np.any(bad_mask):
        return frame
    repaired = np.asarray(frame, dtype=np.float32).copy()
    if fill_mode == "median3":
        replacement = median_filter(repaired, size=3, mode="nearest")
    else:
        raise ValueError(f"Unsupported bad pixel fill mode: {fill_mode}")
    repaired[bad_mask] = replacement[bad_mask]
    return repaired


def apply_pre_shift_detector_corrections(
    binned_data: NDArray,
    *,
    raw_frame_shape: Tuple[int, int],
    bin_y: int,
    bin_x: int,
    dark_map_path: Optional[str] = None,
    gain_map_path: Optional[str] = None,
    gain_map_kind: str = "auto",
    bad_pixel_mask_path: Optional[str] = None,
    bad_pixel_fill: str = "median3",
) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
    corrected = np.asarray(binned_data, dtype=np.float32)
    target_shape = tuple(int(v) for v in corrected.shape[1:])
    report: Dict[str, Any] = {
        "dark_map": None,
        "gain_map": None,
        "bad_pixel_mask": None,
    }

    if dark_map_path:
        dark_map_raw = np.load(dark_map_path)
        dark_map = _prepare_binned_numeric_map(
            dark_map_raw,
            raw_shape=raw_frame_shape,
            target_shape=target_shape,
            bin_y=bin_y,
            bin_x=bin_x,
            reducer="sum",
            label="dark map",
        )
        corrected = corrected - dark_map[None, :, :]
        report["dark_map"] = {
            "path": str(Path(dark_map_path).expanduser().resolve()),
            "shape": [int(v) for v in dark_map.shape],
            "median": float(np.median(dark_map)),
            "std": float(np.std(dark_map)),
        }

    if gain_map_path:
        gain_map_raw = np.load(gain_map_path)
        gain_map = _prepare_binned_numeric_map(
            gain_map_raw,
            raw_shape=raw_frame_shape,
            target_shape=target_shape,
            bin_y=bin_y,
            bin_x=bin_x,
            reducer="mean",
            label="gain map",
        )
        valid = np.isfinite(gain_map) & (gain_map > 0)
        if not np.any(valid):
            raise ValueError("Gain map has no positive finite pixels after binning")

        gain_map_kind_norm = str(gain_map_kind).strip().lower()
        median_gain = float(np.median(gain_map[valid]))
        if gain_map_kind_norm == "auto":
            gain_map_kind_norm = "absolute" if median_gain > 5.0 or median_gain < 0.2 else "relative"
        if gain_map_kind_norm == "absolute":
            gain_map_relative = np.ones_like(gain_map, dtype=np.float32)
            gain_map_relative[valid] = gain_map[valid] / median_gain
        elif gain_map_kind_norm == "relative":
            gain_map_relative = gain_map.astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unsupported gain map kind: {gain_map_kind}")

        gain_valid = np.isfinite(gain_map_relative) & (gain_map_relative > 0)
        gain_denominator = np.ones_like(gain_map_relative, dtype=np.float32)
        gain_denominator[gain_valid] = gain_map_relative[gain_valid]
        corrected = corrected / gain_denominator[None, :, :]
        report["gain_map"] = {
            "path": str(Path(gain_map_path).expanduser().resolve()),
            "shape": [int(v) for v in gain_map_relative.shape],
            "kind": gain_map_kind_norm,
            "median_relative_gain": float(np.median(gain_map_relative[gain_valid])),
            "q01_relative_gain": float(np.quantile(gain_map_relative[gain_valid], 0.01)),
            "q99_relative_gain": float(np.quantile(gain_map_relative[gain_valid], 0.99)),
        }

    if bad_pixel_mask_path:
        bad_mask_raw = np.load(bad_pixel_mask_path)
        bad_mask = _prepare_binned_mask(
            bad_mask_raw,
            raw_shape=raw_frame_shape,
            target_shape=target_shape,
            bin_y=bin_y,
            bin_x=bin_x,
            label="bad pixel mask",
        )
        if np.any(bad_mask) and bad_pixel_fill != "none":
            corrected = np.stack(
                [repair_bad_pixels_in_frame(frame, bad_mask, fill_mode=bad_pixel_fill) for frame in corrected],
                axis=0,
            ).astype(np.float32, copy=False)
        report["bad_pixel_mask"] = {
            "path": str(Path(bad_pixel_mask_path).expanduser().resolve()),
            "shape": [int(v) for v in bad_mask.shape],
            "fill_mode": bad_pixel_fill,
            "bad_pixel_fraction": float(np.mean(bad_mask)),
            "bad_pixel_count": int(np.count_nonzero(bad_mask)),
        }

    return corrected, report


def select_gain_frame_indices(total_frames: int, requested_frames: int) -> List[int]:
    use_frames = max(1, min(int(total_frames), int(requested_frames)))
    if use_frames >= int(total_frames):
        return list(range(int(total_frames)))
    raw = np.linspace(0, int(total_frames) - 1, num=use_frames)
    indices: List[int] = []
    seen = set()
    for value in raw:
        idx = int(round(float(value)))
        idx = max(0, min(int(total_frames) - 1, idx))
        if idx in seen:
            continue
        indices.append(idx)
        seen.add(idx)
    if len(indices) < use_frames:
        for idx in range(int(total_frames)):
            if idx in seen:
                continue
            indices.append(idx)
            seen.add(idx)
            if len(indices) >= use_frames:
                break
    return sorted(indices)

# --- CBF Header Generation ---
def create_cbf_header(template_params: Dict[str, Any],
                       frame_specific_params: Dict[str, Any],
                       applied_pedestal: int = 0,
                       overload_value: int = 1000000) -> str:
    timestamp_iso = (
        frame_specific_params.get("timestamp_iso")
        or template_params.get("timestamp_iso")
        or datetime.now().astimezone().isoformat(timespec="milliseconds")
    )
    exposure_time_s = frame_specific_params.get("exposure_time_s")
    if not _is_finite_positive(exposure_time_s):
        exposure_time_s = template_params.get("exposure_time_s", 0.0)
    exposure_period_s = frame_specific_params.get("exposure_period_s")
    if not _is_finite_positive(exposure_period_s):
        exposure_period_s = template_params.get("exposure_period_s", exposure_time_s)

    lines = [
        f"# Detector: {template_params.get('detector_model', 'GENERIC_MICROED')}, S/N {template_params.get('serial_number', '00-0000')}",
        f"# {timestamp_iso}",
        f"# Pixel_size {template_params['pixel_size_x_m']:.6e} m x {template_params['pixel_size_y_m']:.6e} m",
        *_build_sensor_header_lines(template_params),
        f"# Exposure_time {float(exposure_time_s):.7f} s",
        f"# Exposure_period {float(exposure_period_s):.7f} s",
        f"# Count_cutoff {overload_value} counts",
        f"# Wavelength {template_params['wavelength_A']:.5f} A",
        f"# Detector_distance {template_params['detector_distance_m']:.5f} m",
        f"# Beam_xy ({frame_specific_params['beam_center_x_px']:.2f}, {frame_specific_params['beam_center_y_px']:.2f}) pixels",
        f"# Start_angle {frame_specific_params['start_angle_deg']:.4f} deg.",
        f"# Angle_increment {template_params['angle_increment_deg']:.4f} deg.",
        f"# Applied_Pedestal {applied_pedestal} counts",
    ]

    gain_value = template_params.get("gain")
    if gain_value is not None and np.isfinite(gain_value) and gain_value > 0:
        lines.append(f"# Gain {float(gain_value):.6f} ADU/e")

    probe = template_params.get("probe")
    if probe:
        lines.append(f"# Probe {probe}")

    goniometer_axis = template_params.get("goniometer_axis")
    if goniometer_axis:
        lines.append(f"# Goniometer_axis {goniometer_axis}")

    lines.extend(
        [
            f"# Flux {template_params.get('flux', 0.0):.1f}",
            f"# Filter_transmission {template_params.get('transmission', 1.0):.4f}",
            f"# Polarization {template_params.get('polarization_fraction', 0.0):.3f}",
            "# Detector_2theta 0.0000 deg.",
        ]
    )

    detector_profile = template_params.get("detector_profile")
    if detector_profile:
        lines.append(f"# Detector_profile {detector_profile}")

    return "\n".join(lines)

# --- Second Pass Beam Center Validation/Smoothing ---
def smooth_beam_centers(beam_centers_yx: List[Optional[Tuple[float, float]]],
                        max_jump_pixels: float = 2.0, window_length: int = 11,
                        polyorder: int = 2, fallback_strategy: str = 'previous'
                       ) -> List[Optional[Tuple[float, float]]]:
    num_frames = len(beam_centers_yx)
    if num_frames == 0: 
        return []
    valid_indices = [i for i, bc in enumerate(beam_centers_yx) if bc is not None]
    if len(valid_indices) < 2:
        logging.warning("Smoothing: Less than 2 valid centers, cannot perform jump detection/smoothing.")
        if fallback_strategy == 'global_median' and len(valid_indices) == 1:
             median_y, median_x = beam_centers_yx[valid_indices[0]]
             return [(median_y, median_x) if bc is None else bc for bc in beam_centers_yx]
        return beam_centers_yx

    outlier_indices = set()
    last_valid_idx = valid_indices[0]
    for i in range(1, len(valid_indices)):
        current_idx = valid_indices[i]
        if last_valid_idx < len(beam_centers_yx) and current_idx < len(beam_centers_yx):
            prev_bc, curr_bc = beam_centers_yx[last_valid_idx], beam_centers_yx[current_idx]
            if prev_bc is not None and curr_bc is not None:
                prev_y, prev_x = prev_bc
                curr_y, curr_x = curr_bc
                jump_dist = np.sqrt((curr_y - prev_y)**2 + (curr_x - prev_x)**2)
                if jump_dist > max_jump_pixels:
                    logging.warning(f"Smoothing: Frame {current_idx}: Large jump ({jump_dist:.2f} px) from frame {last_valid_idx}. Marking outlier.")
                    outlier_indices.add(current_idx)
                else:
                    last_valid_idx = current_idx
            else: 
                logging.warning(f"Smoothing: Unexpected None at index {current_idx} or {last_valid_idx}.")
        else: 
            logging.error(f"Smoothing: Invalid index: current={current_idx}, last_valid={last_valid_idx}, len={num_frames}")

    smoothed_centers_yx = list(beam_centers_yx)
    frames_to_fix = sorted(list(set(i for i, bc in enumerate(smoothed_centers_yx) if bc is None or i in outlier_indices)))

    if not frames_to_fix:
         logging.info("Smoothing: No None values or large jumps detected.")
    else:
        logging.info(f"Smoothing: Attempting to fix {len(frames_to_fix)} frames using strategy: {fallback_strategy}")

    non_outlier_indices = [i for i in valid_indices if i not in outlier_indices]
    global_median_y, global_median_x = None, None
    if non_outlier_indices:
        global_median_y = np.median([smoothed_centers_yx[i][0] for i in non_outlier_indices])
        global_median_x = np.median([smoothed_centers_yx[i][1] for i in non_outlier_indices])
    elif valid_indices:
        logging.warning("Smoothing: All valid points were outliers? Using median of all original valid points.")
        global_median_y = np.median([beam_centers_yx[i][0] for i in valid_indices])
        global_median_x = np.median([beam_centers_yx[i][1] for i in valid_indices])

    for i in frames_to_fix:
        fixed = False
        if fallback_strategy == 'previous':
            for j in range(i - 1, -1, -1):
                if 0 <= j < len(smoothed_centers_yx) and smoothed_centers_yx[j] is not None and j not in outlier_indices:
                    smoothed_centers_yx[i] = smoothed_centers_yx[j]
                    logging.debug(f"Smoothing Frame {i}: Used previous valid center from frame {j}.")
                    fixed = True; break
            if not fixed and global_median_y is not None:
                smoothed_centers_yx[i] = (global_median_y, global_median_x)
                logging.debug(f"Smoothing Frame {i}: No preceding valid, used global median.")
                fixed = True
        elif fallback_strategy == 'interpolate':
            prev_valid, next_valid = -1, -1
            for j in range(i - 1, -1, -1):
                 if 0 <= j < len(smoothed_centers_yx) and smoothed_centers_yx[j] is not None and j not in outlier_indices: 
                     prev_valid = j; break
            for j in range(i + 1, num_frames):
                 if 0 <= j < len(smoothed_centers_yx) and smoothed_centers_yx[j] is not None and j not in outlier_indices: 
                     next_valid = j; break
            if prev_valid != -1 and next_valid != -1:
                 y1_t, y2_t = smoothed_centers_yx[prev_valid], smoothed_centers_yx[next_valid]
                 if y1_t is not None and y2_t is not None:
                     y1, x1 = y1_t; y2, x2 = y2_t
                     fraction = (i - prev_valid) / (next_valid - prev_valid)
                     interp_y = y1 + fraction * (y2 - y1); interp_x = x1 + fraction * (x2 - x1)
                     smoothed_centers_yx[i] = (interp_y, interp_x)
                     logging.debug(f"Smoothing Frame {i}: Interpolated between {prev_valid} and {next_valid}.")
                     fixed = True
            if not fixed: 
                 if prev_valid != -1 and smoothed_centers_yx[prev_valid] is not None:
                     smoothed_centers_yx[i] = smoothed_centers_yx[prev_valid]; fixed = True
                     logging.debug(f"Smoothing Frame {i}: Used previous (no next valid).")
                 elif next_valid != -1 and smoothed_centers_yx[next_valid] is not None:
                     smoothed_centers_yx[i] = smoothed_centers_yx[next_valid]; fixed = True
                     logging.debug(f"Smoothing Frame {i}: Used next (no previous valid).")
                 elif global_median_y is not None:
                     smoothed_centers_yx[i] = (global_median_y, global_median_x); fixed = True
                     logging.debug(f"Smoothing Frame {i}: No neighbors, used global median.")
        elif fallback_strategy == 'global_median':
             if global_median_y is not None:
                 smoothed_centers_yx[i] = (global_median_y, global_median_x); fixed = True
                 logging.debug(f"Smoothing Frame {i}: Used global median.")

        if not fixed:
            smoothed_centers_yx[i] = None
            logging.warning(f"Smoothing Frame {i}: Beam center remains None after fallback '{fallback_strategy}'.")

    final_valid_indices = [i for i, bc in enumerate(smoothed_centers_yx) if bc is not None]
    if len(final_valid_indices) >= window_length and polyorder < window_length:
        try:
            if window_length % 2 == 0: 
                logging.warning(f"Smoothing: Sav-Gol window_length {window_length} is even, adjusting.")
                window_length += 1
            if window_length > len(final_valid_indices): 
                raise ValueError("Adjusted window too long.")

            final_valid_y = np.array([smoothed_centers_yx[i][0] for i in final_valid_indices])
            final_valid_x = np.array([smoothed_centers_yx[i][1] for i in final_valid_indices])
            smoothed_y = savgol_filter(final_valid_y, window_length, polyorder)
            smoothed_x = savgol_filter(final_valid_x, window_length, polyorder)

            for k, idx in enumerate(final_valid_indices):
                 original_y, original_x = smoothed_centers_yx[idx]
                 smooth_dist = np.sqrt((smoothed_y[k]-original_y)**2 + (smoothed_x[k]-original_x)**2)
                 smoothing_dev_thresh = max_jump_pixels * 1.5
                 if smooth_dist > smoothing_dev_thresh:
                      logging.warning(f"Smoothing Frame {idx}: Sav-Gol large deviation ({smooth_dist:.2f} px). Retaining pre-smoothed value.")
                 else: 
                      smoothed_centers_yx[idx] = (smoothed_y[k], smoothed_x[k])
            logging.info(f"Smoothing: Applied Savitzky-Golay (win={window_length}, order={polyorder}).")
        except ValueError as e:
             logging.warning(f"Smoothing: Skipping Sav-Gol: {e}")
        except Exception as e:
            logging.warning(f"Smoothing: Sav-Gol failed unexpectedly: {e}.")
    elif len(final_valid_indices) > 0: 
        logging.warning(f"Smoothing: Skipping Sav-Gol: Not enough valid points ({len(final_valid_indices)}) for window {window_length}.")
    else: 
        logging.warning("Smoothing: Skipping Sav-Gol: No valid points.")

    return smoothed_centers_yx

# --- Global variables for worker processes ---
g_worker_binned_data_beamfind = None 
g_worker_beam_find_func = None      

g_worker_binned_data_cbfwrite = None    
g_worker_common_params_cbfwrite = None  
g_worker_frame_params_list_cbfwrite = None
g_worker_final_beam_centers_cbfwrite = None
g_worker_output_dir_cbfwrite = None     


def init_worker_beam_find(bds_main, beam_find_func_main):
    """Initializer for beam finding worker processes."""
    global g_worker_binned_data_beamfind, g_worker_beam_find_func
    init_start_time = time.time()
    g_worker_binned_data_beamfind = bds_main
    g_worker_beam_find_func = beam_find_func_main
    logging.debug(f"BeamFind Worker {os.getpid()} initialized in {time.time() - init_start_time:.4f}s")

def find_beam_center_for_frame_mp(idx: int) -> Tuple[int, Optional[Tuple[float, float]], Optional[Dict]]:
    """Worker function to find beam center for a single frame using global data."""
    global g_worker_binned_data_beamfind, g_worker_beam_find_func
    try:
        if g_worker_binned_data_beamfind is None or g_worker_beam_find_func is None:
            logging.error(f"Frame {idx}: BeamFind worker not properly initialized!")
            return idx, None, {'status': 'Failed (Worker Not Initialized)'}

        frame_data_float = g_worker_binned_data_beamfind[idx].astype(np.float64)
        beam_y, beam_x, fit_details = g_worker_beam_find_func(frame_data_float)
        
        if beam_y is not None and beam_x is not None:
            return idx, (beam_y, beam_x), fit_details
        else:
            return idx, None, fit_details
    except Exception as e:
        logging.error(f"Frame {idx}: Error during parallel beam center finding: {e}", exc_info=True)
        return idx, None, {'status': f'Failed in MP (Exception: {type(e).__name__})'}

def init_worker_cbf_write(bds_main, cp_main, fpl_main, fbc_main, od_main):
    """Initializer for CBF writing worker processes."""
    global g_worker_binned_data_cbfwrite, g_worker_common_params_cbfwrite, g_worker_frame_params_list_cbfwrite, g_worker_final_beam_centers_cbfwrite, g_worker_output_dir_cbfwrite
    init_start_time = time.time()
    g_worker_binned_data_cbfwrite = bds_main
    g_worker_common_params_cbfwrite = cp_main
    g_worker_frame_params_list_cbfwrite = fpl_main
    g_worker_final_beam_centers_cbfwrite = fbc_main
    g_worker_output_dir_cbfwrite = od_main
    logging.debug(f"CBFWrite Worker {os.getpid()} initialized in {time.time() - init_start_time:.4f}s")


def process_frame_cbf_mp(task_args: Tuple[int, bool, bool] 
                        ) -> Tuple[int, Optional[Tuple[float, float]], Optional[Tuple[float, float]], str, Optional[str], Dict[str, float]]:
    global g_worker_binned_data_cbfwrite, g_worker_common_params_cbfwrite, g_worker_frame_params_list_cbfwrite, g_worker_final_beam_centers_cbfwrite, g_worker_output_dir_cbfwrite
    
    frame_idx, diagnostic_plot_frame, apply_shift = task_args

    timings = {
        'slicing_astype': 0.0, 'shift_image': 0.0, 'post_shift_proc': 0.0,
        'cbf_header_prep': 0.0, 'cbf_write': 0.0, 'cbf_fixup': 0.0, 'plotting': 0.0, 'total_frame_proc': 0.0
    }
    overall_frame_start_time = time.time()

    try:
        binned_data_stack_local = g_worker_binned_data_cbfwrite
        common_params_local = g_worker_common_params_cbfwrite
        frame_params_list_local = g_worker_frame_params_list_cbfwrite
        final_beam_centers_local = g_worker_final_beam_centers_cbfwrite
        output_dir_local = g_worker_output_dir_cbfwrite

        if any(v is None for v in [binned_data_stack_local, common_params_local, frame_params_list_local, final_beam_centers_local, output_dir_local]):
            logging.error(f"Frame {frame_idx}: CBFWrite worker not properly initialized with all shared data.")
            timings['total_frame_proc'] = time.time() - overall_frame_start_time
            return frame_idx, None, None, 'Failed (Worker Init Error)', None, timings

        t_start_slicing = time.time()
        frame_data = binned_data_stack_local[frame_idx].astype(np.float32)
        timings['slicing_astype'] = time.time() - t_start_slicing
        
        frame_params = frame_params_list_local[frame_idx]
        num_rows, num_cols = frame_data.shape
        target_center_yx = ((num_rows - 1) / 2.0, (num_cols - 1) / 2.0)

        beam_center_final_local = final_beam_centers_local[frame_idx]

        if beam_center_final_local is None:
            logging.warning(f"Frame {frame_idx}: No valid beam center provided. Skipping CBF generation.")
            timings['total_frame_proc'] = time.time() - overall_frame_start_time
            return frame_idx, None, None, 'Skipped (No Beam Center)', None, timings

        current_center_yx = beam_center_final_local
        beam_y, beam_x = current_center_yx

        final_img_data: NDArray[np.int32]
        shift_vector_yx: Optional[Tuple[float, float]] = None
        should_shift_data = apply_shift
        
        if should_shift_data:
            pedestal_value = common_params_local.get('pedestal', 0.0)
            t_start_shift = time.time()
            if np.allclose(current_center_yx, target_center_yx, atol=1e-3):
                 shifted_img = frame_data 
                 shift_vector_yx = (0.0, 0.0)
            else:
                 shifted_img, shift_vector_yx = shift_image(
                     frame_data,
                     current_center_yx=current_center_yx,
                     target_center_yx=target_center_yx,
                     order=common_params_local.get('shift_interpolation_order', 1),
                     mode='constant',
                     cval=pedestal_value
                 )
            timings['shift_image'] = time.time() - t_start_shift
            
            t_start_post_shift = time.time()
            final_img_data = np.clip(np.round(shifted_img), 0, np.iinfo(np.int32).max).astype(np.int32)
            timings['post_shift_proc'] = time.time() - t_start_post_shift
        else:
            t_start_post_shift = time.time()
            final_img_data = np.clip(np.round(frame_data), 0, np.iinfo(np.int32).max).astype(np.int32)
            timings['post_shift_proc'] = time.time() - t_start_post_shift
            shift_vector_yx = (0.0, 0.0) 

        t_start_header_prep = time.time()
        header_beam_x = target_center_yx[1] if should_shift_data else beam_x
        header_beam_y = target_center_yx[0] if should_shift_data else beam_y
        header_beam_x_1based = header_beam_x + 1.0
        header_beam_y_1based = header_beam_y + 1.0
        actual_pedestal_value = int(common_params_local.get('pedestal', 0)) 
        overload_val = int(common_params_local.get('overload', 1000000)) 
        cbf_header_str = create_cbf_header(
            template_params=common_params_local,
            frame_specific_params={
                'beam_center_x_px': header_beam_x_1based,
                'beam_center_y_px': header_beam_y_1based,
                'start_angle_deg': frame_params['current_angle'],
                'exposure_time_s': frame_params.get('exposure_time_s'),
                'exposure_period_s': frame_params.get('exposure_period_s'),
                'timestamp_iso': frame_params.get('timestamp_iso'),
            },
            applied_pedestal=actual_pedestal_value, 
            overload_value=overload_val 
        )
        cbf_image = CbfImage(data=final_img_data)
        cbf_image.header.update({
            "_array_data.header_convention": common_params_local.get('cbf_header_convention', 'GENERIC_MINI'),
            "_array_data.header_contents": cbf_header_str,
        })
        timings['cbf_header_prep'] = time.time() - t_start_header_prep

        cbf_file_path = os.path.join(output_dir_local, common_params_local['filename_template'].format(frame_idx + 1))
        status = 'Write Failed'
        try:
            t_start_write = time.time()
            cbf_image.write(cbf_file_path)
            timings['cbf_write'] = time.time() - t_start_write

            t_start_fixup = time.time()
            rewrite_fabio_cbf_as_generic_minicbf(
                cbf_file_path,
                cbf_header_str,
                header_convention=common_params_local.get('cbf_header_convention', 'GENERIC_MINI'),
            )
            timings['cbf_fixup'] = time.time() - t_start_fixup
            status = 'Processed'
        except Exception as write_e:
            logging.error(f"Frame {frame_idx}: Failed to write CBF file {cbf_file_path}: {write_e}", exc_info=True)
            cbf_file_path = None
        
        if diagnostic_plot_frame:
            t_start_plot = time.time()
            try:
                plt = get_plt()
                plt.figure(figsize=(10, 8)); gs = plt.GridSpec(2, 2)
                ax1 = plt.subplot(gs[0, 0]); ax1.imshow(frame_data, cmap='inferno', origin='lower'); ax1.set_title("Original Binned")
                ax1.plot(beam_x, beam_y, 'bo', ms=5, mfc='none'); ax1.plot(target_center_yx[1], target_center_yx[0], 'rx', ms=5)
                ax2 = plt.subplot(gs[0, 1]); ax2.imshow(final_img_data, cmap='inferno', origin='lower'); ax2.set_title("Processed for CBF")
                ax2.plot(target_center_yx[1], target_center_yx[0], 'rx', ms=5)
                plt.suptitle(f'Diagnostic Plot - Frame {frame_idx}', fontsize=14); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plot_filename = os.path.join(output_dir_local, f"diagnostic_frame_{frame_idx+1:05d}.png")
                plt.savefig(plot_filename); plt.close()
            except Exception as plot_e: 
                logging.error(f"Frame {frame_idx}: Failed diagnostic plot: {plot_e}")
            timings['plotting'] = time.time() - t_start_plot
        
        timings['total_frame_proc'] = time.time() - overall_frame_start_time
        logging.debug(f"Frame {frame_idx}: Timings (s): Slice+astype: {timings['slicing_astype']:.4f}, Shift: {timings['shift_image']:.4f}, PostShift: {timings['post_shift_proc']:.4f}, CBFHeader: {timings['cbf_header_prep']:.4f}, CBFWrite: {timings['cbf_write']:.4f}, CBFFixup: {timings['cbf_fixup']:.4f}, Plot: {timings['plotting']:.4f}, Total: {timings['total_frame_proc']:.4f}")
        
        return frame_idx, (beam_y, beam_x) if beam_center_final_local else None, shift_vector_yx, status, cbf_file_path, timings

    except Exception as e:
        timings['total_frame_proc'] = time.time() - overall_frame_start_time
        logging.error(f"Error processing frame {frame_idx} in worker: {e}", exc_info=True) 
        return frame_idx, None, None, f'Failed ({type(e).__name__})', None, timings


# --- Helper function for multiprocessing ---
def _worker_wrapper_beam_find(idx: int): 
    return find_beam_center_for_frame_mp(idx)

def _worker_wrapper_cbf_write(args_tuple: Tuple[int, bool, bool]): # CORRECTED
    """ Helper function to pass arguments to process_frame_cbf_mp. """
    return process_frame_cbf_mp(args_tuple) # Pass the tuple directly


# --- Main Pipeline Function ---
def mrc_to_cbf_pipeline_mp_init( 
    mrc_path: str, output_dir: str, pixel_size_mm: float, detector_distance_mm: float,
    wavelength_A: float, start_angle_deg: float, angle_increment_deg: float,
    overload_cli: int,
    bin_x: int = 1, bin_y: int = 1, bin_z: int = 1, pedestal: Optional[int] = None,
    hardware_bin_x: int = 1, hardware_bin_y: int = 1,
    auto_pedestal: bool = True, skip_beam_centering: bool = False, 
    beam_center_roi_size: int = 128, beam_center_sigma_blur: float = 5.0, 
    beam_center_max_initial_deviation: float = 80.0, beam_center_fit_bounds: bool = True,
    beam_center_method: str = "robust",
    dark_map_path: Optional[str] = None,
    gain_map_path: Optional[str] = None,
    gain_map_kind: str = "auto",
    bad_pixel_mask_path: Optional[str] = None,
    bad_pixel_fill: str = "median3",
    trim_edge_bad_frames: bool = True,
    trim_policy: str = "aggressive",
    aggressive_trim_min_strong_frames: int = 5,
    aggressive_trim_signal_ratio: float = 0.65,
    aggressive_trim_std_ratio: float = 0.55,
    aggressive_trim_peak_ratio: float = 0.45,
    auto_pedestal_negative_quantile: float = 0.0,
    xds_offset_mode: str = "storage_pedestal",
    perform_second_pass: bool = True, max_beam_jump_pixels: float = 2.0, 
    smoothing_window_length: int = 11, smoothing_polyorder: int = 2, 
    smoothing_fallback: str = 'previous', apply_image_shift: bool = True, 
    shift_interpolation_order: int = 1, num_workers: Optional[int] = None,
    filename_template: str = "image_{:05d}.cbf",
    gain: Optional[float] = None, probe: str = "electron",
    goniometer_axis: str = "0.999263,-0.0383878,0",
    gain_mode: str = "auto",
    gain_estimation_frames: int = 120,
    gain_kernels: Optional[List[int]] = None,
    gain_grids: Optional[List[Tuple[int, int]]] = None,
    metadata_gain: Optional[float] = None,
    metadata_gain_source: Optional[str] = None,
    detector_model: str = "GENERIC_MICROED", serial_number: str = "00-0000",
    sensor_kind: str = "ccd", sensor_material: Optional[str] = None,
    sensor_thickness_mm: Optional[float] = None,
    detector_profile: Optional[str] = None, detector_profile_reason: Optional[str] = None,
    pixel_size_source: Optional[str] = None, gain_source: Optional[str] = None,
    xml_metadata: Optional[Dict[str, Any]] = None,
    mdoc_metadata: Optional[Dict[str, Any]] = None,
    oscillation_validation_report: Optional[Dict[str, Any]] = None,
    exposure_time_s: Optional[float] = None, exposure_period_s: Optional[float] = None,
    conditioning_mode: str = "radial_reduced",
    xds_offset_quantile: float = 0.001,
    xds_offset_inner_fraction: float = 0.82,
    xds_offset_outer_fraction: float = 0.98,
    radial_store_pedestal: Optional[int] = 100,
    radial_pedestal_quantile: float = 1.0e-5,
    radial_pedestal_margin: float = 20.0,
    radial_buffer: int = 40,
    radial_bin_width_px: float = 2.0,
    radial_quantile: float = 0.50,
    radial_profile_smooth_sigma_bins: float = 4.0,
    radial_mask_smooth_sigma_px: float = 18.0,
    radial_clip_high_sigma: float = 5.0,
    radial_clip_low_sigma: float = 8.0,
    radial_mask_dilation_px: int = 2,
    radial_center_mask_radius_px: float = 28.0,
    radial_fit_plane: bool = True,
    radial_plane_sample_pixels: int = 120_000,
    write_dials_import_helper_file: bool = True, write_conversion_report_file: bool = True,
    write_xds_inp_file: bool = True,
    first_pass_diagnostic_plots: bool = False,
    final_diagnostic_plots: bool = True, diagnostic_plot_specific_frames: Optional[List[int]] = None,
    save_beam_centers_file: bool = True, limit_frames: Optional[int] = None
) -> None:
    start_time = time.time()
    converter_contract_name, converter_contract = resolve_converter_contract(conditioning_mode)
    logging.info(f"Pipeline started for: {mrc_path} -> CBF output (MP Initializer, Parallel Beamfind, v8)")
    logging.info("Converter contract: %s", converter_contract_name)
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Beam centering: {'SKIPPED (using geometric center)' if skip_beam_centering else 'ENABLED'}")
    if not skip_beam_centering: logging.info(f"Second pass smoothing/validation: {'ENABLED' if perform_second_pass else 'DISABLED'}")
    logging.info(f"Image shifting: {'ENABLED' if apply_image_shift else 'DISABLED'}")
    logging.info(
        f"Binning plan: hardware(x={hardware_bin_x}, y={hardware_bin_y}), software(x={bin_x}, y={bin_y}, z={bin_z})"
    )
    if detector_profile:
        reason_suffix = f" ({detector_profile_reason})" if detector_profile_reason else ""
        logging.info(f"Detector profile: {detector_profile}{reason_suffix}")

    limited_input_for_processing = False
    try:
        if not os.path.exists(mrc_path): raise FileNotFoundError(f"MRC file not found: {mrc_path}")
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            input_shape = tuple(int(v) for v in mrc.data.shape)
            if limit_frames is not None and limit_frames > 0:
                raw_frame_limit = int(limit_frames) * int(bin_z)
                if raw_frame_limit < int(input_shape[0]):
                    logging.warning(
                        "Limiting to first %d output frame(s): loading first %d raw input frame(s) before binning.",
                        int(limit_frames),
                        raw_frame_limit,
                    )
                    data = mrc.data[:raw_frame_limit].copy()
                    limited_input_for_processing = True
                else:
                    logging.info("Frame limit (%d) >= available output frames. Processing all.", int(limit_frames))
                    logging.info("Loading MRC data...")
                    data = mrc.data.copy()
            else:
                logging.info("Loading MRC data...")
                data = mrc.data.copy()
            logging.info(f"MRC data loaded. Shape: {data.shape}, Type: {data.dtype}; full input shape: {input_shape}")
    except Exception as e: logging.critical(f"Failed to read MRC file: {e}", exc_info=True); return

    try:
        logging.info(f"Binning data (z={bin_z}, y={bin_y}, x={bin_x})..."); binned_data, new_shape = bin_array(data, bin_z, bin_y, bin_x)
        logging.info(f"Binned data shape: {new_shape}, Type: {binned_data.dtype}"); del data
    except Exception as e: logging.critical(f"Binning failed: {e}", exc_info=True); return

    if limit_frames is not None and limit_frames > 0:
        if limit_frames < binned_data.shape[0]:
            logging.warning(f"Limiting to first {limit_frames} frames.")
            binned_data = binned_data[:limit_frames]
            new_shape = binned_data.shape
        elif not limited_input_for_processing:
            logging.info(f"Frame limit ({limit_frames}) >= total frames. Processing all.")
    num_frames, nrows, ncols = new_shape
    if num_frames == 0: logging.critical("No frames remaining. Exiting."); return

    detector_corrections_report: Dict[str, Any] = {
        "dark_map": None,
        "gain_map": None,
        "bad_pixel_mask": None,
    }
    try:
        if dark_map_path or gain_map_path or bad_pixel_mask_path:
            logging.info("Applying detector-coordinate corrections before pedestal and beam centering...")
            binned_data, detector_corrections_report = apply_pre_shift_detector_corrections(
                binned_data,
                raw_frame_shape=(int(input_shape[1]), int(input_shape[2])),
                bin_y=bin_y,
                bin_x=bin_x,
                dark_map_path=dark_map_path,
                gain_map_path=gain_map_path,
                gain_map_kind=gain_map_kind,
                bad_pixel_mask_path=bad_pixel_mask_path,
                bad_pixel_fill=bad_pixel_fill,
            )
            logging.info(
                "Detector corrections applied: dark=%s gain=%s bad_mask=%s",
                bool(detector_corrections_report.get("dark_map")),
                bool(detector_corrections_report.get("gain_map")),
                bool(detector_corrections_report.get("bad_pixel_mask")),
            )
    except Exception as e:
        logging.critical(f"Detector corrections failed: {e}", exc_info=True)
        return

    raw_frame_metadata, frame_metadata_sources = build_raw_frame_metadata(
        num_input_frames=input_shape[0],
        xml_metadata=xml_metadata,
        fallback_exposure_time_s=exposure_time_s,
        fallback_exposure_period_s=exposure_period_s,
        fallback_timestamp_iso=(xml_metadata or {}).get("info_timestamp_iso"),
    )
    output_frame_metadata = aggregate_output_frame_metadata(raw_frame_metadata, bin_z=bin_z, num_output_frames=num_frames)
    first_output_metadata = next((row for row in output_frame_metadata if row), {})

    effective_gain = float(gain) if _is_finite_positive(gain) else None
    effective_gain_source = gain_source if effective_gain is not None else None
    metadata_gain_value = float(metadata_gain) if _is_finite_positive(metadata_gain) else None
    metadata_gain_value_source = metadata_gain_source if metadata_gain_value is not None else None
    gain_estimation_report = None
    gain_prior_candidates: List[float] = []
    if metadata_gain_value is not None:
        gain_prior_candidates.append(float(metadata_gain_value))
        hardware_scale = int(hardware_bin_x) * int(hardware_bin_y)
        if hardware_scale > 1 and metadata_gain_value_source and metadata_gain_value_source.startswith("mdoc:"):
            gain_prior_candidates.append(float(metadata_gain_value) * float(hardware_scale))
    resolved_gain_kernels = list(gain_kernels or [11, 15, 21, 27, 35])
    resolved_gain_grids = list(gain_grids or [(24, 24), (32, 32), (48, 48), (64, 64), (80, 80)])
    metadata_gain_is_xml = bool(metadata_gain_value_source and metadata_gain_value_source.startswith("xml:"))
    should_run_adaptive_gain = gain_mode == "adaptive" or (gain_mode == "auto" and not metadata_gain_is_xml)

    if effective_gain is not None:
        logging.info(f"Using CLI detector gain: {effective_gain:.6f}")
    elif should_run_adaptive_gain:
        gain_indices = select_gain_frame_indices(num_frames, gain_estimation_frames)
        logging.info(
            "Estimating detector gain from %d binned frames sampled across the scan using kernels=%s grids=%s",
            len(gain_indices),
            resolved_gain_kernels,
            resolved_gain_grids,
        )
        gain_stack = np.asarray(binned_data[gain_indices], dtype=np.float32)
        try:
            gain_estimation_report = estimate_gain_from_binned_stack(
                gain_stack,
                metadata_gain=metadata_gain_value,
                prior_gains=gain_prior_candidates,
                kernels=resolved_gain_kernels,
                grids=resolved_gain_grids,
            )
            effective_gain = float(gain_estimation_report["recommended_gain"])
            effective_gain_source = f"adaptive:{gain_estimation_report.get('selected_source', 'unknown')}"
            gain_estimation_report["input"] = {
                "frame_selection_strategy": "stratified_across_scan",
                "frame_indices_0_based": [int(idx) for idx in gain_indices],
                "frames_used": int(len(gain_indices)),
                "kernels": [int(v) for v in resolved_gain_kernels],
                "grids": [[int(gx), int(gy)] for gx, gy in resolved_gain_grids],
                "prior_gains": [float(value) for value in gain_prior_candidates],
            }
            logging.info("Adaptive detector gain estimate: %.6f (%s)", effective_gain, effective_gain_source)
        except Exception as exc:
            if gain_mode == "adaptive":
                logging.critical("Adaptive gain estimation failed: %s", exc, exc_info=True)
                raise
            logging.warning("Adaptive gain estimation failed, falling back to metadata/profile gain: %s", exc)
    elif metadata_gain_value is not None and metadata_gain_value_source:
        effective_gain = metadata_gain_value
        effective_gain_source = metadata_gain_value_source
        logging.info(
            "Using metadata detector gain directly in %s mode: %.6f (%s)",
            gain_mode,
            metadata_gain_value,
            metadata_gain_value_source,
        )

    if effective_gain is None:
        effective_gain = metadata_gain_value
        effective_gain_source = metadata_gain_value_source
        if effective_gain is not None and effective_gain_source:
            logging.info("Using metadata/profile detector gain: %.6f (%s)", effective_gain, effective_gain_source)
        else:
            logging.warning("No detector gain could be resolved from CLI, adaptive estimation, or metadata.")

    actual_pedestal = 0; common_params_for_cbf_workers_main = {} 
    xds_offset_value = 0
    xds_offset_details: Optional[Dict[str, Any]] = None
    reduced_background_offset_value: Optional[int] = None
    reduced_background_offset_details: Optional[Dict[str, Any]] = None
    storage_pedestal_report: Dict[str, Any] = {}
    background_conditioning_report: Dict[str, Any] = {"mode": str(conditioning_mode or "v7")}
    try:
        if pedestal is not None: actual_pedestal = int(pedestal); logging.info(f"Using user pedestal: {actual_pedestal}")
        elif auto_pedestal:
            actual_pedestal, storage_pedestal_report = estimate_storage_pedestal(
                binned_data,
                negative_quantile=auto_pedestal_negative_quantile,
            )
            if actual_pedestal > 0:
                logging.info(
                    "Calculated storage pedestal: %d using %s at negative quantile %.4f (min %.3f; residual negatives %s).",
                    actual_pedestal,
                    str(storage_pedestal_report.get("method", "unknown")),
                    float(storage_pedestal_report.get("negative_quantile", auto_pedestal_negative_quantile)),
                    float(storage_pedestal_report.get("min_value", 0.0)),
                    storage_pedestal_report.get("residual_negative_count_after_pedestal", 0),
                )
            else: logging.info("Auto-pedestal: No pedestal needed (min value >= 0).")
        else: logging.info("Pedestal: Disabled.")
        if actual_pedestal > 0:
            logging.info(f"Applying pedestal {actual_pedestal}...");
            if np.issubdtype(binned_data.dtype, np.integer):
                 if np.max(binned_data) > np.iinfo(np.int32).max - actual_pedestal: logging.warning(f"Pedestal+max_val may exceed int32 max.")
            binned_data = binned_data.astype(np.int64) + actual_pedestal
        common_params_for_cbf_workers_main['pedestal'] = float(actual_pedestal)
    except Exception as e: logging.critical(f"Error during pedestal handling: {e}", exc_info=True); return

    try:
        xds_offset_mode_norm = str(xds_offset_mode).strip().lower()
        if xds_offset_mode_norm in {"storage_pedestal", "pedestal", "actual", "actual_offset"}:
            xds_offset_value = int(actual_pedestal)
            xds_offset_details = {
                "method": "storage_pedestal",
                "suggested_offset": int(xds_offset_value),
                "storage_pedestal_report": storage_pedestal_report,
            }
        else:
            xds_offset_details = estimate_xds_offset_from_outer_shell(binned_data)
            candidate_offsets = xds_offset_details.get("candidate_offsets", {})
            if xds_offset_mode_norm in {"outer_shell_mode", "mode"}:
                xds_offset_value = int(candidate_offsets.get("outer_shell_mode", xds_offset_details["mode_value"]))
                xds_offset_details["method"] = "outer_shell_mode"
            elif xds_offset_mode_norm in {"outer_shell_median", "median"}:
                xds_offset_value = int(candidate_offsets.get("outer_shell_median", round(float(xds_offset_details["median_value"]))))
                xds_offset_details["method"] = "outer_shell_median"
            elif xds_offset_mode_norm == "outer_shell_mode_minus_sigma":
                xds_offset_value = int(candidate_offsets.get("outer_shell_mode_minus_sigma", xds_offset_details["suggested_offset"]))
                xds_offset_details["method"] = "outer_shell_mode_minus_sigma"
            else:
                logging.warning("Unknown XDS offset mode '%s'; using outer_shell_mode.", xds_offset_mode)
                xds_offset_value = int(candidate_offsets.get("outer_shell_mode", xds_offset_details["mode_value"]))
                xds_offset_details["method"] = "outer_shell_mode"
            xds_offset_details["selected_mode"] = xds_offset_mode_norm
            xds_offset_details["suggested_offset"] = int(xds_offset_value)
        logging.info(
            "Resolved XDS OFFSET=%d using %s",
            xds_offset_value,
            xds_offset_details.get("method", "unknown") if xds_offset_details else "unknown",
        )
    except Exception as e:
        xds_offset_value = int(actual_pedestal)
        xds_offset_details = {
            "method": "pedestal_fallback",
            "suggested_offset": int(xds_offset_value),
            "error": str(e),
        }
        logging.warning("Failed to estimate XDS OFFSET from outer shell; falling back to pedestal: %s", e)

    try:
        reduced_background_offset_details = estimate_xds_offset_from_outer_shell(binned_data)
        reduced_background_offset_value = int(
            reduced_background_offset_details.get("candidate_offsets", {}).get(
                "outer_shell_mode_minus_sigma",
                reduced_background_offset_details.get("suggested_offset", xds_offset_value),
            )
        )
    except Exception as e:
        reduced_background_offset_details = {"error": str(e)}
        reduced_background_offset_value = None
        logging.warning("Could not estimate reduced/background-only XDS offset comment: %s", e)

    new_pixel_size_x_m = (pixel_size_mm / 1000.0) * hardware_bin_x * bin_x
    new_pixel_size_y_m = (pixel_size_mm / 1000.0) * hardware_bin_y * bin_y
    new_angle_increment_deg_binned = angle_increment_deg * bin_z 
    detector_distance_m = detector_distance_mm / 1000.0
    effective_exposure_time_s = (
        float(first_output_metadata["exposure_time_s"])
        if _is_finite_positive(first_output_metadata.get("exposure_time_s"))
        else (float(exposure_time_s) if _is_finite_positive(exposure_time_s) else 0.0)
    )
    effective_exposure_period_s = (
        float(first_output_metadata["exposure_period_s"])
        if _is_finite_positive(first_output_metadata.get("exposure_period_s"))
        else (
            float(exposure_period_s)
            if _is_finite_positive(exposure_period_s)
            else effective_exposure_time_s
        )
    )
    effective_timestamp_iso = (
        first_output_metadata.get("timestamp_iso")
        or (xml_metadata or {}).get("info_timestamp_iso")
        or datetime.now().astimezone().isoformat(timespec="milliseconds")
    )
    
    if num_workers is None:
        try: workers = cpu_count(); logging.info(f"Using {workers} worker processes (cpu_count).")
        except NotImplementedError: workers = 1; logging.warning("cpu_count() not implemented; using 1 worker.")
    elif num_workers >= 1: workers = int(num_workers); logging.info(f"Using {workers} worker process(es).")
    else: workers = 1; logging.warning(f"Invalid num_workers ({num_workers}); using 1.")
    
    effective_workers = workers
    if num_frames > 0 and num_frames < workers: 
        effective_workers = num_frames
        logging.info(f"Adjusted effective workers to {effective_workers} (fewer frames than workers) for some operations.")

    final_beam_centers_yx: List[Optional[Tuple[float, float]]] = [None] * num_frames
    fit_details_list: List[Optional[Dict]] = [None] * num_frames 
    frame_quality_summary: Optional[Dict[str, Any]] = None
    output_frame_origin_0_based = 0

    if skip_beam_centering:
        logging.info("Beam centering skipped by user request.")
        geometric_center_y = (nrows - 1) / 2.0; geometric_center_x = (ncols - 1) / 2.0
        logging.info(f"Using fixed geometric center: ({geometric_center_y:.2f}, {geometric_center_x:.2f}) pixels (0-based).")
        final_beam_centers_yx = [(geometric_center_y, geometric_center_x)] * num_frames
        for i in range(num_frames): fit_details_list[i] = {'status': 'Skipped (Geometric Center Used)'}
        perform_second_pass = False; first_pass_diagnostic_plots = False
    else:
        logging.info("Starting first pass beam center finding (parallel)...")
        first_pass_start_time = time.time()
        beam_center_method_normalized = str(beam_center_method).strip().lower()
        beam_center_func = find_beam_center_robust
        if beam_center_method_normalized == "blurred_peak":
            beam_center_func = find_beam_center_blurred_peak
        elif beam_center_method_normalized == "gaussian":
            beam_center_func = find_beam_center_gaussian_fit
        elif beam_center_method_normalized != "robust":
            logging.warning("Unknown beam center method '%s'; using robust.", beam_center_method)
            beam_center_method_normalized = "robust"
        logging.info("Beam center method: %s", beam_center_method_normalized)
        beam_find_partial_func = partial(
            beam_center_func,
            roi_size=beam_center_roi_size,
            sigma_blur=beam_center_sigma_blur,
            max_initial_deviation=beam_center_max_initial_deviation,
            fit_bounds=beam_center_fit_bounds,
        )
        beam_find_tasks = list(range(num_frames))
        beam_find_workers = min(effective_workers, num_frames) if num_frames > 0 else 1

        if beam_find_workers > 1:
            with Pool(processes=beam_find_workers, 
                      initializer=init_worker_beam_find, 
                      initargs=(binned_data, beam_find_partial_func)) as pool:
                chunk_bf = max(1, num_frames // (beam_find_workers * 4)) if beam_find_workers > 0 else 1
                logging.info(f"BeamFind Pool: Using chunksize: {chunk_bf} for imap_unordered with {beam_find_workers} workers.")
                
                temp_results_beam_find = [None] * num_frames
                for i, result_bf in enumerate(pool.imap_unordered(_worker_wrapper_beam_find, beam_find_tasks, chunksize=chunk_bf)):
                    idx_res_bf, center_res_bf, details_res_bf = result_bf
                    temp_results_beam_find[idx_res_bf] = (center_res_bf, details_res_bf)
                    if (i + 1) % (max(1, num_frames // 10)) == 0 or (i + 1) == num_frames: 
                        logging.info(f"  Beam finding progress (parallel): {i+1}/{num_frames} tasks processed...")
                for idx_bf_final in range(num_frames):
                    if temp_results_beam_find[idx_bf_final] is not None:
                        center_val, details_val = temp_results_beam_find[idx_bf_final]
                        final_beam_centers_yx[idx_bf_final] = center_val
                        fit_details_list[idx_bf_final] = details_val
                        if center_val is None and details_val:
                             logging.warning(f"Frame {idx_bf_final}: Beam center finding failed in MP - {details_val.get('status', 'Unknown')}")
                    else: 
                        logging.error(f"Frame {idx_bf_final}: No result from beam finding worker.")
                        fit_details_list[idx_bf_final] = {'status': 'Failed (No result from worker)'}
        else: 
            logging.info("Running beam center finding sequentially...")
            init_worker_beam_find(binned_data, beam_find_partial_func) 
            for idx in range(num_frames):
                idx_res_bf, center_res_bf, details_res_bf = find_beam_center_for_frame_mp(idx)
                final_beam_centers_yx[idx_res_bf] = center_res_bf
                fit_details_list[idx_res_bf] = details_res_bf
                if center_res_bf is None and details_res_bf:
                     logging.warning(f"Frame {idx_res_bf}: Beam center finding failed - {details_res_bf.get('status', 'Unknown')}")
                if (idx + 1) % 50 == 0 or idx == num_frames - 1: 
                    logging.info(f"  Beam finding progress (sequential): {idx+1}/{num_frames}")

        first_pass_duration = time.time() - first_pass_start_time
        logging.info(f"First pass beam finding finished in {first_pass_duration:.2f}s.")
        valid_centers_count = sum(1 for bc in final_beam_centers_yx if bc is not None)
        logging.info(f"Found {valid_centers_count}/{num_frames} valid centers in first pass.")
        if valid_centers_count == 0: logging.critical("No valid centers found. Cannot proceed."); return
        
        first_pass_centers_for_plot = list(final_beam_centers_yx) 
        if first_pass_diagnostic_plots: 
             logging.info("Generating first pass diagnostic plot...")
             try:
                 plt = get_plt()
                 plt.figure(figsize=(10, 6)); valid_idx_plot = [i for i, bc in enumerate(first_pass_centers_for_plot) if bc is not None]
                 if valid_idx_plot:
                     plt.plot(valid_idx_plot, [first_pass_centers_for_plot[i][1] for i in valid_idx_plot], '.-', label='X')
                     plt.plot(valid_idx_plot, [first_pass_centers_for_plot[i][0] for i in valid_idx_plot], '.-', label='Y')
                 plt.xlabel('Frame Index (0-based)'); plt.ylabel('Center (pixels, 0-based)'); plt.title('Beam Centers - First Pass')
                 plt.legend(); plt.grid(True); plt.xlim(0, max(0, num_frames - 1))
                 plot_path = os.path.join(output_dir, "beam_centers_first_pass.png"); plt.savefig(plot_path)
                 logging.info(f"Saved first pass plot to {plot_path}"); plt.close()
             except Exception as e: logging.error(f"Failed generating first pass plot: {e}")

        if perform_second_pass: 
            logging.info("Starting second pass validation/smoothing...")
            try:
                if smoothing_window_length % 2 == 0: logging.warning(f"Adjusting smoothing window to {smoothing_window_length + 1}."); smoothing_window_length += 1
                final_beam_centers_yx = smooth_beam_centers(final_beam_centers_yx, max_jump_pixels=max_beam_jump_pixels, window_length=smoothing_window_length, polyorder=smoothing_polyorder, fallback_strategy=smoothing_fallback)
                logging.info("Second pass finished.")
            except Exception as e: logging.error(f"Error during second pass smoothing: {e}", exc_info=True); final_beam_centers_yx = first_pass_centers_for_plot
        else: logging.info("Skipping second pass validation/smoothing.")

    final_valid_count = sum(1 for bc in final_beam_centers_yx if bc is not None)
    logging.info(f"Final beam centers available for {final_valid_count}/{num_frames} frames.")
    if final_valid_count == 0 and not skip_beam_centering: logging.critical("No valid beam centers. Cannot proceed."); return
    elif final_valid_count < num_frames: logging.warning(f"{num_frames - final_valid_count} frames lack valid center and will be skipped.")

    try:
        frame_quality_summary = summarize_frame_quality(
            binned_data,
            final_beam_centers_yx,
            window_radius_px=max(24, min(nrows, ncols) // 16),
        )
        flagged_ranges = frame_quality_summary.get("flagged_ranges_1_based", [])
        flagged_frames = frame_quality_summary.get("flagged_frames_1_based", [])
        if flagged_frames:
            logging.warning(
                "Frame-quality screening flagged %d/%d frames in %d range(s): %s",
                len(flagged_frames),
                num_frames,
                len(flagged_ranges),
                flagged_ranges,
            )
        else:
            logging.info("Frame-quality screening did not flag any dead/dying frames.")
    except Exception as e:
        frame_quality_summary = {
            "error": str(e),
            "frames_evaluated": int(num_frames),
        }
        logging.warning("Frame-quality screening failed: %s", e)

    effective_trim_policy = str(trim_policy or "off").strip().lower()
    if not trim_edge_bad_frames:
        effective_trim_policy = "off"
    if frame_quality_summary and "metrics" in frame_quality_summary:
        trim_start, trim_end_exclusive, trim_details = determine_trim_window_from_quality(
            frame_quality_summary,
            num_frames,
            trim_policy=effective_trim_policy,
            aggressive_min_strong_frames=aggressive_trim_min_strong_frames,
            aggressive_signal_ratio=aggressive_trim_signal_ratio,
            aggressive_std_ratio=aggressive_trim_std_ratio,
            aggressive_peak_ratio=aggressive_trim_peak_ratio,
        )
        trimmed_front = int(trim_start)
        trimmed_back = int(num_frames - trim_end_exclusive)
        if effective_trim_policy == "off":
            frame_quality_summary["edge_trim"] = {
                "enabled": False,
                "applied": False,
                "policy": "off",
                "details": trim_details,
            }
        elif trim_start >= trim_end_exclusive:
            logging.warning("Edge trimming would remove all frames; leaving output untrimmed.")
            frame_quality_summary["edge_trim"] = {
                "enabled": True,
                "applied": False,
                "policy": effective_trim_policy,
                "reason": "all_frames_flagged",
                "details": trim_details,
            }
        elif trimmed_front > 0 or trimmed_back > 0:
            original_num_frames = int(num_frames)
            original_flagged_ranges = list(frame_quality_summary.get("flagged_ranges_1_based", []))
            logging.warning(
                "Trimming %d leading and %d trailing frame(s) from output using trim policy '%s'.",
                trimmed_front,
                trimmed_back,
                effective_trim_policy,
            )
            binned_data = np.asarray(binned_data[trim_start:trim_end_exclusive], dtype=binned_data.dtype)
            final_beam_centers_yx = final_beam_centers_yx[trim_start:trim_end_exclusive]
            fit_details_list = fit_details_list[trim_start:trim_end_exclusive]
            output_frame_metadata = output_frame_metadata[trim_start:trim_end_exclusive]
            output_frame_origin_0_based += int(trim_start)
            if not skip_beam_centering:
                first_pass_centers_for_plot = first_pass_centers_for_plot[trim_start:trim_end_exclusive]
            start_angle_deg = float(start_angle_deg) + float(trim_start) * float(new_angle_increment_deg_binned)
            num_frames = int(trim_end_exclusive - trim_start)
            nrows, ncols = binned_data.shape[1:]
            new_shape = (num_frames, nrows, ncols)

            remaining_flagged = [
                int(idx - trim_start)
                for idx in frame_quality_summary.get("flagged_frames_0_based", [])
                if trim_start <= idx < trim_end_exclusive
            ]
            remaining_ranges_0 = _contiguous_ranges_from_indices(remaining_flagged)
            remaining_ranges_1 = [[int(start + 1), int(end + 1)] for start, end in remaining_ranges_0]

            frame_quality_summary["original_frames_evaluated"] = original_num_frames
            frame_quality_summary["original_flagged_ranges_1_based"] = original_flagged_ranges
            frame_quality_summary["written_frame_window_1_based_original"] = [int(trim_start + 1), int(trim_end_exclusive)]
            frame_quality_summary["edge_trim"] = {
                "enabled": True,
                "applied": True,
                "policy": effective_trim_policy,
                "trimmed_leading_frames": trimmed_front,
                "trimmed_trailing_frames": trimmed_back,
                "details": trim_details,
                "trimmed_ranges_1_based_original": (
                    ([[1, trimmed_front]] if trimmed_front > 0 else [])
                    + ([[int(trim_end_exclusive + 1), int(original_num_frames)]] if trimmed_back > 0 else [])
                ),
            }
            frame_quality_summary["frames_evaluated"] = int(num_frames)
            frame_quality_summary["flagged_frames_0_based"] = remaining_flagged
            frame_quality_summary["flagged_frames_1_based"] = [int(idx + 1) for idx in remaining_flagged]
            frame_quality_summary["flagged_ranges_0_based"] = [[int(start), int(end)] for start, end in remaining_ranges_0]
            frame_quality_summary["flagged_ranges_1_based"] = remaining_ranges_1
            frame_quality_summary["flagged_fraction"] = float(len(remaining_flagged) / num_frames) if num_frames else 0.0
            frame_quality_summary["metrics"] = frame_quality_summary.get("metrics", [])[trim_start:trim_end_exclusive]
            for new_idx, metric in enumerate(frame_quality_summary["metrics"]):
                metric["frame_0_based"] = int(new_idx)
                metric["frame_1_based"] = int(new_idx + 1)
        else:
            frame_quality_summary["edge_trim"] = {
                "enabled": effective_trim_policy != "off",
                "applied": False,
                "policy": effective_trim_policy,
                "reason": "no_edge_trim_needed",
                "details": trim_details,
            }

    raw_signed_sample = _sample_stack_values(
        binned_data,
        max_pixels=2_000_000,
        subtract=float(actual_pedestal),
    )
    raw_signed_intensity_summary = summarize_sample_distribution(raw_signed_sample)
    conditioned_storage_sample = np.asarray([], dtype=np.float64)
    conditioned_storage_summary: Dict[str, Any] = {"count": 0}

    try:
        (
            binned_data,
            actual_pedestal,
            xds_offset_value,
            xds_offset_details,
            storage_pedestal_report,
            background_conditioning_report,
        ) = apply_v8_conditioning_policy(
            binned_data,
            final_beam_centers_yx,
            mode=conditioning_mode,
            current_storage_pedestal=int(actual_pedestal),
            current_storage_pedestal_report=storage_pedestal_report,
            xds_offset_quantile=xds_offset_quantile,
            xds_offset_inner_fraction=xds_offset_inner_fraction,
            xds_offset_outer_fraction=xds_offset_outer_fraction,
            radial_store_pedestal=radial_store_pedestal,
            radial_pedestal_quantile=radial_pedestal_quantile,
            radial_pedestal_margin=radial_pedestal_margin,
            radial_buffer=radial_buffer,
            radial_bin_width_px=radial_bin_width_px,
            radial_quantile=radial_quantile,
            radial_profile_smooth_sigma_bins=radial_profile_smooth_sigma_bins,
            radial_mask_smooth_sigma_px=radial_mask_smooth_sigma_px,
            radial_clip_high_sigma=radial_clip_high_sigma,
            radial_clip_low_sigma=radial_clip_low_sigma,
            radial_mask_dilation_px=radial_mask_dilation_px,
            radial_center_mask_radius_px=radial_center_mask_radius_px,
            radial_fit_plane=radial_fit_plane,
            radial_plane_sample_pixels=radial_plane_sample_pixels,
        )
        common_params_for_cbf_workers_main["pedestal"] = float(actual_pedestal)
        conditioned_storage_sample = _sample_stack_values(binned_data, max_pixels=2_000_000)
        conditioned_storage_summary = summarize_sample_distribution(conditioned_storage_sample)
        try:
            reduced_background_offset_details = estimate_highres_xds_offset(
                binned_data,
                final_beam_centers_yx,
                quantile=xds_offset_quantile,
                inner_fraction=xds_offset_inner_fraction,
                outer_fraction=xds_offset_outer_fraction,
            )
            reduced_background_offset_value = int(
                reduced_background_offset_details.get("candidate_offsets", {}).get(
                    "highres_median_minus_sigma",
                    reduced_background_offset_details.get("suggested_offset", xds_offset_value),
                )
            )
        except Exception as exc:
            reduced_background_offset_details = {"error": str(exc)}
            reduced_background_offset_value = None
            logging.warning("Could not estimate v8 reduced/background-only XDS offset comment: %s", exc)
        logging.info(
            "v8 conditioning policy '%s' selected storage pedestal=%d and XDS OFFSET=%d.",
            str(conditioning_mode),
            int(actual_pedestal),
            int(xds_offset_value),
        )
    except Exception as e:
        logging.critical("v8 conditioning policy failed: %s", e, exc_info=True)
        return

    if final_diagnostic_plots: 
        logging.info("Generating final beam center diagnostic plot...")
        try:
            plt = get_plt()
            plt.figure(figsize=(12, 7)); gs = plt.GridSpec(2, 1, height_ratios=[3, 1]); ax1 = plt.subplot(gs[0])
            valid_final_idx_plot = [i for i, bc in enumerate(final_beam_centers_yx) if bc is not None]
            if valid_final_idx_plot:
                label_suffix = "(Geometric)" if skip_beam_centering else f"(n={len(valid_final_idx_plot)})"
                ax1.plot(valid_final_idx_plot, [final_beam_centers_yx[i][1] for i in valid_final_idx_plot], '.-', label=f'Final Beam X {label_suffix}', ms=4, lw=1)
                ax1.plot(valid_final_idx_plot, [final_beam_centers_yx[i][0] for i in valid_final_idx_plot], '.-', label=f'Final Beam Y {label_suffix}', ms=4, lw=1)
            if not skip_beam_centering and (perform_second_pass or first_pass_diagnostic_plots): 
                 valid_first_idx_plot = [i for i, bc in enumerate(first_pass_centers_for_plot) if bc is not None]
                 if valid_first_idx_plot:
                      ax1.plot(valid_first_idx_plot, [first_pass_centers_for_plot[i][1] for i in valid_first_idx_plot], 'x', color='gray', alpha=0.5, label='First Pass X', ms=3)
                      ax1.plot(valid_first_idx_plot, [first_pass_centers_for_plot[i][0] for i in valid_first_idx_plot], '+', color='darkgray', alpha=0.5, label='First Pass Y', ms=3)
            ax1.set_ylabel('Center (pixels, 0-based)'); ax1.set_title('Beam Center: Geometric' if skip_beam_centering else 'Beam Centers: Final (Smoothed)'); ax1.legend(fontsize='small'); ax1.grid(True); plt.setp(ax1.get_xticklabels(), visible=False); ax1.set_xlim(0, max(0, num_frames - 1))
            ax2 = plt.subplot(gs[1], sharex=ax1)
            if not skip_beam_centering and perform_second_pass and len(valid_final_idx_plot) > 1:
                 jumps_dist = np.sqrt(np.diff([final_beam_centers_yx[i][1] for i in valid_final_idx_plot])**2 + np.diff([final_beam_centers_yx[i][0] for i in valid_final_idx_plot])**2)
                 ax2.plot(np.array(valid_final_idx_plot)[1:], jumps_dist, '.-', label='Jump Dist (px)', color='purple', ms=3, lw=0.8)
                 ax2.axhline(max_beam_jump_pixels, color='red', ls='--', lw=1, label=f'Outlier Thr ({max_beam_jump_pixels} px)'); ax2.set_ylabel('Frame-to-Frame\nJump (pixels)'); ax2.legend(fontsize='small')
            else: 
                 ax2.text(0.5, 0.5, 'Jump plot N/A', ha='center', va='center', transform=ax2.transAxes, color='gray'); ax2.set_yticks([])
            ax2.grid(True); plt.xlabel('Frame Index (0-based)'); plt.tight_layout(); 
            final_plot_path = os.path.join(output_dir, "beam_centers_final.png")
            plt.savefig(final_plot_path); plt.close()
            logging.info(f"Saved final beam center plot to {final_plot_path}")
        except Exception as e: logging.error(f"Failed generating final plot: {e}")

    if save_beam_centers_file: 
        bc_file_path = os.path.join(output_dir, "beam_centers_final.txt")
        try:
            with open(bc_file_path, 'w') as f:
                f.write(f"# Beam centers for {os.path.basename(mrc_path)}\n")
                f.write("# Columns: Frame_Index (0-based), Beam_Y (pixels, 0-based), Beam_X (pixels, 0-based), Status\n")
                for i, bc in enumerate(final_beam_centers_yx):
                    status_detail = fit_details_list[i].get('status', 'Unknown') if i < len(fit_details_list) and fit_details_list[i] else "Unknown"
                    if bc is not None: 
                        f.write(f"{i:<18d}  {bc[0]:<22.4f}  {bc[1]:<22.4f}  Valid ({status_detail})\n")
                    else: 
                         if not skip_beam_centering and perform_second_pass and i < len(first_pass_centers_for_plot) and first_pass_centers_for_plot[i] is not None: 
                             status_detail += " (Rejected Outlier)"
                         f.write(f"{i:<18d}  {'None':<22s}  {'None':<22s}  Failed ({status_detail})\n") 
            logging.info(f"Saved final beam center data to {bc_file_path}")
        except Exception as e: logging.error(f"Failed to save beam center file: {e}")

    common_params_for_cbf_workers_main.update({
        'pixel_size_x_m': new_pixel_size_x_m,
        'pixel_size_y_m': new_pixel_size_y_m,
        'detector_distance_m': detector_distance_m,
        'wavelength_A': wavelength_A, 
        'angle_increment_deg': new_angle_increment_deg_binned, 
        'filename_template': filename_template, 
        'shift_interpolation_order': shift_interpolation_order,
        'detector_model': detector_model,
        'serial_number': serial_number,
        'sensor_kind': sensor_kind,
        'sensor_material': sensor_material,
        'sensor_thickness_mm': sensor_thickness_mm,
        'gain': effective_gain,
        'probe': probe,
        'goniometer_axis': goniometer_axis,
        'overload': overload_cli,
        'output_dir_mp': output_dir,
        'exposure_time_s': effective_exposure_time_s,
        'exposure_period_s': effective_exposure_period_s,
        'timestamp_iso': effective_timestamp_iso,
        'flux': 0.0,
        'transmission': 1.0,
        'polarization_fraction': 0.0,
        'detector_profile': detector_profile,
        'cbf_header_convention': 'GENERIC_MINI',
    })

    logging.info(
        "Effective output metadata: pixel_size=(%.6f, %.6f) mm, distance=%.3f mm, wavelength=%.6f A, gain=%s, probe=%s",
        new_pixel_size_x_m * 1000.0,
        new_pixel_size_y_m * 1000.0,
        detector_distance_mm,
        wavelength_A,
        f"{effective_gain:.6f}" if effective_gain is not None else "unset",
        probe,
    )
    if pixel_size_source:
        logging.info(f"Pixel size source: {pixel_size_source}")
    if effective_gain_source:
        logging.info(f"Gain source: {effective_gain_source}")
    
    frame_params_list_main = []
    for idx in range(num_frames):
        frame_metadata = output_frame_metadata[idx] if idx < len(output_frame_metadata) else {}
        frame_params_list_main.append(
            {
                'current_angle': start_angle_deg + idx * new_angle_increment_deg_binned,
                'exposure_time_s': frame_metadata.get('exposure_time_s'),
                'exposure_period_s': frame_metadata.get('exposure_period_s'),
                'timestamp_iso': frame_metadata.get('timestamp_iso'),
                'epoch_s': frame_metadata.get('epoch_s'),
            }
        )
    
    logging.info("Starting CBF writing (MP Initializer version)...")
    process_start_time = time.time()
    
    tasks_cbf_write = []
    for idx in range(num_frames):
        plot_this_frame = diagnostic_plot_specific_frames is not None and idx in diagnostic_plot_specific_frames
        tasks_cbf_write.append((idx, plot_this_frame, apply_image_shift)) 

    all_frame_timings = [] 
    processed_count = 0; skipped_count = 0; failed_count = 0

    cbf_write_workers = min(effective_workers, num_frames) if num_frames > 0 else 1

    if cbf_write_workers > 1 and num_frames > 0 :
        try:
            init_args_cbf_write = (binned_data, common_params_for_cbf_workers_main, frame_params_list_main, final_beam_centers_yx, output_dir)
            chunk_cbf = max(1, num_frames // (cbf_write_workers * 4)) if cbf_write_workers > 0 else 1
            logging.info(f"CBF Write Pool: Using chunksize: {chunk_cbf} for imap_unordered with {cbf_write_workers} workers.")

            with Pool(processes=cbf_write_workers, 
                      initializer=init_worker_cbf_write, 
                      initargs=init_args_cbf_write) as pool:
                result_iterator = pool.imap_unordered(_worker_wrapper_cbf_write, tasks_cbf_write, chunksize=chunk_cbf) # CORRECTED WRAPPER NAME
                logging.info(f"Submitted {len(tasks_cbf_write)} CBF tasks to pool ({cbf_write_workers} workers).")
                for i, result in enumerate(result_iterator):
                    if result is None: 
                        logging.error(f"CBF Task {i} (iterator) returned None")
                        failed_count +=1; continue
                    if len(result) < 6: # Check for the expected number of items including timings dict
                        logging.error(f"CBF Task {i} (iterator) returned malformed result: {result}")
                        failed_count +=1; continue
                        
                    frame_idx_res, _, _, status, _, frame_timings_dict = result 
                    
                    if frame_timings_dict: all_frame_timings.append(frame_timings_dict)
                    else: all_frame_timings.append({'total_frame_proc': -999}) # Should not happen if worker catches errors

                    if status == 'Processed': processed_count += 1
                    elif status.startswith('Skipped'): skipped_count += 1; logging.warning(f"Frame {frame_idx_res}: Skipped ({status})")
                    else: failed_count += 1; logging.error(f"Frame {frame_idx_res}: Failed ({status})")
                    if (i + 1) % (max(1, num_frames // 10)) == 0 or (i + 1) == num_frames: 
                        logging.info(f"  CBF writing progress: {i+1}/{num_frames}...")
        except Exception as e: 
            logging.critical(f"CBF writing multiprocessing pool error: {e}", exc_info=True)
            failed_count = num_frames - processed_count - skipped_count 
    elif num_frames > 0: 
        logging.info("Running CBF writing sequentially (simulating worker init)...")
        init_worker_cbf_write(binned_data, common_params_for_cbf_workers_main, frame_params_list_main, final_beam_centers_yx, output_dir)
        for i, task_args_tuple_seq in enumerate(tasks_cbf_write):
            result = _worker_wrapper_cbf_write(task_args_tuple_seq) # CORRECTED WRAPPER NAME
            if result is None: continue
            if len(result) < 6: continue
            frame_idx_res, _, _, status, _, frame_timings_dict = result
            if frame_timings_dict: all_frame_timings.append(frame_timings_dict)
            else: all_frame_timings.append({'total_frame_proc': -999})

            log_msg = f"Frame {frame_idx_res}: Status={status}"
            if status == 'Processed': 
                processed_count += 1
                if (i + 1) % (max(1, num_frames // 20)) == 0 or (i + 1) == num_frames: 
                    logging.info(f"  CBF sequential progress: {i+1}/{num_frames} processed.")
                else: 
                    logging.debug(log_msg) 
            elif status.startswith('Skipped'): 
                skipped_count += 1; logging.warning(log_msg)
            else: 
                failed_count += 1; logging.error(log_msg)

    logging.info(f"CBF generation finished in {time.time() - process_start_time:.2f} seconds.")
    
    if all_frame_timings:
        logging.info("--- Average Per-Frame CBF Processing Timings (seconds) ---")
        timing_keys_to_log = ['slicing_astype', 'shift_image', 'post_shift_proc', 
                              'cbf_header_prep', 'cbf_write', 'cbf_fixup', 'plotting', 'total_frame_proc']
        for key in timing_keys_to_log:
            valid_times_for_key = [t[key] for t in all_frame_timings if isinstance(t, dict) and key in t and t[key] != -999]
            if valid_times_for_key:
                avg_time = np.mean(valid_times_for_key)
                std_time = np.std(valid_times_for_key)
                logging.info(f"  Avg {key:<18}: {avg_time:.4f} +/- {std_time:.4f}")
            else:
                logging.info(f"  Avg {key:<18}: No data or all failed")
    
    logging.info(f"Summary: {processed_count} processed, {skipped_count} skipped, {failed_count} failed.")
    logging.info(f"Total pipeline execution time: {time.time() - start_time:.2f} seconds.")

    gain_report_path = None
    if gain_estimation_report is not None:
        try:
            gain_report_path = os.path.join(output_dir, "adaptive_gain_report.json")
            _write_text_atomic(gain_report_path, json.dumps(gain_estimation_report, indent=2, sort_keys=True) + "\n")
            logging.info(f"Wrote adaptive gain report: {gain_report_path}")
        except Exception as e:
            logging.error(f"Failed to write adaptive gain report: {e}", exc_info=True)

    helper_script_path = None
    storage_pedestal_int = int(round(float(common_params_for_cbf_workers_main.get('pedestal', 0.0))))
    if write_dials_import_helper_file:
        try:
            helper_script_path = write_dials_import_helper(
                output_dir=output_dir,
                filename_template=filename_template,
                probe=probe,
                gain=effective_gain,
                pedestal=common_params_for_cbf_workers_main.get('pedestal'),
                goniometer_axis=goniometer_axis,
            )
            logging.info(f"Wrote DIALS helper script: {helper_script_path}")
        except Exception as e:
            logging.error(f"Failed to write DIALS helper script: {e}", exc_info=True)

    xds_inp_path = None
    xds_legacy_inp_path = None
    xds_offsets_report_path = None
    xds_fragment_path = None
    xds_beam_center_source = None
    if write_xds_inp_file:
        try:
            stored_beam_center_x_1based = ((ncols - 1) / 2.0) + 1.0 if apply_image_shift else None
            stored_beam_center_y_1based = ((nrows - 1) / 2.0) + 1.0 if apply_image_shift else None
            xds_beam_center_source = "geometric_center_after_image_shift" if apply_image_shift else None
            if not apply_image_shift:
                valid_centers = [
                    (float(bc[0]), float(bc[1]))
                    for bc in final_beam_centers_yx
                    if bc is not None and np.all(np.isfinite(bc))
                ]
                if valid_centers:
                    median_center_yx = np.median(np.asarray(valid_centers, dtype=np.float64), axis=0)
                    stored_beam_center_y_1based = float(median_center_yx[0]) + 1.0
                    stored_beam_center_x_1based = float(median_center_yx[1]) + 1.0
                    xds_beam_center_source = "median_retained_beam_centers_no_shift"
            if stored_beam_center_x_1based is None or stored_beam_center_y_1based is None:
                raise RuntimeError("Unable to determine stored beam center for XDS.INP generation")

            suggested_exclude_ranges = [
                tuple(int(v) for v in item)
                for item in frame_quality_summary.get("flagged_ranges_1_based", [])
            ] if frame_quality_summary and frame_quality_summary.get("flagged_ranges_1_based") else None

            xds_inp_path = write_xds_inp(
                output_dir=output_dir,
                filename_template=filename_template,
                num_frames=num_frames,
                nrows=nrows,
                ncols=ncols,
                pixel_size_x_mm=new_pixel_size_x_m * 1000.0,
                pixel_size_y_mm=new_pixel_size_y_m * 1000.0,
                detector_distance_mm=detector_distance_mm,
                beam_center_x_px_1based=stored_beam_center_x_1based,
                beam_center_y_px_1based=stored_beam_center_y_1based,
                start_angle_deg=start_angle_deg,
                oscillation_range_deg=new_angle_increment_deg_binned,
                wavelength_A=wavelength_A,
                goniometer_axis_imgcif=goniometer_axis,
                overload_value=overload_cli,
                pedestal=storage_pedestal_int,
                xds_offset=int(xds_offset_value),
                legacy_offset=reduced_background_offset_value,
                gain=effective_gain,
                detector_model=detector_model,
                xds_offset_details=xds_offset_details,
                suggested_exclude_ranges=suggested_exclude_ranges,
                output_filename="XDS.INP",
                offset_compatibility_note="For XDS versions after 20230630",
                converter_contract=converter_contract_name,
            )
            logging.info(f"Wrote XDS.INP: {xds_inp_path}")
            xds_fragment_path = write_xds_inp_fragment(output_dir, xds_inp_path)
            xds_legacy_inp_path = write_xds_inp(
                output_dir=output_dir,
                filename_template=filename_template,
                num_frames=num_frames,
                nrows=nrows,
                ncols=ncols,
                pixel_size_x_mm=new_pixel_size_x_m * 1000.0,
                pixel_size_y_mm=new_pixel_size_y_m * 1000.0,
                detector_distance_mm=detector_distance_mm,
                beam_center_x_px_1based=stored_beam_center_x_1based,
                beam_center_y_px_1based=stored_beam_center_y_1based,
                start_angle_deg=start_angle_deg,
                oscillation_range_deg=new_angle_increment_deg_binned,
                wavelength_A=wavelength_A,
                goniometer_axis_imgcif=goniometer_axis,
                overload_value=overload_cli,
                pedestal=storage_pedestal_int,
                xds_offset=int(xds_offset_value),
                legacy_offset=reduced_background_offset_value,
                gain=effective_gain,
                detector_model=detector_model,
                xds_offset_details={"method": "storage_pedestal", "suggested_offset": int(xds_offset_value)},
                suggested_exclude_ranges=suggested_exclude_ranges,
                output_filename="XDS_20230630.INP",
                offset_compatibility_note="For XDS 20230630 and before",
                converter_contract="v7_real_offset",
            )
            logging.info(f"Wrote legacy XDS.INP: {xds_legacy_inp_path}")
            xds_offsets_report_path = write_xds_offsets_report(
                output_dir,
                storage_pedestal=storage_pedestal_int,
                current_xds_offset=int(xds_offset_value),
                legacy_xds_offset=int(xds_offset_value),
                current_xds_inp=xds_inp_path,
                legacy_xds_inp=xds_legacy_inp_path,
                xds_offset_details=xds_offset_details,
                reduced_background_offset=reduced_background_offset_value,
                reduced_background_offset_details=reduced_background_offset_details,
                storage_pedestal_report=storage_pedestal_report,
            )
            logging.info(f"Wrote XDS offset report: {xds_offsets_report_path}")
        except Exception as e:
            logging.error(f"Failed to write XDS.INP: {e}", exc_info=True)

    audit_artifacts: Dict[str, Any] = {}
    offset_confidence: Optional[Dict[str, Any]] = None
    frame_timing_audit_payload: Optional[Dict[str, Any]] = None
    if write_conversion_report_file:
        try:
            pixel_diagnostics = (background_conditioning_report or {}).get("pixel_diagnostics")
            offset_confidence_path, offset_confidence = write_offset_confidence_report(
                output_dir,
                contract_name=converter_contract_name,
                contract=converter_contract,
                storage_pedestal=storage_pedestal_int,
                xds_offset=int(xds_offset_value),
                xds_offset_details=xds_offset_details,
                storage_pedestal_report=storage_pedestal_report,
                raw_signed_summary=raw_signed_intensity_summary,
                conditioned_storage_summary=conditioned_storage_summary,
                pixel_diagnostics=pixel_diagnostics,
            )
            audit_artifacts["offset_confidence_report"] = offset_confidence_path
            frame_timing_audit_path, frame_timing_audit_payload = write_frame_timing_audit(
                output_dir,
                mrc_path=mrc_path,
                filename_template=filename_template,
                input_shape=tuple(int(v) for v in input_shape),
                output_shape=tuple(int(v) for v in new_shape),
                output_frame_origin_0_based=output_frame_origin_0_based,
                bin_z=bin_z,
                start_angle_deg=start_angle_deg,
                oscillation_range_deg=new_angle_increment_deg_binned,
                frame_metadata_sources=frame_metadata_sources,
                oscillation_validation_report={
                    **(oscillation_validation_report or {}),
                    "written_start_angle_deg": float(start_angle_deg),
                    "written_frames": int(num_frames),
                    "written_angle_increment_deg_per_output_frame": float(new_angle_increment_deg_binned),
                },
                xml_metadata=xml_metadata,
                mdoc_metadata=mdoc_metadata,
            )
            audit_artifacts["frame_timing_audit"] = frame_timing_audit_path
            audit_artifacts["background_model_report"] = write_background_model_report(
                output_dir,
                contract_name=converter_contract_name,
                background_conditioning_report=background_conditioning_report,
            )
            histogram_path = write_pixel_histogram_before_after_csv(
                output_dir,
                raw_signed_sample,
                conditioned_storage_sample,
            )
            if histogram_path:
                audit_artifacts["pixel_histogram_before_after"] = histogram_path
            audit_artifacts["xds_smoke_score"] = write_xds_smoke_score_stub(
                output_dir,
                xds_inp_path=xds_inp_path,
                xds_offset=int(xds_offset_value),
            )
            audit_artifacts["model_validation_hook"] = write_model_validation_hook_report(output_dir)
            if xds_fragment_path:
                audit_artifacts["xds_inp_fragment"] = xds_fragment_path
            audit_artifacts["conversion_manifest"] = str(Path(output_dir) / "conversion_manifest.json")
            manifest_payload = {
                "converter": {
                    "name": CONVERTER_VERSION,
                    "contract": converter_contract_name,
                    "contract_details": converter_contract,
                    "code_sha256": _current_file_sha256(),
                    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                },
                "inputs": {
                    "mrc": mrc_path,
                    "mrc_shape": list(input_shape),
                    "xml": (xml_metadata or {}).get("path"),
                    "mdoc_num_subframes": (mdoc_metadata or {}).get("numsubframes") if mdoc_metadata else None,
                },
                "outputs": {
                    "output_dir": output_dir,
                    "filename_template": filename_template,
                    "written_shape": [int(v) for v in new_shape],
                    "processed_frames": int(processed_count),
                    "failed_frames": int(failed_count),
                },
                "offset_contract": {
                    "storage_pedestal": int(storage_pedestal_int),
                    "xds_offset": int(xds_offset_value),
                    "xds_offset_method": (xds_offset_details or {}).get("method"),
                    "offset_confidence_status": (offset_confidence or {}).get("status"),
                    "storage_and_offset_are_distinct_quantities": True,
                },
                "trust_status": {
                    "pre_xds_offset_confidence": (offset_confidence or {}).get("status"),
                    "xds_response_sweep": "not_run",
                    "model_vs_data_validation": "not_run",
                    "final_trust": "pending_xds_response_sweep_and_model_validation",
                },
                "artifacts": audit_artifacts,
                "audit_artifacts": audit_artifacts,
                "frame_timing_audit": frame_timing_audit_payload,
            }
            write_conversion_manifest(output_dir, manifest_payload)
        except Exception as e:
            logging.error("Failed to write self-certification audit reports: %s", e, exc_info=True)

    if write_conversion_report_file:
        try:
            report_path = write_conversion_report(
                output_dir,
                {
                    "input_mrc": mrc_path,
                    "output_dir": output_dir,
                    "output_template": filename_template,
                    "converter": {
                        "name": CONVERTER_VERSION,
                        "contract": converter_contract_name,
                        "contract_details": converter_contract,
                        "code_sha256": _current_file_sha256(),
                    },
                    "input_shape": list(input_shape),
                    "output_shape": [int(v) for v in new_shape],
                    "frames": {
                        "processed": processed_count,
                        "skipped": skipped_count,
                        "failed": failed_count,
                    },
                    "binning": {
                        "hardware_x": hardware_bin_x,
                        "hardware_y": hardware_bin_y,
                        "software_x": bin_x,
                        "software_y": bin_y,
                        "software_z": bin_z,
                    },
                    "metadata": {
                        "pixel_size_mm_unbinned": pixel_size_mm,
                        "pixel_size_mm_effective_x": new_pixel_size_x_m * 1000.0,
                        "pixel_size_mm_effective_y": new_pixel_size_y_m * 1000.0,
                        "pixel_size_source": pixel_size_source,
                        "distance_mm": detector_distance_mm,
                        "wavelength_A": wavelength_A,
                        "start_angle_deg": start_angle_deg,
                        "angle_increment_deg_per_output_frame": new_angle_increment_deg_binned,
                        "oscillation_wedge_validation": {
                            **(oscillation_validation_report or {}),
                            "written_start_angle_deg": float(start_angle_deg),
                            "written_frames": int(num_frames),
                            "written_angle_increment_deg_per_output_frame": float(new_angle_increment_deg_binned),
                        },
                        "exposure_time_s_per_output_frame": effective_exposure_time_s,
                        "exposure_period_s_per_output_frame": effective_exposure_period_s,
                        "exposure_time_source": frame_metadata_sources.get("exposure_time_source"),
                        "exposure_period_source": frame_metadata_sources.get("exposure_period_source"),
                        "timestamp_source": frame_metadata_sources.get("timestamp_source"),
                        "gain_mode": gain_mode,
                        "gain": effective_gain,
                        "gain_source": effective_gain_source,
                        "metadata_gain_prior": metadata_gain_value,
                        "metadata_gain_prior_source": metadata_gain_value_source,
                        "gain_prior_candidates": gain_prior_candidates,
                        "xml_metadata": {
                            "path": (xml_metadata or {}).get("path"),
                            "commercial_name": (xml_metadata or {}).get("commercial_name"),
                            "camera_name": (xml_metadata or {}).get("camera_name"),
                            "serial_number": (xml_metadata or {}).get("serial_number"),
                            "format": (xml_metadata or {}).get("format"),
                            "expected_number_of_fractions": (xml_metadata or {}).get("expected_number_of_fractions"),
                            "recorded_number_of_fractions": (xml_metadata or {}).get("recorded_number_of_fractions"),
                            "fraction_count": (xml_metadata or {}).get("fraction_count"),
                            "fraction_exposure_time_s": (xml_metadata or {}).get("fraction_exposure_time_s"),
                            "fraction_exposure_period_s": (xml_metadata or {}).get("fraction_exposure_period_s"),
                            "fraction_timing_summary": (xml_metadata or {}).get("fraction_timing_summary"),
                            "camera_counts_per_electron": (xml_metadata or {}).get("camera_counts_per_electron"),
                            "saved_counts_per_electron": (xml_metadata or {}).get("saved_counts_per_electron"),
                        },
                        "pedestal": common_params_for_cbf_workers_main.get('pedestal'),
                        "storage_pedestal_details": storage_pedestal_report,
                        "xds_offset": int(xds_offset_value),
                        "xds_offset_details": xds_offset_details,
                        "reduced_background_offset": reduced_background_offset_value,
                        "reduced_background_offset_details": reduced_background_offset_details,
                        "background_conditioning": background_conditioning_report,
                        "xds_beam_center_source": xds_beam_center_source,
                        "probe": probe,
                        "goniometer_axis_imgcif": goniometer_axis,
                        "detector_model": detector_model,
                        "serial_number": serial_number,
                        "sensor_kind": sensor_kind,
                        "sensor_material": sensor_material,
                        "sensor_thickness_mm": sensor_thickness_mm,
                        "detector_profile": detector_profile,
                        "detector_profile_reason": detector_profile_reason,
                        "detector_corrections": detector_corrections_report,
                        "trim_policy": effective_trim_policy,
                        "frame_quality_screen": frame_quality_summary,
                        "raw_signed_intensity_summary": raw_signed_intensity_summary,
                        "conditioned_storage_intensity_summary": conditioned_storage_summary,
                        "offset_confidence": offset_confidence,
                    },
                    "compatibility": {
                        "header_convention": "GENERIC_MINI",
                        "stock_dials_import": "generated helper imports from absolute CBF paths and writes outputs in the caller's work directory; probe/gain/pedestal still need helper options or a custom dxtbx format",
                        "dials_helper_script": helper_script_path,
                        "xds_input": xds_inp_path,
                        "xds_legacy_input": xds_legacy_inp_path,
                        "xds_inp_fragment": xds_fragment_path,
                        "xds_offsets_report": xds_offsets_report_path,
                        "self_certification_artifacts": audit_artifacts,
                    },
                    "adaptive_gain_report": gain_report_path,
                },
            )
            logging.info(f"Wrote conversion report: {report_path}")
        except Exception as e:
            logging.error(f"Failed to write conversion report: {e}", exc_info=True)

    if failed_count > 0: logging.warning(f"{failed_count} frames failed.")
    if skipped_count > 0: logging.warning(f"{skipped_count} frames skipped.")
    if processed_count == 0 and num_frames > 0: logging.error("No frames were successfully processed.")
    elif processed_count > 0: logging.info("Pipeline completed.")

# ==============================================================================
# --- Command Line Argument Parsing and Main Execution ---
# ==============================================================================
def main():
    if sys.platform == "darwin":
        current_method = mp.get_start_method(allow_none=True)
        if current_method is None or current_method != 'fork': 
            try:
                mp.set_start_method('fork', force=True)
            except RuntimeError as e:
                print(f"WARNING: Could not force 'fork' start method on macOS (current/default: {current_method}). Error: {e}. Using current/default.", file=sys.stderr)
            
    parser = argparse.ArgumentParser(
        description="Convert MRC image stacks to CBF format with optional beam centering, "
                    "smoothing, and metadata extraction from .mdoc files. Uses fabio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("mrc_path", help="Path to the input MRC file.")
    parser.add_argument("output_dir", help="Path to the directory for output CBF files and logs.")
    parser.add_argument("--mdoc", dest="mdoc_path", default=None, help="Optional .mdoc path.")
    parser.add_argument("--xml", dest="xml_path", default=None, help="Optional acquisition XML path.")
    parser.add_argument("--pixel-size-mm", type=float, default=None, help="Pixel size (mm, unbinned).")
    parser.add_argument("--detector-distance-mm", type=float, default=None, help="Detector distance (mm).")
    parser.add_argument("--wavelength-A", type=float, default=None, help="Wavelength (Å).")
    parser.add_argument("--start-angle-deg", type=float, default=None, help="Start tilt angle (deg).")
    parser.add_argument("--angle-increment-deg", type=float, default=None, help="Angle increment per original frame (deg).")
    parser.add_argument("--bin-x", type=int, default=None, help="Software binning X. If unset, v7 chooses the smallest bin that keeps the output width at or below --target-size-px.")
    parser.add_argument("--bin-y", type=int, default=None, help="Software binning Y. If unset, v7 chooses the smallest bin that keeps the output height at or below --target-size-px.")
    parser.add_argument("--bin-z", type=int, default=1, help="Binning Z (frames).")
    parser.add_argument("--target-size-px", type=int, default=1024, help="Target maximum output dimension for automatic software binning. Set <=0 to disable auto target binning.")
    parser.add_argument("--hardware-bin-x", type=int, default=None, help="Acquisition/hardware binning factor in X. Used for effective pixel size only.")
    parser.add_argument("--hardware-bin-y", type=int, default=None, help="Acquisition/hardware binning factor in Y. Used for effective pixel size only.")
    parser.add_argument("--detector-profile", default="auto", choices=["auto", "none", "ceta_falcon"], help="Detector profile for metadata defaults.")
    parser.add_argument("--gain", type=float, default=None, help="Detector gain to record in headers and helper outputs.")
    parser.add_argument("--gain-mode", choices=["auto", "metadata", "adaptive"], default="auto", help="How to resolve detector gain when --gain is not provided. 'auto' prefers XML-derived saved-value gain and otherwise falls back to adaptive estimation.")
    parser.add_argument("--gain-estimation-frames", type=int, default=180, help="Number of binned output frames sampled across the scan for adaptive gain estimation.")
    parser.add_argument("--gain-kernels", default="11,15,21,27,35", help="Comma-separated local kernel sizes for adaptive gain estimation.")
    parser.add_argument("--gain-grids", default="24x24,32x32,48x48,64x64,80x80", help="Comma-separated grid sizes (GXxGY) for adaptive gain estimation.")
    parser.add_argument("--dark-map", default=None, help="Optional .npy residual dark/common-mode map applied before pedestal and shifting.")
    parser.add_argument("--gain-map", default=None, help="Optional .npy per-pixel gain/flat-field map applied before pedestal and shifting.")
    parser.add_argument("--gain-map-kind", choices=["auto", "relative", "absolute"], default="auto", help="Interpretation of --gain-map values.")
    parser.add_argument("--bad-pixel-mask", default=None, help="Optional .npy bad-pixel mask. Nonzero pixels are repaired before beam centering and writing.")
    parser.add_argument("--bad-pixel-fill", choices=["median3", "none"], default="median3", help="How to repair pixels flagged by --bad-pixel-mask.")
    parser.add_argument("--probe", default="electron", choices=["x-ray", "electron", "neutron"], help="Beam probe for helper outputs and the optional custom DIALS format.")
    parser.add_argument("--goniometer-axis", default="0.999263,-0.0383878,0", help="Rotation axis in DIALS/imgCIF coordinates for generated import helpers.")
    parser.add_argument("--detector-model", default=None, help="Detector model string written into the CBF header.")
    parser.add_argument("--serial-number", default="00-0000", help="Detector serial number written into the CBF header.")
    parser.add_argument("--sensor-kind", default=None, choices=["ccd", "pad"], help="Sensor kind to advertise in the CBF header.")
    parser.add_argument("--sensor-material", default=None, choices=["Si", "CdTe"], help="Sensor material for PAD-style headers.")
    parser.add_argument("--sensor-thickness-mm", type=float, default=None, help="Sensor thickness in mm for PAD-style headers.")
    parser.add_argument("--exposure-time-s", type=float, default=None, help="Exposure time per output frame.")
    parser.add_argument("--exposure-period-s", type=float, default=None, help="Exposure period per output frame.")
    parser.add_argument("--wedge-validation", choices=["warn", "error", "off"], default="warn", help="Validate the constant per-frame oscillation width against XML timing when present.")
    parser.add_argument("--wedge-tolerance-fraction", type=float, default=1.0e-3, help="Relative tolerance for XML/MDOC constant-wedge validation.")
    parser.add_argument("--wedge-tolerance-seconds", type=float, default=1.0e-4, help="Absolute exposure/period tolerance in seconds for constant-wedge validation.")
    parser.add_argument("--pedestal", type=int, default=None, help="Fixed initial storage pedestal. v8 conditioning modes may replace this before CBF writing.")
    parser.add_argument("--no-auto-pedestal", action="store_false", dest="auto_pedestal", help="Disable auto pedestal.")
    parser.add_argument("--auto-pedestal-negative-quantile", type=float, default=0.0, help="Negative-tail quantile used for auto pedestal. Default 0.0 uses the minimum negative ADU; set above zero only to intentionally ignore an extreme dead/hot-pixel tail.")
    parser.add_argument("--xds-offset-mode", choices=["storage_pedestal", "outer_shell_mode", "outer_shell_mode_minus_sigma", "outer_shell_median", "pedestal"], default="storage_pedestal", help="How to choose active OFFSET in generated XDS inputs. Default uses the actual storage pedestal; outer-shell modes are retained for explicit comparison only.")
    conditioning_group = parser.add_argument_group("v8 Background/Offset Conditioning")
    conditioning_group.add_argument("--conditioning-mode", choices=["radial_reduced", "v7", "quantile_offset", "radial_offset0", "radial_buffer"], default="radial_reduced", help="v8 XDS conditioning policy. Default radial_reduced subtracts a smooth radial/plane background and uses the conditioned high-resolution median-minus-sigma as current-XDS OFFSET.")
    conditioning_group.add_argument("--xds-offset-quantile", type=float, default=0.001, help="High-resolution masked quantile used by quantile_offset and reported for radial modes.")
    conditioning_group.add_argument("--xds-offset-inner-fraction", type=float, default=0.82, help="Inner radius fraction for high-resolution offset/diagnostic pixels.")
    conditioning_group.add_argument("--xds-offset-outer-fraction", type=float, default=0.98, help="Outer radius fraction for high-resolution offset/diagnostic pixels.")
    conditioning_group.add_argument("--radial-store-pedestal", type=int, default=100, help="Fixed storage pedestal after radial conditioning. Default 100 reproduces the validated Tyrosine v8 radial-reduced runs; set to none only by calling the Python API with None.")
    conditioning_group.add_argument("--radial-pedestal-quantile", type=float, default=1.0e-5, help="Residual low quantile used for automatic radial storage pedestal.")
    conditioning_group.add_argument("--radial-pedestal-margin", type=float, default=20.0, help="ADU safety margin above the radial residual low quantile.")
    conditioning_group.add_argument("--radial-buffer", type=int, default=40, help="Positive post-XDS-offset background buffer for radial_buffer mode.")
    conditioning_group.add_argument("--radial-bin-width-px", type=float, default=2.0, help="Radial background bin width in output pixels.")
    conditioning_group.add_argument("--radial-quantile", type=float, default=0.50, help="Per-radius quantile used for the radial background estimate.")
    conditioning_group.add_argument("--radial-profile-smooth-sigma-bins", type=float, default=4.0, help="Gaussian smoothing sigma for the 1D radial profile.")
    conditioning_group.add_argument("--radial-mask-smooth-sigma-px", type=float, default=18.0, help="Large-scale smooth image used to mask Bragg/outlier residuals.")
    conditioning_group.add_argument("--radial-clip-high-sigma", type=float, default=5.0, help="High-side sigma clip for rejecting Bragg pixels from the radial background model.")
    conditioning_group.add_argument("--radial-clip-low-sigma", type=float, default=8.0, help="Low-side sigma clip for rejecting cold/dead artifacts from the radial background model.")
    conditioning_group.add_argument("--radial-mask-dilation-px", type=int, default=2, help="Dilation iterations for rejected radial-model pixels.")
    conditioning_group.add_argument("--radial-center-mask-radius-px", type=float, default=28.0, help="Central direct-beam/beamstop radius excluded from radial background fitting.")
    conditioning_group.add_argument("--no-radial-fit-plane", action="store_false", dest="radial_fit_plane", help="Disable residual tilted-plane fit after the radial profile.")
    conditioning_group.add_argument("--radial-plane-sample-pixels", type=int, default=120000, help="Maximum valid pixels sampled for the residual plane fit per frame.")
    parser.add_argument("--trim-policy", choices=["off", "quality", "aggressive"], default="aggressive", help="Frame trimming policy. 'quality' trims contiguous flagged edges; 'aggressive' also requires strong/full leading and trailing retained frames.")
    parser.add_argument("--trim-edge-bad-frames", action="store_const", const="quality", dest="trim_policy", help="Compatibility alias for --trim-policy quality.")
    parser.add_argument("--no-trim-edge-bad-frames", action="store_const", const="off", dest="trim_policy", help="Compatibility alias for --trim-policy off.")
    parser.add_argument("--aggressive-trim-min-strong-frames", type=int, default=5, help="Minimum consecutive strong frames required at each retained edge in aggressive trim mode.")
    parser.add_argument("--aggressive-trim-signal-ratio", type=float, default=0.65, help="Minimum beam-signal ratio relative to the central stack median for aggressive edge retention.")
    parser.add_argument("--aggressive-trim-std-ratio", type=float, default=0.55, help="Minimum frame-std ratio relative to the central stack median for aggressive edge retention.")
    parser.add_argument("--aggressive-trim-peak-ratio", type=float, default=0.45, help="Minimum beam-peak ratio relative to the central stack median for aggressive edge retention.")
    parser.add_argument("--skip-beam-centering", action="store_true", help="Skip beam centering, use geometric.")
    parser.add_argument("--no-image-shift", action="store_false", dest="apply_image_shift", help="Disable image shifting.")
    parser.add_argument("--shift-order", type=int, default=1, dest="shift_interpolation_order", choices=[0,1,2,3,4,5], help="Shift interpolation order.")
    parser.add_argument("--overload", type=int, default=1000000, dest="overload_value", 
                        help="Detector saturation value (Count_cutoff for CBF header).")
    bcenter_group = parser.add_argument_group('Beam Centering Tuning'); smooth_group = parser.add_argument_group('Beam Center Smoothing Tuning')
    bcenter_group.add_argument("--roi-size", type=int, default=128, dest="beam_center_roi_size")
    bcenter_group.add_argument("--blur-sigma", type=float, default=5.0, dest="beam_center_sigma_blur")
    bcenter_group.add_argument("--max-dev", type=float, default=80.0, dest="beam_center_max_initial_deviation")
    bcenter_group.add_argument("--no-fit-bounds", action="store_false", dest="beam_center_fit_bounds")
    bcenter_group.add_argument("--beam-center-method", choices=["gaussian", "blurred_peak", "robust"], default="robust", dest="beam_center_method")
    smooth_group.add_argument("--no-smoothing", action="store_false", dest="perform_second_pass")
    smooth_group.add_argument("--max-jump", type=float, default=2.0, dest="max_beam_jump_pixels")
    smooth_group.add_argument("--smooth-window", type=int, default=11, dest="smoothing_window_length")
    smooth_group.add_argument("--smooth-order", type=int, default=2, dest="smoothing_polyorder")
    smooth_group.add_argument("--smooth-fallback", default='previous', choices=['previous', 'interpolate', 'global_median'], dest="smoothing_fallback")
    parser.add_argument("-n", "--num-workers", type=int, default=None, help="Number of parallel workers.")
    parser.add_argument("--limit-frames", type=int, default=None, help="Process only first N frames.")
    parser.add_argument("--filename-template", default="image_{:05d}.cbf", help="Output CBF filename template.") 
    diag_group = parser.add_argument_group('Diagnostic Options')
    diag_group.add_argument("--plot-first-pass", action="store_true", dest="first_pass_diagnostic_plots")
    diag_group.add_argument("--no-final-plot", action="store_false", dest="final_diagnostic_plots")
    diag_group.add_argument("--plot-frames", type=str, default=None, dest="diagnostic_plot_specific_frames_str")
    diag_group.add_argument("--no-save-centers", action="store_false", dest="save_beam_centers_file")
    diag_group.add_argument("--no-dials-helper", action="store_false", dest="write_dials_import_helper_file")
    diag_group.add_argument("--no-xds-inp", action="store_false", dest="write_xds_inp_file")
    diag_group.add_argument("--no-report", action="store_false", dest="write_conversion_report_file")
    diag_group.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Console log level.")
    diag_group.add_argument("--log-file-level", default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="File log level.")
    args = parser.parse_args()

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        log_file = os.path.join(args.output_dir, f"mrc2cbf_mp_init_parBeam_{time.strftime('%Y%m%d_%H%M%S')}.log") 
        setup_logging(log_file, console_level=getattr(logging, args.log_level.upper()), file_level=getattr(logging, args.log_file_level.upper()))
    except Exception as e: 
        print(f"CRITICAL ERROR: Failed to set up logging: {e}", file=sys.stderr) 
        sys.exit(1) 

    logging.info(f"Using multiprocessing start method: {mp.get_start_method(allow_none=True)}")
    logging.info("Command line arguments parsed (CBF with MP Initializer & Parallel Beamfind).") 
    logging.debug(f"Arguments: {vars(args)}")

    gain_kernels = parse_int_csv(args.gain_kernels)
    if not gain_kernels:
        logging.critical("No valid values parsed from --gain-kernels")
        sys.exit(1)
    gain_grids = parse_grid_specs(args.gain_grids)
    if not gain_grids:
        logging.critical("No valid values parsed from --gain-grids")
        sys.exit(1)

    mdoc_data = {}
    xml_metadata: Dict[str, Any] = {}
    required_params = ['pixel_size_mm', 'detector_distance_mm', 'wavelength_A',
                       'start_angle_deg', 'angle_increment_deg']

    mdoc_derived_det_dist = None; mdoc_derived_wavelength = None
    mdoc_derived_start_angle = None; mdoc_derived_angle_increment = None

    mrc_path_obj = Path(args.mrc_path)
    resolved_mdoc_path = resolve_sidecar_path(
        mrc_path_obj,
        args.mdoc_path,
        [Path(str(mrc_path_obj) + ".mdoc"), mrc_path_obj.with_suffix(".mdoc"), mrc_path_obj.with_suffix(".MDOC")],
        "mdoc",
    )
    resolved_xml_path = resolve_sidecar_path(
        mrc_path_obj,
        args.xml_path,
        [mrc_path_obj.with_suffix(".xml"), Path(str(mrc_path_obj) + ".xml"), mrc_path_obj.with_suffix(".XML")],
        "xml",
    )

    if resolved_mdoc_path:
        logging.info(f"Parsing mdoc file: {resolved_mdoc_path}"); mdoc_data = parse_mdoc(resolved_mdoc_path)
        if not mdoc_data: logging.warning("Mdoc parsing failed or file was empty.")
        else:
            try:
                if 'voltage' in mdoc_data:
                    mdoc_derived_wavelength = calculate_wavelength_A(mdoc_data['voltage'])
                    logging.info(f"Mdoc Voltage {mdoc_data['voltage']} kV -> Wavelength {mdoc_derived_wavelength:.5f} Å.")
                if 'cameralength' in mdoc_data: mdoc_derived_det_dist = mdoc_data['cameralength']; logging.info(f"Mdoc CameraLength: {mdoc_derived_det_dist:.2f} mm.")
                if 'tiltangle' in mdoc_data: mdoc_derived_start_angle = mdoc_data['tiltangle']; logging.info(f"Mdoc TiltAngle (FrameSet 0): {mdoc_derived_start_angle:.4f} deg.")
                angle_incr_calculated = False 
                if 'degreespersecond' in mdoc_data and 'exposuretime' in mdoc_data and 'numsubframes' in mdoc_data and mdoc_data['numsubframes'] > 0:
                    deg_per_sec = mdoc_data['degreespersecond']; exp_time_sec = mdoc_data['exposuretime']; num_frames_orig = mdoc_data['numsubframes']
                    angle_incr = deg_per_sec * (exp_time_sec / num_frames_orig); mdoc_derived_angle_increment = angle_incr
                    logging.info(f"Mdoc DegreesPerSecond -> Angle Increment {angle_incr:.5f} deg/frame (orig)."); angle_incr_calculated = True
                elif not angle_incr_calculated and 'rotationrate' in mdoc_data and 'exposuretime' in mdoc_data and 'numsubframes' in mdoc_data and mdoc_data['numsubframes'] > 0:
                    rad_per_sec = mdoc_data['rotationrate']; exp_time_sec = mdoc_data['exposuretime']; num_frames_orig = mdoc_data['numsubframes']
                    angle_incr = math.degrees(rad_per_sec) * (exp_time_sec / num_frames_orig); mdoc_derived_angle_increment = angle_incr
                    logging.info(f"Mdoc RotationRate -> Angle Increment {angle_incr:.5f} deg/frame (orig).")
            except Exception as e: logging.error(f"Unexpected error processing mdoc data: {e}", exc_info=True)

    if resolved_xml_path:
        logging.info(f"Parsing acquisition XML: {resolved_xml_path}")
        xml_metadata = parse_acquisition_xml(resolved_xml_path)
        if not xml_metadata:
            logging.warning("XML parsing failed or file was empty.")
        else:
            logging.info(
                "XML metadata: detector=%s camera=%s serial=%s fractions=%s saved_gain=%s",
                xml_metadata.get("commercial_name"),
                xml_metadata.get("camera_name"),
                xml_metadata.get("serial_number"),
                xml_metadata.get("fraction_count"),
                f"{xml_metadata.get('saved_counts_per_electron'):.6f}" if _is_finite_positive(xml_metadata.get("saved_counts_per_electron")) else "unset",
            )

    metadata_input_shape = None
    if args.hardware_bin_x is None or args.hardware_bin_y is None:
        try:
            with mrcfile.open(args.mrc_path, permissive=True) as mrc:
                metadata_input_shape = tuple(int(v) for v in mrc.data.shape)
                logging.info(f"Input shape for metadata inference: {metadata_input_shape}")
        except Exception as e:
            logging.warning(f"Could not inspect MRC shape for metadata inference: {e}")

    detector_profile, detector_profile_reason = infer_detector_profile(args.detector_profile, mdoc_data, xml_metadata)
    hardware_bin_default_x, hardware_bin_default_y, hardware_bin_reason = infer_hardware_binning(
        mdoc_data,
        metadata_input_shape,
        detector_profile,
    )
    hardware_bin_x = args.hardware_bin_x if args.hardware_bin_x is not None else hardware_bin_default_x
    hardware_bin_y = args.hardware_bin_y if args.hardware_bin_y is not None else hardware_bin_default_y
    input_height = int(metadata_input_shape[-2]) if metadata_input_shape is not None else None
    input_width = int(metadata_input_shape[-1]) if metadata_input_shape is not None else None
    software_bin_x = (
        args.bin_x
        if args.bin_x is not None
        else auto_software_bin_for_target(input_width, target_size_px=args.target_size_px)
    )
    software_bin_y = (
        args.bin_y
        if args.bin_y is not None
        else auto_software_bin_for_target(input_height, target_size_px=args.target_size_px)
    )

    logging.info(f"Hardware binning factors: x={hardware_bin_x}, y={hardware_bin_y} ({hardware_bin_reason})")
    logging.info(
        "Software binning factors: x=%s, y=%s, z=%s (%s)",
        software_bin_x,
        software_bin_y,
        args.bin_z,
        "CLI" if args.bin_x is not None or args.bin_y is not None else f"auto target <= {args.target_size_px}px",
    )

    resolved_pixel_size_mm, pixel_size_source = resolve_unbinned_pixel_size_mm(
        args.pixel_size_mm,
        mdoc_data,
        detector_profile,
        hardware_bin_x,
        hardware_bin_y,
    )
    if resolved_pixel_size_mm is not None:
        logging.info(f"Resolved unbinned pixel size: {resolved_pixel_size_mm:.6f} mm ({pixel_size_source})")

    if metadata_input_shape is not None:
        input_frames = int(metadata_input_shape[0])
        mdoc_num_subframes = mdoc_data.get("numsubframes")
        if _is_finite_positive(mdoc_num_subframes) and int(round(float(mdoc_num_subframes))) != input_frames:
            logging.warning(
                "MDOC NumSubFrames (%s) does not match MRC frame count (%d)",
                mdoc_num_subframes,
                input_frames,
            )
        xml_expected = xml_metadata.get("expected_number_of_fractions")
        if _is_finite_positive(xml_expected) and int(round(float(xml_expected))) != input_frames:
            logging.warning(
                "XML ExpectedNumberOfFractions (%s) does not match MRC frame count (%d)",
                xml_expected,
                input_frames,
            )
        xml_fraction_count = xml_metadata.get("fraction_count")
        if _is_finite_positive(xml_fraction_count) and int(round(float(xml_fraction_count))) != input_frames:
            logging.warning(
                "XML parsed fraction count (%s) does not match MRC frame count (%d)",
                xml_fraction_count,
                input_frames,
            )

    metadata_gain_value, metadata_gain_source = resolve_gain_value(None, mdoc_data, xml_metadata, detector_profile)
    if metadata_gain_value is not None and metadata_gain_source:
        logging.info(f"Resolved metadata/profile gain prior: {metadata_gain_value:.6f} ({metadata_gain_source})")

    cli_gain_value = float(args.gain) if _is_finite_positive(args.gain) else None
    cli_gain_source = "CLI" if cli_gain_value is not None else None
    if args.gain is not None and cli_gain_value is None:
        logging.warning(f"Ignoring non-positive --gain value: {args.gain}")

    detector_model = args.detector_model or xml_metadata.get("commercial_name") or (
        detector_profile.detector_model if detector_profile else "GENERIC_MICROED"
    )
    serial_number = args.serial_number
    if serial_number == "00-0000" and xml_metadata.get("serial_number"):
        serial_number = str(xml_metadata["serial_number"])
    sensor_kind = args.sensor_kind or (detector_profile.sensor_kind if detector_profile else "ccd")
    sensor_material = args.sensor_material or (detector_profile.sensor_material if detector_profile else None)
    sensor_thickness_mm = (
        args.sensor_thickness_mm
        if args.sensor_thickness_mm is not None
        else (detector_profile.sensor_thickness_mm if detector_profile else None)
    )
    mdoc_frame_exposure_s = None
    if _is_finite_positive(mdoc_data.get("exposuretime")):
        if _is_finite_positive(mdoc_data.get("numsubframes")):
            mdoc_frame_exposure_s = float(mdoc_data["exposuretime"]) / float(mdoc_data["numsubframes"])
        else:
            mdoc_frame_exposure_s = float(mdoc_data["exposuretime"])
    exposure_time_s = (
        args.exposure_time_s
        if args.exposure_time_s is not None
        else (
            float(xml_metadata["fraction_exposure_time_s"])
            if _is_finite_positive(xml_metadata.get("fraction_exposure_time_s"))
            else mdoc_frame_exposure_s
        )
    )
    exposure_period_s = (
        args.exposure_period_s
        if args.exposure_period_s is not None
        else (
            float(xml_metadata["fraction_exposure_period_s"])
            if _is_finite_positive(xml_metadata.get("fraction_exposure_period_s"))
            else exposure_time_s
        )
    )
    if _is_finite_positive(mdoc_data.get("exposuretime")) and _is_finite_positive(mdoc_data.get("numsubframes")):
        logging.info(
            "MDOC ExposureTime %.7f s interpreted as total-series exposure; per-frame exposure %.7f s from NumSubFrames=%d",
            float(mdoc_data["exposuretime"]),
            float(mdoc_frame_exposure_s),
            int(round(float(mdoc_data["numsubframes"]))),
        )

    resolved_angle_increment_deg = args.angle_increment_deg if args.angle_increment_deg is not None else mdoc_derived_angle_increment
    input_frame_count_for_validation = int(metadata_input_shape[0]) if metadata_input_shape is not None else None
    if args.wedge_validation == "off":
        oscillation_validation_report = {
            "status": "disabled",
            "constant_wedge_required": True,
            "ok_for_constant_wedge": None,
            "warnings": ["Constant-wedge validation was disabled by --wedge-validation off."],
        }
    else:
        oscillation_validation_report = validate_constant_wedge_from_metadata(
            angle_increment_deg_per_input_frame=resolved_angle_increment_deg,
            bin_z=args.bin_z,
            mdoc_data=mdoc_data,
            xml_metadata=xml_metadata,
            input_frame_count=input_frame_count_for_validation,
            tolerance_fraction=args.wedge_tolerance_fraction,
            tolerance_seconds=args.wedge_tolerance_seconds,
        )
        wedge_warnings = list(oscillation_validation_report.get("warnings") or [])
        if oscillation_validation_report.get("status") == "ok":
            logging.info("Constant-wedge validation: OK.")
        elif args.wedge_validation == "error":
            for message in wedge_warnings:
                logging.critical("Constant-wedge validation failed: %s", message)
            sys.exit(1)
        else:
            for message in wedge_warnings:
                logging.warning("Constant-wedge validation: %s", message)

    final_pipeline_args = {
        'mrc_path': args.mrc_path, 'output_dir': args.output_dir,
        'pixel_size_mm': resolved_pixel_size_mm,
        'detector_distance_mm': args.detector_distance_mm if args.detector_distance_mm is not None else mdoc_derived_det_dist,
        'wavelength_A': args.wavelength_A if args.wavelength_A is not None else mdoc_derived_wavelength,
        'start_angle_deg': args.start_angle_deg if args.start_angle_deg is not None else mdoc_derived_start_angle,
        'angle_increment_deg': resolved_angle_increment_deg,
        'overload_cli': args.overload_value, 
        'bin_x': software_bin_x, 'bin_y': software_bin_y, 'bin_z': args.bin_z,
        'hardware_bin_x': hardware_bin_x, 'hardware_bin_y': hardware_bin_y,
        'gain': cli_gain_value, 'probe': args.probe,
        'goniometer_axis': args.goniometer_axis,
        'gain_mode': args.gain_mode,
        'gain_estimation_frames': args.gain_estimation_frames,
        'gain_kernels': gain_kernels,
        'gain_grids': gain_grids,
        'dark_map_path': args.dark_map,
        'gain_map_path': args.gain_map,
        'gain_map_kind': args.gain_map_kind,
        'bad_pixel_mask_path': args.bad_pixel_mask,
        'bad_pixel_fill': args.bad_pixel_fill,
        'metadata_gain': metadata_gain_value,
        'metadata_gain_source': metadata_gain_source,
        'detector_model': detector_model, 'serial_number': serial_number,
        'sensor_kind': sensor_kind, 'sensor_material': sensor_material,
        'sensor_thickness_mm': sensor_thickness_mm,
        'detector_profile': detector_profile.name if detector_profile else None,
        'detector_profile_reason': detector_profile_reason,
        'pixel_size_source': pixel_size_source,
        'gain_source': cli_gain_source,
        'xml_metadata': xml_metadata,
        'mdoc_metadata': mdoc_data,
        'oscillation_validation_report': oscillation_validation_report,
        'exposure_time_s': exposure_time_s,
        'exposure_period_s': exposure_period_s,
        'conditioning_mode': args.conditioning_mode,
        'xds_offset_quantile': args.xds_offset_quantile,
        'xds_offset_inner_fraction': args.xds_offset_inner_fraction,
        'xds_offset_outer_fraction': args.xds_offset_outer_fraction,
        'radial_store_pedestal': args.radial_store_pedestal,
        'radial_pedestal_quantile': args.radial_pedestal_quantile,
        'radial_pedestal_margin': args.radial_pedestal_margin,
        'radial_buffer': args.radial_buffer,
        'radial_bin_width_px': args.radial_bin_width_px,
        'radial_quantile': args.radial_quantile,
        'radial_profile_smooth_sigma_bins': args.radial_profile_smooth_sigma_bins,
        'radial_mask_smooth_sigma_px': args.radial_mask_smooth_sigma_px,
        'radial_clip_high_sigma': args.radial_clip_high_sigma,
        'radial_clip_low_sigma': args.radial_clip_low_sigma,
        'radial_mask_dilation_px': args.radial_mask_dilation_px,
        'radial_center_mask_radius_px': args.radial_center_mask_radius_px,
        'radial_fit_plane': args.radial_fit_plane,
        'radial_plane_sample_pixels': args.radial_plane_sample_pixels,
        'pedestal': args.pedestal, 'auto_pedestal': args.auto_pedestal,
        'auto_pedestal_negative_quantile': args.auto_pedestal_negative_quantile,
        'xds_offset_mode': args.xds_offset_mode,
        'trim_edge_bad_frames': args.trim_policy != "off",
        'trim_policy': args.trim_policy,
        'aggressive_trim_min_strong_frames': args.aggressive_trim_min_strong_frames,
        'aggressive_trim_signal_ratio': args.aggressive_trim_signal_ratio,
        'aggressive_trim_std_ratio': args.aggressive_trim_std_ratio,
        'aggressive_trim_peak_ratio': args.aggressive_trim_peak_ratio,
        'skip_beam_centering': args.skip_beam_centering,
        'beam_center_roi_size': args.beam_center_roi_size,
        'beam_center_sigma_blur': args.beam_center_sigma_blur,
        'beam_center_max_initial_deviation': args.beam_center_max_initial_deviation,
        'beam_center_fit_bounds': args.beam_center_fit_bounds,
        'beam_center_method': args.beam_center_method,
        'perform_second_pass': args.perform_second_pass,
        'max_beam_jump_pixels': args.max_beam_jump_pixels,
        'smoothing_window_length': args.smoothing_window_length,
        'smoothing_polyorder': args.smoothing_polyorder,
        'smoothing_fallback': args.smoothing_fallback,
        'apply_image_shift': args.apply_image_shift,
        'shift_interpolation_order': args.shift_interpolation_order,
        'num_workers': args.num_workers,
        'filename_template': args.filename_template,
        'write_dials_import_helper_file': args.write_dials_import_helper_file,
        'write_xds_inp_file': args.write_xds_inp_file,
        'write_conversion_report_file': args.write_conversion_report_file,
        'first_pass_diagnostic_plots': args.first_pass_diagnostic_plots,
        'final_diagnostic_plots': args.final_diagnostic_plots,
        'save_beam_centers_file': args.save_beam_centers_file,
        'limit_frames': args.limit_frames
    }
    
    if args.diagnostic_plot_specific_frames_str:
        try: 
            frame_indices = [int(f.strip()) for f in args.diagnostic_plot_specific_frames_str.split(',') if f.strip()]
            final_pipeline_args['diagnostic_plot_specific_frames'] = frame_indices
        except ValueError: 
            logging.warning("Could not parse --plot-frames. Ignoring.")
            final_pipeline_args['diagnostic_plot_specific_frames'] = None
    else: 
        final_pipeline_args['diagnostic_plot_specific_frames'] = None

    if not filename_template_looks_dials_friendly(args.filename_template):
        logging.warning(
            "Filename template '%s' does not contain a numeric frame index. "
            "DIALS may not recognise the result as a sweep.",
            args.filename_template,
        )

    missing_final_params = [p for p in required_params if final_pipeline_args.get(p) is None]
    if missing_final_params:
        logging.critical(f"CRITICAL ERROR: Missing required parameters: {', '.join(missing_final_params)}")
        sys.exit(1)
    
    logging.info("Parameter preparation complete. Starting CBF pipeline with MP Initializer and Parallel Beamfind.")
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        debug_pipeline_args = dict(final_pipeline_args)
        if debug_pipeline_args.get("xml_metadata"):
            xml_debug = dict(debug_pipeline_args["xml_metadata"])
            fractions = xml_debug.pop("fractions", None)
            xml_debug["fractions_summary"] = {
                "count": len(fractions) if isinstance(fractions, list) else 0,
                "first_index": fractions[0].get("index") if isinstance(fractions, list) and fractions else None,
                "last_index": fractions[-1].get("index") if isinstance(fractions, list) and fractions else None,
            }
            debug_pipeline_args["xml_metadata"] = xml_debug
        logging.debug(f"Final pipeline arguments: {debug_pipeline_args}")
    try: 
        mrc_to_cbf_pipeline_mp_init(**final_pipeline_args) 
    except Exception as e: 
        logging.critical(f"Pipeline failed with an unhandled exception: {e}", exc_info=True)
        sys.exit(1) 

if __name__ == '__main__':
    if sys.platform == "darwin":
        current_method = mp.get_start_method(allow_none=True)
        # Only attempt to set if not already 'fork' or if not set (None implies it will use default 'spawn' for Pool)
        if current_method is None: 
            try:
                mp.set_start_method('fork')
            except RuntimeError: 
                try:
                    mp.set_start_method('fork', force=True)
                except RuntimeError as e_force:
                    print(f"WARNING: Could not force 'fork' start method on macOS (current/default: {current_method}). Error: {e_force}. Using current/default.", file=sys.stderr)
        elif current_method != 'fork':
             try:
                mp.set_start_method('fork', force=True)
             except RuntimeError as e_force:
                print(f"WARNING: Could not change start method to 'fork' on macOS (current: {current_method}). Error: {e_force}. Using current.", file=sys.stderr)
            
    main()  
