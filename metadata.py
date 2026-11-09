"""Read acquisition geometry. Keep absent values separate from invalid values."""

import copy
import re
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import mrcfile
import numpy as np
from scipy.constants import c, e, h, m_e


LAB_ROTATION_AXIS = (0.999263, -0.0383878, 0.0)
FALCON_CETA_PIXEL_MM = 0.014


def electron_wavelength(voltage_kv):
    """Relativistic electron wavelength in angstroms."""
    energy = voltage_kv * 1000 * e
    return h / np.sqrt(2 * m_e * energy * (1 + energy / (2 * m_e * c**2))) * 1e10


def read_mdoc(path):
    """Accept a single movie, not a multi-image tilt-series autodoc."""
    values = {}
    sections = []
    if path is None:
        return values
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("["):
            if not line.startswith("[T ="):
                sections.append(line)
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    if len(sections) > 1 or any(
        not line.startswith("[FrameSet =") for line in sections
    ):
        raise ValueError(
            "MDOC must describe one FrameSet movie; use explicit geometry for other layouts"
        )
    return values


def read_xml(path):
    if path is None:
        return None, []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        raise ValueError(f"Malformed XML in {path}: {error}") from error
    # Remove namespace prefixes once so the field names stay readable.
    for node in root.iter():
        node.tag = node.tag.split("}")[-1]
    info = root.find(".//Info")
    fractions = root.findall(".//Fractions/Fraction")
    if info is None or not fractions:
        raise ValueError("XML is not a supported Velox fraction sidecar")
    return info, fractions


# Field presence flags from the EPU-D MRC2014 specification, Appendix A.
FEI_FIELDS = {
    "HT": (1, 5),
    "Projection mode": (1, 26),
    "Camera length": (2, 0),
    "Integration time": (2, 16),
    "Binning Width": (2, 17),
    "Binning Height": (2, 18),
    "Camera name": (2, 19),
    "Direct detector electron counting": (2, 26),
    "Start tilt angle": (4, 8),
    "End tilt angle": (4, 9),
    "Tilt per image": (4, 10),
    "Tilt speed": (4, 11),
}


def read_fei(paths):
    """Read only fields marked present. Do not interpret sample pixel size as pitch."""
    rows = []
    for path in paths:
        with mrcfile.open(path, mode="r", header_only=True) as image:
            if bytes(image.header.exttyp) not in (b"FEI1", b"FEI2"):
                rows.extend([{}] * int(image.header.nz))
                continue
            extended = image.indexed_extended_header
            if extended is None or len(extended) != int(image.header.nz):
                raise ValueError(f"Invalid FEI extended header: {path}")
            for frame in extended:
                values = {}
                for key, (mask, bit) in FEI_FIELDS.items():
                    if key in extended.dtype.names and int(frame[f"Bitmask {mask}"]) & (
                        1 << bit
                    ):
                        value = frame[key].item()
                        if isinstance(value, bytes):
                            value = value.split(b"\0", 1)[0].decode(
                                "ascii", errors="replace"
                            )
                        values[key] = value
                rows.append(values)
    result = {}
    for key in FEI_FIELDS:
        values = [row.get(key) for row in rows]
        if not values or any(value is None for value in values):
            continue
        if isinstance(values[0], str):
            uniform = len(set(values)) == 1
        else:
            uniform = np.isfinite(values).all() and np.allclose(
                values, values[0], rtol=1e-6, atol=1e-8
            )
        if not uniform:
            raise ValueError(f"FEI {key} changes within one sweep")
        result[key] = values[0]
    return result


def read_epu(path):
    """Read the acquisition XML saved beside an EPU-D image sequence."""
    if path is None:
        return {}
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        raise ValueError(f"Malformed XML in {path}: {error}") from error
    for node in root.iter():
        node.tag = node.tag.split("}")[-1]
    if root.tag != "AcquisitionDataXml":
        return {}
    return {
        name: float(root.findtext(key))
        for name, key in (
            ("start_angle_deg", "StartAngle"),
            ("angle_step_deg", "TiltPerImage"),
            ("rotation_rate", "TiltSpeed"),
        )
        if root.findtext(key) is not None
    }


def detector_geometry(args, mdoc, info, shape, sources, notes, allow_missing=False):
    """Resolve stored pixel pitch and the v7 lab-axis fallback, with provenance."""

    def xml(key):
        return info.findtext(key) if info is not None else None

    camera = " ".join(
        [
            mdoc.get(key, "")
            for key in ("SubFramePath", "CameraName", "Detector", "DetectorModel")
        ]
        + [xml(key) or "" for key in ("CommercialName", "CameraName")]
    ).lower()
    known_camera = "falcon" in camera or "ceta" in camera
    pixel = args.pixel_size_mm
    sources["pixel_size_mm"] = "command line"
    if pixel is None:
        binning = None
        bin_source = None
        if known_camera:
            for source, value in (
                ("MDOC Binning", mdoc.get("Binning")),
                ("XML Binning", xml("Binning")),
            ):
                if value is None:
                    continue
                factor = float(value)
                if not np.isfinite(factor) or factor < 1 or not factor.is_integer():
                    raise ValueError(f"Invalid {source}; supply --pixel-size-mm X Y")
                if binning is not None and factor != binning:
                    raise ValueError(
                        "MDOC/XML acquisition binning disagrees; supply --pixel-size-mm X Y"
                    )
                if binning is None:
                    binning, bin_source = factor, source

        if "CameraPixelSize" in mdoc:
            pixel = np.array(
                [
                    float(value)
                    for value in mdoc["CameraPixelSize"].replace(",", " ").split()
                ]
            )
            if pixel.size == 1:
                pixel = np.repeat(pixel, 2)
            if pixel.size != 2 or not np.isfinite(pixel).all() or (pixel <= 0).any():
                raise ValueError(
                    "MDOC CameraPixelSize must contain one or two positive values in micrometres"
                )
            pixel /= 1000
            sources["pixel_size_mm"] = "MDOC CameraPixelSize (stored-pixel micrometres)"
            if known_camera and binning is not None:
                expected = FALCON_CETA_PIXEL_MM * binning
                if binning > 1 and np.allclose(
                    pixel, FALCON_CETA_PIXEL_MM, rtol=0.05, atol=0
                ):
                    pixel *= binning
                    sources["pixel_size_mm"] = (
                        f"MDOC CameraPixelSize (native pitch) * {bin_source}"
                    )
                    notes.append(
                        "CameraPixelSize matches the native Falcon/Ceta pitch; acquisition binning was applied once"
                    )
                elif not np.allclose(pixel, expected, rtol=0.05, atol=0):
                    notes.append(
                        "CameraPixelSize disagrees with the Falcon/Ceta pitch and binning; using it as stored-pixel pitch. Check --pixel-size-mm"
                    )
            else:
                notes.append(
                    "Treating CameraPixelSize as stored-pixel micrometres; detector/binning could not confirm it. Check --pixel-size-mm"
                )
        elif known_camera:
            if binning is not None:
                factors = [binning, binning]
                sources["pixel_size_mm"] = (
                    f"Falcon/Ceta 0.014 mm native pitch * {bin_source}"
                )
            else:
                roi = [xml("RegionOfInterest/Width"), xml("RegionOfInterest/Height")]
                if all(value is not None for value in roi):
                    height, width = shape[-2:]
                    factors = np.array(roi, dtype=float) / [width, height]
                    sources["pixel_size_mm"] = (
                        "Falcon/Ceta native pitch * XML ROI / MRC dimensions"
                    )
                else:
                    height, width = shape[-2:]
                    factors = 4096 / np.array([width, height], dtype=float)
                    sources["pixel_size_mm"] = (
                        "Falcon/Ceta native pitch * assumed 4096-pixel sensor / MRC dimensions"
                    )
                    notes.append(
                        "No acquisition binning or ROI: assuming a full 4096 x 4096 Falcon/Ceta image. Override --pixel-size-mm for cropped data"
                    )
                if (
                    not np.isfinite(factors).all()
                    or (factors < 1).any()
                    or not np.allclose(factors, np.rint(factors))
                ):
                    raise ValueError(
                        "Cannot infer integer Falcon/Ceta acquisition binning; supply --pixel-size-mm X Y"
                    )
            pixel = FALCON_CETA_PIXEL_MM * np.asarray(factors)
        elif not allow_missing:
            raise ValueError(
                "Cannot determine stored pixel size: no MDOC CameraPixelSize or recognized Falcon/Ceta detector. Supply --pixel-size-mm X Y"
            )
        else:
            sources["pixel_size_mm"] = "missing"

    axis = args.rotation_axis
    sources["rotation_axis"] = "command line"
    if axis is None:
        axis = LAB_ROTATION_AXIS
        sources["rotation_axis"] = (
            "v7 lab calibration fallback (not measured from metadata)"
        )
        notes.append(
            "Rotation axis is not specified; using the v7 lab default (0.999263, -0.0383878, 0). This default may be wrong for your setup. Supply --rotation-axis X Y Z to override"
        )
    return (np.asarray(pixel, dtype=float) if pixel is not None else None), np.asarray(
        axis, dtype=float
    )


def geometry(args, shape, allow_missing=False, paths=(), first_number=None):
    """Resolve units and reject contradictory constant-wedge metadata."""
    count, height, width = shape
    notes = []
    try:
        mdoc = read_mdoc(args.mdoc)
    except ValueError as error:
        if not allow_missing or "one FrameSet" not in str(error):
            raise
        mdoc = {}
        notes.append(
            "MDOC layout is not supported; per-image tilts were not flattened into a continuous scan"
        )
    epu = read_epu(args.xml)
    try:
        info, fractions = (None, []) if epu else read_xml(args.xml)
    except ValueError as error:
        if not allow_missing or "not a supported" not in str(error):
            raise
        info, fractions = None, []
        notes.append("XML format is not supported; its geometry was not used")
    fei = read_fei(paths) if paths else {}
    sources = {}
    fallback = {}
    pixel_hint = None
    for name, key, scale in (
        ("distance_mm", "Camera length", 1000),
        ("voltage_kv", "HT", 0.001),
        ("start_angle_deg", "Start tilt angle", 1),
        ("angle_step_deg", "Tilt per image", 1),
    ):
        if key in fei:
            fallback[name] = (fei[key] * scale, f"FEI {key}")
    if "Tilt speed" in fei:
        fallback["rotation_rate"] = (fei["Tilt speed"], "FEI Tilt speed")
    for name, value in epu.items():
        if name in fallback and not np.isclose(
            value, fallback[name][0], rtol=1e-3, atol=1e-6
        ):
            raise ValueError(f"EPU XML and FEI {name} disagree")
        fallback[name] = (value, "EPU acquisition XML")
    if (
        first_number is not None
        and first_number > 1
        and args.start_angle_deg is None
        and "start_angle_deg" in fallback
    ):
        fallback.pop("start_angle_deg", None)
        notes.append(
            "Series starts after file number 1; the start angle of the selected files is unknown. Use --start-angle-deg"
        )
    if "Camera name" in fei and not any(
        key in mdoc for key in ("CameraName", "Detector", "SubFramePath")
    ):
        mdoc["CameraName"] = fei["Camera name"]
    if (
        args.pixel_size_mm is None
        and "Binning Width" in fei
        and "Binning Height" in fei
    ):
        camera = fei.get("Camera name", "").lower()
        if "falcon" in camera or "ceta" in camera:
            factors = np.array(
                [fei["Binning Width"], fei["Binning Height"]], dtype=float
            )
            if (factors < 1).any() or not np.equal(factors, np.rint(factors)).all():
                raise ValueError(
                    "FEI acquisition binning must contain positive integers"
                )
            if (
                "Binning" in mdoc
                and not np.equal(factors, float(mdoc["Binning"])).all()
            ):
                raise ValueError(
                    "MDOC/FEI acquisition binning disagrees; supply --pixel-size-mm X Y"
                )
            if "CameraPixelSize" not in mdoc:
                pixel_hint = FALCON_CETA_PIXEL_MM * factors
    if args.counting and fei.get("Direct detector electron counting") is False:
        raise ValueError("--counting contradicts FEI electron counting metadata")

    def choose(name, key):
        value = getattr(args, name)
        sources[name] = "command line" if value is not None else "missing"
        if value is None and key in mdoc:
            value = float(mdoc[key])
            sources[name] = f"MDOC {key}"
        if value is None and name in fallback:
            value, sources[name] = fallback[name]
        return value

    distance = choose("distance_mm", "CameraLength")
    start = choose("start_angle_deg", "TiltAngle")
    voltage = choose("voltage_kv", "Voltage")
    wavelength = args.wavelength_a
    if wavelength is None and voltage is not None:
        if not np.isfinite(voltage) or voltage <= 0:
            raise ValueError("Voltage must be positive and finite")
        wavelength = float(electron_wavelength(voltage))
    sources["wavelength_a"] = (
        "command line" if args.wavelength_a is not None else sources["voltage_kv"]
    )
    if "NumSubFrames" in mdoc and float(mdoc["NumSubFrames"]) != count:
        raise ValueError("MDOC NumSubFrames differs from the MRC stack depth")
    mdoc_period = (
        float(mdoc["ExposureTime"]) / count if "ExposureTime" in mdoc else None
    )
    period = args.frame_time_s if args.frame_time_s is not None else mdoc_period
    rate = epu.get("rotation_rate", fei.get("Tilt speed"))
    fei_step = fallback.get("angle_step_deg", (None,))[0]
    if period is None and rate is not None and rate != 0 and fei_step is not None:
        period = fei_step / rate
    sources["frame_time_s"] = (
        "command line"
        if args.frame_time_s is not None
        else "MDOC ExposureTime / stack depth"
        if mdoc_period is not None
        else "EPU/FEI tilt per image / tilt speed"
        if period is not None
        else "missing"
    )
    timing = {"xml_fractions": len(fractions), "mrc_frames": count}
    timing_issues = []

    def timing_issue(message):
        if not allow_missing:
            raise ValueError(message)
        timing_issues.append(message)

    if fractions:
        if len(fractions) != count:
            raise ValueError("XML fraction count differs from the MRC stack depth")
        for key in ("ExpectedNumberOfFractions", "RecordedNumberOfFractions"):
            if info.findtext(key) is not None and int(info.findtext(key)) != count:
                raise ValueError(f"XML {key} differs from the MRC stack depth")
        for key, size in (("Width", width), ("Height", height)):
            if (
                info.findtext(f"ImageSize/{key}") is not None
                and int(info.findtext(f"ImageSize/{key}")) != size
            ):
                raise ValueError(f"XML image {key} differs from the MRC")
        required = ("Index", "StartFrame", "NumberOfFrames", "ExposureTime")
        if any(node.findtext(key) is None for node in fractions for key in required):
            raise ValueError(
                "XML fraction is missing an index, detector-frame range, or exposure"
            )
        indices = [int(node.findtext("Index")) for node in fractions]
        if indices != list(range(count)):
            raise ValueError(
                "XML fraction indices are missing, duplicated, or out of order"
            )
        exposure = np.array(
            [float(node.findtext("ExposureTime")) for node in fractions]
        )
        if not np.isfinite(exposure).all() or (exposure <= 0).any():
            raise ValueError("XML exposure times must be positive and finite")
        xml_period = float(np.mean(exposure))
        period_source = "XML ExposureTime"
        tolerance = max(1e-4, 1e-3 * xml_period)
        if np.ptp(exposure) > tolerance:
            timing_issue(
                "XML exposures are nonuniform; one constant oscillation width is invalid"
            )
        starts = [int(node.findtext("StartFrame")) for node in fractions]
        lengths = [int(node.findtext("NumberOfFrames")) for node in fractions]
        if any(length <= 0 for length in lengths) or any(
            starts[idx + 1] != starts[idx] + lengths[idx] for idx in range(count - 1)
        ):
            raise ValueError("XML detector-frame ranges have gaps or overlaps")
        if min(starts) < 0:
            raise ValueError("XML detector-frame indices must be nonnegative")
        if len(set(lengths)) != 1:
            timing_issue("XML detector-frame grouping is not uniform")
        timing["frame_exposure_times_s"] = exposure.tolist()
        timing["detector_start_frames"] = starts
        timing["detector_frame_counts"] = lengths
        stamps = [node.findtext("DateTimeWithTimeZone") for node in fractions]
        if all(stamps) and count > 1:
            timestamps = [
                datetime.fromisoformat(
                    re.sub(
                        r"(?<=\d{2}:\d{2}:\d{2})[.,](\d+)",
                        lambda match: "." + match[1][:6].ljust(6, "0"),
                        stamp.replace("Z", "+00:00"),
                    )
                )
                for stamp in stamps
            ]
            times = np.array(
                [(stamp - timestamps[0]).total_seconds() for stamp in timestamps]
            )
            measured_period = float(np.polyfit(np.arange(count), times, 1)[0])
            fitted = np.arange(count) * measured_period
            residual = times - fitted - np.mean(times - fitted)
            timing["max_timestamp_residual_s"] = float(np.max(np.abs(residual)))
            timing["timestamp_period_s"] = measured_period
            timing["exposure_time_s"] = xml_period
            # Software timestamps may jitter; bound their angular error to 0.5% of a frame.
            timestamp_tolerance = max(1e-5, 0.005 * xml_period)
            timing["timestamp_tolerance_s"] = timestamp_tolerance
            if (
                (np.diff(times) <= 0).any()
                or np.max(np.abs(residual)) > timestamp_tolerance
                or abs(measured_period - xml_period) > tolerance
            ):
                timing_issue(
                    "XML timestamps do not support contiguous constant-width exposures"
                )
            xml_period = measured_period
            period_source = "XML fitted timestamp period"
        else:
            notes.append(
                "XML timestamps unavailable; exposure durations alone cannot exclude acquisition gaps"
            )
        if (
            not timing_issues
            and period is not None
            and abs(period - xml_period) > tolerance
        ):
            raise ValueError(
                "MDOC/command-line frame time disagrees with XML exposures"
            )
        period = xml_period
        sources["frame_time_s"] = period_source
        timing["exposure_spread_s"] = float(np.ptp(exposure))
        if args.counting and info.findtext("ElectronCounting", "").lower() == "off":
            raise ValueError("--counting contradicts XML ElectronCounting=Off")
    else:
        notes.append(
            "No XML timing check; scan uniformity relies on the supplied MDOC/geometry"
        )

    if timing_issues:
        timing["constant_scan_valid"] = False
        timing["issues"] = timing_issues
        notes.extend(timing_issues)
        notes.append(
            "Pixels and per-frame exposures preserved. Constant scan timing and angles "
            "are unresolved; XDS/DIALS helpers omitted"
        )
        period = None
        sources["frame_time_s"] = "unresolved XML timing"

    step = choose("angle_step_deg", "AngleIncrement")
    rate = float(mdoc["DegreesPerSecond"]) if "DegreesPerSecond" in mdoc else rate
    derived_step = rate * period if rate is not None and period is not None else None
    if step is not None and derived_step is not None:
        if not np.isclose(step, derived_step, rtol=1e-3, atol=1e-6):
            raise ValueError(
                "--angle-step-deg disagrees with MDOC rotation speed and frame timing"
            )
    if step is None:
        step = derived_step
        sources["angle_step_deg"] = (
            "rotation speed * frame time" if step is not None else "missing"
        )
    if timing_issues:
        # Do not apply a constant angular width to an irregular acquisition.
        step = None
        sources["angle_step_deg"] = "unresolved XML timing"
    for name, value in (
        ("distance-mm", distance),
        ("wavelength-a or voltage-kv", wavelength),
        ("frame-time-s", period),
        ("start-angle-deg", start),
        ("angle-step-deg", step),
    ):
        if value is None:
            if not allow_missing:
                raise ValueError(f"Supply --{name}; usable metadata are missing")
        elif not np.isfinite(value):
            raise ValueError(f"Invalid --{name}; value must be finite")
    if (
        any(
            value is not None and value <= 0 for value in (distance, wavelength, period)
        )
        or step == 0
    ):
        raise ValueError(
            "Distance, wavelength and frame time must be positive; angle step must be nonzero"
        )
    detector_args = copy.copy(args)
    if pixel_hint is not None:
        detector_args.pixel_size_mm = pixel_hint
    pixel, axis = detector_geometry(
        detector_args, mdoc, info, shape, sources, notes, allow_missing
    )
    if pixel_hint is not None:
        sources["pixel_size_mm"] = (
            "Falcon/Ceta 0.014 mm native pitch * FEI acquisition binning"
        )
    if sources["distance_mm"] == "FEI Camera length":
        notes.append(
            "Using the nominal FEI camera length; calibrated distance is unknown. Override --distance-mm if available"
        )
    if pixel is not None and (not np.isfinite(pixel).all() or (pixel <= 0).any()):
        raise ValueError(
            "--pixel-size-mm must contain two positive stored-MRC pixel sizes"
        )
    if not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
        raise ValueError(
            "--rotation-axis must be a finite nonzero vector in the DIALS frame"
        )
    axis /= np.linalg.norm(axis)
    # A positive-width scan about the reversed axis is the same physical rotation.
    if step is not None and step < 0:
        axis = -axis
        start = -start if start is not None else None
        step = -step
        notes.append(
            "Negative scan: axis and start angle negated; frame order preserved"
        )
    exposure_time = (
        float(np.mean(exposure)) if fractions else fei.get("Integration time", period)
    )
    if timing_issues:
        exposure_time = None
    if exposure_time is not None and (
        not np.isfinite(exposure_time) or exposure_time <= 0
    ):
        raise ValueError("Exposure time must be positive and finite")
    if (
        exposure_time is not None
        and period is not None
        and exposure_time > period + max(1e-4, period * 1e-3)
    ):
        raise ValueError("Exposure time exceeds the frame period")
    return dict(
        pixel_size_mm=pixel.tolist() if pixel is not None else None,
        distance_mm=distance,
        wavelength_a=wavelength,
        exposure_time_s=exposure_time,
        frame_time_s=period,
        start_angle_deg=start,
        angle_step_deg=step,
        rotation_axis=axis.tolist(),
        sources=sources,
        timing=timing,
        notes=notes,
    )
