"""Read one SerialEM movie and optional Velox fraction timing."""

from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

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
    root = ET.parse(path).getroot()
    # Remove namespace prefixes once so the field names stay readable.
    for node in root.iter():
        node.tag = node.tag.split("}")[-1]
    info = root.find(".//Info")
    fractions = root.findall(".//Fractions/Fraction")
    if info is None or not fractions:
        raise ValueError("XML is not a supported Velox fraction sidecar")
    return info, fractions


def detector_geometry(args, mdoc, info, shape, sources, notes):
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
        else:
            raise ValueError(
                "Cannot determine stored pixel size: no MDOC CameraPixelSize or recognized Falcon/Ceta detector. Supply --pixel-size-mm X Y"
            )

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
    return np.asarray(pixel, dtype=float), np.asarray(axis, dtype=float)


def geometry(args, shape):
    """Resolve units and reject contradictory constant-wedge metadata."""
    count, height, width = shape
    mdoc = read_mdoc(args.mdoc)
    info, fractions = read_xml(args.xml)
    notes = []
    sources = {}

    def choose(name, key):
        value = getattr(args, name)
        sources[name] = "command line" if value is not None else f"MDOC {key}"
        if value is None and key in mdoc:
            value = float(mdoc[key])
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
    sources["frame_time_s"] = (
        "command line"
        if args.frame_time_s is not None
        else "MDOC ExposureTime / stack depth"
    )
    timing = {"xml_fractions": len(fractions), "mrc_frames": count}
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
            raise ValueError(
                "XML exposures are nonuniform; one constant oscillation width is invalid"
            )
        starts = [int(node.findtext("StartFrame")) for node in fractions]
        lengths = [int(node.findtext("NumberOfFrames")) for node in fractions]
        if any(length <= 0 for length in lengths) or any(
            starts[idx + 1] != starts[idx] + lengths[idx] for idx in range(count - 1)
        ):
            raise ValueError("XML detector-frame ranges have gaps or overlaps")
        if len(set(lengths)) != 1 or min(starts) < 0:
            raise ValueError("XML detector-frame grouping is not uniform")
        stamps = [node.findtext("DateTimeWithTimeZone") for node in fractions]
        if all(stamps) and count > 1:
            timestamps = [
                datetime.fromisoformat(stamp.replace("Z", "+00:00")) for stamp in stamps
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
                raise ValueError(
                    "XML timestamps do not support contiguous constant-width exposures"
                )
            xml_period = measured_period
            period_source = "XML fitted timestamp period"
        else:
            notes.append(
                "XML timestamps unavailable; exposure durations alone cannot exclude acquisition gaps"
            )
        if period is not None and abs(period - xml_period) > tolerance:
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

    step = args.angle_step_deg
    rate = float(mdoc["DegreesPerSecond"]) if "DegreesPerSecond" in mdoc else None
    derived_step = rate * period if rate is not None and period is not None else None
    if step is not None and derived_step is not None:
        if not np.isclose(step, derived_step, rtol=1e-3, atol=1e-6):
            raise ValueError(
                "--angle-step-deg disagrees with MDOC rotation speed and frame timing"
            )
    if step is None:
        step = derived_step
    sources["angle_step_deg"] = (
        "command line"
        if args.angle_step_deg is not None
        else "MDOC DegreesPerSecond * frame time"
    )
    for name, value in (
        ("distance-mm", distance),
        ("wavelength-a or voltage-kv", wavelength),
        ("frame-time-s", period),
        ("start-angle-deg", start),
        ("angle-step-deg", step),
    ):
        if value is None or not np.isfinite(value):
            raise ValueError(f"Supply --{name}; usable metadata are missing")
    if min(distance, wavelength, period) <= 0 or step == 0:
        raise ValueError(
            "Distance, wavelength and frame time must be positive; angle step must be nonzero"
        )
    pixel, axis = detector_geometry(args, mdoc, info, shape, sources, notes)
    if not np.isfinite(pixel).all() or (pixel <= 0).any():
        raise ValueError(
            "--pixel-size-mm must contain two positive stored-MRC pixel sizes"
        )
    if not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
        raise ValueError(
            "--rotation-axis must be a finite nonzero vector in the DIALS frame"
        )
    axis /= np.linalg.norm(axis)
    # A positive-width scan about the reversed axis is the same physical rotation.
    if step < 0:
        axis = -axis
        start = -start
        step = -step
        notes.append(
            "Negative scan: axis and start angle negated; frame order preserved"
        )
    return dict(
        pixel_size_mm=pixel.tolist(),
        distance_mm=distance,
        wavelength_a=wavelength,
        exposure_time_s=float(np.mean(exposure)) if fractions else period,
        frame_time_s=period,
        start_angle_deg=start,
        angle_step_deg=step,
        rotation_axis=axis.tolist(),
        sources=sources,
        timing=timing,
        notes=notes,
    )
