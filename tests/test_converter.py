import contextlib
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import fabio
import mrcfile
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mrc2cbf as converter
from metadata import (
    LAB_ROTATION_AXIS,
    detector_geometry,
    electron_wavelength,
    geometry,
    read_mdoc,
)


class PixelTests(unittest.TestCase):
    def test_sum_binning_conserves_signed_counts(self):
        values = np.arange(-96, 96).reshape(12, 16).astype(np.int16)
        binned, mask = converter.bin_frame(values, 4)
        expected = np.array(
            [
                [values[y : y + 4, x : x + 4].sum() for x in range(0, 16, 4)]
                for y in range(0, 12, 4)
            ]
        )
        np.testing.assert_array_equal(binned, expected)
        self.assertEqual(binned.sum(), values.sum())
        self.assertFalse(mask.any())

    def test_nondivisible_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, "divide"):
            converter.bin_frame(np.zeros((13, 16)), 4)

    def test_mask_and_saturation_propagate_without_repair(self):
        values = np.ones((8, 8))
        values[0, 1], values[7, 6] = np.nan, 100
        bad = np.zeros_like(values, dtype=bool)
        bad[3, 3] = True
        binned, mask = converter.bin_frame(values, 2, bad, 100)
        self.assertEqual(np.count_nonzero(mask), 3)
        pixels = converter.store_pixels(binned, mask, 10)
        self.assertTrue(np.all(pixels[mask] == -1))
        self.assertTrue(np.all(pixels[~mask] == 14))

    def test_pedestal_is_exactly_reversible(self):
        values = np.array([[-381, -3, 0, 19]], dtype=float)
        mask = np.zeros_like(values, dtype=bool)
        stored = converter.store_pixels(values, mask, 381)
        np.testing.assert_array_equal(stored.astype(np.int64) - 381, values)

    def test_undersized_pedestal_and_overflow_fail(self):
        for values, pedestal in (([[-5.0]], 4), ([[float(2**31)]], 0)):
            with self.assertRaisesRegex(ValueError, "clip"):
                converter.store_pixels(
                    np.array(values), np.zeros((1, 1), bool), pedestal
                )

    def test_float_rounding_occurs_after_sum(self):
        values = np.full((2, 2), 0.4)
        binned, mask = converter.bin_frame(values, 2)
        self.assertEqual(converter.store_pixels(binned, mask, 0)[0, 0], 2)

    def test_integer_translation_never_wraps_or_interpolates(self):
        values = np.arange(30).reshape(5, 6)
        mask = np.zeros_like(values, bool)
        for dx, dy in [(2, 1), (-2, -1), (0, 0)]:
            moved, invalid = converter.translate(values, mask, (dx, dy))
            for y, x in np.argwhere(~invalid):
                self.assertEqual(moved[y, x], values[y - dy, x - dx])
            self.assertEqual(np.count_nonzero(~invalid), (5 - abs(dy)) * (6 - abs(dx)))


class BeamTests(unittest.TestCase):
    def test_halo_center_with_core_dip_and_bragg_spot(self):
        yy, xx = np.indices((128, 128))
        center = np.array([65.2, 61.7])
        values = 20 + 5000 * np.exp(
            -((xx - center[0]) ** 2 / 200 + (yy - center[1]) ** 2 / 288)
        )
        values[np.hypot(xx - center[0], yy - center[1]) < 5] = -30
        values[47:49, 84:86] += 100000
        values += np.random.default_rng(51).normal(0, 2, values.shape)
        found = converter.find_beam(values, np.zeros_like(values, bool), 64)
        self.assertIsNotNone(found)
        np.testing.assert_allclose(found, center, atol=0.3)

    def test_blank_frame_does_not_manufacture_center(self):
        self.assertIsNone(
            converter.find_beam(np.ones((128, 128)), np.zeros((128, 128), bool), 64)
        )

    def test_large_drift_rejected(self):
        with self.assertRaisesRegex(ValueError, "varies"):
            converter.robust_center([[1, 3], [5, 3], [9, 3]], 2)

    def test_trim_keeps_internal_weak_frames(self):
        stats = [
            dict(peak=value, signal=value)
            for value in [1, 10, 100, 100, 100, 1, 100, 100, 100, 40, 0]
        ]
        self.assertEqual(converter.trim_frames(stats, "aggressive", 3), (2, 9))
        self.assertEqual(converter.trim_frames(stats, "none", 3), (0, 11))

    def test_no_strong_run_fails(self):
        with self.assertRaises(ValueError):
            converter.trim_frames([dict(peak=0, signal=0)] * 10, "aggressive", 5)


def arguments(source, output, extra=()):
    return converter.parser().parse_args(
        [
            str(source),
            str(output),
            "--pixel-size-mm",
            "0.028",
            "0.031",
            "--rotation-axis",
            "0.98",
            "-0.05",
            "0.03",
            "--distance-mm",
            "961.06",
            "--voltage-kv",
            "200",
            "--start-angle-deg",
            "-60",
            "--angle-step-deg",
            "0.2",
            "--frame-time-s",
            "0.1",
            "--beam-center",
            "31.2",
            "32.7",
            "--trim",
            "none",
            *extra,
        ]
    )


class MetadataTests(unittest.TestCase):
    def resolve_detector(self, mdoc, xml=None, shape=(7, 2048, 2048), overrides=False):
        import xml.etree.ElementTree as ET

        args = arguments("a", "b")
        if not overrides:
            args.pixel_size_mm = None
            args.rotation_axis = None
        sources, notes = {}, []
        pixel, axis = detector_geometry(
            args, mdoc, ET.fromstring(xml) if xml else None, shape, sources, notes
        )
        return pixel, axis, sources, notes

    def test_geometry_flags_are_optional_at_cli(self):
        args = converter.parser().parse_args(["movie.mrc", "cbf"])
        self.assertIsNone(args.pixel_size_mm)
        self.assertIsNone(args.rotation_axis)

    def test_mdoc_stored_pitch_and_v7_axis_defaults(self):
        pixel, axis, sources, notes = self.resolve_detector(
            {
                "CameraPixelSize": "28",
                "Binning": "2",
                "SubFramePath": "BM-Falcon/movie.mrc",
            }
        )
        np.testing.assert_allclose(pixel, [0.028, 0.028])
        np.testing.assert_allclose(axis, LAB_ROTATION_AXIS)
        self.assertIn("CameraPixelSize", sources["pixel_size_mm"])
        self.assertTrue(any("not specified" in note for note in notes))
        self.assertTrue(any("may be wrong" in note for note in notes))

    def test_native_pitch_is_binned_once(self):
        pixel, _, sources, notes = self.resolve_detector(
            {"CameraPixelSize": "14", "Binning": "2", "CameraName": "Ceta"}
        )
        np.testing.assert_allclose(pixel, [0.028, 0.028])
        self.assertIn("native pitch", sources["pixel_size_mm"])
        self.assertTrue(any("applied once" in note for note in notes))

    def test_detector_profile_with_xml_binning(self):
        pixel, _, sources, _ = self.resolve_detector(
            {},
            "<Info><CommercialName>Falcon 4</CommercialName><Binning>4</Binning></Info>",
        )
        np.testing.assert_allclose(pixel, [0.056, 0.056])
        self.assertIn("XML Binning", sources["pixel_size_mm"])

    def test_roi_prevents_mistaking_crop_for_binning(self):
        pixel, _, sources, notes = self.resolve_detector(
            {},
            "<Info><CommercialName>Falcon 4</CommercialName><RegionOfInterest>"
            "<Width>2048</Width><Height>2048</Height></RegionOfInterest></Info>",
        )
        np.testing.assert_allclose(pixel, [0.014, 0.014])
        self.assertIn("ROI", sources["pixel_size_mm"])
        self.assertFalse(any("full 4096" in note for note in notes))

    def test_dimension_fallback_warns_about_cropping(self):
        pixel, _, _, notes = self.resolve_detector({"CameraName": "Ceta"})
        np.testing.assert_allclose(pixel, [0.028, 0.028])
        self.assertTrue(any("full 4096" in note for note in notes))

    def test_unknown_detector_can_use_documented_pixel_pitch(self):
        pixel, _, _, notes = self.resolve_detector({"CameraPixelSize": "55 56"})
        np.testing.assert_allclose(pixel, [0.055, 0.056])
        self.assertTrue(any("could not confirm" in note for note in notes))

    def test_unknown_pixel_pitch_is_not_guessed_from_pixelspacing(self):
        with self.assertRaisesRegex(ValueError, "Cannot determine stored pixel size"):
            self.resolve_detector({"PixelSpacing": "0.00116061"})

    def test_conflicting_binning_needs_explicit_override(self):
        mdoc = {"CameraName": "Falcon 4", "Binning": "2"}
        xml = "<Info><Binning>4</Binning></Info>"
        with self.assertRaisesRegex(ValueError, "binning disagrees"):
            self.resolve_detector(mdoc, xml)
        pixel, axis, sources, notes = self.resolve_detector(mdoc, xml, overrides=True)
        np.testing.assert_allclose(pixel, [0.028, 0.031])
        np.testing.assert_allclose(axis, [0.98, -0.05, 0.03])
        self.assertEqual(
            sources, {"pixel_size_mm": "command line", "rotation_axis": "command line"}
        )
        self.assertEqual(notes, [])

    def test_wavelength(self):
        self.assertAlmostEqual(electron_wavelength(200), 0.02507934, places=7)
        self.assertAlmostEqual(electron_wavelength(300), 0.01968749, places=7)

    def test_axis_conversion_is_proper_rotation(self):
        transform = np.diag([1, -1, -1])
        self.assertEqual(np.linalg.det(transform), 1)
        axis = np.array([0.9, 0.2, 0.3])
        axis /= np.linalg.norm(axis)
        rotation = Rotation.from_rotvec(axis * 0.43).as_matrix()
        expected = Rotation.from_rotvec(transform @ axis * 0.43).as_matrix()
        np.testing.assert_allclose(
            transform @ rotation @ transform.T, expected, atol=1e-14
        )

    def test_negative_scan_preserves_physical_rotation(self):
        args = arguments("a", "b", ["--angle-step-deg", "-0.2"])
        meta = geometry(args, (10, 64, 64))
        axis = np.array(args.rotation_axis)
        axis /= np.linalg.norm(axis)
        for idx in (0, 3, 9):
            original = Rotation.from_rotvec(
                axis * np.deg2rad(-60 - idx * 0.2)
            ).as_matrix()
            converted = Rotation.from_rotvec(
                np.array(meta["rotation_axis"])
                * np.deg2rad(meta["start_angle_deg"] + idx * 0.2)
            ).as_matrix()
            np.testing.assert_allclose(original, converted, atol=1e-14)

    def test_multi_movie_mdoc_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.mdoc"
            path.write_text(
                "[FrameSet = 0]\nTiltAngle = -30\n[FrameSet = 1]\nTiltAngle = 20\n"
            )
            with self.assertRaisesRegex(ValueError, "one FrameSet"):
                read_mdoc(path)

    def test_xml_uniform_and_nonuniform_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.xml"
            args = arguments("a", "b")
            args.xml = path
            for bad in (False, True):
                fractions = "".join(
                    f"<Fraction><Index>{idx}</Index><StartFrame>{idx * 3}</StartFrame><NumberOfFrames>3</NumberOfFrames><ExposureTime>{0.15 if bad and idx == 2 else 0.1}</ExposureTime><DateTimeWithTimeZone>2026-04-10T12:00:00.{idx}00000Z</DateTimeWithTimeZone></Fraction>"
                    for idx in range(5)
                )
                path.write_text(
                    f"<Root><Info/><Fractions>{fractions}</Fractions></Root>"
                )
                if bad:
                    with self.assertRaisesRegex(ValueError, "nonuniform"):
                        geometry(args, (5, 64, 64))
                else:
                    self.assertAlmostEqual(
                        geometry(args, (5, 64, 64))["frame_time_s"], 0.1
                    )

    def test_xml_timestamp_jitter_is_bounded_and_gap_rejected(self):
        from datetime import datetime, timedelta, timezone

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.xml"
            args = arguments("a", "b")
            args.xml = path
            for gap in (0, 0.03):
                rows = []
                for idx in range(20):
                    elapsed = (
                        idx * 0.1
                        + (0.00015 if idx == 8 else 0)
                        + (gap if idx >= 10 else 0)
                    )
                    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
                        seconds=elapsed
                    )
                    rows.append(
                        f"<Fraction><Index>{idx}</Index><StartFrame>{idx * 3}</StartFrame>"
                        f"<NumberOfFrames>3</NumberOfFrames><ExposureTime>0.1</ExposureTime>"
                        f"<DateTimeWithTimeZone>{stamp.isoformat()}</DateTimeWithTimeZone></Fraction>"
                    )
                path.write_text(
                    "<Root><Info/><Fractions>" + "".join(rows) + "</Fractions></Root>"
                )
                if gap:
                    with self.assertRaisesRegex(ValueError, "timestamps"):
                        geometry(args, (20, 64, 64))
                else:
                    meta = geometry(args, (20, 64, 64))
                    self.assertAlmostEqual(meta["exposure_time_s"], 0.1, places=12)
                    self.assertNotEqual(meta["exposure_time_s"], meta["frame_time_s"])
                    header = converter.cbf_header(meta, (31, 31), -60, 30, 10000)
                    self.assertIn("# Exposure_time 0.1 s", header)
                    self.assertLess(
                        meta["timing"]["max_timestamp_residual_s"],
                        meta["timing"]["timestamp_tolerance_s"],
                    )

    def test_missing_xml_fields_fail_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.xml"
            path.write_text("<Root><Info/><Fractions><Fraction/></Fractions></Root>")
            args = arguments("a", "b")
            args.xml = path
            with self.assertRaisesRegex(ValueError, "missing"):
                geometry(args, (1, 64, 64))

    def test_acquisition_binning_and_metadata_gain_do_not_rescale_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.mdoc"
            path.write_text(
                "[FrameSet = 0]\nBinning = 4\nCameraPixelSize = 999\n"
                "CountsPerElectron = 12345\nNumSubFrames = 5\nExposureTime = 0.5\n"
            )
            args = arguments("a", "b")
            args.mdoc = path
            meta = geometry(args, (5, 64, 64))
            self.assertEqual(meta["pixel_size_mm"], [0.028, 0.031])
            self.assertNotIn("gain", meta)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "input.mrc"
        self.data = np.random.default_rng(17).integers(
            -50, 100, (7, 64, 64), dtype=np.int16
        )
        with mrcfile.new(self.source) as movie:
            movie.set_data(self.data)
            movie.set_image_stack()
        self.before = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def tearDown(self):
        self.assertEqual(
            hashlib.sha256(self.source.read_bytes()).hexdigest(), self.before
        )
        self.temp.cleanup()

    def run_conversion(self, name="out", extra=()):
        args = arguments(self.source, self.root / name, extra)
        with contextlib.redirect_stdout(io.StringIO()):
            return converter.convert(args)

    def test_sidecar_only_cli_conversion_matches_explicit_geometry(self):
        source = self.root / "auto.mrc"
        yy, xx = np.indices((64, 64))
        halo = 20 + 5000 * np.exp(-((xx - 32.1) ** 2 + (yy - 30.8) ** 2) / 120)
        with mrcfile.new(source) as movie:
            movie.set_data(np.stack([np.rint(halo)] * 7).astype(np.int16))
            movie.set_image_stack()
        mdoc = Path(str(source) + ".mdoc")
        mdoc.write_text(
            "Voltage = 200\nDegreesPerSecond = 2\n[FrameSet = 0]\n"
            "CameraLength = 961.06\nTiltAngle = -60\nExposureTime = 0.7\n"
            "NumSubFrames = 7\nCameraPixelSize = 28\nBinning = 2\n"
            "SubFramePath = BM-Falcon/auto.mrc\nCountsPerElectron = 999\n"
        )
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        mdoc_hash = hashlib.sha256(mdoc.read_bytes()).hexdigest()
        command = [
            sys.executable,
            str(Path(converter.__file__)),
            str(source),
            str(self.root / "auto"),
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("v7 lab default", result.stdout)
        result = subprocess.run(
            command[:-1]
            + [
                str(self.root / "explicit"),
                "--pixel-size-mm",
                "0.028",
                "0.028",
                "--rotation-axis",
                "0.999263",
                "-0.0383878",
                "0",
            ],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        for idx in range(1, 8):
            self.assertEqual(
                (self.root / "auto" / f"image_{idx:06d}.cbf").read_bytes(),
                (self.root / "explicit" / f"image_{idx:06d}.cbf").read_bytes(),
            )
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), source_hash)
        self.assertEqual(hashlib.sha256(mdoc.read_bytes()).hexdigest(), mdoc_hash)

    def test_signed_roundtrip_and_geometry(self):
        report = self.run_conversion(extra=["--frames", "2", "6", "--bin", "2"])
        self.assertEqual(
            report["storage_pedestal"],
            -min(
                int(np.rint(self.data[idx].reshape(32, 2, 32, 2).sum((1, 3))).min())
                for idx in range(1, 6)
            ),
        )
        self.assertEqual(report["input_frames"], [2, 6])
        np.testing.assert_allclose(report["geometry"]["pixel_size_mm"], [0.056, 0.062])
        np.testing.assert_allclose(
            report["beam_center_xy"], [(31.2 + 0.5) / 2 - 0.5, (32.7 + 0.5) / 2 - 0.5]
        )
        self.assertAlmostEqual(report["geometry"]["start_angle_deg"], -59.8)
        for idx, path in enumerate(sorted((self.root / "out").glob("*.cbf"))):
            with fabio.open(str(path)) as image:
                expected = self.data[idx + 1].reshape(32, 2, 32, 2).sum((1, 3))
                np.testing.assert_array_equal(
                    image.data.astype(np.int64) - report["storage_pedestal"], expected
                )
                header = image.header["_array_data.header_contents"]
                self.assertIn("Beam_xy (15.85000000, 16.60000000)", header)
                self.assertEqual(
                    image.header["_array_data.header_convention"], "GENERIC_MINI"
                )
        text = (self.root / "out" / "XDS_20230630.INP").read_text()
        self.assertIn("ORGX= 16.35000000 ORGY= 17.10000000", text)
        self.assertIn("FRACTION_OF_POLARIZATION= -1", text)
        self.assertIn("AIR= 0", text)
        self.assertTrue((self.root / "out" / "XDS.INP").exists())
        self.assertIn(
            f"OFFSET= {report['storage_pedestal']}",
            (self.root / "out" / "XDS.INP").read_text(),
        )

    def test_worker_count_does_not_change_pixels(self):
        self.run_conversion("one", ["--workers", "1"])
        self.run_conversion("four", ["--workers", "4"])
        for idx in range(1, 8):
            with (
                fabio.open(str(self.root / "one" / f"image_{idx:06d}.cbf")) as one,
                fabio.open(str(self.root / "four" / f"image_{idx:06d}.cbf")) as four,
            ):
                np.testing.assert_array_equal(one.data, four.data)

    def test_explicit_pedestal_is_reported_and_reversible(self):
        self.run_conversion("a", ["--pedestal", "100"])
        report = self.run_conversion("b", ["--pedestal", "200"])
        self.assertEqual(report["xds_offset"], 200)
        for idx in range(1, 8):
            with (
                fabio.open(str(self.root / "a" / f"image_{idx:06d}.cbf")) as a,
                fabio.open(str(self.root / "b" / f"image_{idx:06d}.cbf")) as b,
            ):
                np.testing.assert_array_equal(
                    a.data.astype(np.int64) - 100, b.data.astype(np.int64) - 200
                )

    def test_refuses_overwrite(self):
        self.run_conversion()
        with self.assertRaisesRegex(ValueError, "exists"):
            self.run_conversion()

    def test_counting_rejects_signed_input(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.run_conversion(extra=["--counting"])

    def test_counting_zero_and_no_extra_binning(self):
        source = self.root / "counts.mrc"
        with mrcfile.new(source) as movie:
            movie.set_data(np.maximum(self.data, 0))
            movie.set_image_stack()
        with contextlib.redirect_stdout(io.StringIO()):
            report = converter.convert(
                arguments(source, self.root / "out", ["--counting"])
            )
        self.assertEqual(report["bin"], 1)
        self.assertEqual(report["storage_pedestal"], 0)
        self.assertEqual(report["xds_offset"], 0)

    def test_failure_never_writes_completion_marker(self):
        with patch.object(
            converter.CbfImage, "write", side_effect=OSError("disk full")
        ):
            with self.assertRaises(OSError):
                self.run_conversion()
        self.assertFalse((self.root / "out" / "conversion.json").exists())

    def test_copyable_import_helper(self):
        self.run_conversion()
        work = self.root / "student work"
        work.mkdir()
        helper = work / "import_dials.py"
        helper.write_bytes((self.root / "out" / "import_dials.py").read_bytes())
        result = subprocess.run(
            [
                sys.executable,
                str(helper),
                "--gain",
                "2.5",
                "--pedestal",
                "7",
                "--dry-run",
            ],
            cwd=work,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(self.root / "out" / "image_000001.cbf"), result.stdout)
        self.assertIn("panel.gain=2.5", result.stdout)
        self.assertIn("panel.pedestal=7.0", result.stdout)
        self.assertFalse((work / "imported.expt").exists())

    def test_gain_from_matching_xds_init(self):
        report = self.run_conversion()
        init = self.root / "INIT.LP"
        init.write_text(
            f"NAME_TEMPLATE_OF_DATA_FRAMES={self.root.resolve()}/out/image_??????.cbf\n"
            f"DARK CURRENT LOOK-UP TABLE IS SET CONSTANT TO {report['storage_pedestal']}\n"
            "MEAN GAIN VALUE 12.5\n"
        )
        command = [
            sys.executable,
            str(self.root / "out" / "import_dials.py"),
            "--gain-from-init",
            str(init),
            "--dry-run",
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("panel.gain=12.5", result.stdout)
        init.write_text(
            init.read_text().replace(
                "DARK CURRENT LOOK-UP TABLE IS SET CONSTANT TO",
                'LOOK-UP TABLE "BLANK.cbf" FOR DARK CURRENT IS SET CONSTANT TO OFFSET=',
            )
        )
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        init.write_text(init.read_text().replace("/out/", "/different/"))
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_manual_beam_with_masked_centre(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[16:48, 16:48] = True
        path = self.root / "mask.npy"
        np.save(path, mask)
        report = self.run_conversion(extra=["--bad-pixels", str(path)])
        self.assertEqual(report["status"], "complete")


if __name__ == "__main__":
    unittest.main()
