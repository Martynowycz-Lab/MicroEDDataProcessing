"""Exercise file discovery, conversion, and missing metadata through public entry points."""

import contextlib
import csv
import errno
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import fabio
import mrcfile
from mrcfile.dtypes import get_ext_header_dtype
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mrc2cbf as converter
from inputs import find_inputs
from metadata import FEI_FIELDS, read_fei


def write_mrc(path, pixels, extended=None):
    with mrcfile.new(path) as image:
        image.set_data(pixels)
        if pixels.ndim == 3:
            image.set_image_stack()
        if extended is not None:
            image.set_extended_header(extended)
            image.header.exttyp = b"FEI2"
    return path


def fei_header(count=1, **overrides):
    extended = np.zeros(count, dtype=get_ext_header_dtype(b"FEI2", "<"))
    extended["Metadata size"] = extended.dtype.itemsize
    extended["Metadata version"] = 2
    values = {
        "HT": 200000,
        "Camera length": 1.5,
        "Integration time": 0.1,
        "Binning Width": 2,
        "Binning Height": 2,
        "Camera name": b"BM-Ceta",
        "Start tilt angle": -40,
        "End tilt angle": 40,
        "Tilt per image": 0.2,
        "Tilt speed": 2,
    }
    values.update(overrides)
    for key, value in values.items():
        mask, bit = FEI_FIELDS[key]
        extended[f"Bitmask {mask}"] |= 1 << bit
        extended[key] = value
    return extended


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.pixels = np.arange(-100, 1436, dtype=np.int16).reshape(32, 48)

    def tearDown(self):
        self.temp.cleanup()

    def image(self, name, pixels=None, **kwargs):
        return write_mrc(
            self.raw / name, self.pixels if pixels is None else pixels, **kwargs
        )

    def run_conversion(self, inputs=None, output="cbf", flags=()):
        args = converter.parser().parse_args(
            [
                *(map(str, inputs) if inputs is not None else [str(self.raw)]),
                "-o",
                str(self.root / output),
                "--trim",
                "none",
                *flags,
            ]
        )
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            report = converter.convert(args)
        return report, stream.getvalue()

    def test_directory_2d_and_singleton_3d_roundtrip(self):
        self.image("frame_001.mrc")
        self.image("frame_002.mrc", self.pixels[None])
        before = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.raw.iterdir()
        }
        report, _ = self.run_conversion()
        self.assertEqual(report["output_shape"], [2, 32, 48])
        self.assertIsNone(report["source"])
        for path in sorted((self.root / "cbf").glob("*.cbf")):
            with fabio.open(str(path)) as image:
                np.testing.assert_array_equal(
                    image.data - report["storage_pedestal"], self.pixels
                )
        for path, digest in before.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("check_pixels.py")),
                str(self.root / "cbf/conversion.json"),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_natural_order_and_gaps_are_not_compressed(self):
        for number in (10, 2, 1):
            self.image(f"frame{number}.mrc", self.pixels + number)
        report, _ = self.run_conversion(
            flags=["--start-angle-deg", "-10", "--angle-step-deg", "0.5"]
        )
        with (self.root / "cbf/frames.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            [Path(row["input_file"]).name for row in rows],
            ["frame1.mrc", "frame2.mrc", "frame10.mrc"],
        )
        self.assertEqual(
            [float(row["angle_start_deg"]) for row in rows], [-10, -9.5, -5.5]
        )
        self.assertFalse(report["processing_helpers"])
        self.assertIn("consecutive input frame numbers", report["helper_missing"])

    def test_shell_and_quoted_globs_match(self):
        for number in (1, 2, 3):
            self.image(f"frame{number}.mrc")
        script = str(Path(converter.__file__))
        # shell expansion without a shell: pass the same expanded argument list.
        for name, inputs in (
            ("quoted", [str(self.raw / "*.mrc")]),
            ("expanded", list(map(str, self.raw.glob("*.mrc")))),
        ):
            result = subprocess.run(
                [
                    sys.executable,
                    script,
                    *inputs,
                    "-o",
                    str(self.root / name),
                    "--trim",
                    "none",
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        for number in (1, 2, 3):
            name = f"image_{number:06d}.cbf"
            self.assertEqual(
                (self.root / "quoted" / name).read_bytes(),
                (self.root / "expanded" / name).read_bytes(),
            )

    def test_literal_and_glob_output_names(self):
        first = self.image("frame1.mrc")
        self.image("frame2.mrc")
        report, _ = self.run_conversion(output="named/*.cbf")
        self.assertEqual(report["output_shape"][0], 2)
        self.assertTrue((self.root / "named/frame1.cbf").is_file())
        self.assertTrue((self.root / "named/frame2.cbf").is_file())
        self.assertTrue((self.root / "named/mrc2cbf_report/conversion.json").is_file())
        self.run_conversion([first], output="one.cbf")
        with fabio.open(str(self.root / "one.cbf")) as image:
            np.testing.assert_array_equal(image.data - 100, self.pixels)

    def test_input_output_glob_positional_form(self):
        self.image("frame1.mrc")
        self.image("frame2.mrc")
        args = converter.parser().parse_args(
            [str(self.raw / "*.mrc"), str(self.root / "named/*.cbf"), "--trim", "none"]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            converter.convert(args)
        self.assertTrue((self.root / "named/frame2.cbf").exists())

    def test_stack_glob_keeps_sweeps_separate(self):
        for name, value in (("one", 0), ("two", 300)):
            self.image(name + ".mrc", np.stack([self.pixels + value] * 3))
        report, _ = self.run_conversion([self.raw / "*.mrc"])
        self.assertEqual(len(report["sweeps"]), 2)
        first = json.loads((self.root / "cbf/one/conversion.json").read_text())
        second = json.loads((self.root / "cbf/two/conversion.json").read_text())
        self.assertEqual(first["storage_pedestal"], 100)
        self.assertEqual(second["storage_pedestal"], 0)
        self.assertTrue((self.root / "cbf/batch.json").is_file())

    def test_different_prefixes_are_separate_sweeps(self):
        for name in ("a1", "a2", "b1", "b2"):
            self.image(name + ".mrc")
        report, _ = self.run_conversion()
        self.assertEqual([item["output"] for item in report["sweeps"]], ["a", "b"])

    def test_duplicate_matches_are_read_once(self):
        path = self.image("frame1.mrc")
        paths = find_inputs([str(path), str(self.raw), str(self.raw / "*.mrc")])
        self.assertEqual(paths, [path.resolve()])

    def test_duplicate_frame_numbers_fail(self):
        self.image("frame1.mrc")
        self.image("frame01.mrc")
        with self.assertRaisesRegex(ValueError, "Duplicate frame numbers"):
            self.run_conversion()
        self.assertFalse((self.root / "cbf").exists())

    def test_shape_and_datatype_changes_fail(self):
        self.image("frame1.mrc")
        for pixels in (self.pixels[:16], self.pixels.astype(np.float32)):
            with self.subTest(shape=pixels.shape, dtype=pixels.dtype):
                second = self.raw / "frame2.mrc"
                if second.exists():
                    second.unlink()
                write_mrc(second, pixels)
                with self.assertRaisesRegex(ValueError, "size or datatype"):
                    self.run_conversion()

    def test_empty_directory_missing_file_and_unmatched_glob_fail(self):
        for inputs, message in (
            ([self.raw], "No MRC files"),
            ([self.raw / "missing.mrc"], "does not exist"),
            ([self.raw / "*.mrc"], "No input matches"),
        ):
            with (
                self.subTest(inputs=inputs),
                self.assertRaisesRegex((ValueError, OSError), message),
            ):
                self.run_conversion(inputs)

    def test_truncated_pixels_and_volume_fail(self):
        path = self.image("frame1.mrc")
        path.write_bytes(path.read_bytes()[:-12])
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.run_conversion()
        path.unlink()
        self.image("volume.mrc", np.ones((3, 8, 8), dtype=np.float32))
        with mrcfile.open(self.raw / "volume.mrc", mode="r+") as image:
            image.set_volume()
        with self.assertRaisesRegex(ValueError, "volume"):
            self.run_conversion()

    def test_cannot_overwrite_literal_or_pattern_target(self):
        self.image("frame1.mrc")
        target = self.root / "frame1.cbf"
        target.write_text("keep")
        for output in ("frame1.cbf", "*.cbf"):
            with (
                self.subTest(output=output),
                self.assertRaisesRegex(ValueError, "already exists"),
            ):
                self.run_conversion(output=output)
        self.assertEqual(target.read_text(), "keep")

    def test_atomic_publish_does_not_replace_racing_output(self):
        self.image("frame1.mrc")
        target = self.root / "frame1.cbf"
        original = converter.CbfImage.write

        def write(image, path):
            original(image, path)
            target.write_text("someone else's output")

        with (
            patch.object(converter.CbfImage, "write", write),
            self.assertRaises(FileExistsError),
        ):
            self.run_conversion(output="frame1.cbf")
        self.assertEqual(target.read_text(), "someone else's output")
        self.assertFalse((self.root / "frame1.cbf.report/conversion.json").exists())

    def test_partial_batch_continues_after_one_failure(self):
        self.image("a.mrc", np.ones((3, 32, 48), dtype=np.int16))
        self.image("b.mrc", np.full((3, 32, 48), -1, dtype=np.int16))
        with self.assertRaisesRegex(ValueError, "1 of 2 sweeps failed"):
            self.run_conversion(flags=["--counting"])
        self.assertTrue((self.root / "cbf/a/conversion.json").is_file())
        self.assertFalse((self.root / "cbf/b/conversion.json").is_file())
        self.assertEqual(
            json.loads((self.root / "cbf/batch.json").read_text())["status"], "partial"
        )

    def test_two_input_directories_are_not_a_positional_output(self):
        self.image("one.mrc")
        other = self.root / "other"
        other.mkdir()
        write_mrc(other / "two.mrc", self.pixels)
        # An explicit output removes all ambiguity in the legacy positional form.
        report, _ = self.run_conversion([self.raw, other])
        self.assertEqual(len(report["sweeps"]), 2)

    def test_output_pattern_splits_stack_with_source_names(self):
        path = self.image("stack.mrc", np.stack([self.pixels, self.pixels + 1]))
        report, _ = self.run_conversion([path], output="named/out_*.cbf")
        self.assertEqual(report["output_shape"][0], 2)
        for idx in range(2):
            with fabio.open(
                str(self.root / f"named/out_stack_{idx + 1:06d}.cbf")
            ) as image:
                np.testing.assert_array_equal(image.data - 100, self.pixels + idx)

    def test_spaces_and_unicode_do_not_block_pixel_conversion(self):
        self.image("frame1.mrc")
        report, _ = self.run_conversion(output="space folder/image µ.cbf")
        self.assertEqual(report["status"], "complete")
        with fabio.open(str(self.root / "space folder/image µ.cbf")) as image:
            np.testing.assert_array_equal(image.data - 100, self.pixels)

    def test_parallel_pattern_output_with_spaces_in_filenames(self):
        self.image("frame 1.mrc")
        self.image("frame 2.mrc", self.pixels + 1)
        self.run_conversion(output="named/*.cbf")
        for idx in (1, 2):
            with fabio.open(str(self.root / f"named/frame {idx}.cbf")) as image:
                np.testing.assert_array_equal(image.data - 100, self.pixels + idx - 1)

    def test_removable_drive_without_hard_links(self):
        self.image("frame1.mrc")
        with patch.object(
            converter.os, "link", side_effect=OSError(errno.EOPNOTSUPP, "not supported")
        ):
            report, _ = self.run_conversion()
        with fabio.open(str(self.root / "cbf/image_000001.cbf")) as image:
            np.testing.assert_array_equal(
                image.data - report["storage_pedestal"], self.pixels
            )

    def test_malformed_xml_is_a_short_cli_error(self):
        path = self.image("frame1.mrc")
        (self.raw / "frame1.xml").write_text("<Root>")
        result = subprocess.run(
            [
                sys.executable,
                str(Path(converter.__file__)),
                str(path),
                "-o",
                str(self.root / "cbf"),
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Malformed XML", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.root / "cbf/conversion.json").exists())

    def test_two_directories_without_output_use_default_directory(self):
        self.image("one.mrc")
        other = self.root / "other"
        other.mkdir()
        write_mrc(other / "two.mrc", self.pixels)
        result = subprocess.run(
            [
                sys.executable,
                str(Path(converter.__file__)),
                str(self.raw),
                str(other),
                "--trim",
                "none",
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "cbf/one/conversion.json").is_file())
        self.assertTrue((self.root / "cbf/two/conversion.json").is_file())

    def test_numbered_subset_does_not_reuse_series_start_angle(self):
        self.image("frame20.mrc", extended=fei_header())
        report, _ = self.run_conversion()
        self.assertIsNone(report["geometry"]["start_angle_deg"])
        report, _ = self.run_conversion(
            output="explicit", flags=["--start-angle-deg", "-36.2"]
        )
        self.assertEqual(report["geometry"]["start_angle_deg"], -36.2)

    def test_source_mutation_invalidates_completion(self):
        path = self.image("frame1.mrc")
        original = converter.CbfImage.write

        def write(image, target):
            original(image, target)
            path.touch()

        with (
            patch.object(converter.CbfImage, "write", write),
            self.assertRaisesRegex(ValueError, "Input changed"),
        ):
            self.run_conversion()
        self.assertFalse((self.root / "cbf/conversion.json").exists())

    def test_missing_metadata_does_not_invent_geometry(self):
        self.image("frame1.mrc")
        report, printed = self.run_conversion()
        self.assertFalse(report["metadata_complete"])
        self.assertIn("distance_mm", report["missing_metadata"])
        with fabio.open(str(self.root / "cbf/image_000001.cbf")) as image:
            header = image.header["_array_data.header_contents"]
            for key in (
                "Pixel_size",
                "Detector_distance",
                "Wavelength",
                "Beam_xy",
                "Start_angle",
                "Angle_increment",
                "Exposure_time",
            ):
                self.assertNotIn(key, header)
        self.assertIn("Missing metadata", printed)
        self.assertFalse((self.root / "cbf/XDS.INP").exists())

    def test_default_trim_keeps_frames_without_reliable_illumination(self):
        self.image("frame1.mrc", np.zeros((32, 48), dtype=np.int16))
        report, printed = self.run_conversion(flags=["--trim", "auto"])
        self.assertEqual(report["trim"], "none")
        self.assertEqual(report["output_shape"][0], 1)
        self.assertIn("all selected frames were kept", printed)

    def test_explicit_center_and_strict_geometry_fail_if_unknown(self):
        self.image("frame1.mrc")
        for flag in ("--center", "--require-geometry"):
            with (
                self.subTest(flag=flag),
                self.assertRaisesRegex(
                    ValueError, "beam centre|geometry is incomplete"
                ),
            ):
                self.run_conversion(output=flag[2:], flags=[flag])
            self.assertFalse((self.root / flag[2:] / "conversion.json").exists())

    def test_fei_and_epu_metadata_resolve_with_presence_flags(self):
        for number in range(3):
            self.image(f"frame{number}.mrc", extended=fei_header())
        (self.raw / "frame.xml").write_text(
            "<AcquisitionDataXml><StartAngle>-40</StartAngle><TiltPerImage>0.2</TiltPerImage><TiltSpeed>2</TiltSpeed></AcquisitionDataXml>"
        )
        report, _ = self.run_conversion(flags=["--beam-center", "23", "15"])
        meta = report["geometry"]
        self.assertEqual(meta["input_pixel_size_mm"], [0.028, 0.028])
        self.assertAlmostEqual(meta["distance_mm"], 1500)
        self.assertAlmostEqual(meta["frame_time_s"], 0.1)
        self.assertAlmostEqual(meta["wavelength_a"], 0.02507934, places=7)
        self.assertTrue(report["processing_helpers"])
        self.assertIn("nominal FEI", " ".join(meta["notes"]))
        path = self.raw / "frame0.mrc"
        with mrcfile.open(path, mode="r+") as image:
            image.indexed_extended_header["Bitmask 1"] &= np.uint32(
                0xFFFFFFFF ^ (1 << 5)
            )
        self.assertNotIn("HT", read_fei([path]))

    def test_fei_conflicts_and_unsupported_xml(self):
        self.image("frame1.mrc", extended=fei_header())
        self.image("frame2.mrc", extended=fei_header(HT=300000))
        with self.assertRaisesRegex(ValueError, "HT changes"):
            self.run_conversion()
        (self.raw / "frame2.mrc").unlink()
        (self.raw / "frame1.xml").write_text("<Unrecognized/>")
        report, printed = self.run_conversion()
        self.assertIn("XML format is not supported", printed)
        self.assertEqual(report["status"], "complete")

    def test_epu_xml_conflict_is_not_treated_as_missing(self):
        self.image("frame1.mrc", extended=fei_header())
        (self.raw / "frame1.xml").write_text(
            "<AcquisitionDataXml><StartAngle>-20</StartAngle></AcquisitionDataXml>"
        )
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.run_conversion()

    def test_unsupported_mdoc_layout_keeps_pixels_and_reports_limit(self):
        path = self.image("frame1.mrc")
        Path(str(path) + ".mdoc").write_text(
            "[ZValue = 0]\nTiltAngle = -10\n[ZValue = 1]\nTiltAngle = 20\n"
        )
        report, printed = self.run_conversion()
        self.assertEqual(report["status"], "complete")
        self.assertIsNone(report["geometry"]["start_angle_deg"])
        self.assertIn("MDOC layout is not supported", printed)

    def test_sidecar_mutation_invalidates_completion(self):
        path = self.image("frame1.mrc")
        sidecar = Path(str(path) + ".mdoc")
        sidecar.write_text("Voltage = 200\n")
        original = converter.CbfImage.write

        def write(image, target):
            original(image, target)
            sidecar.write_text("Voltage = 300\n")

        with (
            patch.object(converter.CbfImage, "write", write),
            self.assertRaisesRegex(ValueError, "Metadata or bad-pixel mask changed"),
        ):
            self.run_conversion()
        self.assertFalse((self.root / "cbf/conversion.json").exists())

    def velox_stack(self, name, count, period, suffix=".xml", offset=0):
        from datetime import datetime, timedelta, timezone

        source = self.image(
            name, np.stack([self.pixels + offset + idx for idx in range(count)])
        )
        path = (
            Path(str(source) + ".xml")
            if suffix == ".mrc.xml"
            else source.with_suffix(".xml")
        )
        fractions = []
        for idx in range(count):
            stamp = datetime(2026, 11, 5, tzinfo=timezone.utc) + timedelta(
                seconds=idx * period
            )
            fractions.append(
                f"<Fraction><Index>{idx}</Index><StartFrame>{idx * 3}</StartFrame><NumberOfFrames>3</NumberOfFrames><ExposureTime>{period}</ExposureTime><DateTimeWithTimeZone>{stamp.isoformat()}</DateTimeWithTimeZone></Fraction>"
            )
        path.write_text(
            f'<Root xmlns="urn:velox"><Info><CommercialName>Falcon 4</CommercialName><Binning>2</Binning><ExpectedNumberOfFractions>{count}</ExpectedNumberOfFractions><RecordedNumberOfFractions>{count}</RecordedNumberOfFractions><ImageSize><Width>48</Width><Height>32</Height></ImageSize></Info><Fractions>{"".join(fractions)}</Fractions></Root>'
        )
        return source, path

    def test_velox_irregular_timing_keeps_pixels_and_actual_exposures(self):
        import xml.etree.ElementTree as ET

        for case in ("exposure", "grouping", "timestamps"):
            with self.subTest(case=case):
                source, xml = self.velox_stack(case + ".mrc", 3, 0.1)
                root = ET.fromstring(xml.read_text())
                for node in root.iter():
                    node.tag = node.tag.rsplit("}", 1)[-1]
                rows = root.findall("Fractions/Fraction")
                if case == "exposure":
                    rows[2].find("ExposureTime").text = "0.05"
                elif case == "grouping":
                    rows[2].find("NumberOfFrames").text = "2"
                else:
                    rows[2].find(
                        "DateTimeWithTimeZone"
                    ).text = "2026-11-05T00:00:00.25+00:00"
                xml.write_text(ET.tostring(root, encoding="unicode"))
                report, summary = self.run_conversion([source], output=case)
                self.assertEqual(report["status"], "complete")
                self.assertFalse(report["processing_helpers"])
                self.assertIsNone(report["geometry"]["frame_time_s"])
                self.assertIsNone(report["geometry"]["angle_step_deg"])
                self.assertIn("unresolved", summary)
                for idx, path in enumerate(sorted((self.root / case).glob("*.cbf"))):
                    with fabio.open(str(path)) as image:
                        np.testing.assert_array_equal(
                            image.data - report["storage_pedestal"], self.pixels + idx
                        )
                        header = image.header["_array_data.header_contents"]
                        expected = 0.05 if case == "exposure" and idx == 2 else 0.1
                        self.assertIn(f"# Exposure_time {expected} s", header)
                        self.assertNotIn("# Angle_increment", header)
                        self.assertNotIn("# Exposure_period", header)
                selected, _ = self.run_conversion(
                    [source], output=case + "_selected", flags=("--frames", "3", "3")
                )
                selected_path = self.root / (case + "_selected") / "image_000001.cbf"
                with fabio.open(str(selected_path)) as image:
                    np.testing.assert_array_equal(
                        image.data - selected["storage_pedestal"], self.pixels + 2
                    )
                    expected = 0.05 if case == "exposure" else 0.1
                    self.assertIn(
                        f"# Exposure_time {expected} s",
                        image.header["_array_data.header_contents"],
                    )
                with self.assertRaisesRegex(ValueError, "geometry is incomplete"):
                    self.run_conversion(
                        [source], output=case + "_strict", flags=("--require-geometry",)
                    )

    def test_velox_only_stacks_with_quoted_and_shell_globs(self):
        self.velox_stack("stack1.mrc", 3, 0.1)
        self.velox_stack("stack2.mrc", 4, 0.2, offset=50)
        before = {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.raw.iterdir()
        }
        script = str(Path(converter.__file__))
        commands = [
            [
                sys.executable,
                script,
                "*.mrc",
                "-o",
                "quoted",
                "--bin",
                "1",
                "--trim",
                "none",
            ],
            [
                "/bin/bash",
                "-c",
                'exec "$1" "$2" *.mrc -o expanded --bin 1 --trim none',
                "mrc2cbf-test",
                sys.executable,
                script,
            ],
        ]
        for command in commands:
            result = subprocess.run(
                command,
                cwd=self.raw,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        for name, count, period, pedestal in (
            ("stack1", 3, 0.1, 100),
            ("stack2", 4, 0.2, 50),
        ):
            report = json.loads(
                (self.raw / "quoted" / name / "conversion.json").read_text()
            )
            self.assertEqual(report["output_shape"], [count, 32, 48])
            self.assertEqual(report["storage_pedestal"], pedestal)
            self.assertAlmostEqual(report["geometry"]["frame_time_s"], period)
            self.assertEqual(report["geometry"]["pixel_size_mm"], [0.028, 0.028])
            for key in (
                "distance_mm",
                "wavelength_a",
                "start_angle_deg",
                "angle_step_deg",
            ):
                self.assertIsNone(report["geometry"][key])
                self.assertIn(key, report["missing_metadata"])
            self.assertFalse(report["processing_helpers"])
            self.assertEqual(len(report["sidecars"]), 1)
            for idx in range(count):
                cbf = f"image_{idx + 1:06d}.cbf"
                self.assertEqual(
                    (self.raw / "quoted" / name / cbf).read_bytes(),
                    (self.raw / "expanded" / name / cbf).read_bytes(),
                )
                with fabio.open(str(self.raw / "quoted" / name / cbf)) as image:
                    np.testing.assert_array_equal(
                        image.data - pedestal,
                        self.pixels + (50 if name == "stack2" else 0) + idx,
                    )
        for path, digest in before.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_velox_mrc_xml_name_preserves_dotted_stem(self):
        source, xml = self.velox_stack("scan.001.mrc", 3, 0.1, suffix=".mrc.xml")
        # This unrelated XML must not match the shortened 'scan' stem.
        (self.raw / "scan.xml").write_text("<malformed>")
        report, _ = self.run_conversion([source])
        self.assertEqual(set(report["sidecars"]), {str(xml.resolve())})
        self.assertAlmostEqual(report["geometry"]["frame_time_s"], 0.1)

    def test_generic_velox_xml_does_not_block_two_stacks(self):
        for name in ("one.mrc", "two.mrc"):
            source = self.image(name, np.stack([self.pixels] * 2))
            source.with_suffix(".xml").write_text(
                "<MicroscopeImage><Metadata/></MicroscopeImage>"
            )
        report, printed = self.run_conversion()
        self.assertEqual(report["status"], "complete")
        self.assertIn("XML format is not supported", printed)
        for name in ("one", "two"):
            result = json.loads(
                (self.root / "cbf" / name / "conversion.json").read_text()
            )
            self.assertEqual(result["output_shape"][0], 2)
            self.assertIn("distance_mm", result["missing_metadata"])

    def test_invalid_explicit_metadata_is_not_treated_as_missing(self):
        self.image("frame1.mrc")
        for flags in (
            ["--distance-mm", "0"],
            ["--wavelength-a", "nan"],
            ["--angle-step-deg", "0"],
            ["--pixel-size-mm", "-1", "0.1"],
        ):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                self.run_conversion(flags=flags)


if __name__ == "__main__":
    unittest.main()
