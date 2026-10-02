"""Independently verify every CBF against the original movie and frame manifest.

Usage: python tests/check_pixels.py /path/to/conversion.json
"""

import csv
import hashlib
import json
from pathlib import Path
import sys

import fabio
import mrcfile
import numpy as np

path = Path(sys.argv[1])
report = json.loads(path.read_text())
rows = list(csv.DictReader((path.parent / "frames.csv").open()))
assert len(rows) == report["output_shape"][0]
assert len(list(path.parent.glob("image_*.cbf"))) == len(rows)
factor = report["bin"]
pedestal = report["storage_pedestal"]
mask = np.load(report["bad_pixels"]) if report.get("bad_pixels") else None
height, width = report["output_shape"][1:]
dy, dx = report["beam_shift_xy"][::-1]
yy, xx = np.indices((height, width))
source_y, source_x = yy - dy, xx - dx
inside = (source_y >= 0) & (source_y < height) & (source_x >= 0) & (source_x < width)
negative_pixels = 0
with mrcfile.mmap(report["source"], mode="r") as movie:
    for idx, row in enumerate(rows):
        input_idx = int(row["input_frame"]) - 1
        assert input_idx == report["input_frames"][0] - 1 + idx
        raw = movie.data[input_idx].astype(np.float64)
        invalid = ~np.isfinite(raw)
        if mask is not None:
            invalid |= mask
        if report["saturation"] is not None:
            invalid |= raw >= report["saturation"]
        raw[invalid] = 0
        # Strided sums provide an independent reference for reshape-based binning.
        total = sum(
            raw[y::factor, x::factor] for y in range(factor) for x in range(factor)
        )
        bad = np.logical_or.reduce(
            [
                invalid[y::factor, x::factor]
                for y in range(factor)
                for x in range(factor)
            ]
        )
        expected = np.full((height, width), -1, dtype=np.int64)
        stored = np.rint(total).astype(np.int64) + pedestal
        stored[bad] = -1
        expected[inside] = stored[source_y[inside], source_x[inside]]
        cbf = path.parent / f"image_{idx + 1:06d}.cbf"
        assert hashlib.sha256(cbf.read_bytes()).hexdigest() == row["sha256"]
        with fabio.open(str(cbf)) as image:
            np.testing.assert_array_equal(image.data, expected)
        negative_pixels += int(
            np.count_nonzero((expected >= 0) & (expected < pedestal))
        )
        expected_angle = (
            report["geometry"]["start_angle_deg"]
            + idx * report["geometry"]["angle_step_deg"]
        )
        assert abs(float(row["angle_start_deg"]) - expected_angle) < 1e-9
print(
    f"PASS: {len(rows)} frames, all pixels/hashes/indices/angles; {negative_pixels} signed-negative pixels preserved"
)
