"""Run with dials.python after importing a converted dataset using its helper.

Usage: dials.python tests/check_dials.py imported.expt conversion.json XDS.INP
"""

import json
from pathlib import Path
import sys

import numpy as np
from dxtbx.model.experiment_list import ExperimentListFactory
from dxtbx.serialize.xds import to_xds
from iotbx.xds import xds_inp

experiments = ExperimentListFactory.from_json_file(sys.argv[1], check_format=True)
report = json.loads(Path(sys.argv[2]).read_text())
inp = xds_inp.reader()
inp.read_file(sys.argv[3])
assert len(experiments) == 1
experiment = experiments[0]
assert experiment.beam.get_probe_name() == "electron"
panel = experiment.detector[0]
assert panel.get_px_mm_strategy().name() == "SimplePxMmStrategy"
assert panel.get_thickness() == 0
assert panel.get_pedestal() == report["storage_pedestal"]
assert len(experiment.imageset) == report["output_shape"][0]
exported = to_xds(experiment.imageset)
origin = np.array(panel.get_origin())
px, py = panel.get_pixel_size()
derived_origin = [
    -np.dot(origin, panel.get_fast_axis()) / px + 0.5,
    -np.dot(origin, panel.get_slow_axis()) / py + 0.5,
]
np.testing.assert_allclose(derived_origin, [inp.orgx, inp.orgy], atol=1e-7)
if np.isclose(px, py) or np.allclose(exported.detector_origin, derived_origin):
    np.testing.assert_allclose(exported.detector_origin, derived_origin, atol=1e-7)
else:
    # Some older dxtbx exporters divide the slow-axis distance by the fast pixel pitch.
    np.testing.assert_allclose(
        exported.detector_origin,
        [derived_origin[0], (derived_origin[1] - 0.5) * py / px + 0.5],
        atol=1e-7,
    )
    print(
        "NOTE: installed dxtbx exporter has the y/fast-pitch bug for rectangular pixels; direct detector geometry agrees"
    )
for x, y in [(0, 0), (inp.nx - 1, 0), (0, inp.ny - 1)]:
    ray = np.array(panel.get_pixel_lab_coord((x + 0.5, y + 0.5))) * [1, -1, -1]
    expected = [
        (x + 1 - inp.orgx) * inp.px,
        (y + 1 - inp.orgy) * inp.py,
        inp.detector_distance,
    ]
    np.testing.assert_allclose(ray, expected, atol=1e-8)
np.testing.assert_allclose(exported.rotation_axis, inp.rotation_axis, atol=1e-10)
np.testing.assert_allclose(exported.pixel_size, [inp.px, inp.py], atol=1e-12)
np.testing.assert_allclose(exported.detector_distance, inp.detector_distance, atol=1e-8)
np.testing.assert_allclose(exported.starting_angle, inp.starting_angle, atol=1e-8)
np.testing.assert_allclose(
    exported.oscillation_range, inp.oscillation_range, atol=1e-10
)
np.testing.assert_allclose(exported.wavelength, inp.xray_wavelength, atol=1e-10)
np.testing.assert_allclose(
    exported.beam_vector, inp.incident_beam_direction, atol=1e-10
)
for idx in sorted({0, len(experiment.imageset) // 2, len(experiment.imageset) - 1}):
    raw = experiment.imageset.get_raw_data(idx)[0].as_numpy_array()
    corrected = experiment.imageset.get_corrected_data(idx)[0].as_numpy_array()
    valid = experiment.imageset.get_mask(idx)[0].as_numpy_array()
    assert not np.any(valid & (raw < 0)), (
        "Invalid padding/defects became valid in DIALS"
    )
    np.testing.assert_allclose(
        corrected[valid],
        (raw[valid] - panel.get_pedestal()) / panel.get_gain(),
        atol=1e-10,
    )
    expected_negative = valid & (raw < panel.get_pedestal())
    assert np.all(corrected[expected_negative] < 0), (
        "Signed noise was clipped after pedestal subtraction"
    )
print(
    "PASS: stock DIALS geometry, signed-pixel recovery, masks, and XDS laboratory rays"
)
