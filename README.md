# MicroEDDataProcessing

`mrc2cbf.py` (v9.0.1) converts MRC diffraction movies to miniCBF on Linux and macOS.
It uses `metadata.py` to read one SerialEM movie and optional Velox XML timing.
The published `mrc2cbf_pipeline_v3.py` is unchanged. Its original instructions
and the local v7/v8 code are preserved in [legacy](legacy/README.md).

The converter sum-bins pixels, trims weak ends, estimates a static beam centre,
and adds a reversible storage pedestal. It does not subtract a radial background,
filter diffraction intensities, estimate gain, or tune offsets to improve
integration statistics. Every written CBF is decoded and checked against the
intended integer pixels before the conversion is marked complete.

## Install

Python 3.10 or newer:

```bash
git clone https://github.com/Martynowycz-Lab/MicroEDDataProcessing.git
cd MicroEDDataProcessing
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Keep `mrc2cbf.py` and `metadata.py` together. DIALS and XDS are separate programs;
neither is needed for conversion. No custom DIALS format plugin is required.

## Convert

Matching `movie.mrc.mdoc` and `movie.xml` sidecars are found automatically:

```bash
python mrc2cbf.py movie.mrc cbf
```

Pixel size is resolved from MDOC `CameraPixelSize` in micrometres or the recognized
Falcon/Ceta detector and acquisition binning, as in v7. A stored-pixel value such
as 28 micrometres at binning 2 is used once, not multiplied by binning again.
A value matching the known 14-micrometre native pitch is multiplied by the stated
acquisition binning once, with a message. When the pitch is absent, the known
detector uses MDOC/XML binning, then XML ROI/MRC dimensions. A last-resort 4096-pixel
full-sensor assumption produces a cropping warning. Unknown detectors can use an
explicit `CameraPixelSize`, with a warning about the assumed stored-pixel units.
If no usable pitch can be found, the error asks for `--pixel-size-mm X Y`.

When no rotation axis is supplied, the converter uses the **v7 lab default**
`0.999263 -0.0383878 0` and prints that **it may be wrong for your setup**. This is
a fallback, not an axis measured from MDOC/XML or the diffraction patterns. Its
source is recorded in `conversion.json`. Explicit overrides always take precedence:

```bash
python mrc2cbf.py movie.mrc cbf \
  --pixel-size-mm 0.028 0.028 --rotation-axis 1 0 0
```

Pixel overrides describe **stored MRC pixels**, including acquisition binning.
The axis uses the DIALS frame: fast +x, slow -y, beam travelling -z. Other
microscopes should use their calibrated axis. `PixelSpacing` and `RotationRate`
are still not interpreted as detector pitch or calibrated stage speed. Metadata
sources and fallback warnings are printed before frame processing.

MDOC supplies distance, voltage, start angle and `DegreesPerSecond` when available.
`TiltAngle` is interpreted as the start of input frame 1, not its centre; confirm
this convention in your acquisition software. XML counts, dimensions, exposure
durations, timestamps and detector-frame continuity are checked. Inconsistent
sidecars and nonuniform scans stop conversion. Only a single MDOC FrameSet movie
is supported; multi-image tilt-series autodocs must not be flattened into it.
Explicit sidecar paths can be selected with `--mdoc` and `--xml`.
Exposure agreement uses the larger of 100 microseconds and 0.1% of a frame;
timestamp jitter about a fitted constant period is bounded by the larger of
10 microseconds and 0.5% of a frame. Both measured residuals and the timestamp
tolerance are recorded; this is a tested constant-wedge approximation, not a
measurement of stage motion during every exposure.
CBF exposure duration and frame period are recorded separately when XML provides them.

Without sidecars, supply the complete geometry:

```bash
python mrc2cbf.py movie.mrc cbf \
  --pixel-size-mm 0.028 0.028 --rotation-axis 0.999263 -0.0383878 0 \
  --distance-mm 961.06 --voltage-kv 200 \
  --start-angle-deg -60 --angle-step-deg 0.169 --frame-time-s 0.08444
```

`--wavelength-a` can replace `--voltage-kv`. The output directory must be new.
Raw inputs are opened read-only. Failures return a nonzero exit code and leave
partial output for inspection, without a `conversion.json` completion record.
Choose a fresh directory to retry. Existing files are never overwritten.

## Pixels And Offset

Additional integer sum-binning targets at most 1024 pixels per axis by default.
Smaller images are not enlarged. `--bin 1` preserves size; `--bin 2` sums 2 by 2
blocks. Both dimensions must divide exactly. No detector rows/columns or partial
time bins are silently dropped; temporal binning is not implemented.

Integer sums are exact. Floating-point input is summed in float64, then rounded
once to the nearest integer (ties to even); the maximum rounding error is reported.
`--saturation ADU` marks input values at or above a calibrated limit invalid.
Otherwise the integer datatype ceiling is used; float input has no inferred
saturation limit. `--bad-pixels mask.npy` takes a boolean, input-sized mask where
True means invalid. A bin containing any invalid input is invalid. Invalid pixels
are stored as -1, not filled or included in the pedestal estimate.

For valid binned values Y, `CBF = round(Y) + pedestal`. The automatic pedestal is
`max(0, -min(round(Y)))` over retained valid pixels. `--pedestal N` supplies another
value, but it must preserve every valid pixel. There is no quantile clipping or
automatic classification of negative tails as detector defects.

**The exact value added is written as XDS OFFSET and DIALS panel pedestal.**
Both XDS input files use the same number. XDS GAIN remains unset so INIT can
estimate it. MDOC/XML gain is never applied to either processing program.

For genuine unscaled integer counts, add `--counting`. This validates nonnegative
integer input and forces both pedestal and OFFSET to zero. Counting acquisition
alone does not establish the gain of arbitrarily scaled saved images.

## Trimming And Centre

`--trim aggressive` removes only weak leading/trailing illumination. Each retained
edge requires five consecutive strong frames, using direct-beam peak and summed
signal relative to their 75th percentiles. These are illumination heuristics, not
diffraction-quality measurements. Internal weak frames stay in place to preserve
angles. Use `--trim none` and `--frames FIRST LAST` (inclusive, 1-based) when needed.

Automatic beam finding fits a broad elliptical halo twice with a robust loss,
excluding the central core. The median of up to 21 fits across the retained sweep
defines one static centre. At least 80% must succeed; 95th-percentile drift above
two output pixels stops conversion. This cannot guarantee a correct centre for
obscured beams or nearby strong reflections. Inspect the report and refine
geometry downstream. `--beam-radius` adjusts the central search window.

`--beam-center X Y` supplies a calibrated centre in **zero-based input pixel
centre coordinates**, without an automatic drift check. By default pixel positions
are unchanged. `--center` applies one integer translation to the whole sweep and
marks uncovered borders invalid. It can crop edge reflections and does not correct
a moving beam. There is no subpixel interpolation or per-frame image warping.

## XDS And DIALS

Outputs include `XDS.INP`, `XDS_20230630.INP`, `import_dials.py`, `frames.csv`,
`conversion.json`, and `image_000001.cbf` onwards. Helpers contain absolute CBF
paths. Copy the input file/importer to a writable processing directory while
leaving the CBF files in the read-only data directory:

```bash
python3 import_dials.py
python3 import_dials.py --gain 12.5 --output measured_gain.expt
python3 import_dials.py --source /another/mount/cbf --output other_mount.expt
```

Additional DIALS `name=value` parameters can follow `--`. `--dry-run` prints the
command. `--pedestal` overrides the DIALS subtraction explicitly; normally keep
the generated value. Without a gain option, DIALS uses its own default, which
is **not** a detector-gain measurement.

An explicit scalar approximation from XDS INIT is available:

```bash
python3 import_dials.py --gain-from-init /path/to/xds/INIT.LP --output xds_gain.expt
```

The helper requires the same CBF path and pedestal in INIT.LP and uses its mean
gain. This does not reproduce XDS's spatial gain map or variance model. DIALS
divides corrected pixels by panel gain; do not apply it a second time elsewhere.
A direct GAIN.cbf transfer is not included: its sampling changed between XDS
versions, and a guessed resampling is not a validated map conversion.

Run `xds_par` in the writable directory. For XDS 20230630, use `XDS_20230630.INP`
as `XDS.INP`. Edit the absolute image template if the mount changes. Electron
templates disable X-ray polarization, air-absorption and silicon-thickness
corrections. No resolution cutoff or lattice is selected automatically.

Preserving signed signal does not make it Poisson-distributed or guarantee correct
treatment by every XDS release. Zero high-shell I/sigma with meaningful CC1/2,
sentinel negative Rmeas, or unsupported high-resolution signal still require
investigation. The converter does not tune OFFSET to hide these symptoms.

## Validation

```bash
python -m unittest discover -s tests -v
python tests/check_pixels.py /path/to/cbf/conversion.json
dials.python tests/check_dials.py imported.expt /path/to/cbf/conversion.json /path/to/cbf/XDS.INP
```

The latter commands independently check raw pixels and stock-DIALS geometry. See
[VALIDATION.md](VALIDATION.md) for measured results and evidence limits.
`conversion.json` records a completed file conversion, not a validated structure.
It includes code hashes, source size/mtime, geometry, trim and pedestal. `frames.csv`
maps every output to its input frame/start angle and records each CBF's SHA-256.

License: CC-BY-4.0, as declared by the original repository. The historical method
was described in [the original paper](https://www.biorxiv.org/content/10.1101/2025.07.03.663097v1).
