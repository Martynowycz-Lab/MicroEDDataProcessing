# MicroEDDataProcessing

`mrc2cbf.py` v9.1 converts MRC images and stacks to miniCBF on Linux and macOS.
Give it a file, a folder, or a file pattern. It does not ask questions.
It reads available metadata and lists missing values and assumptions at the end.

## Install

Use Python 3.10 or later:

```bash
git clone https://github.com/Martynowycz-Lab/MicroEDDataProcessing.git
cd MicroEDDataProcessing
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Keep `mrc2cbf.py`, `metadata.py`, and `inputs.py` together.
DIALS and XDS are separate programs. Neither is needed for conversion.

## Convert

A stack, or a folder of individual images:

```bash
python mrc2cbf.py movie.mrc -o cbf
python mrc2cbf.py frames/ -o cbf
```

A shell pattern or a quoted pattern:

```bash
python mrc2cbf.py frames/*.mrc -o cbf
python mrc2cbf.py 'frames/*.mrc' -o cbf
```

For several Velox stacks with XML but no MDOC, run this from the data folder:

```bash
python /path/to/MicroEDDataProcessing/mrc2cbf.py '*.mrc' -o cbf --bin 1 --trim none
```

Each stack gets `cbf/STACK_NAME/image_000001.cbf` and so on. This command keeps
all frames at their input size. Matching `STACK_NAME.xml` or `STACK_NAME.mrc.xml`
is read automatically. Unknown geometry is listed at the end; conversion proceeds.
No voltage, distance, angle, or pixel-size flags are required for pixel conversion.

To keep the input names, use a quoted output pattern:

```bash
python mrc2cbf.py 'frames/*.mrc' 'converted/*.cbf'
python mrc2cbf.py frame.mrc frame.cbf
```

`frame_001.mrc` becomes `converted/frame_001.cbf`. For a stack, the converter adds
`_000001`, `_000002`, and so on to its name. The output accepts one `*` in the
filename. Quote the output pattern so the shell does not expand it.
A literal CBF output needs one input frame.

The original command still works:

```bash
python mrc2cbf.py movie.mrc cbf
```

Without an output argument, files go to a new `cbf` directory in the current
working directory. Existing outputs are never replaced. Raw files stay read-only.
Folders are not searched recursively. Supported extensions are `.mrc`, `.mrcs`,
and `.st`, with either letter case.

Numbered single-image files with the same prefix form one sweep. Numbers set the
order: `frame2` comes before `frame10`. Different prefixes and separate stacks
form separate sweeps, each with its own output folder, metadata, and pedestal.
Repeated input paths are read once. Duplicate frame numbers are an error.
Gaps in file numbers stay in the recorded angles. Processing helpers are omitted
for a sweep with gaps. The converter does not join independent crystals.

## Metadata

Conversion does not require complete geometry. Unknown header fields are omitted.
The summary and `conversion.json` list them. An incomplete header is sufficient
for pixel conversion and inspection. It is not sufficient for reliable integration.

The converter reads matching SerialEM MDOC, Velox fraction XML, EPU-D acquisition
XML, and FEI1/FEI2 extended headers. It checks FEI field-presence flags and all
frames in a series. For a numbered series such as `lam09_0000.mrc`, it looks for
`lam09.xml` and `lam09.mdoc` or `lam09.mrc.mdoc`. It does not select an arbitrary
sidecar when several match. Use `--mdoc FILE` or `--xml FILE` to select one.
An unsupported XML layout is reported and not used. Malformed XML, contradictory
timing, changed image dimensions, and volumes are errors.
Only single-FrameSet MDOC files are supported. Other MDOC layouts are reported
and not used. Per-image tilt series are not flattened into a continuous scan.

Physical pixel pitch comes from MDOC `CameraPixelSize` in micrometres or recognized
Falcon/Ceta metadata and acquisition binning. Acquisition binning is applied once.
`PixelSpacing`, MRC voxel size, and FEI sample pixel size are not detector pitch.
FEI camera length is nominal, not a measured distance calibration. This assumption
is reported. Gain from acquisition metadata is not applied.

The rotation axis falls back to the previous lab value,
`0.999263 -0.0383878 0`, with a warning that it can be wrong for another microscope.
The axis is not measured from the metadata or the images. Supply a calibrated
axis when available. The frame is DIALS/imgCIF: fast +x, slow -y, incident beam -z.

Override values with explicit units:

```bash
python mrc2cbf.py frames/ -o cbf \
  --pixel-size-mm 0.028 0.028 --distance-mm 1488 \
  --voltage-kv 200 --rotation-axis 1 0 0 \
  --start-angle-deg -40 --angle-step-deg 0.25 --frame-time-s 1
```

These example values are not a calibration for your microscope. Pixel pitch
refers to the stored MRC pixels. `--wavelength-a` can replace `--voltage-kv`.
The start angle is the start of the first input frame, not its centre.
A series-wide start angle cannot locate a subset that begins after file number 1;
supply `--start-angle-deg` for that subset. Negative sweeps keep file order and
reverse the axis and angle labels to give an equivalent positive-width scan.
Explicit scan timing must agree with available timing checks.

Use `--require-geometry` to return an error if the processing geometry is incomplete.
No completion record is written for a failed sweep. Use a new output directory
to retry. In a batch, other sweeps continue and `batch.json` lists failures.
The command returns a nonzero exit code if any sweep fails.

## Pixels

The default sum bin targets at most 1024 pixels per axis. Smaller images are not
enlarged. Use `--bin 1` to keep the input size or `--bin 2` to sum 2 by 2 blocks.
Both dimensions must divide exactly. No partial spatial or time bins are dropped.

Integer sums are exact. Float input is summed in float64 and rounded once to the
nearest integer, with ties to even. The report gives the maximum rounding error.
There is no radial subtraction, interpolation, gain correction, or intensity filter.

Bad and saturated pixels invalidate their whole output bin. Invalid pixels are
stored as -1. They are not filled. Use `--bad-pixels mask.npy` for a boolean input
mask, where True means invalid. `--saturation ADU` sets a known threshold.
Without one, the integer datatype limit is used. Float input has no assumed limit.

One storage pedestal preserves all retained valid negative values:

```text
stored CBF = rounded binned signal + pedestal
pedestal = max(0, -minimum rounded binned signal)
```

`--pedestal N` can select another value that preserves every valid pixel.
The exact added value goes to XDS `OFFSET` and the DIALS panel pedestal.
XDS `GAIN` stays unset so INIT can estimate it.
Use `--counting` only for unscaled nonnegative integer counts; it requires zero
pedestal. Counting acquisition alone does not establish the scale of saved data.

## Trim and beam centre

`--trim auto` is the default. It removes weak ends when illumination can be
identified. If that check fails, it keeps all selected frames and reports this.
`--trim aggressive` requires a reliable run of strong frames and fails if absent.
`--trim none` keeps all selected frames. `--frames FIRST LAST` selects an inclusive,
1-based range within each sweep. Internal weak frames remain in place.
These checks measure illumination, not diffraction quality.

Beam finding fits the broad halo in up to 21 frames. It requires 80% of the fits
to succeed and checks drift against `--max-beam-drift` (default: 2 output pixels).
If the centre is not reliable, pixels are converted in place and the centre is
reported as unknown. Fewer than three frames cannot give a reliable drift check.
Use `--beam-center X Y` for a calibrated centre in zero-based input-pixel coordinates.
`--beam-radius` sets the search radius in output pixels.

`--center` applies one integer translation to the whole sweep. It requires a
reliable centre and makes uncovered borders invalid. It can crop edge reflections.
It does not correct beam drift or interpolate pixels.

## Outputs and processing

Directory output uses `image_000001.cbf` and so on. `frames.csv` maps each CBF to
its source file, source frame, scan position, and SHA-256 hash. `conversion.json`
records inputs, metadata sources, missing values, code hashes, and elapsed time.
Every CBF is decoded and compared with the intended integer pixels before it is
published. The completion record certifies conversion, not a solved structure.

Complete, consecutive sweeps with standard output names also get `XDS.INP`,
`XDS_20230630.INP`, and `import_dials.py`. A path with spaces or `!?*` cannot form
an XDS template, so the processing helpers are omitted and this is reported.
CBF filename/pattern output writes its report under `mrc2cbf_report/`, or
`NAME.cbf.report/` for a literal name. It omits the standard-name processing helpers.

Copy the processing input or importer to a writable work directory:

```bash
python3 /path/to/cbf/import_dials.py
python3 /path/to/cbf/import_dials.py --source /another/mount/cbf
python3 /path/to/cbf/import_dials.py --gain 12.5 --output measured_gain.expt
python3 /path/to/cbf/import_dials.py --gain-from-init /path/to/xds/INIT.LP
```

`--dry-run` prints the DIALS command. Additional DIALS parameters follow `--`.
INIT gain transfer requires a matching CBF path and pedestal. It is a scalar
approximation, not XDS's spatial gain map. Without a gain option, DIALS uses its
own default. That default is not a measured detector gain.

Run `xds_par` with `XDS.INP`. For XDS 20230630, use `XDS_20230630.INP` as `XDS.INP`.
Templates disable X-ray polarization, air absorption, and silicon corrections.
No lattice or resolution cutoff is selected. Preserving signed signal does not
make it Poisson-distributed or fix every XDS noise-model limitation.

## Check

```bash
python -m unittest discover -s tests -v
python tests/check_pixels.py /path/to/cbf/conversion.json
dials.python tests/check_dials.py imported.expt /path/to/cbf/conversion.json /path/to/cbf/XDS.INP
```

See [VALIDATION.md](VALIDATION.md) for measured results and limits.
Historical [v3, v7, and v8](legacy/README.md) remain unchanged.
License: CC-BY-4.0, as declared by the original repository.
The original method is described in [the paper](https://www.biorxiv.org/content/10.1101/2025.07.03.663097v1).
