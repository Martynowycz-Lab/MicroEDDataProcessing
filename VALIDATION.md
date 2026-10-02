# v9 Review And Validation

Reviewed on 2026-10-02 against public repository commit
`4cae0be481c1f99f2109fa07d521670049956176` and the local v7/v8 implementations.
This record establishes conversion arithmetic and tested interoperability, not
universal beam calibration or improved refined structures.

## Findings And Changes

Published v3 used MDOC acquisition binning as additional software binning. The
replacement requires the physical pitch of the stored MRC pixels and applies
only the explicitly selected or size-derived additional sum binning. It never
treats diffraction `PixelSpacing` as a detector pitch. SerialEM distinguishes
the detector/camera geometry from reciprocal-space pixel spacing in its
[metadata documentation](https://bio3d.colorado.edu/SerialEM/stableHlp/html/about_formats.htm).

The previous CBF writer added 1 to an array-index beam centre, then used that
same value in XDS. CBF/DIALS and XDS instead differ by half a pixel. For array
index centre `(x,y)`, v9 writes miniCBF `(x+0.5,y+0.5)` and XDS
`(x+1,y+1)`. This agrees with the convention used by the
[dxtbx XDS exporter](https://dials.github.io/_modules/dxtbx/serialize/xds.html).
The DIALS laboratory frame is changed to the chosen XDS laboratory frame by
`diag(1,-1,-1)`, a proper rotation. The previous `(x,-y,z)` axis conversion was
only correct for the special case z=0. No pixels are flipped or transposed.

Interpolation mixes neighbouring measurements and changes their noise and
covariance. v9 leaves positions unchanged by default. Its optional centring is
one integer translation, with uncovered borders invalid. Bad/saturated input
pixels invalidate the corresponding output bin; they are never replaced by a
median. There is no radial subtraction, intensity rescaling, inferred flat field,
or automatic removal of negative tails. Floating input is rounded only after
summation, with that quantization error recorded.

One pedestal is added after binning and trimming. Exactly that number is supplied
to both XDS and DIALS. XDS GAIN remains unset and metadata gain is ignored.
The new code does not inherit v8's radial/quantile offset rules or its provisional
"gold standard" label. These choices preserve measurements; they cannot change
how an integrator models signed detector noise.

The electron XDS template disables polarization with `FRACTION_OF_POLARIZATION=-1`,
air absorption with `AIR=0`, and silicon-thickness correction with
`SENSOR_THICKNESS=0`. The previous polarization fraction 0 with a nonzero plane
normal did not express "disable polarization". Parameter semantics are documented
in the [XDS input reference](https://xds.mr.mpg.de/html_doc/xds_parameters.html).

XML timing is checked against a constant-period fit rather than assuming that
rounded exposure durations exactly equal software timestamp differences. Uniform
exposure, continuous equal detector-frame groups and bounded timestamp residuals
are required. No XML timestamp is silently sorted, discarded or replaced.
Negative-angle sweeps keep file order and negate both the axis and angle labels.
The mapping from every retained input frame to output angle is recorded.

## Tests

The 30 standard-library unittest cases passed on macOS and Linux with no skips.
They cover signed sum conservation, overflow/clipping refusal, masks, floating
quantization, integer translations, halo fitting with a central dip and an
off-centre bright spot, large beam drift, edge trimming, wavelength, reversed
scans, metadata disagreement, timestamp jitter/gaps, copied importers, old/new
INIT.LP gain syntax, parallel determinism, overwrite refusal and write failures.
Ruff checks pass. GitHub Actions runs the suite on macOS and Linux with Python
3.10, 3.12 and 3.13; the workflow result, not its presence, establishes CI status.

A synthetic asymmetric movie was also imported by stock DIALS 3.25.0. It exercises
unequal x/y pixel pitches, a nonzero z rotation-axis component, a negative scan,
noncentral beam, sum-binning, integer translation and masked saturation. Detector
laboratory rays, scan angles, axis, wavelength, signed corrected values and masks
were checked independently against the generated XDS input.

This extended test exposed an issue in the locally installed dxtbx exporter:
it computed the slow-axis origin using the fast-axis pitch. The imported detector
model itself was correct. The checker explicitly recognizes that exact exporter
error for unequal pitches and still requires agreement of independently computed
detector origins and corner rays. Square-pixel data also require agreement with
the stock exporter. No local DIALS installation was modified.

## Real Data

Two complete 688-frame Tyrosine movies from the 2026-04-10 collection were read
from the NAS on cc02 and written only into isolated temporary validation folders.
Both used 0.028 mm stored-input pitch, 2 by 2 sum-binning, aggressive trimming,
automatic beam finding and four threads. No radial conditioning or image shifting
was applied.

| Dataset | Retained input frames | Output | Pedestal = OFFSET | Beam x,y (array index) | Beam drift p95 |
| --- | --- | --- | ---: | --- | ---: |
| 366 | 1-639 | 639 x 1024 x 1024 | 301 | 514.702855, 511.497646 | 1.061901 px |
| 370 | 1-658 | 658 x 1024 x 1024 | 308 | 516.285037, 510.383877 | 0.906141 px |

Initial complete runs, including CBF read-back verification, took 102.6 and
104.8 seconds respectively. These are measured NAS-to-server times, not a portable
speed guarantee: repeated runs during concurrent validation took 314.8 and 299.4
seconds. Their CBF SHA-256 values and complete frame mappings were identical to
the first runs. The converter holds a few working frames per worker instead of
replicating an entire movie into forked processes.

An independent second pass summed the raw input using strided slices rather than
the converter's reshape reduction. It checked every stored pixel, every frame
SHA-256, and every input/output index and angle. It recovered 350,050,027 negative
binned values in 366 and 395,610,005 in 370 after subtracting the stored pedestal.
Those values were preserved, not clipped into zero or invalid pixels.

Both datasets imported as single sequences in stock DIALS 3.25.0 on Linux.
Their detector geometry agreed with the XDS exporter, and DIALS pedestal
subtraction recovered the signed CBF signal. XDS XYCORR/INIT also completed on
both generated inputs. Initial 20-frame background checks gave mean gains of
182.009 and 153.691, illustrating why the MDOC gain is not substituted for INIT.
The released template uses all retained frames for BACKGROUND_RANGE; gain values
depend on that range and are not universal detector constants.
With all retained frames, XDS INIT (Apr 16, 2026, built 20260520) estimated means
of 184.198 for 366 and 161.118 for 370. The 366 mean was passed through the generated
`--gain-from-init` importer into stock DIALS; geometry and the corrected values
`(raw - pedestal) / gain` passed the independent check.

The first full pixel-validation runs used converter SHA-256
`813400535c77ec232337ef69bfa401a7b3b2d1f6fd98f1394f8b73d5c3ced2d5`
and metadata SHA-256
`e34370f20b92856e24ce2a9931639e3002f54d5e2d4937ba7eee7d82dfbb446c`.
Subsequent review tightened completion-marker writing, source reporting, binning
failure messages and the INIT gain helper without changing those pixel results.
The release also distinguishes XML exposure duration from fitted frame period in
the two CBF header fields. Final-revision 12-frame real-data conversions passed
independent raw-pixel checks on macOS (370) and Linux (366), and stock-DIALS import
and geometry checks on macOS. This small metadata correction does not alter pixels.

## Gain And Evidence Limits

`--gain-from-init` deliberately uses a scalar mean from a matching INIT.LP, with
path and pedestal checks. It is not a translation of XDS's full noise model.
XDS documents GAIN.cbf as a local variance/mean table, with revised sampling and
integer scaling in 2024. See the [file definitions](https://xds.mr.mpg.de/html_doc/xds_files.html)
and [release notes](https://xds.mr.mpg.de/html_doc/Release_Notes.html).
The old local routine's generic interpolation of that table has not been promoted.

These tests do not establish improved I/sigma, CC1/2, refinement, model correlation,
or the correctness of a particular microscope calibration. No offset sweep or
statistics-based tuning was used. Automatic beam and trim decisions remain
explicit heuristics. Beamstops, severe drift, detector distortions, nonuniform
rotation, unknown saturation and preprocessed counting data need calibration or
an explicit manual workflow. Retaining the exact pedestal does not resolve known
new-XDS pathologies on signed dark-corrected data.

For paper reproduction, the public v3 script and its original README were checked
byte-for-byte against the original commit. Local v7/v8 and their gain dependency
were checked byte-for-byte against the working copies. Their checksums are in
[legacy/README.md](legacy/README.md); they have not been rewritten to hide defects.
