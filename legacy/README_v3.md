MicroEDDataProcessing. An MRC to miniCBF conversion pipeline.

Convert ED movie stacks (.mrc) into Pilatus-style CBF images with optional beam-center finding, smoothing, image shifting, and metadata extraction from SerialEM .mdoc files. 
Parallelized where it matters. Some bugs. Some things do not work as intended. A work in progress. Built on top of methods from earlier ED papers and 4dstem papers. 

Script: mrc2cbf_pipeline_v3.py
Input: 3D MRC stack (frames × rows × cols)
Output: per-frame .cbf files + logs/diagnostics

Why this exists
	•	Turn raw MRC stacks into CBF files many crystallography tools expect (XDS/DIALS/MOSFLM)
	•	Find beam center per frame (2D Gaussian fit with robust fallbacks). Same approach as 4dstem experiments- nothing new here.
  **THIS DOES NOT WORK WITH A BEAMSTOM**	
  •	Smooth/remove outlier centers (jump detection + Savitzky–Golay). Fails sometimes on big jumps. 
	•	Optionally shift images so the beam sits at the geometric center.
	•	Auto-fill metadata from .mdoc (pixel size, camera length, tilt, rotation rate, etc.). This requires properly set u MDOC files and calibrated SerialEM for ED.
	•	Fast: multiprocessing for beam finding and writing. 

When ran properly, this makes MRC stacks with a hundred to a few hundred images into miniCBF files (uses FabIO) in under a minute. 

**Installation**

# Python ≥3.8 recommended (some odd stuff with numpy and int/int64 depending on versions)
pip install numpy scipy matplotlib mrcfile fabio

If you use conda: conda install -c conda-forge numpy scipy matplotlib mrcfile fabio

**Quick start**

Easiest (with .mdoc)

python mrc2cbf_pipeline_v3.py path/to/stack.mrc out_dir --mdoc path/to/stack.mdoc

Explicit parameters (no .mdoc)

python mrc2cbf_pipeline_v3.py path/to/stack.mrc out_dir \
  --pixel-size-mm 0.015  \
  --detector-distance-mm 1200 \
  --wavelength-A 0.0251 \
  --start-angle-deg -30.0 \
  --angle-increment-deg 0.02

Recommended for 4k images (more tolerant beam finding)

python mrc2cbf_pipeline_v3.py stack.mrc out_dir --mdoc stack.mdoc \
  --roi-size 120 --max-dev 60

What the pipeline does (in order)
	1.	Load MRC stack (frames, y, x).
	2.	Bin data (optional): --bin-z/--bin-y/--bin-x (sum-binning; trailing slices dropped if not divisible).
	3.	Pedestal (optional/auto): offsets negatives to ≥0 (recorded in CBF header).
	4.	Beam center: first pass
	•	ROI around image center → Gaussian blur → peak seed
	•	Bound-constrained 2D Gaussian fit; fallback to blurred peak if fit fails
	5.	Second pass (optional): detect large frame-to-frame jumps, fix with chosen strategy, then Savitzky–Golay smooth if enough valid points.
	6.	Image shift (optional): translate each frame so the beam lands at the geometric center; update header accordingly.
	7.	Write CBF via fabio with Pilatus-style header (pixel size, distance, wavelength, beam, angle series, pedestal, cutoff).
	8.	Diagnostics: plots, per-frame timings, and a text file of final centers.

**Inputs & metadata**

From .mdoc (if provided)
	•	Binning (Binning): default for --bin-x/--bin-y.**This is the wrong behavior. Working on it.**
	•	CameraPixelSize (µm/pix): unbinned --pixel-size-mm.
	•	CameraLength (mm): --detector-distance-mm.
	•	Voltage (kV): --wavelength-A (relativistic).
	•	TiltAngle (FrameSet 0): → --start-angle-deg.
	•	DegreesPerSecond × ExposureTime / NumSubFrames (or RotationRate): --angle-increment-deg (per original frame; adjusted by --bin-z automatically).

Can override any of these using CLI flags.

Key behaviors & conventions
	•	Coordinates: Internally 0-based (y, x).
CBF header uses 1-based (x, y) as # Beam_xy.
	•	When shifting is enabled: images are shifted so the beam is at the geometric center; the header records that center.
When disabled: images are not shifted; header keeps the found beam.
	•	Pixel size in header: assumes square pixels; scaled by --bin-x. Keep --bin-x == --bin-y for correct header geometry.
	•	--bin-z reduces frame count; angle increment is multiplied by --bin-z.

**CLI reference**

Run python mrc2cbf_pipeline_v3.py -h for defaults.

Required (either via flags or .mdoc):
	•	mrc_path — input .mrc
	•	output_dir — folder to write results
	•	Geometry/time series: --pixel-size-mm, --detector-distance-mm, --wavelength-A, --start-angle-deg, --angle-increment-deg
	•	Or: --mdoc path/to/file.mdoc

**General:**
	•	--mdoc path/to/file.mdoc
	•	--bin-x, --bin-y, --bin-z (default 1)
	•	--pedestal INT (else auto if negatives are present)
	•	--no-auto-pedestal
	•	--skip-beam-centering
	•	--no-image-shift
	•	--shift-order {0..5} (scipy spline order; 1 is linear)
	•	--overload INT (CBF Count_cutoff, default 1,000,000)
	•	-n/--num-workers INT
	•	--limit-frames N
	•	--filename-template "image_{:05d}.cbf"
**Beam finding:**
	•	--roi-size INT (size of ROI box around image center)
	•	--blur-sigma FLOAT (Gaussian pre-blur)
	•	--max-dev FLOAT (max initial peak deviation from ROI center, px)
	•	--no-fit-bounds (disable bounds in fit)

**Smoothing / second pass:**
	•	--no-smoothing
	•	--max-jump FLOAT (px; mark outliers above this)
	•	--smooth-window INT (odd; auto-adjusts if even)
	•	--smooth-order INT (Savitzky–Golay poly order)
	•	--smooth-fallback {previous|interpolate|global_median}

**Diagnostics / logging:**
	•	--plot-first-pass
	•	--no-final-plot
	•	--plot-frames "0,10,42" (per-frame diagnostic PNGs)
	•	--no-save-centers
	•	--log-level {DEBUG|INFO|WARNING|ERROR}
	•	--log-file-level {DEBUG|INFO|WARNING|ERROR}

**Typical recipes**
1) Use .mdoc, keep everything default

python mrc2cbf_pipeline_v3.py movie.mrc out --mdoc movie.mdoc

2) Faster & smaller with spatial binning

python mrc2cbf_pipeline_v3.py movie.mrc out --mdoc movie.mdoc --bin-x 2 --bin-y 2

3) Time-bin frames (boost SNR, fewer CBFs)

python mrc2cbf_pipeline_v3.py movie.mrc out --mdoc movie.mdoc --bin-z 5

4) Robust beam finding for large 4k frames

python mrc2cbf_pipeline_v3.py movie.mrc out --mdoc movie.mdoc --roi-size 120 --max-dev 60

5) Don’t shift images; keep native beam position

python mrc2cbf_pipeline_v3.py movie.mrc out --mdoc movie.mdoc --no-image-shift


Outputs
	•	image_00001.cbf, image_00002.cbf, … (rename using --filename-template)
	•	beam_centers_final.txt
Columns: FrameIndex(0-based)  Beam_Y(px,0-based)  Beam_X(px,0-based)  Status
	•	beam_centers_first_pass.png (if --plot-first-pass)
	•	beam_centers_final.png (unless --no-final-plot)
	•	diagnostic_frame_00042.png (if --plot-frames)
	•	Log file: mrc2cbf_mp_init_parBeam_YYYYMMDD_HHMMSS.log in output_dir


**Performance tips**	
•	Use -n <CPU cores> for large stacks. Defaults to cpu_count().
	•	Spatial binning (--bin-x/--bin-y) reduces compute and file size.
	•	--bin-z reduces the number of output frames and increases angular step (done automatically).
	•	--shift-order 1 (linear) is usually a good speed/quality trade-off.

**Troubleshooting**
	•	“Missing required parameters”: supply missing geometry/angle flags or pass --mdoc.
	•	No valid beam centers: increase --roi-size, relax --max-dev, try --blur-sigma 2–4.
	•	Jumpy centers: lower --max-jump, enable smoothing (default), or set --smooth-fallback interpolate.
	•	Weird pixel size in header: keep --bin-x == --bin-y (header assumes square pixels, scales by --bin-x).
	•	Negative intensities / bad CBF: keep auto-pedestal on (default) or set --pedestal manually.
	•	Write failures: check output_dir permissions and free space.

**Notes on units & headers**
•	Pixel size (CBF): meters; computed as pixel_size_mm / 1000 × bin_x.
	•	Detector distance: meters.
	•	Wavelength: Å.
	•	Angles: degrees; start angle + (frame_index × angle_increment_after_binZ).
	•	Beam_xy: 1-based pixel indices in the header.

**Reproducibility**
	•	Turn on verbose logging:

python mrc2cbf_pipeline_v3.py movie.mrc out --mdoc movie.mdoc --log-level DEBUG
	•	The log captures parameters, progress, and timing summaries.



**Development**
	•	macOS: the script prefers the fork start method for multiprocessing; warnings are printed if it can’t switch.
	•	Cross-platform: Linux/macOS tested; Windows uses default spawn.

**Contributing**

Contribution is welcome (bug reports, feature requests, docs).

**License**

CC-BY-4.0

**Citation**

If this pipeline helps your work, consider citing the repo in Methods/Software. This was first described in:
https://www.biorxiv.org/content/10.1101/2025.07.03.663097v1 

