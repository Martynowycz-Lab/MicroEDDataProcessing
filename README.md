To run the converter in the simplest way:
python mrc2cbf_pipeline_v3.py path/to/mrc path/to/CBF/folder

Recommendation is to run as below to guarantee the mdoc file is located.
python mrc2cbf_pipeline_v3.py path/to/mrc path/to/CBF/folder --mdoc path/to/mdoc

Running just 'python mrc2cbf_pipeline_v3.py' will output all options.
--bin-x and --bin-y can reduce the amount of time to process and reduce size of the output images. Time savings are fairly minor with this version.

For 4k images, it may be wise to run --roi-size [integer greater than 100] and --max-dev [integer greater than 50] to prevent beamfinding from failing. 120 and 60 should be sufficient.
For smaller mrc files, increasing these values may cause problems with finding beam centers properly.
