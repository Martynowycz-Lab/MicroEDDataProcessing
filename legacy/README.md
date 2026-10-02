# Historical Converters

The published `../mrc2cbf_pipeline_v3.py` is unchanged from repository commit
`4cae0be481c1f99f2109fa07d521670049956176`. Its original documentation is
[README_v3.md](README_v3.md). Use that script and recorded parameters to reproduce
published processing. v9 does not claim identical numerical output.

The v7/v8 scripts are byte-for-byte copies of the local implementations present
on 2026-10-02. Their sibling `adaptive_gain_estimator.py` is included. v8 defaults
to experimental radial conditioning, unlike v9. These files retain their original
behavior and bugs; the new converter's correctness claims do not cover them.

SHA-256 checksums:

```text
d3d8dcdc49918cd44f8382948094a626f33c226fa785dde7e5ef11bb783f6344  ../mrc2cbf_pipeline_v3.py
d21edb449e6979f728795d172f67bbd84b54b3f2a541bdf78d27b0e8fa6447ae  mrc2cbf_pipeline_v7.py
7d42a7e39ed5cb9d77f95669f3c4f058f6d81b52427009bda70464b2529a2915  mrc2cbf_pipeline_v8.py
3690bcdd560a37054e36f6fbd0697b1d9cce47eb99b93f4f0891e4f686ba9cef  adaptive_gain_estimator.py
```
