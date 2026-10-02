"""Create a small, asymmetric signed movie for the stock-DIALS geometry check."""

from pathlib import Path
import sys

import mrcfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mrc2cbf import convert, parser

root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=False)
pixels = np.random.default_rng(195).integers(-30, 70, (8, 64, 96), dtype=np.int16)
pixels[:, 9, 21] = 2000
pixels[:, 51, 70] = 32767
source = root / "signed.mrc"
with mrcfile.new(source) as movie:
    movie.set_data(pixels)
    movie.set_image_stack()
args = parser().parse_args(
    [
        str(source),
        str(root / "cbf"),
        "--pixel-size-mm",
        "0.02",
        "0.03",
        "--rotation-axis",
        "0.98",
        "-0.05",
        "0.03",
        "--distance-mm",
        "900",
        "--voltage-kv",
        "200",
        "--start-angle-deg",
        "-30",
        "--angle-step-deg",
        "-0.2",
        "--frame-time-s",
        "0.1",
        "--beam-center",
        "41.2",
        "34.3",
        "--trim",
        "none",
        "--frames",
        "2",
        "7",
        "--center",
        "--bin",
        "2",
    ]
)
convert(args)
