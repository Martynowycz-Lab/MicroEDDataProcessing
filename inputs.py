"""Find MRC inputs and read one sweep without loading it into memory."""

from bisect import bisect_right
import glob
from pathlib import Path
import re

import mrcfile
import numpy as np
from mrcfile.utils import data_dtype_from_header


MRC_SUFFIXES = {".mrc", ".mrcs", ".st"}


def natural_key(path):
    return [
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", str(path))
    ]


def numbered_name(path):
    match = re.fullmatch(r"(.*?)(\d+)(\D*)", path.stem)
    return (match[1], int(match[2]), match[3]) if match else None


def find_inputs(arguments):
    """Accept files, directories, and quoted globs. Read each resolved file once."""
    paths = []
    for argument in arguments:
        path = Path(argument).expanduser()
        if path.exists():
            matches = [path]
        elif glob.has_magic(str(path)):
            matches = [Path(value) for value in glob.glob(str(path))]
            if not matches:
                raise ValueError(f"No input matches: {argument}")
        else:
            raise FileNotFoundError(f"Input does not exist: {argument}")
        for match in sorted(matches, key=natural_key):
            if match.is_dir():
                files = sorted(
                    (
                        value
                        for value in match.iterdir()
                        if value.is_file() and value.suffix.lower() in MRC_SUFFIXES
                    ),
                    key=natural_key,
                )
                if not files:
                    raise ValueError(f"No MRC files in directory: {match}")
                paths.extend(files)
            elif match.is_file() and match.suffix.lower() in MRC_SUFFIXES:
                paths.append(match)
            else:
                raise ValueError(f"Input is not an MRC file or directory: {match}")
    return sorted(set(path.resolve(strict=True) for path in paths), key=natural_key)


def describe(path):
    before = path.stat()
    with mrcfile.open(path, mode="r", header_only=True, permissive=False) as image:
        header = image.header
        if int(header.ispg) != 0:
            raise ValueError(f"MRC is marked as a volume, not detector images: {path}")
        if tuple(int(header[key]) for key in ("mapc", "mapr", "maps")) != (1, 2, 3):
            raise ValueError(f"Nonstandard MRC axis mapping: {path}")
        count, height, width = (int(header[key]) for key in ("nz", "ny", "nx"))
        dtype = data_dtype_from_header(header)
        if min(count, height, width) <= 0 or dtype.kind not in "iuf":
            raise ValueError(f"MRC must contain real detector images: {path}")
        expected = 1024 + int(header.nsymbt) + count * height * width * dtype.itemsize
        if before.st_size < expected:
            raise ValueError(f"MRC pixel data are truncated: {path}")
    return dict(
        path=path,
        count=count,
        shape=(height, width),
        dtype=dtype,
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
    )


def group_inputs(paths):
    """Keep stacks separate. Group numbered, single-image files by their prefix."""
    groups = {}
    for path in paths:
        item = describe(path)
        number = numbered_name(path)
        key = (
            (path.parent, number[0], number[2])
            if item["count"] == 1 and number
            else (path,)
        )
        groups.setdefault(key, []).append(item)
    return list(groups.values())


class MrcSeries:
    def __init__(self, items):
        self.items = items
        self.paths = [item["path"] for item in items]
        self.dtype = items[0]["dtype"]
        self.shape = (sum(item["count"] for item in items), *items[0]["shape"])
        if any(
            item["shape"] != self.shape[1:] or item["dtype"] != self.dtype
            for item in items
        ):
            raise ValueError("MRC image size or datatype changes within one series")
        self.ends = np.cumsum([item["count"] for item in items]).tolist()
        self.numbers = [numbered_name(path) for path in self.paths]
        self.offsets = list(range(self.shape[0]))
        self.notes = []
        if len(items) > 1:
            numbers = [value[1] for value in self.numbers]
            if len(set(numbers)) != len(numbers):
                raise ValueError("Duplicate frame numbers in MRC series")
            self.offsets = [number - numbers[0] for number in numbers]
            if self.offsets != list(range(len(items))):
                self.notes.append(
                    "Input frame numbers have gaps; angles retain those gaps. XDS/DIALS helpers are omitted"
                )
        self.movie = None

    def __len__(self):
        return self.shape[0]

    def __enter__(self):
        if len(self.paths) == 1:
            self.movie = mrcfile.mmap(self.paths[0], mode="r", permissive=False)
        return self

    def __exit__(self, *error):
        if self.movie is not None:
            self.movie.close()
            self.movie = None

    def location(self, idx):
        file_idx = bisect_right(self.ends, idx)
        local_idx = idx - (self.ends[file_idx - 1] if file_idx else 0)
        return self.paths[file_idx], local_idx

    def __getitem__(self, idx):
        if self.movie is not None:
            return (
                self.movie.data if self.movie.data.ndim == 2 else self.movie.data[idx]
            )
        path, local_idx = self.location(idx)
        with mrcfile.mmap(path, mode="r", permissive=False) as image:
            return np.array(
                image.data if image.data.ndim == 2 else image.data[local_idx]
            )

    def check_unchanged(self):
        for item in self.items:
            after = item["path"].stat()
            if (after.st_size, after.st_mtime_ns) != (item["size"], item["mtime_ns"]):
                raise ValueError(f"Input changed during conversion: {item['path']}")

    def records(self):
        return [
            dict(
                path=str(item["path"]),
                frames=item["count"],
                size=item["size"],
                mtime_ns=item["mtime_ns"],
            )
            for item in self.items
        ]
