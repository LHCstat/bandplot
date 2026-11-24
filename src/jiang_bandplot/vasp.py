from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from .procar import NUMBER, number


@dataclass
class Structure:
    reciprocal: np.ndarray
    scalar_scale: float | None
    elements: dict
    nions: int


def structure_from(path, nions):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        scales = np.array([number(item) for item in lines[1].split()])
        lattice = np.array([[number(item) for item in line.split()[:3]] for line in lines[2:5]])
        volume = abs(np.linalg.det(lattice))
        if not np.isfinite(volume) or volume == 0:
            raise ValueError("lattice must have a finite nonzero volume")
        scale = None
        if len(scales) == 1 and scales[0] != 0:
            scale = scales[0] if scales[0] > 0 else (-scales[0] / volume) ** (1 / 3)
            lattice *= scale
        elif len(scales) == 3 and np.all(scales > 0):
            lattice *= scales
        else:
            raise ValueError("invalid lattice scaling factor")
        fields = lines[5].split("!")[0].split()
        if not fields:
            raise ValueError("missing ion counts or species")
        if all(field.isdecimal() for field in fields):
            species = []
            counts = list(map(int, fields))
        else:
            species = fields
            counts = list(map(int, lines[6].split("!")[0].split()))
            if len(species) != len(counts):
                raise ValueError("species and ion counts differ")
        if any(count < 1 for count in counts) or sum(counts) != nions:
            raise ValueError(f"ion count does not match the {nions} ions in PROCAR")
        groups = {}
        for name, start, stop in zip(species, np.cumsum([0] + counts), np.cumsum(counts)):
            groups.setdefault(name, []).extend(range(int(start) + 1, int(stop) + 1))
        return Structure(2 * np.pi * np.linalg.inv(lattice).T, scale, groups, nions)
    except (IndexError, ValueError, np.linalg.LinAlgError) as error:
        raise ValueError(f"{path.name}: {error}") from error


@dataclass
class KPath:
    x: np.ndarray
    slices: list
    ticks: list
    labels: list

    def report(self):
        return {"slices": self.slices, "labels": self.labels, "ticks_inv_angstrom": self.ticks}


def clean_label(label):
    stripped = label.strip().strip("$")
    if stripped.lower() in {"g", "gamma", "gam", r"\gamma", "γ"}:
        return "Γ"
    return stripped


def endpoints_from(path):
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        count = int(lines[1].split()[0])
        if count < 2 or not lines[2].strip().lower().startswith("l"):
            raise ValueError("requires a line-mode path with at least two points per segment")
        mode = lines[3].strip().lower()
        if not mode.startswith(("r", "c", "k")):
            raise ValueError("unknown path coordinate system")
        points = []
        for line in lines[4:]:
            raw, separator, comment = line.partition("!")
            fields = raw.split()
            if not fields:
                continue
            point = [number(item) for item in fields[:3]]
            if len(point) != 3:
                raise ValueError("path endpoint must have three coordinates")
            trailing = fields[3:]
            if trailing and re.fullmatch(NUMBER, trailing[0]):
                trailing = trailing[1:]
            points.append((point, clean_label(comment if separator else " ".join(trailing))))
        if not points or len(points) % 2:
            raise ValueError("there must be two endpoints per segment")
        return count, mode, points
    except (IndexError, ValueError) as error:
        raise ValueError(f"{path.name}: {error}") from error


def path_from(path, structure, kpoints):
    count, mode, points = endpoints_from(path)
    coords, labels = zip(*points)
    coords = np.asarray(coords)
    if mode.startswith(("c", "k")):
        if structure.scalar_scale is None:
            raise ValueError("KPOINTS: Cartesian endpoints need a scalar POSCAR scale")
        coords = (coords * (2 * np.pi / structure.scalar_scale)) @ np.linalg.inv(structure.reciprocal)

    positions = []
    ticks = [0.0]
    names = [labels[0]]
    slices = []
    stop = 0
    tolerance = 5e-6
    for i, (first, last) in enumerate(zip(coords[::2], coords[1::2])):
        expected = np.linspace(first, last, count)
        distances = np.linalg.norm(np.diff(expected @ structure.reciprocal, axis=0), axis=1)
        if not np.all(np.isfinite(distances)) or np.sum(distances) == 0:
            raise ValueError(f"KPOINTS: segment {i + 1} has zero or invalid length")
        start = stop
        candidates = [start]
        if i and np.allclose(first, coords[2 * i - 1], atol=tolerance, rtol=0):
            candidates.append(start - 1)
        for start in candidates:
            observed = kpoints[start:start + count]
            if len(observed) == count and np.allclose(observed, expected, atol=tolerance, rtol=0):
                break
        else:
            raise ValueError(f"KPOINTS: segment {i + 1} does not match PROCAR k-point coordinates")
        actual_distances = np.linalg.norm(np.diff(observed @ structure.reciprocal, axis=0), axis=1)
        x = ticks[-1] + np.r_[0.0, np.cumsum(actual_distances)]
        positions.extend(x[stop - start:].tolist())
        stop = start + count
        slices.append([start, stop])
        ticks.append(float(x[-1]))
        start_label, end_label = labels[2 * i:2 * i + 2]
        names[-1] = " | ".join(dict.fromkeys(label for label in (names[-1], start_label) if label))
        names.append(end_label)
    if stop != len(kpoints):
        raise ValueError(f"KPOINTS: path accounts for {stop} of {len(kpoints)} PROCAR k-points")
    return KPath(np.asarray(positions), slices, ticks, names)


def read_fermi(path):
    pattern = re.compile(rf"\bE-fermi\s*:\s*({NUMBER})", re.I)
    value = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                value = number(match[1])
    return value
