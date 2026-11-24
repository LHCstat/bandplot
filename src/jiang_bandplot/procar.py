from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import tempfile
from zipfile import BadZipFile

import numpy as np


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
DIMENSIONS = re.compile(r"#\s*of k-points:\s*(\d+)\s+#\s*of bands:\s*(\d+)\s+#\s*of ions:\s*(\d+)", re.I)
KPOINT = re.compile(r"k-point\s+(\d+)\s*:\s*(.*?)\s*weight\s*=", re.I)
BAND = re.compile(rf"band\s+(\d+)\s*#\s*energy\s+({NUMBER})\s*#\s*occ", re.I)
SPIN = re.compile(r"spin component\s+(\d+)", re.I)
CACHE_VERSION = "procar-v2"


def number(value):
    result = float(value.replace("D", "E").replace("d", "e"))
    if not np.isfinite(result):
        raise ValueError("non-finite number")
    return result


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_digest(path)}


def signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


@dataclass
class Bands:
    kpoints: np.ndarray
    energy: np.ndarray
    weights: np.ndarray
    orbitals: tuple

    @property
    def spins(self):
        return ("up", "down") if self.energy.shape[0] == 2 else ("scalar",)


class Lines:
    def __init__(self, handle):
        self.handle = handle
        self.index = 0

    def next(self):
        for line in self.handle:
            self.index += 1
            if line.strip():
                return line.strip()
        return ""

    def fail(self, message):
        raise ValueError(f"PROCAR near line {self.index}: {message}")

    def block_start(self, line, spin, dimensions):
        while line.lower().startswith("procar"):
            if "phase" in line.lower():
                self.fail("phase-resolved projections are unsupported")
            line = self.next()
        while line:
            spin_header = SPIN.fullmatch(line)
            if spin_header:
                if int(spin_header[1]) != spin:
                    self.fail("unexpected spin component")
            elif DIMENSIONS.fullmatch(line):
                if tuple(map(int, DIMENSIONS.fullmatch(line).groups())) != dimensions:
                    self.fail("inconsistent dimensions across spin blocks")
            else:
                break
            line = self.next()
        return line


def read_table(lines, line, nions, orbitals):
    fields = line.split()
    if not fields or fields[0].lower() != "ion" or fields[-1].lower() != "tot":
        lines.fail("expected orbital table with ion and tot columns")
    labels = tuple(label.lower() for label in fields[1:-1])
    if not labels or len(set(labels)) != len(labels) or (orbitals and labels != orbitals):
        lines.fail("invalid or inconsistent orbital columns")

    width = len(labels) + 2
    values = np.empty((nions, len(labels)), dtype=np.float32)
    for atom in range(nions):
        row = lines.next().split()
        if len(row) != width or row[0] != str(atom + 1):
            lines.fail(f"expected ion {atom + 1} with {width - 1} projection values")
        try:
            values[atom] = row[1:-1]
            total = float(row[-1])
        except ValueError:
            lines.fail("invalid projection value")
        if not np.isfinite(total) or total < 0:
            lines.fail("invalid scalar projection total")

    total_row = lines.next().split()
    if len(total_row) != width or total_row[0].lower() != "tot":
        lines.fail("missing orbital totals")
    try:
        totals = np.asarray(total_row[1:], dtype=float)
    except ValueError:
        lines.fail("invalid orbital totals")
    if not np.all(np.isfinite(values)) or np.any(values < 0) or not np.all(np.isfinite(totals)) or np.any(totals < 0):
        lines.fail("invalid scalar weights. Noncollinear/SOC projections are unsupported")
    return labels, values


def parse_procar(path):
    with path.open(encoding="ascii", errors="replace") as handle:
        lines = Lines(handle)
        line = lines.next()
        while line and not DIMENSIONS.fullmatch(line):
            if not (line.lower().startswith("procar") or SPIN.fullmatch(line)):
                lines.fail("unexpected header before dimensions")
            if "phase" in line.lower():
                lines.fail("phase-resolved projections are unsupported")
            if SPIN.fullmatch(line) and int(SPIN.fullmatch(line)[1]) != 1:
                lines.fail("invalid first spin component")
            line = lines.next()
        if not line:
            lines.fail("missing k-point, band and ion counts")
        dimensions = tuple(map(int, DIMENSIONS.fullmatch(line).groups()))
        nk, nb, nions = dimensions
        if min(dimensions) < 1:
            lines.fail("empty data dimensions")

        kpoints = np.empty((nk, 3))
        orbitals = ()
        energies = []
        weights = []
        line = lines.next()
        for spin in (1, 2):
            if not line and spin == 2:
                break
            line = lines.block_start(line, spin, dimensions)
            energy = np.empty((nk, nb))
            projection = None
            for k in range(nk):
                match = KPOINT.fullmatch(line.split("=", 1)[0] + "=") if "=" in line else None
                if not match or int(match[1]) != k + 1:
                    lines.fail(f"expected k-point {k + 1} in spin block {spin}")
                try:
                    coords = [number(val) for val in re.findall(NUMBER, match[2])]
                    if len(coords) != 3 or re.sub(NUMBER, "", match[2]).strip():
                        lines.fail("expected three coordinates")
                except ValueError as error:
                    lines.fail(str(error))
                if spin == 1:
                    kpoints[k] = coords
                elif not np.allclose(kpoints[k], coords, rtol=0, atol=5e-7):
                    lines.fail("k-points differ between spin blocks")

                for b in range(nb):
                    match = BAND.match(lines.next() if b == 0 else line)
                    if not match or int(match[1]) != b + 1:
                        lines.fail(f"expected band {b + 1} at k-point {k + 1}")
                    energy[k, b] = number(match[2])
                    orbitals, values = read_table(lines, lines.next(), nions, orbitals)
                    if projection is None:
                        projection = np.empty((nk, nb, nions, len(orbitals)), dtype=np.float32)
                    projection[k, b] = values
                    line = lines.next()
                    if line.split()[:1] == ["ion"]:
                        lines.fail("extra orbital table: noncollinear/SOC or phase projections are unsupported")
            energies.append(energy)
            weights.append(projection)
        if line:
            lines.fail("unexpected data after collinear spin blocks")
    if len(energies) == 1:
        return Bands(kpoints, energies[0][None], weights[0][None], orbitals)
    return Bands(kpoints, np.stack(energies), np.stack(weights), orbitals)


def load_procar(path, cache_dir):
    before = signature(path)
    source = fingerprint(path)
    if signature(path) != before:
        raise ValueError("PROCAR changed while computing its fingerprint")
    if cache_dir is None:
        target = None
    else:
        target = cache_dir / f"{CACHE_VERSION}-{source['sha256']}.npz"
        try:
            with np.load(target, allow_pickle=False) as stored:
                data = Bands(
                    stored["kpoints"], stored["energy"], stored["weights"],
                    tuple(stored["orbitals"].tolist()),
                )
            if data.weights.shape[:3] != data.energy.shape or data.kpoints.shape != (data.energy.shape[1], 3):
                raise ValueError("invalid cache dimensions")
            if signature(path) != before:
                raise RuntimeError("PROCAR changed while reading its cache")
            return data, source, True
        except (OSError, ValueError, KeyError, IndexError, BadZipFile):
            pass

    try:
        data = parse_procar(path)
    except EOFError as error:
        raise ValueError(f"incomplete {path.name}: {error}") from error
    if signature(path) != before:
        raise ValueError("PROCAR changed while being parsed")
    if target is not None:
        temporary = None
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez(handle, kpoints=data.kpoints, energy=data.energy,
                         weights=data.weights, orbitals=data.orbitals)
            temporary.replace(target)
        except OSError:
            # Reading and plotting can still succeed when caching is unavailable.
            pass
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
    return data, source, False
