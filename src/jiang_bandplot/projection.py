from dataclasses import dataclass
import re

import numpy as np


ALIASES = {"x2-y2": "dx2-y2", "z2": "dz2", "dz^2": "dz2", "d3z2-r2": "dz2"}


def normalized(name):
    name = name.strip().lower().replace("−", "-")
    return ALIASES.get(name, name)


@dataclass
class Channel:
    name: str
    atoms: list
    orbitals: list
    weights: np.ndarray


def select_channel(name, weights, labels, structure, normalization):
    selector, sep, orbital = name.partition(":")
    if not sep or not selector or not orbital:
        raise ValueError(f"invalid orbital channel {name!r}. Use Element:orbital or 1,2:orbital")
    if selector == "*":
        atoms = list(range(1, structure.nions + 1))
    elif re.fullmatch(r"\d+(?:,\d+)*", selector):
        atoms = list(dict.fromkeys(map(int, selector.split(","))))
    else:
        atoms = [atom for key, group in structure.elements.items() if key.casefold() == selector.casefold()
                 for atom in group]
    if not atoms or min(atoms) < 1 or max(atoms) > structure.nions:
        raise ValueError(f"unknown element or out-of-range ion selection: {selector!r}")

    query = normalized(orbital)
    orbitals = [i for i, label in enumerate(labels) if normalized(label) == query]
    if not orbitals and query in {"p", "d", "f"}:
        orbitals = [i for i, label in enumerate(labels) if normalized(label).startswith(query)]
    if not orbitals:
        available = ", ".join(labels)
        raise ValueError(f"unknown orbital {orbital!r}. PROCAR columns: {available}")
    selected = np.zeros(weights.shape[:3], dtype=float)
    for atom in atoms:
        for index in orbitals:
            selected += weights[:, :, :, atom - 1, index]
    if normalization == "mean":
        selected /= len(atoms)
    return Channel(name, atoms, [labels[i] for i in orbitals], selected)


def area_scale(channels, visible, requested, panel_area):
    if requested != "auto":
        scale = float(requested)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("--scale must be auto or a positive finite number")
        return scale
    if not channels:
        return None
    values = np.concatenate([channel.weights[visible & (channel.weights > 0)] for channel in channels])
    if not len(values):
        return 1.0
    # The percentile controls typical marks. A shared upper bound keeps every
    # outlier proportional without clipping its area.
    target = float(np.clip(24 * np.sqrt(panel_area / len(values)), 7, 24))
    return min(target / np.quantile(values, .95), 64.0 / values.max())


def rank_orbitals(weights, energy, visible, path, labels, structure, decay, limit):
    """Rank ion/column projections with energy and trapezoidal k-path weights."""
    if not visible.any():
        return []
    k_weights = np.zeros(len(path.x))
    for start, stop in path.slices:
        half_steps = np.diff(path.x[start:stop]) / 2
        k_weights[start:stop - 1] += half_steps
        k_weights[start + 1:stop] += half_steps
    contributes = visible & np.any(weights, axis=(-2, -1)) & (k_weights[None, :, None] > 0)
    if not contributes.any():
        return []
    distance = np.abs(energy[contributes])
    # Subtract a common offset before exponentiation; it cancels in all shares.
    state_weight = np.zeros_like(energy)
    with np.errstate(over="ignore"):
        state_weight[contributes] = np.exp(-(distance - distance.min()) / decay)
    state_weight *= k_weights[None, :, None]

    by_ion = np.einsum("skb,skbao->ao", state_weight, weights, dtype=np.float64)
    total = float(by_ion.sum())
    if total <= 0:
        return []
    groups = structure.elements or {str(i): [i] for i in range(1, structure.nions + 1)}
    entries = []
    for group, atoms in groups.items():
        values = by_ion[np.asarray(atoms) - 1].sum(axis=0)
        entries.extend((f"{group}:{normalized(label)}", float(value / total))
                       for label, value in zip(labels, values) if value > 0)
    entries.sort(key=lambda row: (-row[1], row[0]))
    return [{"channel": name, "weighted_share": share} for name, share in entries[:limit]]
