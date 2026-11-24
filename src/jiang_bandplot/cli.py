import argparse
from itertools import count
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import tempfile

import numpy as np

from . import __version__
from .help_text import HELP_TEXT
from .procar import fingerprint, load_procar, signature
from .projection import area_scale, rank_orbitals, select_channel
from .vasp import path_from, read_fermi, structure_from


FORMATS = ("png", "pdf", "svg", "png-pdf")
DEFAULT_WINDOW = (-2.0, 3.0)


class BandplotParser(argparse.ArgumentParser):
    def format_help(self):
        return HELP_TEXT


def arguments():
    parser = BandplotParser(prog="bandplot", allow_abbrev=False)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=argparse.ArgumentParser)
    inspect = sub.add_parser("inspect", allow_abbrev=False, description="检查轨道、自旋、k 路径和费米能。")
    inspect.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    plot = sub.add_parser("plot", allow_abbrev=False, description="绘制普通能带或胖能带，可同时打印排名。")
    rank = sub.add_parser("rank", allow_abbrev=False, description="打印能窗内的轨道排名。")
    for cmd in (inspect, plot, rank):
        cmd.add_argument("directory", nargs="?", type=Path, metavar="目录", help="VASP 计算目录，默认当前目录")
        cmd.add_argument("--structure", type=Path, metavar="文件", help="使用匹配的 POSCAR 或 CONTCAR")
        cmd.add_argument("--kpoints", type=Path, metavar="文件", help="使用匹配的线模式 KPOINTS")
        cmd.add_argument("--cache-dir", type=Path, metavar="目录", help="指定 PROCAR 缓存目录")
        cmd.add_argument("--no-cache", action="store_true", help="本次跳过缓存的读取和写入")
    channels = plot.add_mutually_exclusive_group()
    channels.add_argument("--plain", action="store_true", default=None, help="画普通能带，默认开启")
    channels.add_argument("--orbitals", "-orbitals", nargs="+", metavar="通道", help="选择胖能带通道，如 Ni:dxy O:p 1,2:dz2")
    for cmd in (plot, rank):
        reference = cmd.add_mutually_exclusive_group()
        reference.add_argument("--fermi", type=float, metavar="eV", help="手动指定费米能")
        reference.add_argument("--fermi-outcar", nargs="?", const="OUTCAR", metavar="文件",
                               help="从指定 OUTCAR 读费米能，默认用计算目录的 OUTCAR")
        cmd.set_defaults(raw_energy=None)
        if cmd is plot:
            reference.add_argument("--raw-energy", action="store_true", default=None, help="用 PROCAR 原始能量绘图")
        cmd.add_argument("--energy", "-e", nargs=2, type=float, metavar=("下限", "上限"),
                         help="能窗，默认相对费米能 -2 到 3 eV")
        cmd.add_argument("--spin", choices=("all", "up", "down"), help="选择自旋，默认 all")
        cmd.add_argument("--rank-decay", type=float, metavar="eV", help="排名的能量衰减尺度，默认 0.5 eV")
        cmd.add_argument("--rank-limit", type=int, metavar="整数", help="显示前几项，默认 8")
    ranking = plot.add_mutually_exclusive_group()
    ranking.add_argument("--rank", action="store_true", dest="ranking", default=None, help="绘图时打印轨道排名")
    ranking.add_argument("--no-rank", action="store_false", dest="ranking", help="绘图时跳过轨道排名")
    plot.add_argument("--scale", metavar="auto|数值", help="自动调节或固定点面积比例")
    plot.add_argument("--layout", choices=("auto", "overlay", "panels"), help="自动布局、叠加或分面，默认 auto")
    plot.add_argument("--normalization", choices=("sum", "mean"), help="对原子投影求和或取平均，默认 sum")
    plot.add_argument("--dpi", type=int, metavar="整数", help="PNG 分辨率，默认 300")
    plot.add_argument("--title", metavar="文字", help="设置图的标题")
    plot.add_argument("--formats", nargs="+", choices=FORMATS, help="选择输出格式，默认全输出")
    plot.add_argument("--output", type=Path, metavar="目录", help="使用新目录保存图和 plot.json")
    plot.add_argument("--config", type=Path, metavar="plot.json", help="沿用已有设置复画，并校验输入文件")
    return parser


def command_line(argv):
    if argv and argv[0] in {"inspect", "plot", "rank", "-h", "--help", "--version"}:
        return argv
    flags = {item.split("=", 1)[0] for item in argv}
    rank_options = {"--energy", "-e", "--rank", "--rank-decay", "--rank-limit"}
    plot_options = {"--orbitals", "-orbitals", "--plain", "--no-rank", "--scale", "--raw-energy",
                    "--output", "--config", "--layout", "--normalization", "--dpi", "--title", "--formats"}
    if flags & rank_options and not flags & plot_options:
        return ["rank", *(item for item in argv if item != "--rank")]
    return ["plot", *argv]


def json_text(value):
    return json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n"


def cache_location(args):
    if args.no_cache:
        return None
    if args.cache_dir is not None:
        return args.cache_dir.expanduser().resolve()
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "jiang-bandplot"


def relocated_file(path, origin, directory):
    path_type = PureWindowsPath if PureWindowsPath(origin).drive else PurePosixPath
    try:
        relative = path_type(path).relative_to(path_type(origin))
    except ValueError:
        return Path(path)
    return directory.joinpath(*relative.parts)


def calc_files(args, replay):
    directory = (args.directory or Path(".")).expanduser().resolve()
    if replay and args.directory is None:
        directory = Path(replay["directory"])
    structure_file = args.structure or directory / "POSCAR"
    if not structure_file.is_file() and args.structure is None and (directory / "CONTCAR").is_file():
        structure_file = directory / "CONTCAR"
    kpath_file = args.kpoints or directory / "KPOINTS"
    if replay:
        origin = replay["directory"]
        structure_file = args.structure or relocated_file(replay["sources"]["structure"]["path"], origin, directory)
        kpath_file = args.kpoints or relocated_file(replay["sources"]["KPOINTS"]["path"], origin, directory)
    return directory, structure_file.resolve(), kpath_file.resolve()


def stable_input(reader, path, *args):
    before = signature(path)
    value = reader(path, *args)
    source = fingerprint(path)
    if signature(path) != before:
        raise ValueError(f"{path.name} changed while being read")
    return value, source


def load_inputs(args, directory, structure_file, kpath_file):
    procar = directory / "PROCAR"
    if not procar.is_file():
        raise ValueError(f"expected uncompressed PROCAR at {procar}")
    bands, provenance, hit = load_procar(procar, cache_location(args))
    sources = {"PROCAR": provenance}
    structure = path = None
    notes = []
    try:
        structure, sources["structure"] = stable_input(structure_from, structure_file, bands.weights.shape[3])
        if not structure.elements:
            notes.append("POSCAR has no species labels. Use numeric ion selectors.")
    except (OSError, ValueError) as error:
        notes.append(str(error))
    if structure is not None:
        try:
            path, sources["KPOINTS"] = stable_input(path_from, kpath_file, structure, bands.kpoints)
        except (OSError, ValueError) as error:
            notes.append(str(error))
    return bands, structure, path, sources, hit, notes


def inspect_inputs(args, directory, loaded):
    bands, structure, path, sources, hit, notes = loaded
    candidate = directory / "OUTCAR"
    fermi = read_fermi(candidate) if candidate.is_file() else None
    report = {
        "directory": str(directory),
        "kpoints": bands.energy.shape[1],
        "bands": bands.energy.shape[2],
        "ions": bands.weights.shape[3],
        "orbitals": bands.orbitals,
        "elements": structure.elements if structure else {},
        "spins": bands.spins,
        "outcar_fermi_eV": fermi,
        "outcar_fermi_path": str(candidate) if fermi is not None else None,
        "path": path.report() if path else None,
        "cache_hit": hit,
        "sources": sources,
        "notes": notes,
    }
    if args.json:
        print(json_text(report), end="")
    else:
        print(f"{report['kpoints']} k-points · {report['bands']} bands · {report['ions']} ions")
        print("Spin:", ", ".join(bands.spins))
        print("Orbitals:", ", ".join(bands.orbitals))
        if report["elements"]:
            print("Elements (1-based ions):")
            for name, indices in report["elements"].items():
                print(f"  {name}: {', '.join(map(str, indices))}")
        if path:
            print("K path:", " → ".join(label or "?" for label in path.labels))
        if fermi is not None:
            print(f"Fermi energy candidate (OUTCAR): {fermi:g} eV")
        print("PROCAR cache:", "hit" if hit else "read source")
        for note in notes:
            print(f"Note: {note}")


def energy_reference(args, directory, replay, sources):
    recorded = replay["energy_reference"] if replay else None
    if args.raw_energy or recorded and recorded["kind"] == "raw":
        return {"kind": "raw", "value_eV": None}
    outcar = args.fermi_outcar
    if recorded and recorded["kind"] == "outcar":
        outcar = outcar or relocated_file(recorded["path"], replay["directory"], directory)
    elif recorded and outcar:
        raise ValueError("--fermi-outcar can only relocate the OUTCAR recorded in plot.json")
    fermi = recorded["value_eV"] if recorded else args.fermi
    if outcar or fermi is None:
        default_outcar = not outcar
        file = directory / "OUTCAR" if not outcar or outcar == "OUTCAR" else Path(outcar).expanduser().resolve()
        if default_outcar and not file.is_file():
            raise ValueError("no OUTCAR found. Provide --fermi EV or --fermi-outcar FILE")
        fermi, sources["fermi_OUTCAR"] = stable_input(read_fermi, file)
        if fermi is None:
            raise ValueError(f"{file} has no E-fermi value. Provide --fermi EV or --fermi-outcar FILE")
        return {"kind": "outcar", "path": str(file), "value_eV": fermi}
    if not np.isfinite(fermi):
        raise ValueError("--fermi must be finite")
    return {"kind": "manual", "value_eV": fermi}


def energy_selection(args, bands, ref, replay):
    spins = bands.spins
    if replay:
        spins = replay["spins"]
    elif args.spin in ("up", "down"):
        spins = [args.spin]
    spin = slice(None)
    if tuple(spins) != bands.spins:
        if len(spins) != 1 or spins[0] not in bands.spins:
            raise ValueError(f"requested spin is unavailable. Choose from: {', '.join(bands.spins)}")
        index = bands.spins.index(spins[0])
        spin = slice(index, index + 1)
    energy = bands.energy[spin] - (ref["value_eV"] or 0)
    limits = replay["energy_range_eV"] if replay else args.energy
    if limits is None:
        limits = DEFAULT_WINDOW if ref["kind"] != "raw" else [float(energy.min()), float(energy.max())]
        if limits[0] == limits[1]:
            limits = [limits[0] - .5, limits[1] + .5]
    if len(limits) != 2 or not np.all(np.isfinite(limits)) or limits[0] >= limits[1]:
        raise ValueError("--energy requires finite MIN < MAX")
    return spins, bands.weights[spin], energy, limits


def print_reference(ref):
    if ref["kind"] == "outcar":
        print(f"E-fermi = {ref['value_eV']:g} eV ({ref['path']})", flush=True)
    elif ref["kind"] == "manual":
        print(f"E-fermi = {ref['value_eV']:g} eV (explicit)", flush=True)


def print_ranking(args, weights, energy, limits, path, labels, structure, settings=None):
    decay = settings["decay_eV"] if settings else (.5 if args.rank_decay is None else args.rank_decay)
    limit = settings["limit"] if settings else (8 if args.rank_limit is None else args.rank_limit)
    if not np.isfinite(decay) or decay <= 0:
        raise ValueError("--rank-decay must be a positive finite number")
    if limit <= 0:
        raise ValueError("--rank-limit must be a positive integer")
    visible = (energy >= limits[0]) & (energy <= limits[1])
    results = rank_orbitals(weights, energy, visible, path, labels, structure, decay, limit)
    print(f"Orbital ranking in [{limits[0]:g}, {limits[1]:g}] eV relative to E-fermi (decay {decay:g} eV)")
    if results:
        for index, row in enumerate(results, 1):
            print(f"  {index:>2}. {row['channel']:<16} {row['weighted_share']:>7.2%}")
        print("Shares of all projections in the window, weighted by k-path length and Fermi proximity.", flush=True)
    else:
        print("  No nonzero orbital projections in this window.", flush=True)
    return {"decay_eV": decay, "limit": limit, "channels": results}


def read_config(args):
    if args.config is None:
        return None
    overrides = ("orbitals", "plain", "fermi", "raw_energy", "energy", "scale", "spin",
                 "ranking", "rank_decay", "rank_limit",
                 "layout", "normalization", "dpi", "title", "formats")
    if any(getattr(args, name) is not None for name in overrides):
        raise ValueError("--config already defines the plot settings; only input paths, output and cache can be overridden")
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or config.get("schema") != 1:
        raise ValueError("unsupported plot.json schema")
    return config


def rank_inputs(args, directory, loaded):
    bands, structure, path, sources, _, notes = loaded
    if structure is None or path is None:
        raise ValueError("; ".join(notes))
    ref = energy_reference(args, directory, None, sources)
    _, weights, energy, limits = energy_selection(args, bands, ref, None)
    print_reference(ref)
    print_ranking(args, weights, energy, limits, path, bands.orbitals, structure)


def plot_inputs(args, directory, loaded, replay):
    from .render import geometry, write_figures

    bands, structure, path, sources, hit, notes = loaded
    if structure is None or path is None:
        raise ValueError("; ".join(notes))
    ref = energy_reference(args, directory, replay, sources)
    if replay:
        for name, source in replay["sources"].items():
            if name not in sources or sources[name]["sha256"] != source["sha256"]:
                raise ValueError(f"input changed since plot.json was saved: {name}")
        if ref["kind"] != replay["energy_reference"]["kind"] or ref["value_eV"] != replay["energy_reference"]["value_eV"]:
            raise ValueError("the energy reference no longer matches plot.json")
    normalization = replay["normalization"] if replay else (args.normalization or "sum")
    names = [channel["name"] for channel in replay["channels"]] if replay else (args.orbitals or [])
    if len(set(names)) != len(names):
        raise ValueError("orbital channel names must be unique")
    spins, weights, energy, limits = energy_selection(args, bands, ref, replay)
    visible = (energy >= limits[0]) & (energy <= limits[1])
    if not visible.any():
        print("Note: no sampled band energies inside the requested window.", file=sys.stderr)
    channels = [select_channel(name, weights, bands.orbitals, structure, normalization) for name in names]
    settings = replay.get("ranking") if replay else None
    rank_enabled = settings is not None if replay else args.ranking
    if rank_enabled is None:
        modified_ranking = args.rank_decay is not None or args.rank_limit is not None
        rank_enabled = modified_ranking or (bool(names) and ref["kind"] != "raw")
    if rank_enabled and ref["kind"] == "raw":
        raise ValueError("orbital ranking requires a Fermi energy. Use --fermi or --fermi-outcar")
    print_reference(ref)
    ranking = print_ranking(args, weights, energy, limits, path, bands.orbitals, structure, settings) if rank_enabled else None
    layout = replay["layout"] if replay else (args.layout or "auto")
    if layout == "auto":
        layout = "panels" if len(channels) > 4 else "overlay"
    if not channels:
        layout = "overlay"
    rows, columns, panel = geometry(layout, len(channels), len(spins))
    requested_scale = replay["marker"]["requested"] if replay else (args.scale or "auto")
    fixed = replay["marker"]["scale_pt2_per_weight"] if replay else None
    effective_scale = area_scale(channels, visible, fixed if fixed is not None else requested_scale,
                                 rows * columns * panel[0] * panel[1] / 30)
    dpi = replay["dpi"] if replay else (300 if args.dpi is None else args.dpi)
    if not 36 <= dpi <= 1200:
        raise ValueError("--dpi must be between 36 and 1200")
    formats = replay["formats"] if replay else (args.formats or FORMATS)
    record = {
        "schema": 1,
        "version": __version__,
        "directory": str(directory),
        "sources": sources,
        "cache_hit": hit,
        "energy_reference": ref,
        "energy_range_eV": limits,
        "ranking": ranking,
        "path": path.report(),
        "spins": spins,
        "normalization": normalization,
        "layout": layout,
        "channels": [
            {"name": ch.name, "atoms_1based": ch.atoms, "orbitals": ch.orbitals,
             "max_visible_weight": float(ch.weights[visible].max(initial=0))}
            for ch in channels
        ],
        "marker": {"formula": "area_pt2 = scale * weight",
                   "requested": requested_scale, "scale_pt2_per_weight": effective_scale},
        "dpi": dpi,
        "formats": formats,
        "title": replay["title"] if replay else args.title,
    }
    output = (args.output or directory / "figures").expanduser().absolute()
    if args.output and (output.exists() or output.is_symlink()):
        raise ValueError(f"output directory already exists: {output}. Choose a new --output")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bandplot-", dir=output.parent) as staging:
        working = Path(staging) / "figure"
        working.mkdir()
        write_figures(working, path, energy, channels, spins, record, effective_scale)
        (working / "plot.json").write_text(json_text(record), encoding="utf-8")
        if args.output:
            working.rename(output)
        else:
            for number in count(1):
                output = directory / ("figures" if number == 1 else f"figures-{number}")
                if output.exists() or output.is_symlink():
                    continue
                try:
                    working.rename(output)
                except OSError:
                    if output.exists() or output.is_symlink():
                        continue
                    raise
                break
    print(f"Saved figures and plot.json to {output}")


def main(argv=None):
    parser = arguments()
    args = parser.parse_args(command_line(list(sys.argv[1:] if argv is None else argv)))
    try:
        replay = read_config(args) if args.command == "plot" else None
        directory, structure_file, kpath_file = calc_files(args, replay)
        loaded = load_inputs(args, directory, structure_file, kpath_file)
        if args.command == "inspect":
            inspect_inputs(args, directory, loaded)
        elif args.command == "rank":
            rank_inputs(args, directory, loaded)
        else:
            plot_inputs(args, directory, loaded, replay)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(2, f"bandplot: {error}\n")
    return 0
