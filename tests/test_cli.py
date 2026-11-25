import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample(tmp_path):
    folder = tmp_path / "calculation"
    folder.mkdir()
    (folder / "POSCAR").write_text(
        "test cell\n2\n1 0 0\n0 1 0\n0 0 1\nNi O\n2 1\nDirect\n"
        "0 0 0\n0 .5 .5\n.5 0 0\n"
    )
    (folder / "KPOINTS").write_text(
        "two disjoint segments\n3\nLine-mode\nReciprocal\n"
        "0 0 0 ! G\n.5 0 0 ! X\n\n.5 .5 0 ! M\n0 .5 0 ! Y\n"
    )
    (folder / "OUTCAR").write_text(" E-fermi : 7.0\n E-fermi : 8.0 XC(G=0): 0\n")
    write_procar(folder)
    return folder


def write_procar(folder, *, spins=1, noncollinear=False, shared=False):
    points = [(0, 0), (.25, 0), (.5, 0), (.5, .5), (.25, .5), (0, .5)]
    if shared:
        points = [(0, 0), (.25, 0), (.5, 0), (.5, .25), (.5, .5)]
    table = (
        "ion s py pz px dxy dyz dz2 dxz x2-y2 tot\n"
        "1 .010 .020 .030 .040 .060 .000 .040 .000 .080 .280\n"
        "2 .020 .010 .010 .010 .020 .000 .040 .000 .000 .110\n"
        "3 .200 .050 .010 .010 .000 .000 .000 .000 .000 .270\n"
        "tot .230 .080 .050 .060 .080 .000 .080 .000 .080 .660\n"
    )
    lines = ["PROCAR lm decomposed\n"]
    for spin in range(spins):
        lines.append(f"# of k-points: {len(points)} # of bands: 2 # of ions: 3\n")
        for i, (x, y) in enumerate(points, 1):
            lines.append(f"\n k-point {i} : {x:.8f} {y:.8f} 0.0 weight = 0.0\n")
            for band in (1, 2):
                lines.append(f"band {band} # energy {8 + (i-1)*.1 + band-1 + .2*spin:.5f} # occ. 1\n")
                lines.append(table)
                if noncollinear:
                    lines.extend([table] * 3)
    (folder / "PROCAR").write_text("".join(lines))


def invoke(sample, *args, in_calculation=False):
    env = dict(
        os.environ,
        PYTHONPATH=str(PROJECT / "src"),
        PYTHONIOENCODING="utf-8",
        XDG_CACHE_HOME=str(sample.parent / "cache"),
        MPLCONFIGDIR=str(sample.parent / "mpl"),
    )
    return subprocess.run(
        [sys.executable, "-m", "jiang_bandplot", *map(str, args)],
        cwd=sample if in_calculation else sample.parent, env=env, encoding="utf-8", capture_output=True,
    )


def succeed(run):
    assert run.returncode == 0, run.stderr
    return run.stdout


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_explains_the_public_options_in_chinese_without_calculation_files(sample, flag):
    help_text = succeed(invoke(sample, flag))
    assert help_text.index("快速开始") < help_text.index("常用参数") < help_text.index("进阶参数")
    assert "bandplot -orbitals Ni:dxy O:p" in help_text
    assert "bandplot --energy -0.5 0.5" in help_text
    for option in (
        "--orbitals", "--energy", "--fermi", "--fermi-outcar", "--plain",
        "--rank", "--no-rank", "--output", "--scale", "--formats", "--rank-limit",
        "--rank-decay", "--normalization", "--layout", "--spin", "--dpi", "--title", "--config",
        "--raw-energy", "--structure", "--kpoints", "--cache-dir", "--no-cache", "--json", "--version",
    ):
        assert option in help_text
    assert "未压缩" in help_text and "OUTCAR" in help_text


@pytest.mark.parametrize("command,text", [("plot", "普通能带"), ("rank", "排名"), ("inspect", "轨道")])
def test_subcommand_help_has_chinese_descriptions(sample, command, text):
    assert text in succeed(invoke(sample, command, "--help"))


def test_inspect_checks_path_and_identifies_elements(sample):
    report = json.loads(succeed(invoke(sample, "inspect", sample, "--json")))
    assert report["orbitals"][-1] == "x2-y2"
    assert report["elements"] == {"Ni": [1, 2], "O": [3]}
    assert report["outcar_fermi_eV"] == 8
    assert report["path"]["labels"][1] == "X | M"
    assert report["path"]["slices"] == [[0, 3], [3, 6]]
    assert report["path"]["ticks_inv_angstrom"] == pytest.approx([0, 1.5707963, 3.1415927])


def test_kpoints_numeric_fourth_column_is_not_a_symmetry_label(sample):
    file = sample / "KPOINTS"
    file.write_text(file.read_text().replace("! G", "1").replace("! X", "1 X")
                    .replace("! M", "1").replace("! Y", "! 1"))
    report = json.loads(succeed(invoke(sample, "inspect", sample, "--json")))
    assert report["path"]["labels"] == ["", "X", "1"]


def test_plot_many_channels_uses_common_linear_scale_and_writes_exports(sample):
    out = sample.parent / "many"
    succeed(invoke(sample, "plot", sample, "--orbitals", "Ni:dxy", "Ni:dx2-y2",
                   "O:p", "Ni:dz2", "1:dxz", "O:s", "--normalization", "mean",
                   "--energy", "-1", "2", "--fermi", "8", "--output", out,
                   "--formats", "png", "pdf", "svg", "png-pdf"))
    record = json.loads((out / "plot.json").read_text())
    assert record["layout"] == "panels"
    assert record["energy_reference"]["value_eV"] == 8
    assert record["channels"][0]["max_visible_weight"] == pytest.approx(.04)
    assert record["channels"][2]["max_visible_weight"] == pytest.approx(.07)
    assert record["channels"][4]["max_visible_weight"] == 0
    assert record["channels"][5]["max_visible_weight"] == pytest.approx(.2)
    assert record["marker"]["formula"] == "area_pt2 = scale * weight"
    assert record["marker"]["scale_pt2_per_weight"] > 0
    assert record["sources"]["PROCAR"]["sha256"] == hashlib.sha256((sample / "PROCAR").read_bytes()).hexdigest()
    assert {file.name for file in out.iterdir()} == {
        "bands.png", "bands.svg", "bands.pdf", "bands_png.pdf", "plot.json"
    }
    assert "<image" not in (out / "bands.svg").read_text(encoding="utf-8")
    other = sample.parent / "repeat"
    succeed(invoke(sample, "plot", "--config", out / "plot.json", "--output", other))
    assert {file.name for file in other.iterdir()} == {file.name for file in out.iterdir()}
    assert (out / "bands.png").read_bytes() == (other / "bands.png").read_bytes()


def test_one_channel_fixed_scale_and_site_sum(sample):
    out = sample.parent / "single"
    succeed(invoke(sample, "plot", sample, "--orbitals", "1,2:dxy", "--scale", "96",
                   "--fermi-outcar", sample / "OUTCAR", "--output", out, "--formats", "png"))
    record = json.loads((out / "plot.json").read_text())
    assert record["layout"] == "overlay"
    assert record["channels"][0]["max_visible_weight"] == pytest.approx(.08)
    assert record["marker"]["scale_pt2_per_weight"] == 96


def test_collinear_shared_endpoint_and_cache_invalidation(sample):
    write_procar(sample, spins=2, shared=True)
    (sample / "KPOINTS").write_text(
        "shared\n3\nLine-mode\nReciprocal\n"
        "0 0 0 ! G\n.5 0 0 ! X\n.5 0 0 ! X\n.5 .5 0 ! M\n"
    )
    args = ("inspect", sample, "--json")
    first = json.loads(succeed(invoke(sample, *args)))
    assert first["spins"] == ["up", "down"]
    assert first["path"]["slices"] == [[0, 3], [2, 5]]
    assert not first["cache_hit"]
    assert json.loads(succeed(invoke(sample, *args)))["cache_hit"]
    with (sample / "PROCAR").open("a") as handle:
        handle.write("\n")
    assert not json.loads(succeed(invoke(sample, *args)))["cache_hit"]
    out = sample.parent / "spin"
    succeed(invoke(sample, "plot", sample, "--plain", "--spin", "down", "--fermi", "8",
                   "--output", out, "--formats", "svg"))
    assert json.loads((out / "plot.json").read_text())["spins"] == ["down"]


def test_noncollinear_and_truncated_inputs_are_rejected(sample):
    write_procar(sample, noncollinear=True)
    result = invoke(sample, "inspect", sample)
    assert result.returncode == 2 and "noncollinear" in result.stderr.lower()
    write_procar(sample)
    file = sample / "PROCAR"
    file.write_text(file.read_text()[:500])
    result = invoke(sample, "inspect", sample)
    assert result.returncode == 2 and "PROCAR" in result.stderr


def test_truncated_second_spin_header_is_not_scalar(sample):
    with (sample / "PROCAR").open("a") as handle:
        handle.write("# of k-points: 6 # of bands: 2 # of ions: 3\n")
    result = invoke(sample, "inspect", sample)
    assert result.returncode == 2 and "spin block 2" in result.stderr


def test_only_uncompressed_procar_is_read(sample):
    source = sample / "PROCAR"
    (sample / "PROCAR.gz").write_bytes(gzip.compress(source.read_bytes()))
    report = json.loads(succeed(invoke(sample, "inspect", sample, "--json", "--no-cache")))
    assert report["sources"]["PROCAR"]["path"].endswith("PROCAR")
    source.unlink()
    result = invoke(sample, "inspect", sample, "--json", "--no-cache")
    assert result.returncode == 2 and "uncompressed PROCAR" in result.stderr
    assert not source.exists()


def test_replay_can_verify_relocated_input_files(sample):
    out = sample.parent / "original"
    succeed(invoke(sample, "plot", sample, "--plain", "--fermi-outcar", "--output", out, "--formats", "png"))
    relocated = sample.parent / "relocated"
    sample.rename(relocated)
    other = relocated.parent / "replay"
    succeed(invoke(relocated, "plot", relocated, "--config", out / "plot.json", "--output", other))
    assert (out / "bands.png").read_bytes() == (other / "bands.png").read_bytes()
    updated = relocated.parent / "edited"
    shutil.copytree(relocated, updated)
    with (updated / "PROCAR").open("a") as handle:
        handle.write("\n")
    denied = invoke(updated, "plot", updated, "--config", out / "plot.json", "--output", updated / "out")
    assert denied.returncode == 2 and "input changed" in denied.stderr


def test_no_implicit_fermi_and_no_mismatched_path(sample):
    (sample / "OUTCAR").unlink()
    result = invoke(sample, "plot", sample, "--plain")
    assert result.returncode == 2 and "--fermi" in result.stderr
    path = sample / "KPOINTS"
    path.write_text(path.read_text().replace(".5 0 0 ! X", ".4 0 0 ! X"))
    result = invoke(sample, "plot", sample, "--plain", "--fermi", "8")
    assert result.returncode == 2 and "KPOINTS" in result.stderr
    assert not (sample / "figures").exists()


def test_short_plot_command_defaults_to_local_inputs_and_lists_orbitals(sample):
    result = succeed(invoke(sample, "-orbitals", "Ni:dxy", "O:p", in_calculation=True))
    record = json.loads((sample / "figures" / "plot.json").read_text())
    assert "OUTCAR" in result and "Orbital ranking" in result
    assert record["energy_reference"]["value_eV"] == 8
    assert record["energy_range_eV"] == [-2, 3]
    assert record["marker"]["requested"] == "auto"
    assert record["sources"]["PROCAR"]["path"].endswith("PROCAR")
    assert record["formats"] == ["png"]
    assert {file.name for file in (sample / "figures").iterdir()} == {"bands.png", "plot.json"}
    succeed(invoke(sample, "--orbitals", "Ni:dxy", "--no-rank", "--formats", "png", in_calculation=True))
    assert (sample / "figures-2" / "bands.png").exists()


def test_range_only_ranks_and_near_fermi_can_outrank_larger_far_weight(sample):
    procar = sample / "PROCAR"
    lines = procar.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.startswith("band 1") or line.startswith("band 2"):
            near = line.startswith("band 1")
            lines[index] = f"band {1 if near else 2} # energy {8 if near else 9} # occ. 1"
            lines[index+2:index+6] = [
                "1 0 0 0 0 0 0 0 0 0 0",
                f"2 0 0 0 0 {'.1' if near else '0'} 0 0 0 0 {'.1' if near else '0'}",
                f"3 {'0' if near else '.3'} 0 0 0 0 0 0 0 0 {'0' if near else '.3'}",
                f"tot {'0 0 0 0 .1 0 0 0 0 .1' if near else '.3 0 0 0 0 0 0 0 0 .3'}",
            ]
    procar.write_text("\n".join(lines) + "\n")
    ranked = succeed(invoke(sample, "--energy", "-0.1", "1.1", in_calculation=True))
    assert ranked.index("Ni:dxy") < ranked.index("O:s")
    broad = succeed(invoke(sample, "rank", "--energy", "-0.1", "1.1", "--rank-decay", "4",
                           in_calculation=True))
    assert broad.index("O:s") < broad.index("Ni:dxy")
    empty = succeed(invoke(sample, "rank", "--energy", "2", "3", in_calculation=True))
    assert "No nonzero orbital projections" in empty
    procar.write_text(procar.read_text().replace("2 0 0 0 0 .1 0 0 0 0 .1", "2 0 0 0 0 0 0 0 0 0 0")
                      .replace("tot 0 0 0 0 .1 0 0 0 0 .1", "tot 0 0 0 0 0 0 0 0 0 0"))
    distant = succeed(invoke(sample, "rank", "--energy", "-0.1", "1.1", "--rank-decay", "0.001",
                             in_calculation=True))
    assert "O:s" in distant and "100.00%" in distant
    assert not (sample / "figures").exists()


def test_output_collision_preserves_existing_files(sample):
    out = sample.parent / "existing"
    out.mkdir()
    (out / "bands.svg").write_text("existing")
    result = invoke(sample, "plot", sample, "--plain", "--fermi", "8", "--output", out)
    assert result.returncode == 2 and (out / "bands.svg").read_text() == "existing"
