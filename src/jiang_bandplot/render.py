from colorsys import hsv_to_rgb
from math import ceil

from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from PIL import Image
import numpy as np


COLORS = ("#B01C63", "#006D9C", "#147D65", "#B65317", "#6445A2", "#8A6B17")


def channel_colors(count):
    if count <= len(COLORS):
        return COLORS[:count]
    return [hsv_to_rgb((index * .61803398875) % 1, .73, .67 + .19 * (index % 2))
            for index in range(count)]


def geometry(layout, nchannels, nspins):
    columns = min(3, nchannels) if layout == "panels" else 1
    rows = ceil(nchannels / columns) * nspins if layout == "panels" else nspins
    panel = (4.25, 3.65) if layout == "panels" else (7.2, 4.65)
    return rows, columns, panel


def draw_axis(ax, path, energy, energy_limits, reference_kind):
    traces = [
        np.stack((np.broadcast_to(path.x[start:stop], energy[start:stop].T.shape),
                  energy[start:stop].T), axis=-1)
        for start, stop in path.slices
    ]
    ax.add_collection(LineCollection(
        np.concatenate(traces) if len({trace.shape[1] for trace in traces}) == 1
        else [row for trace in traces for row in trace],
        colors="#77818A", alpha=.58, linewidths=.6, zorder=1,
    ))
    for tick in path.ticks[1:-1]:
        ax.axvline(tick, color="#C5C9CF", linewidth=.75, zorder=0)
    if reference_kind != "raw" and energy_limits[0] <= 0 <= energy_limits[1]:
        ax.axhline(0, color="#454A52", linewidth=.8, linestyle="--", zorder=2)
    ax.set(
        xlim=(path.ticks[0], path.ticks[-1]),
        ylim=energy_limits,
        xticks=path.ticks,
        xticklabels=path.labels,
    )
    ax.tick_params(direction="in", top=True, right=True, length=3.5)


def draw_weights(ax, channel, spin, energy, path, limits, scale, color):
    weights = channel.weights[spin]
    visible = (weights > 0) & (energy >= limits[0]) & (energy <= limits[1])
    k, b = np.nonzero(visible)
    ax.scatter(path.x[k], energy[k, b], s=scale * weights[k, b], color=color,
               alpha=.72, linewidths=0, zorder=3, clip_on=True)


def make_figure(path, energy, channels, spins, options, marker_scale):
    layout, limits = options["layout"], options["energy_range_eV"]
    rows, cols, (width, height) = geometry(layout, len(channels), len(spins))
    figure = Figure(figsize=(width * cols, height * rows + .45), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, cols, squeeze=False)
    palette = list(zip(channels, channel_colors(len(channels))))
    rows_per_spin = rows // len(spins)
    for index, spin in enumerate(spins):
        grid = axes[index * rows_per_spin:(index + 1) * rows_per_spin].flat
        for cell, ax in enumerate(grid):
            if layout == "panels" and cell >= len(channels):
                ax.set_axis_off()
                continue
            draw_axis(ax, path, energy[index], limits, options["energy_reference"]["kind"])
            active = [palette[cell]] if layout == "panels" else palette
            for channel, color in active:
                draw_weights(ax, channel, index, energy[index], path, limits, marker_scale, color)
            if layout == "panels":
                channel, color = active[0]
                title = channel.name + (f"  ·  spin {spin}" if spin != "scalar" else "")
                ax.set_title(title, color=color, loc="left")
            else:
                if spin != "scalar":
                    ax.set_title(f"Spin {spin}", loc="left")
                for channel, color in active:
                    ax.scatter([], [], s=26, color=color, linewidths=0, label=channel.name)
                if active:
                    ax.legend(frameon=False, fontsize=9, loc="lower center",
                              bbox_to_anchor=(.5, 1.01), ncol=min(4, len(active)))
    label = "E (eV)" if options["energy_reference"]["kind"] == "raw" else r"$E-E_{\mathrm{F}}$ (eV)"
    figure.supylabel(label)
    if options["title"]:
        figure.suptitle(options["title"])
    if marker_scale is not None and channels:
        figure.supxlabel(
            f"{options['normalization'].capitalize()} projection  ·  "
            f"weight 0.10 → {marker_scale * .10:.3g} pt²",
            fontsize=9, color="#4A5159",
        )
    return figure


def png_pdf(png_path, pdf_path, dpi):
    with Image.open(png_path) as source:
        rgb = np.asarray(source.convert("RGB"))
    height, width, _ = rgb.shape
    page = Figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    page.figimage(rgb, origin="upper")
    page.savefig(pdf_path, dpi=dpi)


def write_figures(out, path, energy, channels, spins, options, marker_scale):
    style = {
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.edgecolor": "#20242B",
        "axes.labelcolor": "#20242B", "path.simplify": False,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
    }
    with rc_context(style):
        figure = make_figure(path, energy, channels, spins, options, marker_scale)
        png = out / "bands.png"
        formats = options["formats"]
        dpi = options["dpi"]
        if "png" in formats or "png-pdf" in formats:
            figure.savefig(png, dpi=dpi)
        for format_name in ("pdf", "svg"):
            if format_name in formats:
                figure.savefig(out / f"bands.{format_name}", dpi=dpi)
        if "png-pdf" in formats:
            png_pdf(png, out / "bands_png.pdf", dpi)
            if "png" not in formats:
                png.unlink()
