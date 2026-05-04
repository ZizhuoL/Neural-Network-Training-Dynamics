"""Dependency-light SVG plotting helpers for experiment artifacts."""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np


COLORS = [
    "#1b4965",
    "#d1495b",
    "#2a9d8f",
    "#f4a261",
    "#6d597a",
    "#457b9d",
    "#bc4749",
]


def _scale(values, lo, hi, out_lo, out_hi):
    values = np.asarray(values, dtype=float)
    if abs(hi - lo) < 1e-12:
        return np.full_like(values, (out_lo + out_hi) / 2.0)
    return out_lo + (values - lo) * (out_hi - out_lo) / (hi - lo)


def _svg_header(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
    ]


def write_multi_line_plot(series, path, title, x_label="epoch", y_label="value", width=900, height=560):
    path = Path(path)
    margin = {"left": 74, "right": 28, "top": 64, "bottom": 66}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    all_x = np.concatenate([np.asarray(item["x"], dtype=float) for item in series])
    all_y = np.concatenate([np.asarray(item["y"], dtype=float) for item in series])
    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    pad = 0.05 * (y_max - y_min + 1e-12)
    y_min -= pad
    y_max += pad

    lines = _svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" fill="#1f2933">{escape(title)}</text>')
    x0, y0 = margin["left"], height - margin["bottom"]
    lines.append(f'<line x1="{x0}" y1="{margin["top"]}" x2="{x0}" y2="{y0}" stroke="#222" stroke-width="1.2"/>')
    lines.append(f'<line x1="{x0}" y1="{y0}" x2="{width-margin["right"]}" y2="{y0}" stroke="#222" stroke-width="1.2"/>')

    for tick in np.linspace(x_min, x_max, 5):
        x = float(_scale(tick, x_min, x_max, x0, x0 + plot_w))
        lines.append(f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y0+5}" stroke="#222"/>')
        lines.append(f'<text x="{x}" y="{y0+24}" text-anchor="middle" font-family="Arial" font-size="12" fill="#333">{tick:.0f}</text>')
    for tick in np.linspace(y_min, y_max, 5):
        y = float(_scale(tick, y_min, y_max, y0, margin["top"]))
        lines.append(f'<line x1="{x0-5}" y1="{y}" x2="{x0}" y2="{y}" stroke="#222"/>')
        lines.append(f'<text x="{x0-10}" y="{y+4}" text-anchor="end" font-family="Arial" font-size="12" fill="#333">{tick:.3g}</text>')

    lines.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="14" fill="#333">{escape(x_label)}</text>')
    lines.append(f'<text x="20" y="{height/2}" text-anchor="middle" transform="rotate(-90 20 {height/2})" font-family="Arial" font-size="14" fill="#333">{escape(y_label)}</text>')

    legend_x = width - margin["right"] - 170
    legend_y = margin["top"] + 4
    for i, item in enumerate(series):
        color = COLORS[i % len(COLORS)]
        xs = _scale(item["x"], x_min, x_max, x0, x0 + plot_w)
        ys = _scale(item["y"], y_min, y_max, y0, margin["top"])
        points = " ".join(f"{float(x):.2f},{float(y):.2f}" for x, y in zip(xs, ys))
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{points}"/>')
        ly = legend_y + i * 22
        lines.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x+22}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x+28}" y="{ly+4}" font-family="Arial" font-size="13" fill="#333">{escape(item["label"])}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_bar_chart(labels, values, path, title, y_label="value", width=850, height=520):
    path = Path(path)
    margin = {"left": 74, "right": 28, "top": 64, "bottom": 110}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    values = np.asarray(values, dtype=float)
    y_max = float(max(values.max(), 1e-9)) * 1.12
    lines = _svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" fill="#1f2933">{escape(title)}</text>')
    x0, y0 = margin["left"], height - margin["bottom"]
    lines.append(f'<line x1="{x0}" y1="{margin["top"]}" x2="{x0}" y2="{y0}" stroke="#222" stroke-width="1.2"/>')
    lines.append(f'<line x1="{x0}" y1="{y0}" x2="{width-margin["right"]}" y2="{y0}" stroke="#222" stroke-width="1.2"/>')
    lines.append(f'<text x="20" y="{height/2}" text-anchor="middle" transform="rotate(-90 20 {height/2})" font-family="Arial" font-size="14" fill="#333">{escape(y_label)}</text>')

    slot = plot_w / len(labels)
    bar_w = slot * 0.62
    for i, (label, value) in enumerate(zip(labels, values)):
        h = float(value / y_max * plot_h)
        x = x0 + i * slot + (slot - bar_w) / 2.0
        y = y0 - h
        color = COLORS[i % len(COLORS)]
        lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}"/>')
        lines.append(f'<text x="{x + bar_w/2:.2f}" y="{y-6:.2f}" text-anchor="middle" font-family="Arial" font-size="12" fill="#333">{value:.3g}</text>')
        lines.append(f'<text x="{x + bar_w/2:.2f}" y="{y0+20}" text-anchor="end" transform="rotate(-35 {x + bar_w/2:.2f} {y0+20})" font-family="Arial" font-size="12" fill="#333">{escape(label)}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_decision_boundary(model, X, y, path, title, width=620, height=560, grid=80):
    path = Path(path)
    margin = 46
    x_min, x_max = float(X[:, 0].min() - 0.5), float(X[:, 0].max() + 0.5)
    y_min, y_max = float(X[:, 1].min() - 0.5), float(X[:, 1].max() + 0.5)
    xs = np.linspace(x_min, x_max, grid)
    ys = np.linspace(y_min, y_max, grid)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    pred, _ = model.forward(points)
    pred = pred.reshape(grid, grid)

    lines = _svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="20" fill="#1f2933">{escape(title)}</text>')
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    cell_w = plot_w / grid
    cell_h = plot_h / grid
    for i in range(grid):
        for j in range(grid):
            p = float(pred[i, j])
            color = "#e9f5f2" if p >= 0.5 else "#f8e9ec"
            x = margin + j * cell_w
            ypix = margin + i * cell_h
            lines.append(f'<rect x="{x:.2f}" y="{ypix:.2f}" width="{cell_w+0.2:.2f}" height="{cell_h+0.2:.2f}" fill="{color}"/>')
    lines.append(f'<rect x="{margin}" y="{margin}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222" stroke-width="1"/>')

    sx = _scale(X[:, 0], x_min, x_max, margin, margin + plot_w)
    sy = _scale(X[:, 1], y_min, y_max, margin + plot_h, margin)
    labels = y.ravel().astype(int)
    for xpix, ypix, label in zip(sx, sy, labels):
        color = "#0f766e" if label == 1 else "#be123c"
        lines.append(f'<circle cx="{float(xpix):.2f}" cy="{float(ypix):.2f}" r="3.1" fill="{color}" fill-opacity="0.78"/>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")

