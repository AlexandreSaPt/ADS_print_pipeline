"""S1P Plotter - run with:  streamlit run app.py"""
import json
import os
import tempfile
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skrf as rf
import streamlit as st

st.set_page_config(page_title="S1P Plotter", layout="wide")

SMITH_IMG = Path(__file__).parent / "smith_diagram.png"
FREQ_DIV = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
# variable -> allowed units (first one is the default)
VARS = {
    "Frequency": ["GHz", "MHz", "kHz", "Hz"],
    "S11 re": ["linear"],
    "S11 im": ["linear"],
    "S11 mag": ["dB", "linear"],
    "S11 phase": ["deg", "rad"],
}
LINESTYLES = {"line": "-", "dashed": "--", "dot": ":"}
# image edges in data units [left, right, bottom, top], calibrated for smith_diagram.png
SMITH_EXTENT = [-1.18423, 1.18980, -1.18918, 1.18485]
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


# ---------- data ----------
def load_s1p(uploaded) -> pd.DataFrame:
    """Read one uploaded .s1p with scikit-rf and return a DataFrame."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, uploaded.name)
        with open(path, "wb") as fh:
            fh.write(uploaded.getvalue())
        ntw = rf.Network(path)
    s = ntw.s[:, 0, 0]
    return pd.DataFrame({
        "Sim name": uploaded.name,
        "Frequency": ntw.f,  # Hz
        "S11 re": s.real,
        "S11 im": s.imag,
        "S11 mag": np.abs(s),  # linear
        "S11 phase": np.angle(s, deg=True),  # deg
    })


def convert(values, var, unit):
    v = np.asarray(values, dtype=float)
    if var == "Frequency":
        return v / FREQ_DIV[unit]
    if var == "S11 mag" and unit == "dB":
        return 20 * np.log10(np.maximum(v, 1e-12))
    if var == "S11 phase" and unit == "rad":
        return np.deg2rad(v)
    return v


# ---------- state ----------
ss = st.session_state
ss.setdefault("df", pd.DataFrame())
ss.setdefault("plots", {})
ss.setdefault("next_id", 1)
ss.setdefault("active", None)


def new_plot():
    pid = ss.next_id
    ss.next_id += 1
    ss.plots[pid] = dict(
        name=f"Plot {pid}", type="Normal", sims=[], x="Frequency", x_unit="GHz",
        x_scale="linear", y=["S11 mag"], y_units={"S11 mag": "dB"},
        y_scale="linear", lines={},
    )
    return pid


# ---------- template export / import ----------
TPL_KEYS = ("name", "type", "x", "x_unit", "x_scale", "y", "y_units", "y_scale")


def export_template() -> str:
    """Plot layout only: no sims and no legend names. Line styles are kept in order."""
    out = []
    for c in ss.plots.values():
        if c["type"] == "Smith":
            combos = [(s, "S11") for s in c["sims"]]
        else:
            combos = [(s, y) for s in c["sims"] for y in c["y"]]
        styles = [{k: c["lines"][f"{s}|{y}"][k] for k in ("color", "width", "style", "marker_size")
                   if k in c["lines"][f"{s}|{y}"]}
                  for s, y in combos if f"{s}|{y}" in c["lines"]]
        out.append({**{k: c[k] for k in TPL_KEYS}, "styles": styles})
    return json.dumps({"n_plots": len(out), "plots": out}, indent=2)


# ---------- drawing ----------
def draw(c):
    df = ss.df
    fig, ax = plt.subplots(figsize=(8, 6))
    if c["type"] == "Smith":
        if SMITH_IMG.exists():
            ax.imshow(mpimg.imread(SMITH_IMG), extent=SMITH_EXTENT, zorder=0)
        else:
            st.warning(f"{SMITH_IMG.name} not found next to app.py")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.axis("off")
        combos = [(s, "S11") for s in c["sims"]]
    else:
        combos = [(s, y) for s in c["sims"] for y in c["y"]]
    for s, y in combos:
        d = df[df["Sim name"] == s]
        ln = c["lines"].get(f"{s}|{y}", {})
        kw = dict(color=ln.get("color", "#1f77b4"), lw=ln.get("width", 1.5),
                  ls=LINESTYLES[ln.get("style", "line")],
                  marker="o", ms=ln.get("marker_size", 3), label=ln.get("label", s), zorder=2)
        if c["type"] == "Smith":
            ax.plot(d["S11 re"], d["S11 im"], **kw)
        else:
            ax.plot(convert(d[c["x"]], c["x"], c["x_unit"]),
                    convert(d[y], y, c["y_units"].get(y, VARS[y][0])), **kw)
    if c["type"] == "Normal":
        ax.set_xscale(c["x_scale"])
        ax.set_yscale(c["y_scale"])
        ax.set_xlabel(f'{c["x"]} ({c["x_unit"]})')
        ax.set_ylabel(", ".join(f'{y} ({c["y_units"].get(y, VARS[y][0])})' for y in c["y"]))
        ax.grid(True, which="both", alpha=0.3)
    ax.set_title(c["name"])
    if combos:
        ax.legend()
    return fig


# ---------- editor dialog ----------
@st.dialog("Edit plot", width="large")
def editor(pid):
    c = ss.plots[pid]
    sims = list(ss.df["Sim name"].unique())
    k = f"p{pid}_"
    c["name"] = st.text_input("Plot name", c["name"], key=k + "name")
    c["type"] = st.radio("Diagram type", ["Normal", "Smith"], horizontal=True,
                         index=["Normal", "Smith"].index(c["type"]), key=k + "type")
    c["sims"] = st.multiselect("Sims", sims, default=[s for s in c["sims"] if s in sims],
                               key=k + "sims")

    if c["type"] == "Normal":
        cx, cxu, cxs = st.columns(3)
        c["x"] = cx.selectbox("X axis", list(VARS), index=list(VARS).index(c["x"]), key=k + "x")
        units = VARS[c["x"]]
        c["x_unit"] = cxu.selectbox("X unit", units,
                                    index=units.index(c["x_unit"]) if c["x_unit"] in units else 0,
                                    key=k + "xu" + c["x"])
        c["x_scale"] = cxs.selectbox("X scale", ["linear", "log"],
                                     index=["linear", "log"].index(c["x_scale"]), key=k + "xs")
        cy, cys = st.columns([3, 1])
        c["y"] = cy.multiselect("Y axes", list(VARS), default=c["y"], key=k + "y")
        c["y_scale"] = cys.selectbox("Y scale", ["linear", "log"],
                                     index=["linear", "log"].index(c["y_scale"]), key=k + "ys")
        for y in c["y"]:
            units = VARS[y]
            cur = c["y_units"].get(y, units[0])
            c["y_units"][y] = st.selectbox(f"Unit for {y}", units,
                                           index=units.index(cur) if cur in units else 0,
                                           key=f"{k}yu{y}")
        combos = [(s, y) for s in c["sims"] for y in c["y"]]
    else:
        st.caption("Smith diagram plots S11 (real vs imaginary) over smith_diagram.png")
        combos = [(s, "S11") for s in c["sims"]]

    st.subheader("Lines")
    for i, (s, y) in enumerate(combos):
        key = f"{s}|{y}"
        seq = c.get("style_seq", [])
        base = seq[i] if i < len(seq) else {}
        ln = c["lines"].setdefault(key, dict(
            label=s if c["type"] == "Smith" else f"{s} - {y}",
            color=base.get("color", PALETTE[i % len(PALETTE)]),
            width=base.get("width", 1.5), style=base.get("style", "line"),
            marker_size=base.get("marker_size", 3)))
        with st.expander(key, expanded=True):
            a, b, w, t, m = st.columns([3, 1, 2, 2, 2])
            ln["label"] = a.text_input("Legend name", ln["label"], key=f"{k}{key}l")
            ln["color"] = b.color_picker("Color", ln["color"], key=f"{k}{key}c")
            ln["width"] = w.slider("Weight", 0.5, 6.0, float(ln["width"]), 0.5, key=f"{k}{key}w")
            ln["style"] = t.selectbox("Style", list(LINESTYLES),
                                      index=list(LINESTYLES).index(ln["style"]), key=f"{k}{key}s")
            ln["marker_size"] = m.slider("Point size", 0.0, 12.0, float(ln.get("marker_size", 3)),
                                         0.5, key=f"{k}{key}m")
    if st.button("Save & plot", type="primary"):
        ss.active = pid
        st.rerun()


# ---------- sidebar: file loading ----------
with st.sidebar:
    st.header("Files")
    files = st.file_uploader("Select .s1p files", type=["s1p"], accept_multiple_files=True)
    if files:
        ss.df = pd.concat([load_s1p(f) for f in files], ignore_index=True)
    else:
        ss.df = pd.DataFrame()
    for f in files:
        st.write(f"• {f.name}")
    if not ss.df.empty:
        with st.expander("DataFrame preview"):
            st.dataframe(ss.df)

with st.sidebar:
    st.header("Plot template")
    st.download_button("Export plots (.json)", export_template(), "plot_template.json",
                       "application/json", disabled=not ss.plots)
    tpl = st.file_uploader("Import plots (.json)", type=["json"])
    if tpl and ss.get("tpl_id") != (tpl.name, tpl.size):
        ss.tpl_id = (tpl.name, tpl.size)
        for p in json.loads(tpl.getvalue())["plots"]:
            pid = new_plot()
            ss.plots[pid].update({k: p[k] for k in TPL_KEYS if k in p})
            ss.plots[pid]["style_seq"] = p.get("styles", [])

# ---------- main: gallery ----------
st.title("S1P Plotter")
if ss.df.empty:
    st.info("Load one or more .s1p files from the sidebar to start.")
    st.stop()

COLS = 4
items = list(ss.plots.items())
for row in range(0, len(items) + 1, COLS):
    cols = st.columns(COLS)
    for col, idx in zip(cols, range(row, row + COLS)):
        with col:
            if idx < len(items):
                pid, c = items[idx]
                with st.container(border=True):
                    st.markdown(f"**{c['name']}**")
                    st.caption(f"{c['type']} · {len(c['sims'])} sim(s)")
                    b1, b2 = st.columns(2)
                    if b1.button("Generate", key=f"gen{pid}"):
                        ss.active = pid
                    if b2.button("Edit", key=f"edit{pid}"):
                        editor(pid)
            elif idx == len(items):
                with st.container(border=True):
                    st.markdown("&nbsp;")
                    if st.button("➕ Add plot", key="add"):
                        editor(new_plot())

if ss.active in ss.plots:
    st.divider()
    st.pyplot(draw(ss.plots[ss.active]))