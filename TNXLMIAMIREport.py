# =======================
# TNXL MIAMI – Full App
# =======================

import os
import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Flowable,
    Image, KeepInFrame
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS / SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="TNXL MIAMI Report", layout="wide")

styles = getSampleStyleSheet()
HEADER_HEIGHT = 1.85 * inch
LOGO_RATIO = 3.0 / 3.7
LOGO_SIZE  = HEADER_HEIGHT * LOGO_RATIO
BAR_WIDTH  = 80
BAR_HEIGHT = 8

DATABASE_FILENAME = "player_database.csv"
NOTES_FILENAME    = "scout_notes.csv"

AGE_LABELS = [
    "youth (12–13)",
    "jv (14–15)",
    "varsity (16–18)",
    "college (18+)",
]

# Metrics where LOWER = better (running times, etc.)
LOWER_IS_BETTER = {
    "30yd Time", "60yd Time", "5-5-10 Shuttle Time",
    "Time to Contact (sec)",
}

# ─────────────────────────────────────────────────────────────────────────────
# SESSION BOOTSTRAP FOR NOTES
# ─────────────────────────────────────────────────────────────────────────────
if "notes_df" not in st.session_state:
    if os.path.exists(NOTES_FILENAME):
        st.session_state.notes_df = pd.read_csv(NOTES_FILENAME, parse_dates=["Date"])
    else:
        st.session_state.notes_df = pd.DataFrame(columns=["Name", "Date", "Note"])

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

def broadcast_metrics_to_ages(base_metrics: dict, age_labels=AGE_LABELS) -> dict:
    return {
        age: {metric: cuts.copy() for metric, cuts in base_metrics.items()}
        for age in age_labels
    }

def flatten_thresholds(thr: dict, *, pad_factor: float = 0.1) -> pd.DataFrame:
    rows = []
    for grp, metrics in thr.items():
        for metric, cuts in metrics.items():
            if not (isinstance(cuts, dict) and {"below_avg", "avg", "above_avg"}.issubset(cuts)):
                try:
                    mid = float(cuts)
                except Exception:
                    mid = 0.0
                cuts = {
                    "below_avg": round(mid * (1 - pad_factor), 2),
                    "avg":       round(mid,                   2),
                    "above_avg": round(mid * (1 + pad_factor), 2),
                }
                thr[grp][metric] = cuts

            rows.append({
                "Age Group": grp,
                "Metric":     metric,
                "below_avg":  cuts["below_avg"],
                "avg":        cuts["avg"],
                "above_avg":  cuts["above_avg"],
            })
    return pd.DataFrame(rows)

metric_thresholds = {
    "Plane Score": {"above_avg": 70, "avg":60,"below_avg":40},
    "Connection Score":{"above_avg": 70, "avg":60,"below_avg":40},
    "Rotation Score" :{"above_avg": 70, "avg":60,"below_avg":40},
    "Attack Angle (deg)":       {"above_avg": 10, "avg":7,"below_avg":5},
    "On Plane Efficiency (%)":  {"above_avg": 70, "avg":60,"below_avg":40},
    "Time to Contact (sec)":    {"above_avg": 0.14, "avg":0.10, "below_avg":0.08},
    "Bat Speed (mph)":          {"above_avg": 70, "avg":60, "below_avg":50},
    "Rotational Acceleration (g)": {"above_avg":15,"avg":12,"below_avg":10},
    "Peak Hand Speed (mph)":    {"above_avg": 20, "avg":18,"below_avg":15},
    "Connection at Impact (deg)": {"above_avg":80,"avg":75,"below_avg":65},
    "Early Connection (deg)":   {"above_avg":95,"avg":80,"below_avg":70},
    "Vertical Bat Angle (deg)": {"above_avg":-20,"avg":-30,"below_avg":-40},
    "Max EV (mph)":             {"above_avg":95,"avg":90,"below_avg":85},
    "90th % EV (mph)":          {"above_avg": 95, "avg":85,"below_avg":80},
    "Positional Throw Velocity":{"above_avg": 60, "avg":50, "below_avg":45},
    "Pulldown Velocity":        {"above_avg": 65, "avg":55, "below_avg":45},
    "30yd Time":                {"above_avg":3.0, "avg":3.5, "below_avg":4.0},
    "60yd Time":                {"above_avg":6.0, "avg":6.5, "below_avg":7.0},
    "5-5-10 Shuttle Time":      {"above_avg":10.0,"avg":11.0, "below_avg":13.0},
    "Ankle":    {"above_avg":4, "avg":3, "below_avg":1},
    "Thoracic": {"above_avg":4, "avg":3, "below_avg":1},
    "Lumbar":   {"above_avg":4, "avg":3, "below_avg":1},
}

if "thresholds" not in st.session_state:
    st.session_state["thresholds"] = broadcast_metrics_to_ages(metric_thresholds)

# ─────────────────────────────────────────────────────────────────────────────
# AGE GROUPING + HELPERS
# ─────────────────────────────────────────────────────────────────────────────
age_groups = {
    "youth (12–13)":   lambda age: 12 <= age <= 13,
    "jv (14–15)":      lambda age: 14 <= age <= 15,
    "varsity (16–18)": lambda age: 16 <= age <= 18,
    "college (18+)":   lambda age: age >= 18,
}

def get_group(age: int) -> str:
    for grp, fn in age_groups.items():
        if fn(age):
            return grp
    return "unknown"

def order_cuts(metric, lo, mid, hi):
    if metric in LOWER_IS_BETTER:
        return dict(
            lbl_lo="Best (fastest) / Above",
            lbl_mid="Avg",
            lbl_hi="Worst (slowest) / Below",
            lo=hi, mid=mid, hi=lo  # flip ends so the number_inputs flow best→avg→worst
        )
    return dict(
        lbl_lo="Below",
        lbl_mid="Avg",
        lbl_hi="Above",
        lo=lo, mid=mid, hi=hi
    )

# ─────────────────────────────────────────────────────────────────────────────
# COLOR SCHEME FOR BARS (uses session thresholds)
# ─────────────────────────────────────────────────────────────────────────────
def get_bar_color(metric_name: str, value: float, age_group: str) -> str:
    all_th = st.session_state.get("thresholds", {})
    thr = all_th.get(age_group, {}).get(metric_name)
    if not thr or value is None:
        return "#7f8c8d"  # gray

    if metric_name in LOWER_IS_BETTER:
        if value <= thr["above_avg"]:
            return "#3498db"  # best
        elif value <= thr["avg"]:
            return "#2ecc71"
        else:
            return "#f1c40f"
    else:
        if value >= thr["above_avg"]:
            return "#3498db"
        elif value >= thr["avg"]:
            return "#2ecc71"
        else:
            return "#f1c40f"

# ─────────────────────────────────────────────────────────────────────────────
# REPORTLAB VISUAL WIDGETS
# ─────────────────────────────────────────────────────────────────────────────
class RangeBar(Flowable):
    def __init__(self, value, min_value, max_value, width=100, height=6,
                 fill_color=colors.grey, handle_radius=3, show_range=True):
        super().__init__()
        self.value         = value
        self.min_value     = min_value
        self.max_value     = max_value
        self.width         = width
        self.height        = height
        self.fill_color    = fill_color
        self.handle_radius = handle_radius
        self.show_range    = show_range

    def draw(self):
        c = self.canv
        x, y = 0, self.height/2
        c.setStrokeColor(colors.lightgrey)
        c.setLineWidth(self.height/3)
        c.line(x, y, x + self.width, y)

        pct = 0
        if self.max_value > self.min_value:
            pct = (self.value - self.min_value) / (self.max_value - self.min_value)
            pct = max(0, min(pct, 1))
        filled_width = pct * self.width

        c.setStrokeColor(self.fill_color)
        c.setLineWidth(self.height/3)
        c.line(x, y, x + filled_width, y)

        c.setFillColor(self.fill_color)
        c.circle(x + filled_width, y, self.handle_radius, stroke=0, fill=1)

# ─────────────────────────────────────────────────────────────────────────────
# CSV READ / UTILS
# ─────────────────────────────────────────────────────────────────────────────
def smart_read_csv(file_obj, **read_kwargs):
    from pandas.errors import EmptyDataError, ParserError
    for enc in (None, "utf-8", "cp1252", "latin-1"):
        try:
            file_obj.seek(0)
            return pd.read_csv(file_obj, encoding=enc, **read_kwargs) if enc else pd.read_csv(file_obj, **read_kwargs)
        except (UnicodeDecodeError, EmptyDataError, ParserError):
            continue
    raise UnicodeDecodeError("Unable to decode file with common encodings.")

def safe_float(val):
    try:
        return float(val)
    except:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# PDF HEADER + DECOR
# ─────────────────────────────────────────────────────────────────────────────
styles.add(ParagraphStyle(
    name="HeaderWhite",
    parent=styles["Heading1"],
    textColor=colors.white,
    fontSize=24,
    leading=28,
))
styles.add(ParagraphStyle(
    name="SubheaderWhite",
    parent=styles["Heading2"],
    textColor=colors.white,
    fontSize=16,
    leading=20,
))
styles.add(ParagraphStyle(
    name="ProgramTitle",
    parent=styles["Heading2"],
    textColor=colors.white,
    fontSize=18,
    leading=22,
))
styles.add(ParagraphStyle(
    name="AssessmentDate",
    parent=styles["Normal"],
    textColor=colors.white,
    fontSize=12,
    leading=14,
))

def draw_header_bg(canvas, doc):
    w, h = doc.pagesize
    header_h = HEADER_HEIGHT
    x0 = doc.leftMargin + doc.width * 0.20
    canvas.saveState()
    canvas.setFillColor(colors.black)
    canvas.rect(0, h - header_h, w, header_h, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#D4AF37"))
    path = canvas.beginPath()
    path.moveTo(w, h)
    path.lineTo(w, h - header_h)
    path.lineTo(x0, h)
    path.close()
    canvas.drawPath(path, fill=1, stroke=0)
    canvas.restoreState()

def build_header_with_logo_and_player_info(
    logo_path,
    player_info,
    width,
    name_style=None,
    info_style=None,
    program_style=None,
    date_style=None,
):
    name_style = name_style or ParagraphStyle(
        name="HeaderSmall",
        parent=styles["HeaderWhite"],
        fontSize=16,
        leading=19.2,
        spaceAfter=2,
        textColor=colors.white
    )
    info_style = info_style or ParagraphStyle(
        name="SubheaderSmall",
        parent=styles["SubheaderWhite"],
        fontSize=12,
        leading=14,
        spaceAfter=2,
        textColor=colors.white
    )
    program_style = program_style or ParagraphStyle(
        name="Program",
        parent=styles["ProgramTitle"],
        fontName= "Helvetica-Bold",
        fontSize=20,
        leading=24,
        tracking=1.0,
        textColor=colors.black
    )
    date_style = date_style or ParagraphStyle(
        name="DateBlack",
        parent=styles["AssessmentDate"],
        fontSize=10,
        leading=14,
        textColor=colors.white
    )

    if os.path.exists(logo_path):
        logo = Image(logo_path, width=LOGO_SIZE, height=LOGO_SIZE)
    else:
        logo = Paragraph("LOGO MISSING", info_style)

    name = Paragraph(player_info.get("Name", ""), name_style)
    pos_and_school = Paragraph(
        f"{player_info.get('Position','')} | "
        f"{player_info.get('High School','')} | "
        f"{player_info.get('Class','')}",
        info_style
    )

    raw_h = player_info.get("Height", 0) or 0
    try:
        ft, inch = divmod(int(raw_h), 12)
        height_text = f"{ft}′{inch}″"
    except Exception:
        height_text = f"{raw_h} in"

    height_wt = Paragraph(
        f"Height: {height_text} | Weight: {player_info.get('Weight','')} lbs",
        info_style
    )
    bt = player_info.get("B/T","")
    bat_throw = Paragraph(f"B/T:{bt}",info_style)
    dob       = Paragraph(f"DOB: {player_info.get('DOB','')}", info_style)

    middle = [name, Spacer(1, 4), pos_and_school, height_wt, bat_throw, dob]

    program = Paragraph("Summer Development Program", program_style)
    assess  = Paragraph(f"Assessment Date: {player_info.get('AssessmentDate','')}", date_style)
    right = [program, Spacer(1, 4), assess]

    tbl = Table(
        [[logo, middle, right]],
        colWidths=[width * 0.15, width * 0.55, width * 0.30],
        rowHeights=[HEADER_HEIGHT]
    )
    tbl.setStyle(TableStyle([
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (2, 0),    0),
        ("BOTTOMPADDING",(0, 0), (2, 0),    0),
    ]))

    return tbl

# ─────────────────────────────────────────────────────────────────────────────
# METRIC CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────
def calculate_blast_metrics(df):
    cols = [c for c in df.columns if c.lower() in {
        "plane score","connection score","rotation score","bat speed (mph)",
        "rotational acceleration (g)","on plane efficiency (%)","attack angle (deg)",
        "early connection (deg)","connection at impact (deg)","vertical bat angle (deg)",
        "power (kw)","time to contact (sec)","peak hand speed (mph)"
    }]
    if not cols:
        return {},{}
    avgs = df[cols].mean().to_dict()
    rngs = {c:(df[c].min(),df[c].max()) for c in cols}
    return avgs, rngs

def calculate_throwing_velocities(df):
    if df is None:
        return {}
    out = {}
    for col in df.columns:
        if "velocity" in col.lower():
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if not vals.empty:
                out[col] = float(vals.mean())
    return out

def calculate_running_speeds(df):
    means, ranges = {}, {}
    if df is None or df.empty:
        return means, ranges
    for col in df.columns:
        low = col.lower()
        if any(x in low for x in ["30yd", "60yd", "shuttle"]):
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if not vals.empty:
                means[col]  = float(vals.mean())
                ranges[col] = (float(vals.min()), float(vals.max()))
    return means, ranges

def calculate_flightscope_metrics(data):
    exit_speed_column = None
    if data is None:
        return None, None
    for col in data.columns:
        if "exit" in col.lower() and "speed" in col.lower():
            exit_speed_column = col
            break
    if exit_speed_column is None:
        return None, None
    data[exit_speed_column] = pd.to_numeric(data[exit_speed_column], errors="coerce")
    cleaned = data.dropna(subset=[exit_speed_column])
    if cleaned.empty:
        return None, None
    max_ev = cleaned[exit_speed_column].max()
    percentile_90_ev = cleaned[exit_speed_column].quantile(0.9)
    return max_ev, percentile_90_ev

# poly helpers (guard for numeric)
def safe_est_poly_at_t(t, poly_val):
    """poly_val may be a string 'a;b;c;d;e' or numeric/NaN → return None in non-string."""
    if not isinstance(poly_val, str):
        return None
    try:
        coeffs = [float(x.strip()) for x in poly_val.split(";")]
        if len(coeffs) < 5 or any(pd.isna(coeff) for coeff in coeffs):
            return None
        return sum(coeffs[i] * (t**i) for i in range(5))
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# TABLE BUILDERS FOR PDF
# ─────────────────────────────────────────────────────────────────────────────
def build_gameplay_data_table(
    averages: dict,
    ranges: dict,
    max_ev: float,
    percentile_90_ev: float,
    velocities: dict,
    width: float,
    thresholds: dict,
    age_group: str
):
    data = [["Metric", "Value", "Range / Visual"]]
    blast_metrics = [
        ("Plane Score",              "Plane Score"),
        ("Connection Score",         "Connection Score"),
        ("Rotation Score",           "Rotation Score"),
        ("Attack Angle (°)",         "Attack Angle (deg)"),
        ("On-Plane Efficiency (%)",  "On Plane Efficiency (%)"),
        ("Time to Contact (s)",      "Time to Contact (sec)"),
        ("Bat Speed (mph)",          "Bat Speed (mph)"),
        ("Rotational Acceleration (g)", "Rotational Acceleration (g)"),
        ("Peak Hand Speed (mph)",    "Peak Hand Speed (mph)"),
        ("Connection at Impact (°)", "Connection at Impact (deg)"),
        ("Early Connection (°)",     "Early Connection (deg)"),
        ("Vertical Bat Angle (°)",   "Vertical Bat Angle (deg)"),
    ]
    for label, key in blast_metrics:
        value = averages.get(key)
        value_str = f"{value:.2f}" if value is not None else "N/A"
        cuts = thresholds.get(age_group, {}).get(key)
        if cuts:
            rmin, rmax = cuts["below_avg"], cuts["above_avg"]
        else:
            rmin, rmax = ranges.get(key, (None, None))
        if value is not None and rmin is not None and rmax is not None:
            color  = get_bar_color(key, value, age_group)
            visual = RangeBar(value, rmin, rmax, width=BAR_WIDTH, height=BAR_HEIGHT,
                              fill_color=colors.HexColor(color), show_range=False)
        else:
            visual = "—"
        data.append([Paragraph(label, styles["Normal"]), value_str, visual])

    # Max EV
    if max_ev is not None:
        key = "Max EV (mph)"
        cuts = thresholds.get(age_group, {}).get(key, {})
        rmin, rmax = cuts.get("below_avg"), cuts.get("above_avg")
        if rmin is not None and rmax is not None:
            color  = get_bar_color(key, max_ev, age_group)
            visual = RangeBar(max_ev, rmin, rmax, width=BAR_WIDTH, height=BAR_HEIGHT,
                              fill_color=colors.HexColor(color), show_range=False)
        else:
            visual = "—"
        data.append([key, f"{max_ev:.1f}", visual])

    # 90th % EV
    if percentile_90_ev is not None:
        key = "90th % EV (mph)"
        cuts = thresholds.get(age_group, {}).get(key, {})
        rmin, rmax = cuts.get("below_avg"), cuts.get("above_avg")
        if rmin is not None and rmax is not None:
            color  = get_bar_color(key, percentile_90_ev, age_group)
            visual = RangeBar(percentile_90_ev, rmin, rmax, width=BAR_WIDTH, height=BAR_HEIGHT,
                              fill_color=colors.HexColor(color), show_range=False)
        else:
            visual = "—"
        data.append([key, f"{percentile_90_ev:.1f}", visual])

    # Throwing velocities (any column with 'velocity')
    for pitch, velo in velocities.items():
        cuts = thresholds.get(age_group, {}).get(pitch, {})
        rmin, rmax = cuts.get("below_avg"), cuts.get("above_avg")
        if velo is not None and rmin is not None and rmax is not None:
            color  = get_bar_color(pitch, velo, age_group)
            visual = RangeBar(velo, rmin, rmax, width=BAR_WIDTH, height=BAR_HEIGHT,
                              fill_color=colors.HexColor(color), show_range=False)
        else:
            visual = "—"
        data.append([pitch, f"{velo:.1f} mph" if velo is not None else "N/A", visual])

    table = Table(data, colWidths=[width*0.30, width*0.12, width*0.30], hAlign="LEFT")
    table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#D4AF37')),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.black),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 10),
        ('ALIGN',         (1,1), (1,-1), 'RIGHT'),
        ('ALIGN',         (2,1), (2,-1), 'CENTER'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [None, '#FAFAFA']),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.lightgrey),
    ]))
    return table

def build_dynamo_table(dynamo_data, player_info, width):
    from reportlab.lib.styles import getSampleStyleSheet
    if dynamo_data is None or dynamo_data.empty:
        return Paragraph("No Dynamo Data", getSampleStyleSheet()["Normal"])
    name = player_info.get("Name", "").lower()
    df = dynamo_data[dynamo_data["Name"].str.lower() == name]
    if df.empty:
        return Paragraph("No Dynamo Data for this player", getSampleStyleSheet()["Normal"])

    numeric_cols = [
        "ROM Asymmetry (%)","Force Asymmetry (%)",
        "L Max ROM (°)","R Max ROM (°)",
        "L Max Force (N)","R Max Force (N)",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    agg = df.groupby(["Movement","Type"], as_index=False).agg({
        "ROM Asymmetry (%)": "mean",
        "Force Asymmetry (%)": "mean",
        "L Max ROM (°)":      "mean",
        "R Max ROM (°)":      "mean",
        "L Max Force (N)":    "mean",
        "R Max Force (N)":    "mean",
    })
    data = [["Movement", "Type", "ROM Asym", "Force Asym", "L Max", "R Max"]]
    for _, r in agg.iterrows():
        data.append([
            r["Movement"], r["Type"],
            f"{r.get('ROM Asymmetry (%)'):.1f}" if not pd.isna(r.get("ROM Asymmetry (%)")) else "N/A",
            f"{r.get('Force Asymmetry (%)'):.1f}" if not pd.isna(r.get("Force Asymmetry (%)")) else "N/A",
            f"{r.get('L Max ROM (°)'):.1f}" if not pd.isna(r.get("L Max ROM (°)")) else "N/A",
            f"{r.get('R Max ROM (°)'):.1f}" if not pd.isna(r.get("R Max ROM (°)")) else "N/A"
        ])
    tbl = Table(data, colWidths=[width/6]*6)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#D4AF37')),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.black),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 10),
        ('ALIGN',         (1,1), (1,-1), 'RIGHT'),
        ('ALIGN',         (2,1), (2,-1), 'CENTER'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [None, '#FAFAFA']),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.lightgrey),
    ]))
    return tbl

def build_profile_table(
    mobility: dict,
    speeds: dict,
    speed_ranges: dict,
    width: float,
    thresholds: dict,
    age_group: str
):
    data = [["Metric", "Value", "Δ (max–min)"]]

    for key in ["Ankle", "Thoracic", "Lumbar"]:
        score = mobility.get(key)
        if score is None:
            val_str, delta = "N/A", "—"
        else:
            cuts = thresholds.get(age_group, {}).get(key, {})
            rmin, rmax = cuts.get("below_avg"), cuts.get("above_avg")
            val_str = f"{float(score):.2f}"
            if rmin is not None and rmax is not None and rmax != rmin:
                delta = f"{(rmax - rmin):.2f}"
            else:
                delta = "—"
        data.append([Paragraph(key, styles["Normal"]), val_str, Paragraph(delta, styles["Normal"])])

    for key in ["30yd Time", "60yd Time", "5-5-10 Shuttle Time"]:
        avg   = speeds.get(key)
        rmin, rmax = speed_ranges.get(key, (None, None))
        if avg is None:
            val_str, delta = "N/A", "—"
        else:
            val_str = f"{avg:.2f} sec"
            if rmin is not None and rmax is not None and rmax != rmin:
                delta = f"{(rmax - rmin):.2f}"
            else:
                delta = "—"
        data.append([Paragraph(key, styles["Normal"]), val_str, Paragraph(delta, styles["Normal"])])

    tbl = Table(data, colWidths=[width*0.30, width*0.12, width*0.30], hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#D4AF37')),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.black),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 10),
        ('ALIGN',         (1,1), (1,-1), 'RIGHT'),
        ('ALIGN',         (2,1), (2,-1), 'CENTER'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [None, '#FAFAFA']),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.lightgrey),
    ]))
    return tbl

# Heatmap used in PDF
def generate_exit_velo_heatmap(df):
    import numpy as np
    import matplotlib.pyplot as plt
    from reportlab.platypus import Image
    from io import BytesIO
    if df is None or df.empty:
        return None
    if not set(["Parsed_X","Parsed_Z","Exit_Speed"]).issubset(df.columns):
        return None

    x = df["Parsed_X"].values * 12
    y = df["Parsed_Z"].values * 12
    c = df["Exit_Speed"].values

    Zoom_ext = [-18, 18, 0, 60]
    fig, ax = plt.subplots(figsize=(5,5))
    hb = ax.hexbin(
        x, y, C=c, reduce_C_function=np.mean, gridsize=(8,8),
        cmap="coolwarm", mincnt=1, extent=Zoom_ext
    )
    ax.set_xlim(Zoom_ext[0], Zoom_ext[1])
    ax.set_ylim(Zoom_ext[2], Zoom_ext[3])
    ax.set_aspect("equal", "box")
    ax.axis("off")

    offsets = hb.get_offsets()
    values  = hb.get_array()
    for (cx, cy), v in zip(offsets, values):
        ax.text(cx, cy, f"{v:.1f}", ha="center", va="center", fontsize=10, color="white")

    sz_w, sz_h = 17, 25
    left, bottom = -sz_w/2, 16
    ax.add_patch(plt.Rectangle((left,bottom), sz_w, sz_h, fill=False, lw=2, edgecolor="black"))
    ax.add_patch(plt.Rectangle((left,bottom), sz_w, sz_h, fill=False, lw=1, linestyle="--", edgecolor="black"))

    buf = BytesIO()
    plt.tight_layout(pad=0)
    fig.savefig(buf, format="png", dpi=150, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=280, height=280)

# ─────────────────────────────────────────────────────────────────────────────
# PDF CREATION
# ─────────────────────────────────────────────────────────────────────────────
def create_combined_pdf(
    max_ev,
    percentile_90_ev,
    averages,
    ranges,
    velocities,
    speeds,
    speed_ranges,
    player_info,
    flightscope_data,
    mobility=None,
    dynamo_data=None,
):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A3),
        rightMargin=30, leftMargin=30, topMargin=10, bottomMargin=30
    )
    elements = []

    header = build_header_with_logo_and_player_info(
        "TNXL Miami - Updated Logo.png",  # put logo in repo root for Streamlit Cloud
        player_info,
        doc.width
    )
    elements.append(header)
    elements.append(Spacer(1, 12))

    # Prepare heatmap
    heatmap_img = None
    if flightscope_data is not None and not flightscope_data.empty:
        tmp = flightscope_data.copy()
        tmp["Parsed_X"] = tmp.get("Hit_Poly_X", pd.Series([None]*len(tmp))).apply(
            lambda p: safe_est_poly_at_t(0, p)
        )
        tmp["Parsed_Z"] = tmp.get("Hit_Poly_Z", pd.Series([None]*len(tmp))).apply(
            lambda p: safe_est_poly_at_t(0, p)
        )
        tmp["Exit_Speed"] = pd.to_numeric(tmp.get("Exit_Speed"), errors="coerce")
        valid = (
            tmp.dropna(subset=["Parsed_X", "Parsed_Z", "Exit_Speed"])
               .query("Exit_Speed > 0")
               .copy()
        )
        # catcher view
        valid["PlateLocSide"]   = -valid["Parsed_X"] * 12.0
        valid["PlateLocHeight"] =  valid["Parsed_Z"] * 12.0
        heatmap_img = generate_exit_velo_heatmap(valid)

    # Row 1: Gameplay vs Heatmap
    default_left_w  = doc.width * 0.61
    default_right_w = doc.width - default_left_w

    gameplay_frame = KeepInFrame(
        default_left_w, doc.height,
        [
            Paragraph("Gameplay Data", styles["Heading3"]),
            Spacer(1, 6),
            build_gameplay_data_table(
                averages, ranges, max_ev, percentile_90_ev, velocities,
                default_left_w, thresholds=st.session_state["thresholds"],
                age_group=player_info["Age Group"]
            )
        ],
        hAlign="LEFT", mergeSpace=True
    )

    right_contents = [Paragraph("AVG Exit Velocity by Zone", styles["Heading3"]), Spacer(1,6)]
    right_contents.append(heatmap_img if heatmap_img else Paragraph("No heatmap data", styles["Normal"]))
    right_frame = KeepInFrame(default_right_w, doc.height, right_contents, hAlign="LEFT", mergeSpace=True)

    elements.append(Table([[gameplay_frame, right_frame]], colWidths=[default_left_w, default_right_w]))
    elements.append(Spacer(1, 12))

    # Row 2: Physical Profile & Notes
    notes_txt = player_info.get("LatestNoteText")
    notes_txt = str(notes_txt) if notes_txt not in (None, "nan", "NaN") else "No scout notes available."

    left_w2 = default_left_w
    notes_w = doc.width - left_w2

    physical_frame = KeepInFrame(
        left_w2, doc.height,
        [
            Paragraph("Physical Profile", styles["Heading3"]),
            Spacer(1, 6),
            build_profile_table(
                mobility or {}, speeds or {}, speed_ranges or {}, left_w2,
                thresholds=st.session_state.get("thresholds", {}),
                age_group=player_info.get("Age Group"),
            ),
        ],
        hAlign="LEFT", mergeSpace=True,
    )

    notes_tbl = Table(
        [[Paragraph("Scout Notes", styles["Heading3"])],
         [Paragraph(notes_txt, styles["Normal"])]],
        colWidths=[notes_w],
    )
    notes_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D4AF37")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN",      (0, 0), (-1, 0), "CENTER"),
        ("VALIGN",     (0, 1), (-1, 1), "TOP"),
        ("BOX",        (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID",  (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("LEFTPADDING",(0, 0), (-1, -1), 6),
        ("RIGHTPADDING",(0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))

    elements.extend([Table([[physical_frame, notes_tbl]], colWidths=[left_w2, notes_w]), Spacer(1, 12)])

    # Row 3: Dynamo
    dynamo_frame = KeepInFrame(
        default_left_w, doc.height,
        [Paragraph("Dynamo Summary", styles["Heading3"]), Spacer(1,6),
         build_dynamo_table(dynamo_data, player_info, default_left_w)],
        hAlign="LEFT", mergeSpace=True
    )
    elements.append(Table([[dynamo_frame, '']], colWidths=[default_left_w, default_right_w]))

    doc.build(elements, onFirstPage=draw_header_bg, onLaterPages=draw_header_bg)
    buffer.seek(0)
    return buffer

# ─────────────────────────────────────────────────────────────────────────────
# PLAYER DB – LOAD/INIT
# ─────────────────────────────────────────────────────────────────────────────
expected_columns = [
    "Name", "DOB", "Age", "Class", "High School", "Height", "Weight",
    "Position", "BattingHandedness", "ThrowingHandedness"
]

@st.cache_data
def load_player_db(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=expected_columns)
    return df

if "player_db" not in st.session_state:
    st.session_state.player_db = load_player_db(DATABASE_FILENAME)

# Always prepare a lowercase name key for merges
def ensure_nm(df):
    if df is not None and not df.empty:
        if "Name" in df.columns:
            df["nm"] = df["Name"].astype(str).str.lower().str.strip()
        elif "Player Name" in df.columns:
            df["nm"] = df["Player Name"].astype(str).str.lower().str.strip()
        elif "Batter" in df.columns:
            df["nm"] = df["Batter"].astype(str).str.lower().str.strip()
        else:
            df["nm"] = None
    return df

st.title("TNXL MIAMI - Athlete Performance Data Uploader, Report Generator & CSV Utilities")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "CSV Merge",
    "Player Database",
    "Scout Notes",
    "Thresholds",
    "Reports & Templates"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: CSV MERGE
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("CSV Merger")
    csv_categories = ["Blast", "Flightscope", "Throwing Velocities", "Running Speed", "Mobility", "Dynamo"]
    csv_type = st.selectbox("Select CSV Type to Merge", csv_categories)
    files = st.file_uploader(f"Upload {csv_type} CSV Files", type="csv", accept_multiple_files=True, key="merge_files")

    if files:
        merged_dfs = []
        st.markdown("### Configure Each File")
        for idx, uploaded in enumerate(files):
            st.subheader(f"File {idx+1}: {uploaded.name}")
            df = pd.read_csv(uploaded)
            st.write("Columns detected:", df.columns.tolist())

            if csv_type == "Blast":
                if "Name" not in df.columns:
                    df.insert(0, "Name", "")
                st.markdown("**Edit Blast data (e.g. player names)**")
                df = st.data_editor(df, use_container_width=True, num_rows="dynamic", key=f"blast_edit_{idx}")

            default_label = os.path.splitext(uploaded.name)[0]
            label = st.text_input(f"Label for {uploaded.name}", value=default_label, key=f"label_{idx}")

            df["Type"]  = csv_type
            df["Label"] = label
            merged_dfs.append(df)

        if st.button("Merge Selected Files"):
            merged = pd.concat(merged_dfs, ignore_index=True)
            st.success(f"Merged {len(merged_dfs)} files of type {csv_type}!")
            st.dataframe(merged.head())
            csv_bytes = merged.to_csv(index=False).encode("utf-8")
            st.download_button("Download Merged CSV", data=csv_bytes,
                               file_name=f"merged_{csv_type.lower().replace(' ','_')}.csv", mime="text/csv")
    else:
        st.info(f"Upload two or more {csv_type} CSVs above to merge them.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: PLAYER DATABASE (new layout)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Player Database")
    st.caption("Data stored on disk in **player_database.csv**")

    add_tab, edit_tab = st.tabs(["➕ Add Player", "✏️ Edit / Delete"])

    with add_tab:
        with st.form("add_player_form", clear_on_submit=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                add_name = st.text_input("Name", key="add_name")
                add_dob  = st.date_input("Date of Birth",
                                         min_value=datetime.date(1900,1,1),
                                         max_value=datetime.date.today(),
                                         key="add_dob")
            with c2:
                add_class = st.text_input("Class", key="add_class")
                add_hs    = st.text_input("High School", key="add_hs")
            with c3:
                h_ft  = st.number_input("Height (ft)", 0, 8, key="add_h_ft")
                h_in  = st.number_input("Height (in)", 0, 11, key="add_h_in")
                add_height = h_ft*12 + h_in
                add_weight = st.number_input("Weight (lbs)", 0, 500, key="add_weight")
            with c4:
                pos_opts = ["Pitcher","Catcher","1B","2B","3B","SS","LF","CF","RF","DH"]
                bat_opts = ["Left","Right","Switch"]
                thr_opts = ["Left","Right"]
                add_pos   = st.selectbox("Position", pos_opts, key="add_pos")
                add_bat   = st.selectbox("Batting Handedness", bat_opts, key="add_bat")
                add_throw = st.selectbox("Throwing Handedness", thr_opts, key="add_throw")

            add_submit = st.form_submit_button("Add Player", use_container_width=True)

        if add_submit:
            today     = datetime.date.today()
            age       = (today.year - add_dob.year
                         - ((today.month, today.day) < (add_dob.month, add_dob.day)))
            age_group = get_group(age)
            new_row   = {
                "Name":               add_name,
                "DOB":                add_dob.strftime("%m/%d/%Y"),
                "Age":                int(age),
                "Age Group":          age_group,
                "Class":              add_class,
                "High School":        add_hs,
                "Height":             add_height,
                "Weight":             add_weight,
                "Position":           add_pos,
                "BattingHandedness":  add_bat,
                "ThrowingHandedness": add_throw,
            }
            st.session_state.player_db = pd.concat(
                [st.session_state.player_db, pd.DataFrame([new_row])],
                ignore_index=True
            )
            st.session_state.player_db.to_csv(DATABASE_FILENAME, index=False)
            st.success(f"✅ Added {add_name}")

    with edit_tab:
        st.session_state.player_db.reset_index(drop=True, inplace=True)
        db = st.session_state.player_db
        if db.empty:
            st.info("Database is empty—add a player first.")
        else:
            idx = st.selectbox("Select player to edit / delete", db.index,
                               format_func=lambda i: db.at[i, "Name"], key="edit_selectbox")
            sel = db.loc[idx]

            with st.form("edit_player_form", clear_on_submit=True):
                parsed_dob = pd.to_datetime(sel["DOB"], errors="coerce")
                init_dob   = parsed_dob.date() if pd.notna(parsed_dob) else datetime.date.today()

                e_name = st.text_input("Name", sel["Name"], key="e_name")
                e_dob  = st.date_input("Date of Birth", init_dob, key="e_dob")

                today   = datetime.date.today()
                e_age   = (today.year - e_dob.year
                           - ((today.month, today.day) < (e_dob.month, e_dob.day)))
                e_group = get_group(e_age)

                e_class = st.text_input("Class", sel["Class"], key="e_class")
                e_hs    = st.text_input("High School", sel["High School"], key="e_hs")

                raw_h   = int(sel.get("Height", 0) or 0)
                ft0, in0 = divmod(raw_h, 12)
                raw_w   = int(sel.get("Weight", 0) or 0)

                ft   = st.number_input("Height (ft)", 0, 8, value=ft0, key="e_h_ft")
                inch = st.number_input("Height (in)", 0,11, value=in0, key="e_h_in")
                e_height = ft*12 + inch
                e_weight = st.number_input("Weight (lbs)", 0, 500, value=raw_w, key="e_weight")

                pos_opts = ["Pitcher","Catcher","1B","2B","3B","SS","LF","CF","RF","DH"]
                bat_opts = ["Left","Right","Switch"]
                thr_opts = ["Left","Right"]

                e_pos   = st.selectbox("Position", pos_opts,
                                       index=pos_opts.index(sel["Position"]), key="e_pos")
                e_bat   = st.selectbox("Batting Handedness", bat_opts,
                                       index=bat_opts.index(sel["BattingHandedness"]), key="e_bat")
                e_throw = st.selectbox("Throwing Handedness", thr_opts,
                                       index=thr_opts.index(sel["ThrowingHandedness"]), key="e_throw")

                col1, col2 = st.columns(2)
                with col1:
                    update_submit = st.form_submit_button("Update Player", use_container_width=True)
                with col2:
                    delete_submit = st.form_submit_button("Delete Player", type="primary", use_container_width=True)

            if update_submit:
                updates = {
                    "Name":               e_name,
                    "DOB":                e_dob.strftime("%m/%d/%Y"),
                    "Age":                int(e_age),
                    "Age Group":          e_group,
                    "Class":              e_class,
                    "High School":        e_hs,
                    "Height":             e_height,
                    "Weight":             e_weight,
                    "Position":           e_pos,
                    "BattingHandedness":  e_bat,
                    "ThrowingHandedness": e_throw,
                }
                for col, val in updates.items():
                    st.session_state.player_db.at[idx, col] = val
                st.session_state.player_db.to_csv(DATABASE_FILENAME, index=False)
                st.success("✅ Player updated")

            if delete_submit:
                st.session_state.player_db = (
                    st.session_state.player_db.drop(idx).reset_index(drop=True)
                )
                st.session_state.player_db.to_csv(DATABASE_FILENAME, index=False)
                st.success("🗑️ Player deleted")

    # live table
    st.markdown("### Current Database")
    db_view = st.session_state.player_db.copy()
    if "Age" in db_view.columns:
        db_view["Age"] = pd.to_numeric(db_view["Age"], errors="coerce").astype("Int64")
    st.dataframe(db_view, use_container_width=True)

    # import / export
    st.divider()
    st.subheader("⬇️⬆️  Import / Export")
    csv_bytes = st.session_state.player_db.to_csv(index=False).encode("utf-8")
    st.download_button("Download Current DB as CSV", data=csv_bytes,
                       file_name="player_database.csv", mime="text/csv", key="dl_db")

    uploaded_db = st.file_uploader("Upload Player Database CSV", type="csv",
                                   key="upload_db",
                                   help="CSV must include the same columns as the table above.")
    if uploaded_db is not None:
        try:
            imported = pd.read_csv(uploaded_db)
            missing  = [c for c in expected_columns if c not in imported.columns]
            if missing:
                st.error(f"CSV missing required columns: {', '.join(missing)}")
            else:
                mode = st.radio("Import mode:",
                                ["Replace existing DB", "Merge (append & deduplicate by Name)"],
                                horizontal=True, key="import_mode")
                if mode == "Replace existing DB":
                    st.session_state.player_db = imported.copy()
                    st.success("✅ Replaced database with uploaded CSV.")
                else:
                    combined = pd.concat([st.session_state.player_db, imported], ignore_index=True)
                    st.session_state.player_db = combined.drop_duplicates(subset=["Name"], keep="last")
                    st.success("✅ Merged uploaded CSV into current database.")
                st.session_state.player_db.to_csv(DATABASE_FILENAME, index=False)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")

    st.divider()
    if st.button("Clear Entire Player Database", type="primary", key="clear_db"):
        st.session_state.player_db = pd.DataFrame(columns=expected_columns)
        if os.path.exists(DATABASE_FILENAME):
            os.remove(DATABASE_FILENAME)
        st.success("🚮 Database cleared from disk and memory")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: SCOUT NOTES (new layout)
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Scout Notes")
    st.info("Add, preview, bulk-upload or delete notes per player.")

    if st.button("Clear ALL Notes 🗑️", type="primary", key="clear_notes"):
        st.session_state.notes_df = pd.DataFrame(columns=["Name", "Date", "Note"])
        st.session_state.notes_df.to_csv(NOTES_FILENAME, index=False)
        st.success("All notes removed from disk and memory.")

    player = st.selectbox("Select Player", st.session_state.player_db["Name"].tolist(), key="notes_player")

    player_notes = (
        st.session_state.notes_df[st.session_state.notes_df["Name"] == player]
        .sort_values("Date", ascending=False)
    )

    st.subheader("Existing Notes")
    if player_notes.empty:
        st.write("No notes yet for this player.")
    else:
        idx = st.radio("Select note to include in PDF:",
                       options=player_notes.index.tolist(),
                       format_func=lambda i: player_notes.loc[i, "Date"].strftime("%Y-%m-%d"),
                       key="select_note")
        st.markdown(f"**Preview ({player_notes.loc[idx,'Date'].strftime('%Y-%m-%d')}):**")
        st.write(player_notes.loc[idx, "Note"])

        if st.button("Delete this note", key="delete_note"):
            st.session_state.notes_df = (
                st.session_state.notes_df.drop(idx).reset_index(drop=True)
            )
            st.session_state.notes_df.to_csv(NOTES_FILENAME, index=False)
            st.success("Note deleted.")
            st.rerun()

    st.divider()
    st.subheader("⬇️⬆️  Bulk-upload / Download Notes")
    csv_bytes = st.session_state.notes_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Current Notes CSV", data=csv_bytes,
                       file_name="scout_notes.csv", mime="text/csv", key="dl_notes")

    bulk_file = st.file_uploader("Upload Notes CSV (columns: Name, Date, Note)",
                                 type="csv", key="bulk_notes_csv")
    if bulk_file is not None:
        try:
            incoming = smart_read_csv(bulk_file, parse_dates=["Date"])
            required = {"Name", "Date", "Note"}
            if not required.issubset(incoming.columns):
                st.error(f"CSV must contain columns: {', '.join(required)}")
            else:
                mode = st.radio(
                    "Import mode",
                    ["Merge (append & deduplicate by Name+Date+Note)", "Replace ALL existing notes"],
                    horizontal=True, key="bulk_note_mode"
                )
                if mode.startswith("Merge"):
                    combined = pd.concat([st.session_state.notes_df, incoming], ignore_index=True)
                    st.session_state.notes_df = combined.drop_duplicates(
                        subset=["Name", "Date", "Note"], keep="last"
                    )
                    st.success(f"✅ Merged {len(incoming)} notes.")
                else:
                    st.session_state.notes_df = incoming.copy()
                    st.success(f"✅ Replaced with {len(incoming)} notes.")
                st.session_state.notes_df.to_csv(NOTES_FILENAME, index=False)
        except Exception as exc:
            st.error(f"Could not read CSV – {exc}")

    st.divider()
    st.subheader("✍️  Add a New Note")
    note_date = st.date_input("Note Date", value=datetime.date.today(), key="new_note_date")
    note_text = st.text_area("Note Text", key="new_note_text")
    if st.button("Save Note", key="save_note"):
        new_row = {"Name": player, "Date": pd.to_datetime(note_date), "Note": note_text}
        st.session_state.notes_df = pd.concat(
            [st.session_state.notes_df, pd.DataFrame([new_row])],
            ignore_index=True
        )
        st.session_state.notes_df.to_csv(NOTES_FILENAME, index=False)
        st.success("Note saved.")
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: THRESHOLDS (new layout; number_inputs + CSV import/export)
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("🔧 Metric Thresholds by Age-Group")

    # One-time fix if a flat dict somehow exists
    if not any(k in AGE_LABELS for k in st.session_state["thresholds"].keys()):
        st.session_state["thresholds"] = broadcast_metrics_to_ages(st.session_state["thresholds"])
    thresholds = st.session_state["thresholds"]

    def order_for_ui(metric: str, lo: float, mid: float, hi: float) -> dict:
        if metric in LOWER_IS_BETTER:
            return {
                "lbl_lo": "Above Avg (best/fastest)",
                "lbl_mid": "Avg",
                "lbl_hi": "Below Avg (worst/slowest)",
                "lo": hi, "mid": mid, "hi": lo
            }
        return {"lbl_lo":"Below Avg", "lbl_mid":"Avg", "lbl_hi":"Above Avg", "lo":lo, "mid":mid, "hi":hi}

    edit_mode = st.toggle("Edit mode", value=False)
    st.caption("Browse in **View**; switch to **Edit** to change values, then press **Save thresholds**.")

    age_groups_keys = list(thresholds.keys())
    if not age_groups_keys:
        st.error("⚠️ No age-group data found.")
        st.stop()
    tabs_age = st.tabs(age_groups_keys)

    for grp, grp_tab in zip(age_groups_keys, tabs_age):
        metrics = thresholds[grp]
        with grp_tab:
            sel_metric = st.selectbox("Choose metric", sorted(metrics.keys()), key=f"sel_{grp}")
            cuts = metrics[sel_metric]
            if isinstance(cuts, dict) and {"below_avg", "avg", "above_avg"}.issubset(cuts):
                lo, mid, hi = cuts["below_avg"], cuts["avg"], cuts["above_avg"]
            else:
                mid = float(cuts) if cuts is not None else 0.0
                lo, hi = round(mid * 0.9, 2), round(mid * 1.1, 2)

            cfg = order_for_ui(sel_metric, float(lo), float(mid), float(hi))
            rng_min = min(cfg["lo"], cfg["mid"], cfg["hi"]) * 0.50
            rng_max = max(cfg["lo"], cfg["mid"], cfg["hi"]) * 1.50

            col_lbl, col_lo, col_mid, col_hi = st.columns([2,1,1,1])
            col_lbl.markdown(f"### {sel_metric}")

            if edit_mode:
                new_lo  = col_lo.number_input(cfg["lbl_lo"],  value=float(cfg["lo"]),
                                              min_value=float(rng_min), max_value=float(cfg["mid"]),
                                              step=0.01, key=f"{grp}_{sel_metric}_lo")
                new_mid = col_mid.number_input(cfg["lbl_mid"], value=float(cfg["mid"]),
                                              min_value=min(new_lo, cfg["mid"]),
                                              max_value=max(new_lo, cfg["hi"]),
                                              step=0.01, key=f"{grp}_{sel_metric}_mid")
                new_hi  = col_hi.number_input(cfg["lbl_hi"],  value=float(cfg["hi"]),
                                              min_value=new_mid, max_value=float(rng_max),
                                              step=0.01, key=f"{grp}_{sel_metric}_hi")

                if sel_metric in LOWER_IS_BETTER:
                    thresholds[grp][sel_metric] = {
                        "below_avg": new_hi, "avg": new_mid, "above_avg": new_lo
                    }
                else:
                    thresholds[grp][sel_metric] = {
                        "below_avg": new_lo, "avg": new_mid, "above_avg": new_hi
                    }
            else:
                col_lo.metric(cfg["lbl_lo"],  f"{cfg['lo']}")
                col_mid.metric(cfg["lbl_mid"], f"{cfg['mid']}")
                col_hi.metric(cfg["lbl_hi"],  f"{cfg['hi']}")

    if edit_mode:
        col_save, col_dl, col_up = st.columns(3)
        with col_save:
            if st.button("💾 Save thresholds"):
                flatten_thresholds(thresholds).to_csv("thresholds.csv", index=False)
                st.success("Saved → thresholds.csv")
        with col_dl:
            dl_bytes = flatten_thresholds(thresholds).to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", dl_bytes, file_name="thresholds.csv", mime="text/csv", key="dl_thresh")
        with col_up:
            up_file = st.file_uploader("⬆️ Upload CSV", type="csv",
                                       help="Columns: Age Group, Metric, below_avg, avg, above_avg")
            if up_file:
                try:
                    df_up = pd.read_csv(up_file)
                    req = {"Age Group", "Metric", "below_avg", "avg", "above_avg"}
                    if not req.issubset(df_up.columns):
                        st.error("CSV missing required columns.")
                    else:
                        new = {}
                        for _, r in df_up.iterrows():
                            g, m = r["Age Group"], r["Metric"]
                            new.setdefault(g, {})[m] = {
                                "below_avg": float(r["below_avg"]),
                                "avg":       float(r["avg"]),
                                "above_avg": float(r["above_avg"]),
                            }
                        st.session_state["thresholds"] = new
                        st.success("Imported thresholds.")
                        st.rerun()
                except Exception as exc:
                    st.error(f"Failed to read CSV: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: REPORTS & TEMPLATES (uploads inside tab, two sub-tabs)
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("📄 Reports & Templates")

    rep_tab, tmpl_tab = st.tabs(["Generate Report", "Template's"])

    # A) Generate Report
    with rep_tab:
        import difflib
        from pandas.errors import EmptyDataError

        def safe_read_csv(file_obj):
            if file_obj is None:
                return pd.DataFrame()
            for enc in (None, "cp1252", "latin-1"):
                try:
                    file_obj.seek(0)
                    return pd.read_csv(file_obj, encoding=enc) if enc else pd.read_csv(file_obj)
                except (UnicodeDecodeError, EmptyDataError):
                    continue
            return pd.DataFrame()

        def normalize_dashes(s):
            s = "" if s is None else str(s)
            return s.replace("\u0096", "-").replace("–", "-").replace("—", "-")

        with st.expander("1️⃣  Upload CSVs & Map Names", expanded=True):
            up_cols = st.columns(3)
            with up_cols[0]:
                fs_file    = st.file_uploader("Flightscope CSV",         type="csv")
                throw_file = st.file_uploader("Throwing Velocities CSV", type="csv")
            with up_cols[1]:
                blast_file = st.file_uploader("Blast CSV",               type="csv")
                run_file   = st.file_uploader("Running Speed CSV",       type="csv")
            with up_cols[2]:
                mob_file   = st.file_uploader("Mobility CSV",            type="csv")
                dyn_file   = st.file_uploader("Dynamo CSV",              type="csv")

            flightscope_data = safe_read_csv(fs_file)
            blast_data       = safe_read_csv(blast_file)
            throwing_data    = safe_read_csv(throw_file)
            running_data     = safe_read_csv(run_file)
            mobility_data    = safe_read_csv(mob_file)
            dynamo_data      = safe_read_csv(dyn_file)

            for df, col in [
                (running_data,  "AthleteID"),
                (mobility_data, "Player Name"),
                (throwing_data, "Player Name"),
            ]:
                if df is not None and not df.empty and col in df.columns:
                    df[col] = df[col].astype(str).apply(normalize_dashes)

            player_db  = st.session_state.player_db.copy()
            player_db["nm"] = player_db["Name"].astype(str).str.lower().str.strip()
            canonical  = player_db["Name"].tolist()

            def mapper_ui(df, raw_col, label):
                if df is None or df.empty or raw_col not in df.columns:
                    return {}
                m = {}
                st.subheader(f"{label} name mapping")
                for raw in df[raw_col].dropna().unique():
                    best = difflib.get_close_matches(raw, canonical, n=3, cutoff=0.6)
                    default = best[0] if best else "<leave as is>"
                    choice = st.selectbox(f" {raw}",
                                          ["<leave as is>"] + canonical,
                                          index=(["<leave as is>"] + canonical).index(default),
                                          key=f"map_{label}_{raw}")
                    m[raw] = raw if choice == "<leave as is>" else choice
                return m

            run_map   = mapper_ui(running_data,  "AthleteID",   "Running")
            mob_map   = mapper_ui(mobility_data, "Player Name", "Mobility")
            throw_map = mapper_ui(throwing_data, "Player Name", "Throwing")

            if running_data is not None and not running_data.empty and "AthleteID" in running_data.columns:
                running_data["AthleteID"] = running_data["AthleteID"].map(lambda x: run_map.get(x, x))
            if mobility_data is not None and not mobility_data.empty and "Player Name" in mobility_data.columns:
                mobility_data["Player Name"] = mobility_data["Player Name"].map(lambda x: mob_map.get(x, x))
            if throwing_data is not None and not throwing_data.empty and "Player Name" in throwing_data.columns:
                throwing_data["Player Name"] = throwing_data["Player Name"].map(lambda x: throw_map.get(x, x))

        st.markdown("### 2️⃣  Select Player & Date")
        if "Age Group" not in st.session_state.player_db.columns:
            st.session_state.player_db["Age Group"] = st.session_state.player_db["Age"].apply(get_group)

        if st.session_state.player_db.empty:
            st.warning("Add players first on the **Player Database** tab.")
            st.stop()

        sel_idx = st.selectbox("Player", st.session_state.player_db.index,
                               format_func=lambda i: st.session_state.player_db.at[i, "Name"])
        prow = st.session_state.player_db.loc[sel_idx]
        assess_date = st.date_input("Assessment Date", datetime.date.today())

        player_info = {
            "Name": prow["Name"],
            "Age": int(prow["Age"]),
            "Age Group": prow["Age Group"],
            "Position": prow["Position"],
            "Class": prow["Class"],
            "High School": prow["High School"],
            "Height": prow["Height"],
            "Weight": prow["Weight"],
            "B/T": f"{prow.get('BattingHandedness','')}/{prow.get('ThrowingHandedness','')}".rstrip("/"),
            "DOB": prow["DOB"],
            "AssessmentDate": assess_date.strftime("%m/%d/%Y"),
        }

        ndf = st.session_state.get("notes_df", pd.DataFrame())
        last_note = (
            ndf.query("Name == @player_info['Name']")
               .sort_values("Date", ascending=False)
               .Note.head(1)
        )
        player_info["LatestNoteText"] = last_note.iat[0] if not last_note.empty else ""

        grp = player_info["Age Group"]

        def safe_merge_all(df, name_cols):
            if df is None or df.empty:
                return df
            tmp = df.copy()
            # choose first present column to derive "nm"
            for c in name_cols:
                if c in tmp.columns:
                    tmp["nm"] = tmp[c].astype(str).str.lower().str.strip()
                    break
            else:
                tmp["nm"] = None
            return tmp.merge(
                st.session_state.player_db.assign(nm=st.session_state.player_db["Name"].astype(str).str.lower().str.strip())[
                    ["nm", "DOB", "Age", "Age Group"]
                ],
                on="nm", how="left"
            )

        blast_data       = safe_merge_all(blast_data,       ["Name"])
        flightscope_data = safe_merge_all(flightscope_data, ["Name","Player Name","Batter"])
        throwing_data    = safe_merge_all(throwing_data,    ["Name","Player Name"])
        running_data     = safe_merge_all(running_data,     ["Name","Player Name","AthleteID"])
        mobility_data    = safe_merge_all(mobility_data,    ["Name","Batter","Player Name"])
        dynamo_data      = safe_merge_all(dynamo_data,      ["Name"])

        key = str(player_info["Name"]).lower().strip()
        def by_age(df):
            if df is None or df.empty or "Age Group" not in df.columns:
                return df
            return df[df["Age Group"] == grp]

        def by_name(df):
            if df is None or df.empty or "nm" not in df.columns:
                return df
            return df[df["nm"] == key]

        grp_blast = by_age(blast_data)
        grp_fs    = by_age(flightscope_data)
        grp_dyn   = by_age(dynamo_data)

        grp_throw = by_name(throwing_data)
        grp_run   = by_name(running_data)
        grp_mob   = by_name(mobility_data)

        max_ev, p90_ev        = calculate_flightscope_metrics(grp_fs) if (grp_fs is not None and not grp_fs.empty) else (None, None)
        averages, ranges      = calculate_blast_metrics(grp_blast)    if (grp_blast is not None and not grp_blast.empty) else ({}, {})
        velocities            = calculate_throwing_velocities(grp_throw)
        speeds, speed_ranges  = calculate_running_speeds(grp_run)

        mobility_dict = {}
        if grp_mob is not None and not grp_mob.empty:
            r = grp_mob.iloc[0]
            mobility_dict = {
                "Ankle":    r.get("Ankle Mobility"),
                "Thoracic": r.get("Thoracic Mobility"),
                "Lumbar":   r.get("Lumbar Mobility"),
            }

        if st.checkbox("Show debug preview"):
            for lbl, df in [("Blast", grp_blast), ("Flightscope", grp_fs),
                            ("Throwing", grp_throw), ("Running", grp_run),
                            ("Mobility", grp_mob), ("Dynamo", grp_dyn)]:
                st.markdown(f"**{lbl}** *(first 3 rows)*")
                if df is None:
                    st.write("None")
                elif df.empty:
                    st.write("Empty")
                else:
                    st.dataframe(df.head(3))

        st.markdown("---")
        if st.button("Generate Combined PDF", use_container_width=True):
            with st.spinner("Building PDF…"):
                pdf_buf = create_combined_pdf(
                    max_ev, p90_ev, averages, ranges,
                    velocities, speeds, speed_ranges,
                    player_info, grp_fs,
                    mobility=mobility_dict, dynamo_data=grp_dyn
                )
            st.success("PDF ready!")
            st.download_button("⬇️  Download",
                               data=pdf_buf,
                               file_name=f"{player_info['Name'].replace(' ','')}.pdf",
                               mime="application/pdf")

    # B) Template's
    with tmpl_tab:
        st.subheader("📥  Blank CSV Templates")
        def template_btn(fname, cols):
            csv = pd.DataFrame(columns=cols).to_csv(index=False).encode("utf-8")
            st.download_button(fname, csv, file_name=fname, mime="text/csv")

        colA, colB = st.columns(2)
        with colA:
            template_btn("running_speed_template.csv",
                         ["Player Name", "30yd Time", "60yd Time", "5-5-10 Shuttle Time"])
            template_btn("core_strength_template.csv",
                         ["Player Name", "Core Strength Measurement"])
        with colB:
            template_btn("throwing_velocities_template.csv",
                         ["Player Name", "Positional Throw Velocity", "Pulldown Velocity",
                          "FB Velocity", "SL Velocity", "CB Velocity", "CH Velocity"])
            template_btn("mobility_template.csv",
                         ["Player Name", "Ankle Mobility", "Thoracic Mobility", "Lumbar Mobility"])
