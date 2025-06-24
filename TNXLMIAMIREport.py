# =======================
# GLOBAL SETUP
# =======================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize
import os
import datetime
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, landscape
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Flowable, Image, KeepInFrame, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from pathlib import Path

# build one global stylesheet and register all your custom styles immediately
styles = getSampleStyleSheet()

HEADER_HEIGHT = 1.85 * inch
LOGO_RATIO = 3.0 / 3.7
LOGO_SIZE  = HEADER_HEIGHT * LOGO_RATIO
BAR_WIDTH  = 80
BAR_HEIGHT = 8
# ✅ Database filename
DATABASE_FILENAME = "player_database.csv"
NOTES_FILENAME = "scout_notes.csv"

# load or initialize notes DataFrame
if "notes_df" not in st.session_state:
    if os.path.exists(NOTES_FILENAME):
        st.session_state.notes_df = pd.read_csv(NOTES_FILENAME, parse_dates=["Date"])
    else:
        st.session_state.notes_df = pd.DataFrame(columns=["Name","Date","Note"])


LOGO_PATH = Path(__file__).parent / "assets" / "tnxl_logo.png" 

# ── LOGO DEBUG (delete after we’re done) ───────────────────
st.markdown("### Logo debug (cloud)")
st.write("LOGO_PATH:", LOGO_PATH)
st.write("Exists in this container →", LOGO_PATH.exists())
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), caption="Preview logo", width=120)
# ───────────────────────────────────────────────────────────

# thresholds for coloring bars:
# ──────────────────────────────────────────────────────────────────────────────
# 1) Thresholds + get_bar_color
# ──────────────────────────────────────────────────────────────────────────────
# ——— 1) Define age-groups and your base thresholds ———



from pandas.errors import EmptyDataError, ParserError
# (keep smart_read_csv here if you added it earlier)


AGE_LABELS = [
    "youth (12–13)",
    "jv (14–15)",
    "varsity (16–18)",
    "college (18+)",
]

# Metrics in which LOWER numbers mean better performance
lower_is_better = {
    "30yd Time",
    "60yd Time",
    "5-5-10 Shuttle Time",
    "Time to Contact (sec)",   # add more if you need
}


def broadcast_metrics_to_ages(base_metrics: dict, age_labels=AGE_LABELS) -> dict:
    """
    Take your flat `metric → {below, avg, above}` dict and copy those
    same cuts into every age group:
        {"Plane Score": {...}, ...}
    →  {
         "youth (12–13)":  {"Plane Score": {...}, ...},
         "jv (14–15)":     {"Plane Score": {...}, ...},
         ...
       }
    """
    return {
        age: {metric: cuts.copy() for metric, cuts in base_metrics.items()}
        for age in age_labels
    }


def flatten_thresholds(thr: dict, *, pad_factor: float = 0.1) -> pd.DataFrame:
    """
    Flatten age-group-centric thresholds into a DataFrame.
    Any scalar cut is padded to a 3-key dict:
        avg = scalar,
        below = avg * (1-pad_factor),
        above = avg * (1+pad_factor)
    """
    rows = []
    for grp, metrics in thr.items():
        for metric, cuts in metrics.items():

            # coerce scalar → dict if needed
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
                thr[grp][metric] = cuts            # write back the fixed version

            rows.append({
                "Age Group": grp,
                "Metric":     metric,
                "below_avg":  cuts["below_avg"],
                "avg":        cuts["avg"],
                "above_avg":  cuts["above_avg"],
            })
    return pd.DataFrame(rows)

# -------------------------------------------------
# Safe re-run helper (works on *all* Streamlit versions)
# -------------------------------------------------

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


age_groups = {
    "youth (12–13)":   lambda age: 12 <= age <= 13,
    "jv (14–15)":      lambda age: 14 <= age <= 15,
    "varsity (16–18)": lambda age: 16 <= age <= 18,
    "college (18+)":   lambda age: age >= 18,
}

def order_cuts(metric, lo, mid, hi):
    """
    Arrange the three cut‐points in the numeric order we want the sliders
    to travel and return a label/value mapping.

    For metrics where *lower is better* (times), the logical order is
        best (hi)  <  mid  <  worst (lo)
    and we rename the labels accordingly.
    """
    if metric in lower_is_better:
        return dict(
            lbl_lo="Worst (slowest)",
            lbl_mid="Avg",
            lbl_hi="Best (fastest)",
            lo=hi, mid=mid, hi=lo       # flip the ends
        )
    else:
        return dict(
            lbl_lo="Below",
            lbl_mid="Avg",
            lbl_hi="Above",
            lo=lo, mid=mid, hi=hi
        )


def get_group(age: int) -> str:
    """Return the age‐group label for a given age."""
    for grp, fn in age_groups.items():
        if fn(age):
            return grp
    return "unknown"


from reportlab.lib import colors

styles = getSampleStyleSheet()

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



def get_bar_color(metric_name: str, value: float, age_group: str) -> str:
    # grab the nested thresholds dict for this age_group
    all_th = st.session_state.get("thresholds", {})
    thr = all_th.get(age_group, {}).get(metric_name)

    # if no thresholds or no value, fallback grey
    if not thr or value is None:
        return "#7f8c8d"

    # list of metrics where lower is better
    lower_is_better = {
        "30yd Time", "60yd Time", "5-5-10 Shuttle Time"
    }

    if metric_name in lower_is_better:
        # smaller is better: above_avg = fastest
        if value <= thr["above_avg"]:
            return "#3498db"
        elif value <= thr["avg"]:
            return "#2ecc71"
        else:
            return "#f1c40f"
    else:
        # higher is better
        if value >= thr["above_avg"]:
            return "#3498db"
        elif value >= thr["avg"]:
            return "#2ecc71"
        else:
            return "#f1c40f"


from reportlab.lib import colors
from reportlab.lib.units import inch

def draw_header_bg(canvas, doc):
    w, h = doc.pagesize
    header_h = HEADER_HEIGHT
    x0 = doc.leftMargin + doc.width * 0.20

    canvas.saveState()
    # 1) Fill the entire header band black
    canvas.setFillColor(colors.black)
    canvas.rect(0, h - header_h, w, header_h, fill=1, stroke=0)

    # 2) Draw a gold triangle on the right side only
    canvas.setFillColor(colors.HexColor("#DDC38B"))
    path = canvas.beginPath()
    path.moveTo(w, h)                 # top‐right
    path.lineTo(w, h - header_h)      # bottom‐right of header
    path.lineTo(x0, h)                # back up to top at x0
    path.close()
    canvas.drawPath(path, fill=1, stroke=0)

    canvas.restoreState()

# Ensure player_db is initialized from file or fallback
if "player_db" not in st.session_state:
    if os.path.exists("player_database.csv"):
        try:
            st.session_state.player_db = pd.read_csv("player_database.csv")
            st.write("✅ Loaded player database from file.")
        except Exception as e:
            st.session_state.player_db = pd.DataFrame(columns=[
                "Name", "DOB", "Age", "Class", "High School", "Height", "Weight",
                "Position", "BattingHandedness", "ThrowingHandedness", "Age Group"
            ])
            st.warning("⚠️ Failed to load player_database.csv. Initialized empty DB.")
            st.exception(e)
    else:
        st.session_state.player_db = pd.DataFrame(columns=[
            "Name", "DOB", "Age", "Class", "High School", "Height", "Weight",
            "Position", "BattingHandedness", "ThrowingHandedness", "Age Group"
        ])
        st.info("ℹ️ No player_database.csv found. Starting with empty database.")

# Ensure thresholds are initialized from file or default
if "thresholds" not in st.session_state:
    if os.path.exists("thresholds.csv"):
        try:
            df = pd.read_csv("thresholds.csv")
            grouped = {}
            for _, row in df.iterrows():
                age_grp = row["Age Group"]
                metric = row["Metric"]
                grouped.setdefault(age_grp, {})[metric] = {
                    "below_avg": float(row["below_avg"]),
                    "avg": float(row["avg"]),
                    "above_avg": float(row["above_avg"])
                }
            st.session_state["thresholds"] = grouped
            st.write("✅ Loaded thresholds from thresholds.csv.")
        except Exception as e:
            st.session_state["thresholds"] = broadcast_metrics_to_ages(metric_thresholds)
            st.warning("⚠️ Failed to load thresholds.csv. Using default thresholds.")
            st.exception(e)
    else:
        st.session_state["thresholds"] = broadcast_metrics_to_ages(metric_thresholds)
        st.info("ℹ️ No thresholds.csv found. Using default thresholds.")



def smart_read_csv(file_obj, **read_kwargs):
    """
    Try reading a CSV as UTF-8 first; if that fails, fall back to Windows-1252
    and Latin-1.  Raises UnicodeDecodeError if none succeed.
    """
    for enc in (None, "utf-8", "cp1252", "latin-1"):
        try:
            file_obj.seek(0)  # rewind for each attempt
            if enc:
                return pd.read_csv(file_obj, encoding=enc, **read_kwargs)
            else:
                return pd.read_csv(file_obj, **read_kwargs)       # default UTF-8
        except (UnicodeDecodeError, EmptyDataError, ParserError):
            continue
    raise UnicodeDecodeError("Unable to decode file with common encodings.")




# ──────────────────────────────────────────────────────────────────────────────
# 2) RangeBar
# ──────────────────────────────────────────────────────────────────────────────
from reportlab.platypus import Flowable
from reportlab.lib import colors

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

        # ── background track
        c.setStrokeColor(colors.lightgrey)
        c.setLineWidth(self.height/3)
        c.line(x, y, x + self.width, y)

        # ── filled portion
        if self.max_value > self.min_value:
            pct = (self.value - self.min_value) / (self.max_value - self.min_value)
            pct = max(0, min(pct, 1))
        else:
            pct = 0
        filled_width = pct * self.width

        c.setStrokeColor(self.fill_color)
        c.setLineWidth(self.height/3)
        c.line(x, y, x + filled_width, y)

        # ── handle circle
        c.setFillColor(self.fill_color)
        c.circle(x + filled_width, y, self.handle_radius, stroke=0, fill=1)

# ──────────────────────────────────────────────────────────────────────────────
# 3) calculate_blast_metrics
# ──────────────────────────────────────────────────────────────────────────────
def calculate_blast_metrics(df):
    # pick the columns you actually want to average
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

# ──────────────────────────────────────────────────────────────────────────────
# 4) calculate_throwing_velocities
# ──────────────────────────────────────────────────────────────────────────────
def calculate_throwing_velocities(df):
    """
    Scan for *any* column with “velocity” in its header and return the
    mean of that column.  This way you don’t have to hard-code two keys.
    """
    if df is None: 
        return {}
    out = {}
    for col in df.columns:
        if "velocity" in col.lower():
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if not vals.empty:
                out[col] = float(vals.mean())
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 5) calculate_running_speeds
# ──────────────────────────────────────────────────────────────────────────────
def calculate_running_speeds(df):
    """
    Scan for any column with “30yd”, “60yd” or “shuttle” in its name,
    and return two dicts:
      - means:   average of each metric
      - ranges:  (min, max) of each metric
    """
    means  = {}
    ranges = {}
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

def calculate_mobility(df):
    if df is None: return {}
    out = {}
    for col in df.columns:
        low = col.lower()
        if "mobility" in low:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if not vals.empty:
                out[col] = float(vals.mean())
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Now **all** of those are defined, so when you paste in the Tab 3 block from my last 
# message, none of them will be “undefined” any more.
# ──────────────────────────────────────────────────────────────────────────────



# ✅ Expected columns for player database
expected_columns = [
    "Name", "DOB", "Age", "Class", "High School", "Height", "Weight",
    "Position", "BattingHandedness", "ThrowingHandedness"
]

# ✅ Function to update Age from DOB
def update_age_from_dob(df):
    import datetime  # SAFE import if not at the top already

    if "DOB" not in df.columns:
        df["DOB"] = ""
    if "Age" not in df.columns:
        df["Age"] = ""

    today = datetime.date.today()

    def calculate_age(dob):
        try:
            dob = pd.to_datetime(dob, errors='coerce')
            if pd.isna(dob):
                return ""
            return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        except:
            return ""

    df["Age"] = df["DOB"].apply(calculate_age)
    return df


# ✅ Load or initialize database safely
if "player_db" not in st.session_state:
    if os.path.exists(DATABASE_FILENAME):
        df = pd.read_csv(DATABASE_FILENAME)
        for col in expected_columns:
            if col not in df.columns:
                df[col] = ""
        df = update_age_from_dob(df)
        st.session_state.player_db = df
    else:
        st.session_state.player_db = pd.DataFrame(columns=expected_columns)

from reportlab.platypus import Flowable
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, Image, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import os

def build_header_with_logo_and_player_info(
    logo_path,
    player_info,
    width,
    name_style=None,
    info_style=None,
    program_style=None,
    date_style=None,
):
    global styles

    # 1) define or override your text styles
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
    textColor=colors.white   # ← and here
)

    # 2) Left: logo
    if os.path.exists(logo_path):
        logo = Image(logo_path, width=LOGO_SIZE, height=LOGO_SIZE)
    else:
        logo = Paragraph("LOGO MISSING", info_style)

    # 3) Middle: player name + combined position/class/HS + details
    name = Paragraph(player_info.get("Name", ""), name_style)
    pos_and_school = Paragraph(
        f"{player_info.get('Position','')} | "
        f"{player_info.get('High School','')} | "
        f"{player_info.get('Class','')}",
        info_style
    )
    h_inches = player_info.get("Height",0) or 0
    h_ft,h_in = divmod(int(h_inches),12)
    height_str = f"{h_ft}'{h_in}"
    
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

    middle = [
        name,
        Spacer(1, 4),
        pos_and_school,
        height_wt,
        bat_throw,
        dob
    ]

    # 4) Right: program title & date
    program = Paragraph("Summer Development Program", program_style)
    assess  = Paragraph(
        f"Assessment Date: {player_info.get('AssessmentDate','')}",
        date_style
    )
    right = [program, Spacer(1, 4), assess]

    # 5) Assemble into a single‐row, three‐column table
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





def calculate_flightscope_metrics(data):
    """
    Given a Flightscope DataFrame, find the max exit velocity and the 90th percentile.
    Returns (max_ev, percentile_90_ev).
    """
    # find the first column whose name contains both “exit” and “speed”
    exit_speed_column = None
    for col in data.columns:
        if "exit" in col.lower() and "speed" in col.lower():
            exit_speed_column = col
            break

    if exit_speed_column is None:
        st.error("No Exit Speed column found in Flightscope data.")
        return None, None

    # coerce to numeric & drop
    data[exit_speed_column] = pd.to_numeric(data[exit_speed_column], errors="coerce")
    cleaned = data.dropna(subset=[exit_speed_column])
    if cleaned.empty:
        return None, None

    max_ev = cleaned[exit_speed_column].max()
    percentile_90_ev = cleaned[exit_speed_column].quantile(0.9)
    return max_ev, percentile_90_ev


def safe_est_poly_at_t(t, poly_str):
    """
    Evaluate a 5th-order polynomial string safely at time t.
    Handles missing or incomplete polynomial coefficients.
    """
    try:
        coeffs = [float(x.strip()) for x in poly_str.split(";")]
        if len(coeffs) < 5 or any(pd.isna(coeff) for coeff in coeffs):
            return None
        return sum(coeffs[i] * t**i for i in range(5))
    except Exception:
        return None




from reportlab.platypus import Table as RLTable, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Flowable
from reportlab.lib import colors


# --- Main function to build gameplay data table
from reportlab.platypus import Table as RLTable, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import Spacer, Table as RLTable, TableStyle
from reportlab.lib import colors
# ─── 1. Improved build_gameplay_data_table ────────────────────────────────────
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
    from reportlab.platypus import Table as RLTable, TableStyle, Spacer
    from reportlab.lib import colors

    data = [["Metric", "Value", "Range / Visual"]]

    blast_metrics = [
        ("Plane Score",              "Plane Score"),
        ("Connection Score",         "Connection Score"),
        ("Rotation Score",           "Rotation Score"),
        ("Attack Angle (°)",         "Attack Angle (deg)"),
        ("On-Plane Efficiency (%)",  "On Plane Efficiency (%)"),
        ("Time to Contact (s)",      "Time to Contact (sec)"),
        ("Bat Speed (mph)",          "Bat Speed (mph)"),
        ("Rotational Acceleration (g)","Rotational Acceleration (g)"),
        ("Peak Hand Speed (mph)",    "Peak Hand Speed (mph)"),
        ("Connection at Impact (°)", "Connection at Impact (deg)"),
        ("Early Connection (°)",     "Early Connection (deg)"),
        ("Vertical Bat Angle (°)",   "Vertical Bat Angle (deg)")
    ]

    for label, key in blast_metrics:
        value     = averages.get(key)
        value_str = f"{value:.2f}" if value is not None else "N/A"

        # look up the cuts for this age_group + metric
        cuts = thresholds.get(age_group, {}).get(key)
        if cuts:
            rmin, rmax = cuts["below_avg"], cuts["above_avg"]
        else:
            rmin, rmax = ranges.get(key, (None, None))

        if value is not None and rmin is not None and rmax is not None:
            color  = get_bar_color(key,value,age_group)
            visual = RangeBar(
                value, rmin, rmax,
                width=BAR_WIDTH,
                height=BAR_HEIGHT,
                fill_color=colors.HexColor(color),
                show_range=False
            )
        else:
            visual = "—"

        data.append([Paragraph (label,styles["Normal"]), value_str, visual])

    # Max EV
    if max_ev is not None:
        key = "Max EV (mph)"
        cuts = thresholds.get(age_group, {}).get(key, {})
        rmin, rmax = cuts.get("below_avg"), cuts.get("above_avg")
        if rmin is not None and rmax is not None:
            color  = get_bar_color(key,max_ev,age_group)
            visual = RangeBar(max_ev, rmin, rmax,
                              width=BAR_WIDTH, height=BAR_HEIGHT,
                              fill_color=colors.HexColor(color),
                              show_range=False)
        else:
            visual = "—"
        data.append([key, f"{max_ev:.1f}", visual])

    # 90th % EV
    if percentile_90_ev is not None:
        key = "90th % EV (mph)"
        cuts = thresholds.get(age_group, {}).get(key, {})
        rmin, rmax = cuts.get("below_avg"), cuts.get("above_avg")
        if rmin is not None and rmax is not None:
            color  = get_bar_color(key,percentile_90_ev, age_group)
            visual = RangeBar(percentile_90_ev, rmin, rmax,
                              width=BAR_WIDTH, height=BAR_HEIGHT,
                              fill_color=colors.HexColor(color),
                              show_range=False)
        else:
            visual = "—"
        data.append([key, f"{percentile_90_ev:.1f}", visual])

    # Throwing velocities
    for pitch, velo in velocities.items():
        value_str = f"{velo:.1f} mph" if velo is not None else "N/A"
        cuts = thresholds.get(age_group, {}).get(pitch, {})
        rmin, rmax = cuts.get("below_avg"), cuts.get("above_avg")

        if velo is not None and rmin is not None and rmax is not None:
            color  = get_bar_color(pitch,velo,age_group)
            visual = RangeBar(velo, rmin, rmax,
                              width=BAR_WIDTH, height=BAR_HEIGHT,
                              fill_color=colors.HexColor(color),
                              show_range=False)
        else:
            visual = "—"

        data.append([pitch, value_str, visual])

    table = RLTable(data,
                    colWidths=[width*0.30, width*0.12, width*0.30],
                    hAlign="LEFT")
    table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#DDC38B')),
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



def get_player_forcedecks_data(force_df, player_name):
    if force_df is None or player_name is None:
        return None

    player_data = force_df[force_df['Name'].str.lower() == player_name.lower()]
    if player_data.empty:
        return None

    output = []
    grouped = player_data.groupby('Test Type')

    for test_type, group in grouped:
        row = [test_type]

        # Handle each metric safely
        jh = group.get('Jump Height (Imp-Mom) in Inches [in]', pd.Series([None])).values[0]
        pp = group.get('Peak Power / BM [W/kg]', pd.Series([None])).values[0]
        rsi = group.get('RSI-modified [m/s]', pd.Series([None])).values[0]
        rfd = group.get('Concentric RFD % (Asym) (%)', pd.Series([None])).values[0]

        row.append(f"{jh:.1f}" if pd.notnull(jh) else "")
        row.append(f"{pp:.1f}" if pd.notnull(pp) else "")
        row.append(f"{rsi:.2f}" if pd.notnull(rsi) else "")
        row.append(rfd.strip() if isinstance(rfd, str) else f"{rfd:.1f}" if pd.notnull(rfd) else "")

        output.append(row)

    return output


def generate_exit_velo_heatmap(df):
    import numpy as np
    import matplotlib.pyplot as plt
    from reportlab.platypus import Image
    from io import BytesIO

    # 1) Filter to true events
    df = df.dropna(subset=["Parsed_X","Parsed_Z","Exit_Speed"])
    if df.empty:
        return None

    # 2) Convert to inches
    x = df["Parsed_X"].values * 12
    y = df["Parsed_Z"].values * 12
    c = df["Exit_Speed"].values   # ← define c here

    # 3) Debug ranges _after_ you have x,y,c
    st.write(f"💡 heatmap rows: {len(df)}")
    st.write(f"💡 x range: {x.min():.1f}→{x.max():.1f}, y range: {y.min():.1f}→{y.max():.1f}")
    
    # 4) Hexbin plot
    Zoom_ext = [-18, 18, 0, 60]
    fig, ax = plt.subplots(figsize=(5,5))
    hb = ax.hexbin(
        x, y, C=c,
        reduce_C_function=np.mean,
        gridsize=(8,8),
        cmap="coolwarm",
        mincnt=1,
        extent=Zoom_ext
    )
    ax.set_xlim(Zoom_ext[0], Zoom_ext[1])
    ax.set_ylim(Zoom_ext[2], Zoom_ext[3])
    ax.set_aspect("equal", "box")
    ax.axis("off")


    offsets = hb.get_offsets()    # Nx2 array of hex centers (in data coords)
    values  = hb.get_array()      # N aggregated C‐values
    for (cx, cy), v in zip(offsets, values):
        ax.text(
            cx, cy,
            f"{v:.1f}",           # format to one decimal
            ha="center", va="center",
            fontsize=10,
            color="white"         # pick a contrasting color
        )


    # 5) Strike‐zone overlay
    sz_w, sz_h = 17, 25
    left, bottom = -sz_w/2, 16
    ax.add_patch(plt.Rectangle((left,bottom), sz_w, sz_h,
                  fill=False, lw=2, edgecolor="black"))
    ax.add_patch(plt.Rectangle((left,bottom), sz_w, sz_h,
                  fill=False, lw=1, linestyle="--", edgecolor="black"))

    # 6) Export to ReportLab
    buf = BytesIO()
    plt.tight_layout(pad=0)
    fig.savefig(buf, format="png", dpi=150, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=280, height=280)


def build_dynamo_table(dynamo_data, player_info, width):
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    # if there’s no dynamo_data at all, bail out with a message
    if dynamo_data is None or dynamo_data.empty:
        return Paragraph("No Dynamo Data", getSampleStyleSheet()["Normal"])

    # filter to only this player
    name = player_info.get("Name", "").lower()
    df = dynamo_data[dynamo_data["Name"].str.lower() == name]
    if df.empty:
        return Paragraph("No Dynamo Data for this player", getSampleStyleSheet()["Normal"])
     
    df = dynamo_data [ dynamo_data["Name"].str.lower() == name]
    numeric_cols = [
        "ROM Asymmetry (%)","Force Asymmetry (%)",
        "L Max ROM (°)","R Max ROM (°)",
        "L Max Force (N)","R Max Force (N)",
    ] 
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # build the rows
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
            r["Movement"],
            r["Type"],
            f"{r.get('ROM Asymmetry (%)'):.1f}" if not pd.isna(r.get("ROM Asymmetry (%)")) else "N/A",
            f"{r.get('Force Asymmetry (%)'):.1f}" if not pd.isna(r.get("Force Asymmetry (%)")) else "N/A",
            f"{r.get('L Max ROM (°)'):.1f}" if not pd.isna(r.get("L Max ROM (°)")) else "N/A",
            f"{r.get('R Max ROM (°)'):.1f}" if not pd.isna(r.get("R Max ROM (°)")) else "N/A"
        ])
    # create the table with the same look as your other tables
    tbl = Table(data, colWidths=[width/6]*6)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#DDC38B')),
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

from reportlab.platypus import PageBreak

def build_forcedecks_table(forcedecks_df, player_name, width):
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib import colors

    if forcedecks_df is None or forcedecks_df.empty:
        return Paragraph("No ForceDecks data available.", styles["Normal"])

    # Filter and group by Test Type
    df = forcedecks_df[forcedecks_df["Name"].str.lower().str.strip() == player_name.lower().strip()]

    if df.empty:
        return Paragraph("No ForceDecks data for this player.", styles["Normal"])

    # Columns we want
    columns = [
        "Test Type",
        "Jump Height (Imp-Mom) in Inches [in]",
        "Peak Power / BM [W/kg]",
        "RSI-modified [m/s]",
        "Concentric RFD % (Asym) (%)"
    ]
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        return Paragraph(f"Missing columns in ForceDecks CSV: {', '.join(missing_cols)}", styles["Normal"])

    # Table rows
    data = [columns]
    for _, row in df.iterrows():
        data.append([
            row["Test Type"],
            f"{row[columns[1]]:.1f}" if pd.notnull(row[columns[1]]) else "",
            f"{row[columns[2]]:.1f}" if pd.notnull(row[columns[2]]) else "",
            f"{row[columns[3]]:.2f}" if pd.notnull(row[columns[3]]) else "",
            str(row[columns[4]])
        ])

    tbl = Table(data, colWidths=[width/5]*5, hAlign='LEFT')
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0E0A06")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))
    return tbl



# ================================
# PDF Report Building
# ================================

from io import BytesIO
from reportlab.lib.pagesizes import A3, landscape
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepInFrame
)
from reportlab.lib import colors

def create_combined_pdf(
    max_ev,
    percentile_90_ev,
    averages,
    ranges,
    velocities,
    speeds,
    speed_ranges,          # ← NEW
    player_info,
    flightscope_data,
    mobility=None,
    dynamo_data=None,
    forcedecks_data=None, 
    
):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A3),
        rightMargin=30,
        leftMargin=30,
        topMargin=10,
        bottomMargin=30,
    )
    elements = []
    
     
    # ── Header ──────────────────────────────────────────────────────────
    header = build_header_with_logo_and_player_info(
        str(LOGO_PATH),
        player_info,
        doc.width,
    )
    elements += [header, Spacer(1, 12)]

    # ── Exit-velo heat-map (if Flightscope present) ─────────────────────
    heatmap_img = None
    if flightscope_data is not None and not flightscope_data.empty:
        flightscope_data["Parsed_X"] = flightscope_data["Hit_Poly_X"].apply(
            lambda p: safe_est_poly_at_t(0, p) if isinstance(p, str) else None
        )
        flightscope_data["Parsed_Z"] = flightscope_data["Hit_Poly_Z"].apply(
            lambda p: safe_est_poly_at_t(0, p) if isinstance(p, str) else None
        )
        flightscope_data["Exit_Speed"] = pd.to_numeric(
            flightscope_data["Exit_Speed"], errors="coerce"
        )
        valid = (
            flightscope_data.dropna(subset=["Parsed_X", "Parsed_Z", "Exit_Speed"])
            .query("Exit_Speed > 0")
            .copy()
        )
        valid["PlateLocSide"] = -valid["Parsed_X"] * 12.0
        valid["PlateLocHeight"] = valid["Parsed_Z"] * 12.0
        heatmap_img = generate_exit_velo_heatmap(valid)

    # ── Row 1 : Gameplay data + heat-map ────────────────────────────────
    left_w  = doc.width * 0.61
    right_w = doc.width - left_w

    gameplay_frame = KeepInFrame(
        left_w,
        doc.height,
        [
            Paragraph("Gameplay Data", styles["Heading3"]),
            Spacer(1, 6),
            build_gameplay_data_table(
                averages,
                ranges,
                max_ev,
                percentile_90_ev,
                velocities,
                left_w,
                thresholds=st.session_state["thresholds"],
                age_group=player_info["Age Group"],
            ),
        ],
        hAlign="LEFT",
        mergeSpace=True,
    )

    right_contents = [
        Paragraph("AVG Exit Velocity by Zone", styles["Heading3"]),
        Spacer(1, 6),
        heatmap_img
        if heatmap_img
        else Paragraph("No heat-map data", styles["Normal"]),
    ]
    right_frame = KeepInFrame(right_w, doc.height, right_contents, hAlign="LEFT")

    elements += [
        Table([[gameplay_frame, right_frame]], colWidths=[left_w, right_w]),
        Spacer(1, 12),
    ]

    # ── Row 2 : Physical Profile & Scout Notes ──────────────────────────
    notes_txt = player_info.get("LatestNoteText")
    notes_txt = (
        str(notes_txt)
        if notes_txt not in (None, "nan", "NaN")
        else "No scout notes available."
    )

    phys_w   = left_w
    notes_w  = doc.width - phys_w

    physical_frame = KeepInFrame(
        phys_w,
        doc.height,
        [
            Paragraph("Physical Profile", styles["Heading3"]),
            Spacer(1, 6),
            build_profile_table(
                mobility,
                speeds,
                speed_ranges,
                phys_w,
                thresholds=st.session_state.get("thresholds"),
                age_group=player_info["Age Group"],
            ),
        ],
        hAlign="LEFT",
        mergeSpace=True,
    )

    notes_tbl = Table(
        [
            [Paragraph("Scout Notes", styles["Heading3"])],
            [Paragraph(notes_txt, styles["Normal"])],
        ],
        colWidths=[notes_w],
    )
    notes_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDC38B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("VALIGN", (0, 1), (-1, 1), "TOP"),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    elements += [
        Table([[physical_frame, notes_tbl]], colWidths=[phys_w, notes_w]),
        Spacer(1, 12),
    ]

    # ── Row 3 : Dynamo summary ───────────────────────────────────────────
    dynamo_frame = KeepInFrame(
        left_w,
        doc.height,
        [
            Paragraph("Dynamo Summary", styles["Heading3"]),
            Spacer(1, 6),
            build_dynamo_table(dynamo_data, player_info, left_w),
        ],
        hAlign="LEFT",
        mergeSpace=True,
    )
    elements.append(Table([[dynamo_frame, ""]], colWidths=[left_w, right_w]))
    
    # ForceDecks table (like Dynamo)
    elements.append(PageBreak())
    elements.append(Paragraph("ForceDecks Summary", styles["Heading3"]))
    elements.append(Spacer(1, 6))

    if forcedecks_data is not None and not forcedecks_data.empty:
      force_tbl = build_forcedecks_table(forcedecks_data, player_info.get("Name", ""), doc.width)
    else:
      force_tbl = Paragraph("No ForceDecks data available.", styles["Normal"])

    elements.append(force_tbl)


    # ── Build PDF ────────────────────────────────────────────────────────
    doc.build(elements, onFirstPage=draw_header_bg, onLaterPages=draw_header_bg)
    buffer.seek(0)
    return buffer









from reportlab.platypus import Table as RLTable, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

def build_profile_table(
    mobility: dict,
    speeds: dict,
    speed_ranges: dict,
    width: float,
    thresholds: dict,
    age_group: str
):
    from reportlab.platypus import Table as RLTable, TableStyle, Paragraph
    from reportlab.lib import colors

    data = [["Metric", "Value", "Δ (max–min)"]]

    # — Mobility metrics —
    for key in ["Ankle", "Thoracic", "Lumbar"]:
        score = mobility.get(key)
        if score is None:
            val_str, delta = "N/A", "—"
        else:
            cuts = thresholds.get(age_group, {}).get(key, {})
            rmin, rmax = cuts.get("below_avg"), cuts.get("above_avg")
            val_str = f"{score:.2f}"
            if rmin is not None and rmax is not None and rmax != rmin:
                delta = f"{(rmax - rmin):.2f}"
            else:
                delta = "—"
        data.append([
            Paragraph(key, styles["Normal"]),
            val_str,
            Paragraph(delta, styles["Normal"])
        ])

    # — Running‐speed metrics —
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

        data.append([
            Paragraph(key, styles["Normal"]),
            val_str,
            Paragraph(delta, styles["Normal"])
        ])

    tbl = RLTable(
        data,
        colWidths=[width*0.30, width*0.12, width*0.30],
        hAlign="LEFT"
    )
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#DDC38B')),
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







@st.cache_data
def load_player_db(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame(
            columns=["Name","DOB","Age","Class","High School","Height","Weight","Position","BattingHandedness","ThrowingHandedness"]
        )
    return df

DATABASE_FILENAME = "player_database.csv"
if "player_db" not in st.session_state:
    st.session_state.player_db = load_player_db(DATABASE_FILENAME)

if "player_db" not in st.session_state or st.session_state["player_db"].empty:
    st.warning("⚠️ Player database is empty. Tabs 4 and 5 may not function.")
    st.session_state["player_db"] = pd.DataFrame(columns=[
        "Name", "Age", "Class", "High School", "Height", "Weight",
        "Position", "BattingHandedness", "ThrowingHandedness", "DOB"
    ])





# =======================
# Streamlit App Tabs
# =======================
from datetime import date
import traceback
st.title("TNXL MIAMI - Athlete Performance Data Uploader, Report Generator & CSV Utilities")
st.markdown("✅ Tabs are initializing")

# ───────────────────────────────────────────────────────────
# Tab labels and flags
# ───────────────────────────────────────────────────────────
tab_labels = ["CSV Merge", "Player Database", "Scout Notes", "Thresholds", "Report Generation"]

page = st.radio(
    "Jump to section:",
    tab_labels,
    horizontal=True,
    key="main_nav"          # <— remembers choice across reruns
)

# right after the st.radio() line
st.markdown(
    """
    <style>
    /* ───────── make the new Streamlit-1.46 radio look like tabs ───────── */

    /* 1️⃣  Lay the options horizontally with no gaps */
    .stRadio > div               { flex-direction: row; gap: 0 !important; }

    /* 2️⃣  Hide the default radio circles (inputs) */
    .stRadio input[type="radio"] { display: none; }

    /* 3️⃣  Pill container */
    .stRadio div[role="radiogroup"] > label {
        background: #eeeeee;
        padding: 0.35rem 0.9rem;
        border-radius: 8px;
        margin-right: 0.5rem;        /* spacing between pills */
        cursor: pointer;
        transition: background 0.25s;
        color: #333333;
        font-weight: 500;
    }

    /* 4️⃣  Hover effect */
    .stRadio div[role="radiogroup"] > label:hover {
        background: #ddc38b66;       /* 40 % opaque gold */
    }

    /* 5️⃣  Selected pill (checked input sits *inside* the label) */
    .stRadio input:checked + div   { display: none; }   /* hide leftover glyph */
    .stRadio input:checked + div + span {
        background: #ddc38b;
        color: #000;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Set active tab based on session or default to tab1
# --- initialise once ---
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "CSV Merge"


def show_csv_merge_ui():
    """Tab 1 - CSV merger"""
    
    # 1) Choose your CSV category
    csv_categories = [
        "Blast",
        "Flightscope",
        "Throwing Velocities",
        "Running Speed",
        "Mobility",
        "Dynamo",
    ]
    csv_type = st.selectbox("Select CSV Type to Merge", csv_categories)

    # 2) Upload one or more files of that type
    files = st.file_uploader(
        f"Upload {csv_type} CSV Files",
        type="csv",
        accept_multiple_files=True,
        key="merge_files"
    )

    if files:
        merged_dfs = []
        st.markdown("### Configure Each File")
        for idx, uploaded in enumerate(files):
            st.subheader(f"File {idx+1}: {uploaded.name}")
            df = pd.read_csv(uploaded)

            # Show detected columns
            st.write("Columns detected:", df.columns.tolist())

            # If it’s a Blast CSV, ensure there's a Name column and allow editing
            if csv_type == "Blast":
                if "Name" not in df.columns:
                    df.insert(0, "Name", "")  # add empty Name column at front
                st.markdown("**Edit Blast data (e.g. player names)**")
                df = st.data_editor(
                    df,
                    use_container_width=True,
                    num_rows="dynamic",
                    key=f"blast_edit_{idx}"
                )

            # Let the user supply a label for this file
            default_label = os.path.splitext(uploaded.name)[0]
            label = st.text_input(
                f"Label for {uploaded.name}",
                value=default_label,
                key=f"label_{idx}"
            )

            # Tag the DataFrame
            df["Type"]  = csv_type
            df["Label"] = label

            merged_dfs.append(df)

        # 3) Merge and download
        if st.button("Merge Selected Files"):
            merged = pd.concat(merged_dfs, ignore_index=True)
            st.success(f"Merged {len(merged_dfs)} files of type {csv_type}!")

            st.dataframe(merged.head())
            csv_bytes = merged.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Merged CSV",
                data=csv_bytes,
                file_name=f"merged_{csv_type.lower().replace(' ','_')}.csv",
                mime="text/csv"
            )
    else:
        st.info(f"Upload two or more {csv_type} CSVs above to merge them.")


pass

import datetime


# ─── Tab 2 : Player Database ──────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────
# TAB 2 – PLAYER DATABASE
# ────────────────────────────────────────────────────────────────────
# 

def show_player_db_ui():
    st.subheader("👥 Player database")
    st.caption("Rows live in **player_database.csv**")

    # — Four sub-tabs ---------------------------------------------------
    view_tab, add_tab, edit_tab, io_tab = st.tabs(
        ["📋 View", "➕ Add", "✏️ Edit / Delete", "⬇️⬆️ Import / Export"]
    )

    # 1️⃣ VIEW  ────────────────────────────────────────────────────────
    with view_tab:
        db_view = st.session_state.player_db.copy()
        db_view["Age"] = pd.to_numeric(db_view["Age"], errors="coerce").astype("Int64")
        st.dataframe(db_view, use_container_width=True, height=430)

    # 2️⃣ ADD  ─────────────────────────────────────────────────────────
    with add_tab:
        with st.form("add_player_form", clear_on_submit=True):
            c1, c2, c3, c4 = st.columns(4)

            # ─ column-1 ─
            with c1:
                add_name = st.text_input("Name")
                add_dob  = st.date_input(
                    "Date of Birth",
                    min_value=datetime.date(1900,1,1),
                    max_value=datetime.date.today()
                )

            # ─ column-2 ─
            with c2:
                add_class = st.text_input("Class")
                add_hs    = st.text_input("High School")

            # ─ column-3 ─
            with c3:
                ft = st.number_input("Height (ft)", 0, 8, value=0)
                inch = st.number_input("Height (in)", 0, 11, value=0)
                add_height = ft*12 + inch
                add_weight = st.number_input("Weight (lbs)", 0, 500, value=0)

            # ─ column-4 ─
            pos_opts = ["Pitcher","Catcher","1B","2B","3B","SS","LF","CF","RF","DH"]
            bat_opts = ["Left","Right","Switch"]
            thr_opts = ["Left","Right"]
            with c4:
                add_pos   = st.selectbox("Position", pos_opts)
                add_bat   = st.selectbox("Batting Side", bat_opts)
                add_throw = st.selectbox("Throwing Arm", thr_opts)

            submitted = st.form_submit_button("Save ➕")
        if submitted:
            today = datetime.date.today()
            age   = today.year - add_dob.year - ((today.month, today.day) < (add_dob.month, add_dob.day))
            new_row = {
                "Name": add_name,
                "DOB":  add_dob.strftime("%m/%d/%Y"),
                "Age":  age,
                "Age Group": get_group(age),
                "Class": add_class,
                "High School": add_hs,
                "Height": add_height,
                "Weight": add_weight,
                "Position": add_pos,
                "BattingHandedness": add_bat,
                "ThrowingHandedness": add_throw,
            }
            st.session_state.player_db = pd.concat(
                [st.session_state.player_db, pd.DataFrame([new_row])],
                ignore_index=True
            )
            st.session_state.player_db.to_csv(DATABASE_FILENAME, index=False)
            st.success(f"✅ Added {add_name}")

    # 3️⃣ EDIT / DELETE  ───────────────────────────────────────────────
    with edit_tab:
        db = st.session_state.player_db
        if db.empty:
            st.info("Database is empty — add a player first.")
        else:
            idx = st.selectbox("Choose player", db.index,
                               format_func=lambda i: db.at[i, "Name"])
            sel = db.loc[idx]

            with st.form("edit_player_form", clear_on_submit=True):
                # pre-fill values
                c1,c2,c3,c4 = st.columns(4)

                # name / dob
                with c1:
                    e_name = st.text_input("Name", sel["Name"])
                    init_dob = pd.to_datetime(sel["DOB"], errors="coerce")
                    e_dob  = st.date_input("DOB", init_dob.date() if pd.notna(init_dob) else datetime.date.today())

                # school
                with c2:
                    e_class = st.text_input("Class", sel["Class"])
                    e_hs    = st.text_input("High School", sel["High School"])

                # body
                raw_h = int(sel.get("Height",0) or 0)
                ft0, in0 = divmod(raw_h, 12)
                with c3:
                    ft  = st.number_input("Height (ft)", 0, 8, value=ft0)
                    inch = st.number_input("Height (in)", 0, 11, value=in0)
                    e_height = ft*12 + inch
                    e_weight = st.number_input("Weight (lbs)", 0, 500, value=int(sel.get("Weight",0) or 0))

                # handedness / pos
                pos_opts = ["Pitcher","Catcher","1B","2B","3B","SS","LF","CF","RF","DH"]
                bat_opts = ["Left","Right","Switch"]
                thr_opts = ["Left","Right"]
                with c4:
                    e_pos = st.selectbox("Position", pos_opts, pos_opts.index(sel["Position"]))
                    e_bat = st.selectbox("Bat side", bat_opts, bat_opts.index(sel["BattingHandedness"]))
                    e_thr = st.selectbox("Throw arm", thr_opts, thr_opts.index(sel["ThrowingHandedness"]))

                col_u, col_d = st.columns(2)
                update_btn = col_u.form_submit_button("💾 Update")
                delete_btn = col_d.form_submit_button("🗑️ Delete", type="primary")

            if update_btn:
                today  = datetime.date.today()
                e_age  = today.year - e_dob.year - ((today.month, today.day) < (e_dob.month, e_dob.day))
                updates = {
                    "Name": e_name, "DOB": e_dob.strftime("%m/%d/%Y"),
                    "Age": e_age, "Age Group": get_group(e_age),
                    "Class": e_class, "High School": e_hs,
                    "Height": e_height, "Weight": e_weight,
                    "Position": e_pos, "BattingHandedness": e_bat,
                    "ThrowingHandedness": e_thr
                }
                for col,val in updates.items():
                    st.session_state.player_db.at[idx, col] = val
                st.session_state.player_db.to_csv(DATABASE_FILENAME, index=False)
                st.success("✅ Updated")

            if delete_btn:
                st.session_state.player_db = db.drop(idx).reset_index(drop=True)
                st.session_state.player_db.to_csv(DATABASE_FILENAME, index=False)
                st.success("🗑️ Deleted")

    # 4️⃣ IMPORT / EXPORT  ──────────────────────────────────────────────
    with io_tab:
        st.markdown("#### Download")
        st.download_button(
            "⬇️ Current DB CSV",
            st.session_state.player_db.to_csv(index=False).encode(),
            "player_database.csv",
            "text/csv"
        )

        st.markdown("#### Upload")
        up_file = st.file_uploader("Upload CSV", "csv")
        if up_file:
            try:
                imported = pd.read_csv(up_file)
                missing  = [c for c in expected_columns if c not in imported.columns]
                if missing:
                    st.error("CSV missing columns: " + ", ".join(missing))
                else:
                    mode = st.radio("Import mode",
                                    ["Replace", "Merge (dedupe by Name)"],
                                    horizontal=True)
                    if st.button("Import"):
                        if mode == "Replace":
                            st.session_state.player_db = imported.copy()
                        else:
                            merged = pd.concat([st.session_state.player_db, imported], ignore_index=True)
                            st.session_state.player_db = merged.drop_duplicates("Name", keep="last")
                        st.session_state.player_db.to_csv(DATABASE_FILENAME, index=False)
                        st.success("✅ Import complete")
            except Exception as e:
                st.error(f"Couldn’t read CSV — {e}")

        # Danger zone
        with st.expander("🛑 Danger zone – clear entire database"):
            if st.button("Delete ALL players", type="primary"):
                st.session_state.player_db = pd.DataFrame(columns=expected_columns)
                if os.path.exists(DATABASE_FILENAME):
                    os.remove(DATABASE_FILENAME)
                st.success("Database cleared")
pass

# ─── Tab 3 : Scout Notes ────────────────────────────────────────────────
# ─── Tab 3 : Scout Notes ────────────────────────────────────────────────

def show_scout_notes_ui():

    import datetime
    st.subheader("📋 Scout Notes")

    # ── 0) ensure notes_df exists in session_state ──────────────────────
    if "notes_df" not in st.session_state:
        if os.path.exists(NOTES_FILENAME):
            st.session_state.notes_df = pd.read_csv(
                NOTES_FILENAME, parse_dates=["Date"])
        else:
            st.session_state.notes_df = pd.DataFrame(
                columns=["Name", "Date", "Note"])

    # ── 1) player selector & “🆕 New note” button in one row ────────────
    top_l, top_r = st.columns([3, 1])
    with top_l:
        sel_player = st.selectbox(
            "Player", st.session_state.player_db["Name"].tolist(),
            key="notes_player")
    with top_r:
        if st.button("🆕 New note"):
            new_idx = len(st.session_state.notes_df)
            st.session_state.notes_df.loc[new_idx] = {
                "Name": sel_player,
                "Date": pd.Timestamp.today().normalize(),
                "Note": ""}
            st.session_state.active_note = new_idx

    # ── 2) filter notes for that player & sort newest-first ─────────────
    notes_view = (
        st.session_state.notes_df
        .query("Name == @sel_player")
        .sort_values("Date", ascending=False)
        .reset_index(drop=True)
    )

    if notes_view.empty:
        st.info("No notes yet — click **New note** above to add one.")
        st.stop()

    # remember which note is “active” across reruns
    st.session_state.active_note = st.session_state.get(
        "active_note", 0) % len(notes_view)

    # ── 3) two-column work area ─────────────────────────────────────────
    col_tl, col_tr = st.columns([1, 2])

    ## left → timeline list
    with col_tl:
        sel_idx = st.radio(
            "Timeline",
            options=notes_view.index,
            format_func=lambda i: notes_view.loc[i, "Date"].strftime("%Y-%m-%d"),
            index=st.session_state.active_note)
        st.session_state.active_note = sel_idx

    ## right → editor / preview
    row = notes_view.loc[sel_idx]
    with col_tr:
        new_date = st.date_input("Date", row["Date"].date(), key="note_date")
        new_text = st.text_area("Note text", row["Note"], height=200, key="note_body")

        btn_save, btn_del = st.columns(2)
        with btn_save:
            if st.button("💾 Save"):
                st.session_state.notes_df.loc[row.name, "Date"] = pd.to_datetime(new_date)
                st.session_state.notes_df.loc[row.name, "Note"] = new_text
                st.session_state.notes_df.to_csv(NOTES_FILENAME, index=False)
                st.success("Saved ✔︎")
        with btn_del:
            if st.button("🗑️ Delete", type="primary"):
                st.session_state.notes_df = st.session_state.notes_df.drop(row.name)
                st.session_state.notes_df.to_csv(NOTES_FILENAME, index=False)
                st.success("Deleted")
                (getattr(st, "rerun", getattr(st, "experimental_rerun")))()

    # ── 4) bulk utilities & danger-zone inside fold-aways ───────────────
    with st.expander("⬇️⬆️ Bulk upload / download", expanded=False):
        dl_bytes = st.session_state.notes_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download all notes CSV", dl_bytes,
                           file_name="scout_notes.csv", mime="text/csv")

        up_file = st.file_uploader("Upload CSV (Name, Date, Note)", type="csv")
        if up_file:
            incoming = smart_read_csv(up_file, parse_dates=["Date"])
            req_cols = {"Name", "Date", "Note"}
            if not req_cols.issubset(incoming.columns):
                st.error("CSV must have columns: Name, Date, Note")
            else:
                st.session_state.notes_df = (
                    pd.concat([st.session_state.notes_df, incoming],
                              ignore_index=True)
                    .drop_duplicates(["Name", "Date", "Note"], keep="last")
                )
                st.session_state.notes_df.to_csv(NOTES_FILENAME, index=False)
                st.success("Merged file → notes database")
                (getattr(st, "rerun", getattr(st, "experimental_rerun")))()

    with st.expander("🛑 Danger zone – clear ALL notes", expanded=False):
        if st.button("Delete every note", type="primary"):
            st.session_state.notes_df = pd.DataFrame(columns=["Name", "Date", "Note"])
            st.session_state.notes_df.to_csv(NOTES_FILENAME, index=False)
            st.success("All notes removed")
pass

# ─── Tab 4 : Thresholds ───────────────────────────────────────────────
def show_thresholds_ui():
    st.session_state.activate_tab = "Thresholds"
    try:
        st.header("🔧 Metric Thresholds by Age-Group")
        st.markdown("✅ Tab 4 logic started")

        if "tab4_csv_uploaded" not in st.session_state:
            st.session_state["tab4_csv_uploaded"] = False

        if "thresholds" not in st.session_state:
            st.session_state["thresholds"] = broadcast_metrics_to_ages(metric_thresholds)

        # Ensure thresholds match expected structure
        if not any(k in AGE_LABELS for k in st.session_state["thresholds"].keys()):
            st.session_state["thresholds"] = broadcast_metrics_to_ages(st.session_state["thresholds"])

        thresholds = st.session_state["thresholds"]
        lower_is_better = {"30yd Time", "60yd Time", "5-5-10 Shuttle Time"}

        def order_cuts(metric: str, lo: float, mid: float, hi: float) -> dict:
            if metric in lower_is_better:
                return dict(
                    lbl_lo="Above Avg (fastest/best)",
                    lbl_mid="Avg",
                    lbl_hi="Below Avg (slowest/worst)",
                    lo=hi, mid=mid, hi=lo
                )
            return dict(
                lbl_lo="Below Avg",
                lbl_mid="Avg",
                lbl_hi="Above Avg",
                lo=lo, mid=mid, hi=hi
            )

        edit_mode = st.toggle("Edit mode", value=False)
        st.caption("Switch to **Edit** to change values, then press **Save thresholds**.")

        age_groups = list(thresholds.keys())
        if not age_groups:
            st.warning("⚠️ No age-group data found. Please upload thresholds.")
        else:
            for grp, grp_tab in zip(age_groups, st.tabs(age_groups)):
                with grp_tab:
                    metrics = thresholds[grp]
                    sel_metric = st.selectbox("Metric", sorted(metrics.keys()), key=f"sel_{grp}")
                    cuts = metrics[sel_metric]
                    if isinstance(cuts, dict) and {"below_avg", "avg", "above_avg"} <= cuts.keys():
                        lo, mid, hi = cuts["below_avg"], cuts["avg"], cuts["above_avg"]
                    else:
                        mid = float(cuts) if cuts else 0.0
                        lo, hi = round(mid * 0.9, 2), round(mid * 1.1, 2)

                    cfg = order_cuts(sel_metric, float(lo), float(mid), float(hi))
                    rng_min = min(cfg["lo"], cfg["mid"], cfg["hi"]) * 0.50
                    rng_max = max(cfg["lo"], cfg["mid"], cfg["hi"]) * 1.50

                    col_lbl, col_lo, col_mid, col_hi = st.columns([2, 1, 1, 1])
                    col_lbl.markdown(f"### {sel_metric}")

                    if edit_mode:
                        new_lo = col_lo.number_input(cfg["lbl_lo"], value=float(cfg["lo"]),
                                                     min_value=float(rng_min), max_value=float(cfg["mid"]),
                                                     step=0.01, key=f"{grp}_{sel_metric}_lo")
                        new_mid = col_mid.number_input(cfg["lbl_mid"], value=float(cfg["mid"]),
                                                       min_value=min(new_lo, cfg["mid"]),
                                                       max_value=max(new_lo, cfg["hi"]),
                                                       step=0.01, key=f"{grp}_{sel_metric}_mid")
                        new_hi = col_hi.number_input(cfg["lbl_hi"], value=float(cfg["hi"]),
                                                     min_value=new_mid, max_value=float(rng_max),
                                                     step=0.01, key=f"{grp}_{sel_metric}_hi")

                        # write back
                        if sel_metric in lower_is_better:
                            thresholds[grp][sel_metric] = {
                                "below_avg": new_hi,
                                "avg": new_mid,
                                "above_avg": new_lo
                            }
                        else:
                            thresholds[grp][sel_metric] = {
                                "below_avg": new_lo,
                                "avg": new_mid,
                                "above_avg": new_hi
                            }
                    else:
                        col_lo.metric(cfg["lbl_lo"], f"{cfg['lo']}")
                        col_mid.metric(cfg["lbl_mid"], f"{cfg['mid']}")
                        col_hi.metric(cfg["lbl_hi"], f"{cfg['hi']}")

            if edit_mode:
                col_save, col_dl, col_up = st.columns(3)

                with col_save:
                    if st.button("💾 Save thresholds"):
                        flatten_thresholds(thresholds).to_csv("thresholds.csv", index=False)
                        st.success("Saved → thresholds.csv")

                with col_dl:
                    dl_bytes = flatten_thresholds(thresholds).to_csv(index=False).encode()
                    st.download_button("⬇️ Download CSV", dl_bytes,
                                       file_name="thresholds.csv", mime="text/csv", key="dl_thresh")

                with col_up:
                    uploaded_csv = st.file_uploader("⬆️ Upload CSV", type="csv",
                                                    help="Columns: Age Group, Metric, below_avg, avg, above_avg",
                                                    key="threshold_uploader")

                    if uploaded_csv:
                        try:
                            df_up = pd.read_csv(uploaded_csv)
                            required_cols = {"Age Group", "Metric", "below_avg", "avg", "above_avg"}

                            if not required_cols.issubset(df_up.columns):
                                st.error("CSV missing required columns.")
                            else:
                                st.success("✅ File ready. Click below to apply and update thresholds.")

                                if st.button("🔁 Update Thresholds"):
                                    new = {}
                                    for _, r in df_up.iterrows():
                                        g, m = r["Age Group"], r["Metric"]
                                        new.setdefault(g, {})[m] = {
                                            "below_avg": float(r["below_avg"]),
                                            "avg": float(r["avg"]),
                                            "above_avg": float(r["above_avg"]),
                                        }
                                    st.session_state["thresholds"] = new
                                    st.session_state["active_tab"] = "Report Generation"  # force return to Tab 5
                                    st.success("✅ Thresholds updated. Ready to continue.")

                        except Exception as exc:
                            st.error(f"❌ Failed to read CSV: {exc}")

    except Exception as e:
        st.error("❌ Tab 4 crashed.")
        st.exception(e)
pass






# ─────────────────────────────────────────────────────────────────────────────
# TAB-5  ➜  Reports & Templates
# ─────────────────────────────────────────────────────────────────────────────

def show_report_gen_ui():

    st.session_state.activate_tab = "Report Generation"
    try:
        st.header("📄 Reports & Templates")
        st.markdown("### ✅ Tab 5 is rendering...")

        # Internal tabs
        rep_tab, tmpl_tab = st.tabs(["Generate Report", "Template’s"])

        # =========================================================================
        # A)  GENERATE REPORT
        # =========================================================================
        with rep_tab:
            import pandas as pd
            import datetime, difflib
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
                return (s or "").replace("\u0096", "-").replace("–", "-").replace("—", "-")

            with st.expander("1️⃣  Upload CSVs & Map Names", expanded=True):
                up1, up2, up3, up4 = st.columns(4)
                with up1:
                    fs_file = st.file_uploader("Flightscope CSV", type="csv")
                    throw_file = st.file_uploader("Throwing Velocities CSV", type="csv")
                with up2:
                    blast_file = st.file_uploader("Blast CSV", type="csv")
                    run_file = st.file_uploader("Running Speed CSV", type="csv")
                with up3:
                    mob_file = st.file_uploader("Mobility CSV", type="csv")
                    dyn_file = st.file_uploader("Dynamo CSV", type="csv")
                with up4:
                    force_file = st.file_uploader("ForceDesk CSV", type="csv")    

                flightscope_data = safe_read_csv(fs_file)
                blast_data = safe_read_csv(blast_file)
                throwing_data = safe_read_csv(throw_file)
                running_data = safe_read_csv(run_file)
                mobility_data = safe_read_csv(mob_file)
                dynamo_data = safe_read_csv(dyn_file)
                forcedecks_data = safe_read_csv(force_file)

                for df, col in [
                    (running_data, "AthleteID"),
                    (mobility_data, "Player Name"),
                    (throwing_data, "Player Name")
                ]:
                    if col in df.columns:
                        df[col] = df[col].astype(str).apply(normalize_dashes)

                player_db = st.session_state.player_db.copy()
                canonical = player_db["Name"].tolist()

                def mapper_ui(df, raw_col, label):
                    if df.empty or raw_col not in df.columns:
                        return {}
                    m = {}
                    st.subheader(f"{label} – name mapping")
                    for raw in df[raw_col].dropna().unique():
                        matches = difflib.get_close_matches(raw, canonical, n=3, cutoff=0.6)
                        default = matches[0] if matches else "<leave as is>"
                        choice = st.selectbox(
                            raw,
                            ["<leave as is>"] + canonical,
                            index=(["<leave as is>"] + canonical).index(default),
                            key=f"{label}_{raw}"
                        )
                        m[raw] = raw if choice == "<leave as is>" else choice
                    return m

                run_map = mapper_ui(running_data, "AthleteID", "Running")
                mob_map = mapper_ui(mobility_data, "Player Name", "Mobility")
                throw_map = mapper_ui(throwing_data, "Player Name", "Throwing")

                if not running_data.empty:
                    running_data["AthleteID"] = running_data["AthleteID"].map(lambda x: run_map.get(x, x))
                if not mobility_data.empty:
                    mobility_data["Player Name"] = mobility_data["Player Name"].map(lambda x: mob_map.get(x, x))
                if not throwing_data.empty:
                    throwing_data["Player Name"] = throwing_data["Player Name"].map(lambda x: throw_map.get(x, x))

            # 2️⃣ Select Player & Date
            st.markdown("### 2️⃣  Select Player & Date")
            player_db["nm"] = player_db["Name"].str.lower().str.strip()
            if "Age Group" not in player_db.columns:
                player_db["Age Group"] = player_db["Age"].apply(get_group)

            if player_db.empty:
             st.warning("Add players first (tab **Player Database**). Tab 4 & 5 may not function without it.")
            else:
             


             sel_idx = st.selectbox("Player", player_db.index, format_func=lambda i: player_db.at[i, "Name"])
             prow = player_db.loc[sel_idx]
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

            notes_df = st.session_state.get("notes_df", pd.DataFrame())
            last_note = notes_df.query("Name == @player_info['Name']").sort_values("Date", ascending=False)["Note"].head(1)
            player_info["LatestNoteText"] = last_note.iat[0] if not last_note.empty else ""

            grp = player_info["Age Group"]
            key = normalize_dashes(player_info["Name"]).lower().strip()

            def by_age(df):
                if df is None or df.empty or "Age Group" not in df.columns:
                    return df
                return df[df["Age Group"] == grp]

            def by_name(df):
                if df is None or df.empty or "nm" not in df.columns:
                    return df
                return df[df["nm"] == key]

            for df, raw_cols in [
                (throwing_data, ["Player Name"]),
                (running_data, ["AthleteID", "Player Name"]),
                (mobility_data, ["Player Name"])
            ]:
                if df is not None and not df.empty and "nm" not in df.columns:
                    for c in raw_cols:
                        if c in df.columns:
                            df["nm"] = df[c].str.lower().str.strip()
                            break

            def safe_merge_all(df, name_cols):
                if df is None or df.empty:
                    return df
                tmp = df.copy()
                if "nm" not in tmp.columns:
                    for c in name_cols:
                        if c in tmp.columns:
                            tmp["nm"] = tmp[c].str.lower().str.strip()
                            break
                return tmp.merge(player_db[["nm", "DOB", "Age", "Age Group"]],
                                 on="nm", how="left", suffixes=("", "_x"))

            blast_data = safe_merge_all(blast_data, ["Name"])
            flightscope_data = safe_merge_all(flightscope_data, ["Name", "Player Name", "Batter"])
            throwing_data = safe_merge_all(throwing_data, ["Player Name"])
            running_data = safe_merge_all(running_data, ["Player Name"])
            mobility_data = safe_merge_all(mobility_data, ["Player Name"])
            dynamo_data = safe_merge_all(dynamo_data, ["Name"])

            grp_blast = by_age(blast_data)
            grp_fs = by_age(flightscope_data)
            grp_dyn = by_age(dynamo_data)
            grp_throw = by_name(throwing_data)
            grp_run = by_name(running_data)
            grp_mob = by_name(mobility_data)

            max_ev, p90_ev = calculate_flightscope_metrics(grp_fs) if grp_fs is not None and not grp_fs.empty else (None, None)
            averages, ranges = calculate_blast_metrics(grp_blast) if grp_blast is not None and not grp_blast.empty else ({}, {})
            velocities = calculate_throwing_velocities(grp_throw)
            speeds, speed_ranges = calculate_running_speeds(grp_run)

            mobility_dict = {}
            if grp_mob is not None and not grp_mob.empty:
                r = grp_mob.iloc[0]
                mobility_dict = {
                    "Ankle": r.get("Ankle Mobility"),
                    "Thoracic": r.get("Thoracic Mobility"),
                    "Lumbar": r.get("Lumbar Mobility")
                }

            if st.checkbox("Show debug preview"):
                for lbl, df in [("Blast", grp_blast), ("Flightscope", grp_fs),
                                ("Throwing", grp_throw), ("Running", grp_run),
                                ("Mobility", grp_mob), ("Dynamo", grp_dyn)]:
                    st.markdown(f"**{lbl}** *(first 3 rows)*")
                    st.dataframe(df.head(3))

            st.write("ForceDecks data preview:")
            st.dataframe(forcedecks_data.head())

            if forcedecks_data is not None and not forcedecks_data.empty and "Name" in forcedecks_data.columns:
              st.write("Player name:", player_info["Name"]) 
              st.write("Matches in ForceDecks CSV:",
               forcedecks_data[forcedecks_data["Name"].str.lower() == player_info["Name"].lower()])
            else:
             st.warning("⚠️ No ForceDecks CSV uploaded yet or 'Name' column missing.")

            st.markdown("---")
            if st.button("Generate Combined PDF", use_container_width=True):
                with st.spinner("Building PDF…"):
                    pdf_buf = create_combined_pdf(
                        max_ev=max_ev,
                        percentile_90_ev=p90_ev,
                        averages=averages,
                        ranges=ranges,
                        velocities=velocities,
                        speeds=speeds,
                        speed_ranges=speed_ranges,
                        player_info=player_info,
                        flightscope_data=grp_fs,
                        mobility=mobility_dict,
                        dynamo_data=grp_dyn,
                        forcedecks_data=safe_read_csv(force_file)
                    )
                st.success("PDF ready!")
                st.download_button("⬇️ Download",
                                   data=pdf_buf,
                                   file_name=f"{player_info['Name'].replace(' ','')}.pdf",
                                   mime="application/pdf")

        # =========================================================================
        # B)  BLANK CSV TEMPLATES
        # =========================================================================
        with tmpl_tab:
            st.subheader("📥 Blank CSV templates")
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

    except Exception as e:
        st.error("❌ Tab 5 crashed.")
        st.exception(e)

if page == "CSV Merge":
    show_csv_merge_ui()
elif page == "Player Database":
    show_player_db_ui()
elif page == "Scout Notes":
    show_scout_notes_ui()
elif page == "Thresholds":
    show_thresholds_ui()
else:   # "Report Generation"
    show_report_gen_ui()









