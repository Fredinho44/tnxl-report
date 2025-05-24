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


# thresholds for coloring bars:
# ──────────────────────────────────────────────────────────────────────────────
# 1) Thresholds + get_bar_color
# ──────────────────────────────────────────────────────────────────────────────
# ——— 1) Define age-groups and your base thresholds ———


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

def get_group(age: int) -> str:
    """Return the age‐group label for a given age."""
    for grp, fn in age_groups.items():
        if fn(age):
            return grp
    return "unknown"

from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
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




def draw_sz(sz_top=3.5, sz_bot=1.5, ls='k-'):
    plt.plot([-0.708,  0.708], [sz_bot, sz_bot], ls)
    plt.plot([-0.708, -0.708], [sz_bot, sz_top], ls)
    plt.plot([ 0.708,  0.708], [sz_bot, sz_top], ls)
    plt.plot([-0.708,  0.708], [sz_top, sz_top], ls)

def draw_home_plate(catcher_perspective=True, ls='k-'):
    if catcher_perspective:
        plt.plot([-0.708, 0.708], [0, 0], ls)
        plt.plot([-0.708,-0.708], [0, -0.3], ls)
        plt.plot([ 0.708, 0.708], [0, -0.3], ls)
        plt.plot([-0.708, 0.   ], [-0.3, -0.6], ls)
        plt.plot([ 0.708, 0.   ], [-0.3, -0.6], ls)
    else:
        plt.plot([-0.708, 0.708], [0, 0], ls)
        plt.plot([-0.708,-0.708], [0, 0.1], ls)
        plt.plot([ 0.708, 0.708], [0, 0.1], ls)
        plt.plot([-0.708, 0.   ], [0.1, 0.3], ls)
        plt.plot([ 0.708, 0.   ], [0.1, 0.3], ls)


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
    canvas.setFillColor(colors.HexColor("#D4AF37"))
    path = canvas.beginPath()
    path.moveTo(w, h)                 # top‐right
    path.lineTo(w, h - header_h)      # bottom‐right of header
    path.lineTo(x0, h)                # back up to top at x0
    path.close()
    canvas.drawPath(path, fill=1, stroke=0)

    canvas.restoreState()






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

class DiagonalBackground(Flowable):
    def __init__(self, width, height, color1=colors.black, color2=colors.HexColor("#D4AF37")):
        super().__init__()
        self.width = width
        self.height = height
        self.color1 = color1
        self.color2 = color2

    def draw(self):
        c = self.canv

        # draw the lower-left triangle
        c.setFillColor(self.color1)
        p = c.beginPath()
        p.moveTo(0, 0)
        p.lineTo(self.width, 0)
        p.lineTo(0, self.height)
        p.close()
        c.drawPath(p, fill=1, stroke=0)

        # draw the upper-right triangle
        c.setFillColor(self.color2)
        p = c.beginPath()
        p.moveTo(self.width, 0)
        p.lineTo(self.width, self.height)
        p.lineTo(0, self.height)
        p.close()
        c.drawPath(p, fill=1, stroke=0)







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

def find_t_for_poly_crossing(poly_str, crossing=17.0/12.0):
    """
    Find the time `t` at which a polynomial crosses a specific Y-value.
    """
    try:
        coeffs = [float(x.strip()) for x in poly_str.split(";")]
        if len(coeffs) < 5:
            return None
        
        def min_fun(t): 
            return (sum(coeffs[i] * t**i for i in range(5)) - crossing)**2

        result = scipy.optimize.minimize(min_fun, [0])
        return result.x[0] if result.success else None
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



# ================================
# PDF Report Building
# ================================
def safe_float(val):
    try:
        return float(val)
    except:
        return None




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
    player_info,
    flightscope_data,
    mobility=None,
    dynamo_data=None
):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A3),
        rightMargin=30, leftMargin=30,
        topMargin=10, bottomMargin=30
    )
    elements = []

    # Header
    header = build_header_with_logo_and_player_info(
        r"C:\Users\AlfredoCaraballo\OneDrive - DSSports\Desktop\TNXL Miami\TNXL MIAMI\TNXL Miami - Updated Logo.png",
        player_info,
        doc.width
    )
    elements.append(header)
    elements.append(Spacer(1, 12))

    # Prepare the Exit Velocity heatmap image
    heatmap_img = None
    if flightscope_data is not None:
        # 1) Evaluate polynomials at t = 0 (contact point), in feet
        flightscope_data["Parsed_X"] = flightscope_data["Hit_Poly_X"].apply(
            lambda p: safe_est_poly_at_t(0, p) if isinstance(p, str) else None
        )
        flightscope_data["Parsed_Z"] = flightscope_data["Hit_Poly_Z"].apply(
            lambda p: safe_est_poly_at_t(0, p) if isinstance(p, str) else None
        )

        # 2) Coerce and filter to only true batted‐ball events
        flightscope_data["Exit_Speed"] = pd.to_numeric(
            flightscope_data["Exit_Speed"], errors="coerce"
        )
        valid = (
            flightscope_data
            .dropna(subset=["Parsed_X", "Parsed_Z", "Exit_Speed"])
            .query("Exit_Speed > 0")
            .copy()
        )

        # 3) Convert to inches and flip for catcher’s view
        valid["PlateLocSide"]   = -valid["Parsed_X"] * 12.0
        valid["PlateLocHeight"] = valid["Parsed_Z"] * 12.0

        # 4) Generate the flowable heatmap
        heatmap_img = generate_exit_velo_heatmap(valid)

    # Layout: two columns, left = gameplay data, right = heatmap
    default_left_w  = doc.width * 0.61
    default_right_w = doc.width - default_left_w

    gameplay_frame = KeepInFrame(
        default_left_w, doc.height,
        [
            Paragraph("Gameplay Data", styles["Heading3"]),
            Spacer(1, 6),
            build_gameplay_data_table(
                averages,
                ranges,
                max_ev,
                percentile_90_ev,
                velocities,
                default_left_w,
                thresholds=st.session_state["thresholds"],
                age_group=player_info["Age Group"]
            )
        ],
        hAlign="LEFT", mergeSpace=True
    )

    right_contents = [
        Paragraph("AVG Exit Velocity by Zone", styles["Heading3"]),
        Spacer(1,6)
    ]
    if heatmap_img:
        right_contents.append(heatmap_img)
    else:
        right_contents.append(Paragraph("No heatmap data", styles["Normal"]))    
    right_frame = KeepInFrame(
        default_right_w, doc.height,
        right_contents,
        hAlign="LEFT",mergeSpace=True
    )

    elements.append(
        Table(
            [[gameplay_frame, right_frame]],
            colWidths=[default_left_w, default_right_w]
        )
    )
    elements.append(Spacer(1, 12))

    # Row 2: Physical Profile & Scout Notes
    notes_txt = player_info.get("LatestNoteText") or "No scout notes available."
    left_w2 = default_left_w
    notes_w = doc.width - left_w2

    physical_frame = KeepInFrame(
        left_w2, doc.height,
        [
            Paragraph("Physical Profile", styles["Heading3"]),
            Spacer(1,6),
            build_profile_table(
                mobility, speeds,speed_ranges, left_w2,
                thresholds=st.session_state.get("thresholds"),
                age_group=player_info.get("Age Group")
            )
        ],
        hAlign="LEFT", mergeSpace=True
    )

    notes_tbl = Table(
        [
            [Paragraph("Scout Notes", styles["Heading3"])],
            [Paragraph(notes_txt, styles["Normal"])]
        ],
        colWidths=[notes_w]
    )
    notes_tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#D4AF37')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('ALIGN',(0,0),(-1,0),'CENTER'),
        ('BACKGROUND',(0,1),(-1,1),colors.whitesmoke),
        ('VALIGN',(0,1),(-1,1),'TOP'),
        ('BOX',(0,0),(-1,-1),0.5,colors.grey),
        ('INNERGRID',(0,0),(-1,-1),0.5,colors.lightgrey),
        ('LEFTPADDING',(0,0),(-1,-1),6),
        ('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),4),
        ('BOTTOMPADDING',(0,0),(-1,-1),4),
    ]))

    elements.extend([Table([[physical_frame, notes_tbl]], colWidths=[left_w2, notes_w]),
                     Spacer(1,12)])

    # Row 3: Dynamo & blank
    dynamo_frame = KeepInFrame(
        default_left_w, doc.height,
        [
            Paragraph("Dynamo Summary", styles["Heading3"]),
            Spacer(1,6),
            build_dynamo_table(dynamo_data, player_info, default_left_w)
        ],
        hAlign="LEFT", mergeSpace=True
    )
    elements.append(
        Table(
            [[dynamo_frame, '']],
            colWidths=[default_left_w, default_right_w]
        )
    )

    # Build PDF
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



# =======================
# Streamlit App Tabs
# =======================
from datetime import date
import traceback
st.title("TNXL MIAMI - Athlete Performance Data Uploader, Report Generator & CSV Utilities")

tab1, tab2, tab3, tab4,tab5 = st.tabs([
    "Blast CSV Merge",
    "Player Database",
    "Report Generation",
    "Scout Notes", 
    "Thresholds"
])
with tab1:
    st.header("CSV Merger")

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


import pandas as pd
from datetime import date

# … earlier you have st.session_state.player_db loaded …

from datetime import date
import pandas as pd

import datetime
import pandas as pd

# Tab 2: Player Database
import datetime
import pandas as pd

import datetime
import pandas as pd

# ─── Tab 2: Player Database ────────────────────────────────────────────────
with tab2:
    st.header("Player Database")
    st.info("Add, edit or delete players. The database is saved persistently.")

    # ─── ADD PLAYER FORM ────────────────────────────────────────────────────
    with st.form("add_player_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            add_name   = st.text_input("Name", key="add_name")
            add_dob    = st.date_input(
                "Date of Birth",
                min_value=datetime.date(1900,1,1),
                max_value=datetime.date.today(),
                key="add_dob"
            )
        with c2:
            add_class  = st.text_input("Class", key="add_class")
            add_hs     = st.text_input("High School", key="add_hs")
        with c3:
            h_ft = st.number_input("Height (ft)",0,8,key="add_height_ft")
            h_in = st.number_input("Height (in)", 0,11, key="add_height_in")
            add_height = h_ft*12 + h_in
            add_weight = st.number_input("Weight (lbs)", 0,500, key="add_weight")
        with c4:
            pos_opts     = ["Pitcher","Catcher","1B","2B","3B","SS","LF","CF","RF","DH"]
            add_position = st.selectbox("Position", pos_opts, key="add_position")
            bat_opts     = ["Left","Right","Switch"]
            add_bat      = st.selectbox("Batting Handedness", bat_opts, key="add_bat")
            throw_opts   = ["Left","Right"]
            add_throw    = st.selectbox("Throwing Handedness", throw_opts, key="add_throw")

        add_submitted = st.form_submit_button("Add Player")

    if add_submitted:
        today     = datetime.date.today()
        age       = (today.year - add_dob.year
                     - ((today.month,today.day) < (add_dob.month,add_dob.day)))
        age_group = get_group(age)
        new = {
            "Name":              add_name,
            "DOB":               add_dob.strftime("%m/%d/%Y"),
            "Age":               int(age),
            "Age Group":         age_group,
            "Class":             add_class,
            "High School":       add_hs,
            "Height":            add_height,
            "Weight":            add_weight,
            "Position":          add_position,
            "BattingHandedness": add_bat,
            "ThrowingHandedness":add_throw,
        }
        st.session_state.player_db = pd.concat(
            [st.session_state.player_db, pd.DataFrame([new])],
            ignore_index=True
        )
        st.session_state.player_db.to_csv(DATABASE_FILENAME, index=False)
        st.success(f"Player '{add_name}' added.")

    # ─── SHOW DATABASE ───────────────────────────────────────────────────────
    st.write("### Current Player Database")
    df = st.session_state.player_db.copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").astype("Int64")
    st.dataframe(df)

    # ─── EDIT / DELETE FORM ─────────────────────────────────────────────────
    if not st.session_state.player_db.empty:
        idx = st.selectbox(
            "Select player to edit/delete",
            st.session_state.player_db.index,
            format_func=lambda i: st.session_state.player_db.loc[i, "Name"],
            key="edit_select"
        )
        sel = st.session_state.player_db.loc[idx]

        with st.form("edit_player_form", clear_on_submit=True):
            parsed   = pd.to_datetime(sel["DOB"], errors="coerce")
            init_dob = parsed.date() if pd.notna(parsed) else datetime.date.today()

            e_name   = st.text_input("Name",          value=sel["Name"], key="edit_name")
            e_dob    = st.date_input("Date of Birth", value=init_dob,    key="edit_dob")

            today    = datetime.date.today()
            e_age    = (today.year - e_dob.year
                        - ((today.month,today.day) < (e_dob.month,e_dob.day)))
            e_group  = get_group(e_age)

            e_class  = st.text_input("Class",           sel["Class"], key="edit_class")
            e_hs     = st.text_input("High School",     sel["High School"], key="edit_hs")

            # sanitize any junk in sel["Height"] / sel["Weight"]
            raw_h   = int(sel.get("Height", 0) or 0)
            init_ft,init_in = divmod(raw_h,12)
            try: default_h = int(raw_h)
            except: default_h = 0
            raw_w   = sel.get("Weight", 0)
            try: default_w = int(raw_w)
            except: default_w = 0
             
            ft = st.number_input("Height(ft)",0,8, value=init_ft,key="edit_height_ft")
            inch = st.number_input("Height (in)",0,11, value=init_in, key="edit_height_in")
            e_height = ft*12 + inch
            e_weight = st.number_input(
                "Weight (lbs)", min_value=0, max_value=5000,
                value=default_w, key="edit_weight"
            )

            pos_opts   = ["Pitcher","Catcher","1B","2B","3B","SS","LF","CF","RF","DH"]
            curr_pos   = sel["Position"] if sel["Position"] in pos_opts else pos_opts[0]
            e_position = st.selectbox("Position", pos_opts,
                                      index=pos_opts.index(curr_pos),
                                      key="edit_position")

            bat_opts  = ["Left","Right","Switch"]
            b0        = sel["BattingHandedness"] if sel["BattingHandedness"] in bat_opts else bat_opts[0]
            e_bat     = st.selectbox("Batting Handedness", bat_opts,
                                      index=bat_opts.index(b0),
                                      key="edit_bat")

            throw_opts = ["Left","Right"]
            t0         = sel["ThrowingHandedness"] if sel["ThrowingHandedness"] in throw_opts else throw_opts[0]
            e_throw    = st.selectbox("Throwing Handedness", throw_opts,
                                       index=throw_opts.index(t0),
                                       key="edit_throw")

            update_submitted = st.form_submit_button("Update Player")
            delete_submitted = st.form_submit_button("Delete Player")

        if update_submitted:
            updates = {
                "Name":               e_name,
                "DOB":                e_dob.strftime("%m/%d/%Y"),
                "Age":                int(e_age),
                "Age Group":          e_group,
                "Class":              e_class,
                "High School":        e_hs,
                "Height":             e_height,
                "Weight":             e_weight,
                "Position":           e_position,
                "BattingHandedness":  e_bat,
                "ThrowingHandedness": e_throw,
            }
            for col, val in updates.items():
                st.session_state.player_db.at[idx, col] = val

            st.session_state.player_db.to_csv(DATABASE_FILENAME, index=False)
            st.success("Player updated successfully!")

        if delete_submitted:
            st.session_state.player_db = (
                st.session_state.player_db.drop(idx).reset_index(drop=True)
            )
            st.session_state.player_db.to_csv(DATABASE_FILENAME, index=False)
            st.success("Player deleted successfully!")

    # ─── CLEAR ENTIRE DATABASE BUTTON ───────────────────────────────────────
    if st.button("Clear Player Database"):
        st.session_state.player_db = pd.DataFrame(columns=expected_columns)
        if os.path.exists(DATABASE_FILENAME):
            os.remove(DATABASE_FILENAME)
        st.success("All players have been deleted.")




# ─── 2. Full Tab3 Script ───────────────────────────────────────────────────────

with tab3:
    st.header("Report Generation")
    import datetime
    import difflib
    from pandas.errors import EmptyDataError

    # ─── 1) Safe CSV reader (reset file pointer on each try) ──────────────
    def safe_read_csv(f):
        import pandas as pd
        from pandas.errors import EmptyDataError

        if f is None:
            return pd.DataFrame()

        for enc in (None, "cp1252", "latin-1"):
            try:
                f.seek(0)
                return pd.read_csv(f, encoding=enc) if enc else pd.read_csv(f)
            except (UnicodeDecodeError, EmptyDataError):
                continue
        return pd.DataFrame()

    # ─── 2) Upload your CSVs ─────────────────────────────────────────────
    st.sidebar.header("Upload Your Data Files")
    fs_file    = st.sidebar.file_uploader("Flightscope CSV",         type="csv", key="fs_csv")
    blast_file = st.sidebar.file_uploader("Blast CSV",               type="csv", key="blast_csv")
    throw_file = st.sidebar.file_uploader("Throwing Velocities CSV", type="csv", key="throw_csv")
    run_file   = st.sidebar.file_uploader("Running Speed CSV",       type="csv", key="run_csv")
    mob_file   = st.sidebar.file_uploader("Mobility CSV",            type="csv", key="mobility_csv")
    dyn_file   = st.sidebar.file_uploader("Dynamo CSV",              type="csv", key="dynamo_csv")

    # ─── 3) Read in (None out empty) ─────────────────────────────────────
    flightscope_data = safe_read_csv(fs_file)    if fs_file    else None
    blast_data       = safe_read_csv(blast_file) if blast_file else None
    throwing_data    = safe_read_csv(throw_file) if throw_file else None
    running_data     = safe_read_csv(run_file)   if run_file   else None
    mobility_data    = safe_read_csv(mob_file)   if mob_file   else None
    dynamo_data      = safe_read_csv(dyn_file)   if dyn_file   else None

    # ─── 3b) Normalize en/em-dashes in name columns ──────────────────────
    def normalize_dashes(s):
        return (s or "").replace("\u0096", "-").replace("–", "-").replace("—", "-")

    for df, col in [
        (running_data,  "AthleteID"),
        (mobility_data, "Player Name"),
        (throwing_data, "Player Name"),
    ]:
        if df is not None and col in df.columns:
            df[col] = df[col].astype(str).apply(normalize_dashes)

    # ─── 3c) Fuzzy‐map raw names → canonical DB names in sidebar ─────────
    player_db  = st.session_state.player_db.copy()
    canonical  = player_db["Name"].tolist()

    def build_name_mapper(df, raw_col, label):
        """Return a dict mapping each unique raw name → chosen canonical name."""
        if df is None or raw_col not in df.columns:
            return {}
        mapper = {}
        st.sidebar.subheader(f"{label} Name Mapping")
        for raw in df[raw_col].dropna().unique():
            # show up to 3 close matches, cutoff 0.6
            matches = difflib.get_close_matches(raw, canonical, n=3, cutoff=0.6)
            default = matches[0] if matches else "<none>"
            choice = st.sidebar.selectbox(
                f"Map “{raw}” to:", ["<none>"] + canonical,
                index=(["<none>"] + canonical).index(default),
                key=f"map_{label}_{raw}"
            )
            mapper[raw] = choice if choice != "<none>" else raw
        return mapper

    run_map   = build_name_mapper(running_data,  "AthleteID",     "Running")
    mob_map   = build_name_mapper(mobility_data, "Player Name",   "Mobility")
    throw_map = build_name_mapper(throwing_data, "Player Name",   "Throwing")

    # ─── 3d) Apply mappings □──────────────────────────────────────────────
    if running_data is not None:
        running_data["AthleteID"] = running_data["AthleteID"].map(lambda x: run_map.get(x, x))
    if mobility_data is not None:
        mobility_data["Player Name"] = mobility_data["Player Name"].map(lambda x: mob_map.get(x, x))
    if throwing_data is not None:
        throwing_data["Player Name"] = throwing_data["Player Name"].map(lambda x: throw_map.get(x, x))

    # ─── 4) Rename AthleteID → Player Name if needed ────────────────────
    for df in (running_data, mobility_data, throwing_data):
        if df is not None and "AthleteID" in df.columns:
            df.rename(columns={"AthleteID": "Player Name"}, inplace=True)

    # ─── 5) Merge in DOB/Age/Age Group via first matching name column ─────
    player_db["nm"] = player_db["Name"].str.lower().str.strip()
    def safe_merge_all(df, name_cols):
        if df is None:
            return None
        tmp = df.copy()
        for c in name_cols:
            if c in tmp.columns:
                tmp["nm"] = tmp[c].str.lower().str.strip()
                break
        else:
            tmp["nm"] = None
        return tmp.merge(player_db[["nm","DOB","Age","Age Group"]],
                        on="nm", how="left")

    blast_data       = safe_merge_all(blast_data,       ["Name"])
    flightscope_data = safe_merge_all(flightscope_data, ["Name","Player Name","Batter"])
    throwing_data    = safe_merge_all(throwing_data,    ["Name","Player Name"])
    running_data     = safe_merge_all(running_data,     ["Name","Player Name"])
    mobility_data    = safe_merge_all(mobility_data,    ["Name","Batter","Player Name"])
    dynamo_data      = safe_merge_all(dynamo_data,      ["Name"])

    # ─── 6) Pick player & date ────────────────────────────────────────────
    assess_date = st.date_input("Assessment Date",
                                value=datetime.date.today(),
                                key="report_assess_date")
    idx = st.selectbox("Select Player",
                       player_db.index,
                       format_func=lambda i: player_db.loc[i,"Name"],
                       key="report_select_player")
    row = player_db.loc[idx]
    bat = row.get("BattingHandedness") or ""
    thr = row.get("ThrowingHandedness") or ""
    player_info = {
        "Name":          row["Name"],
        "Age":           int(row["Age"]),
        "Age Group":     row["Age Group"],
        "Position":      row["Position"],
        "Class":         row["Class"],
        "High School":   row["High School"],
        "Height":        row["Height"],
        "Weight":        row["Weight"],
        "B/T":           f"{bat}/{thr}".rstrip("/"),
        "DOB":           row["DOB"],
        "AssessmentDate": assess_date.strftime("%m/%d/%Y"),
    }
    notes_df     = st.session_state.get("notes_df", pd.DataFrame())
    player_notes = notes_df[notes_df["Name"] == player_info["Name"]]
    if not player_notes.empty:
         latest_note = (
         player_notes
         .sort_values("Date", ascending=False)
         .iloc[0]["Note"]
        )
         player_info["LatestNoteText"] = latest_note
    else:
        player_info["LatestNoteText"] = ""
    # ─── 7) Two slice helpers ─────────────────────────────────────────────
    def slice_by_age(df):
        if df is None: return None
        return df[df["Age Group"] == player_info["Age Group"]]

    def slice_by_name(df):
        if df is None: return None
        key = normalize_dashes(player_info["Name"]).lower().strip()
        return df[df["nm"] == key]

    # ─── 8) Slice each dataset appropriately ─────────────────────────────
    grp_blast = slice_by_age(blast_data)
    grp_fs    = slice_by_age(flightscope_data)
    grp_dyn   = slice_by_age(dynamo_data)

    grp_throw = slice_by_name(throwing_data)
    grp_run   = slice_by_name(running_data)
    grp_mob   = slice_by_name(mobility_data)

    # ─── 9) Compute your metrics ──────────────────────────────────────────
    max_ev, percentile_90_ev = (
        calculate_flightscope_metrics(grp_fs)
        if grp_fs is not None and not grp_fs.empty else (None, None)
    )
    averages, ranges = (
        calculate_blast_metrics(grp_blast)
        if grp_blast is not None and not grp_blast.empty else ({}, {})
    )
    velocities = (
        calculate_throwing_velocities(grp_throw)
        if grp_throw is not None and not grp_throw.empty else {}
    )
    if grp_run is not None and not grp_run.empty:
       speeds, speed_ranges = calculate_running_speeds(grp_run)
    else:
     speeds, speed_ranges = {}, {}
    
    mobility_dict = {}
    if grp_mob is not None and not grp_mob.empty:
        r = grp_mob.iloc[0]
        mobility_dict = {
            "Ankle":    r.get("Ankle Mobility"),
            "Thoracic": r.get("Thoracic Mobility"),
            "Lumbar":   r.get("Lumbar Mobility"),
        }

    # ─── 🔟 Debug: show each group’s columns & sample ─────────────────────
    for name, df in [
        ("grp_blast", grp_blast),
        ("grp_fs",    grp_fs),
        ("grp_throw", grp_throw),
        ("grp_run",   grp_run),
        ("grp_mob",   grp_mob),
        ("grp_dyn",   grp_dyn),
    ]:
        if df is not None:
            st.write(f"▶️ {name} columns:", df.columns.tolist())
            st.write(f"▶️ {name} sample:", df.head())
        else:
            st.write(f"▶️ {name} is None")

    st.write("▶️ throwing velocities dict:", velocities)
    st.write("▶️ running speeds dict:",      speeds)
    st.write("▶️ mobility dict:",            mobility_dict)

    # ─── ⓫ Generate PDF ───────────────────────────────────────────────────
    if st.button("Generate Combined PDF Report"):
        with st.spinner("Generating PDF…"):
            pdf_buffer = create_combined_pdf(
                max_ev=max_ev,
                percentile_90_ev=percentile_90_ev,
                averages=averages,
                ranges=ranges,
                velocities=velocities,
                speeds=speeds,
                player_info=player_info,
                flightscope_data=grp_fs,
                mobility=mobility_dict,
                dynamo_data=grp_dyn
            )
        st.success("✅ PDF generated successfully!")
        out_name = player_info["Name"].replace(" ", "") + ".pdf"
        st.download_button(
            "Download PDF",
            data=pdf_buffer,
            file_name=out_name,
            mime="application/pdf"
        )



#Tab4 - Scouts notes Section
# In your Streamlit app, after tab3:

with tab4:
    st.header("Scout Notes")
    st.info("Add, preview, select and delete notes by player.")

    # 1) Make sure there's a DataFrame in session_state
    if "notes_df" not in st.session_state:
        if os.path.exists(NOTES_FILENAME):
            st.session_state.notes_df = pd.read_csv(NOTES_FILENAME, parse_dates=["Date"])
        else:
            st.session_state.notes_df = pd.DataFrame(columns=["Name","Date","Note"])

    # 2) Clear All Notes button
    if st.button("Clear All Notes"):
        # reset session state
        st.session_state.notes_df = pd.DataFrame(columns=["Name","Date","Note"])
        # overwrite the on‐disk file with empty CSV
        st.session_state.notes_df.to_csv(NOTES_FILENAME, index=False)
        st.success("All notes have been deleted persistently.")

    # 2) Pick a player
    player = st.selectbox(
        "Select Player",
        st.session_state.player_db["Name"].tolist(),
        key="notes_player"
    )

    # 3) Show existing notes for that player
    player_notes = st.session_state.notes_df[
        st.session_state.notes_df["Name"] == player
    ].sort_values("Date", ascending=False)

    st.subheader("Existing Notes")
    if player_notes.empty:
        st.write("No notes yet for this player.")
    else:
        # Let the user pick which note to use in the report
        idx = st.radio(
            "Select note to include in PDF:",
            options=player_notes.index.tolist(),
            format_func=lambda i: player_notes.loc[i, "Date"].strftime("%Y-%m-%d")
        )
        st.markdown(f"**Preview ({player_notes.loc[idx,'Date'].strftime('%Y-%m-%d')}):**")
        st.write(player_notes.loc[idx, "Note"])

        # Offer a delete button for that note
        if st.button("Delete this note", key="delete_note"):
            st.session_state.notes_df = (
                st.session_state.notes_df.drop(idx).reset_index(drop=True)
            )
            st.success("Note deleted.")
            \

    st.markdown("---")

    # 4) Add a new note
    st.subheader("Add a New Note")
    note_date = st.date_input(
        "Note Date",
        value=datetime.date.today(),
        key="new_note_date"
    )
    note_text = st.text_area("Note Text", key="new_note_text")
    if st.button("Save Note", key="save_note"):
        new = {
            "Name": player,
            "Date": pd.to_datetime(note_date),
            "Note": note_text,
        }
        st.session_state.notes_df = pd.concat([
            st.session_state.notes_df,
            pd.DataFrame([new])
        ], ignore_index=True)
        st.success("Note saved.")
        st.rerun()

#tab 5 - Thresholds
with tab5:
    st.header("🔧 Edit Metric Thresholds by Age-Group")

    # — 0) Base age-groups and initial metric_thresholds come from up top
    age_groups = {
        "youth (12–13)":   lambda age: 12 <= age <= 13,
        "jv (14–15)":      lambda age: 14 <= age <= 15,
        "varsity (16–18)": lambda age: 16 <= age <= 18,
        "college (18+)":   lambda age: age >= 18,
    }
    base_thresholds = metric_thresholds  # your dict from the top of the file

    # — 1) Upload an edited thresholds CSV if you have one —
    uploaded = st.file_uploader(
        "Upload Thresholds CSV",
        type="csv",
        help="Choose a CSV you previously downloaded, edit it, and re-upload to set new thresholds."
    )
    if uploaded:
        df_thresh = pd.read_csv(uploaded)
        st.success("🔄 Loaded thresholds from CSV. You can tweak them below and click Save.")
    else:
        # — 2) Otherwise build the flat DataFrame from the in-memory dict —
        records = []
        for grp in age_groups:
            for metric, cuts in base_thresholds.items():
                records.append({
                    "Age Group": grp,
                    "Metric":     metric,
                    "above_avg":  cuts["above_avg"],
                    "avg":        cuts["avg"],
                    "below_avg":  cuts["below_avg"],
                })
        df_thresh = pd.DataFrame(records)

    # — 3) Show the editor and let them tweak (or accept the uploaded) —
    edited = st.data_editor(
        df_thresh,
        num_rows="dynamic",
        use_container_width=True,
        key="threshold_editor"
    )

    # — 4) Offer a download of whatever they see here —
    csv_bytes = edited.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Thresholds CSV",
        data=csv_bytes,
        file_name="thresholds.csv",
        mime="text/csv",
        help="Get a CSV you can edit offline and re-upload above."
    )

    # — 5) Rebuild your nested thresholds dict from the edited DataFrame —
    new_thresholds = {}
    for _, row in edited.iterrows():
        grp    = row["Age Group"]
        metric = row["Metric"]
        new_thresholds.setdefault(grp, {})[metric] = {
            "above_avg": float(row["above_avg"]),
            "avg":       float(row["avg"]),
            "below_avg": float(row["below_avg"]),
        }

    # — 6) Store for Tab 3 to consume —
    st.session_state["thresholds"] = new_thresholds

    st.success("✅ Thresholds updated!")



# ================================
# Sidebar: Download CSV Templates
# ================================

st.sidebar.header("Download Data Templates")

def create_template(file_name, columns):
    template_df = pd.DataFrame(columns=columns)
    csv_data = template_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label=f"Download {file_name} Template",
        data=csv_data,
        file_name=file_name,
        mime='text/csv'
    )

# ➡️ Templates for Running Speed, Core Strength, Throwing Velocities, Mobility
create_template("running_speed_template.csv", ["Player Name", "30yd Time", "60yd Time", "5-5-10 Shuttle Time"])
create_template("core_strength_template.csv", ["Player Name", "Core Strength Measurement"])
create_template("throwing_velocities_template.csv", ["Player Name", "Positional Throw Velocity", "Pulldown Velocity", "FB Velocity", "SL Velocity", "CB Velocity", "CH Velocity"])
create_template("mobility_template.csv", ["Player Name", "Ankle Mobility", "Thoracic Mobility", "Lumbar Mobility"])
