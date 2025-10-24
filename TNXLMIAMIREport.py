import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import re
import string
import scipy.optimize
import io
from io import BytesIO
from reportlab.pdfgen import canvas


st.title(" TNXL MIAMI - Athlete Performance Data Uploader, Report Generator & CSV Utilities")



# ================================
# Helper Functions for Flightscope CSV Conversion & Blast Merge
# (Flightscope conversion functions, Blast CSV merge functions, etc.)
# ================================

# --- Flightscope CSV Conversion Functions ---

YT_HEADER_STRING = (
    "PitchNo,Date,Time,PAofInning,PitchofPA,Pitcher,PitcherId,PitcherThrows,PitcherTeam,Batter,"
    "BatterId,BatterSide,BatterTeam,PitcherSet,Inning,Top/Bottom,Outs,Balls,Strikes,TaggedPitchType,"
    "AutoPitchType,PitchCall,KorBB,HitType,PlayResult,OutsOnPlay,RunsScored,Notes,RelSpeed,"
    "VertRelAngle,HorzRelAngle,SpinRate,SpinAxis,Tilt,RelHeight,RelSide,Extension,VertBreak,"
    "InducedVertBreak,HorzBreak,PlateLocHeight,PlateLocSide,ZoneSpeed,VertApprAngle,"
    "HorzApprAngle,ZoneTime,ExitSpeed,Angle,Direction,HitSpinRate,PositionAt110X,"
    "PositionAt110Y,PositionAt110Z,Distance,LastTrackedDistance,Bearing,HangTime,"
    "pfxx,pfxz,x0,y0,z0,vx0,vy0,vz0,ax0,ay0,az0,HomeTeam,AwayTeam,Stadium,Level,"
    "League,GameID,PitchUUID,yt_RelSpeed,yt_RelHeight,yt_RelSide,yt_VertRelAngle,"
    "yt_HorzRelAngle,yt_ZoneSpeed,yt_PlateLocHeight,yt_PlateLocSide,yt_VertApprAngle,"
    "yt_HorzApprAngle,yt_ZoneTime,yt_HorzBreak,yt_InducedVertBreak,yt_OutOfPlane,"
    "yt_FSRI,yt_EffectiveSpin,yt_GyroSpin,yt_Efficiency,yt_SpinComponentX,yt_SpinComponentY,"
    "yt_SpinComponentZ,yt_HitVelocityX,yt_HitVelocityY,yt_HitVelocityZ,yt_HitLocationX,"
    "yt_HitLocationY,yt_HitLocationZ,yt_GroundLocationX,yt_GroundLocationY,yt_HitBreakX,"
    "yt_HitBreakY,yt_HitBreakT,yt_HitSpinComponentX,yt_HitSpinComponentY,yt_HitSpinComponentZ,"
    "yt_SessionName,Note,yt_PitchSpinConfidence,yt_PitchReleaseConfidence,yt_HitSpinConfidence,"
    "yt_EffectiveBattingSpeed,yt_ReleaseAccuracy,yt_ZoneAccuracy,yt_SeamLat,yt_SeamLong,"
    "yt_ReleaseDistance,Catcher,CatcherId,CatcherTeam"
)
def yt_header_list():
    return YT_HEADER_STRING.split(",")

FS_TO_YT_MAPPING = {
    'Batter_ID': 'BatterId',
    'Batter_Team': 'BatterTeam',
    'Exit_Speed': 'ExitSpeed',
    'Launch_Angle_Vertical': 'Angle',
    'Launch_Angle_Horizontal': 'Direction',
    'Carry': 'Distance',
    'Hit_Track_Distance': 'LastTrackedDistance',
    'Flight_Time': 'HangTime',
}

HIT_TYPE_MAPPING = {
    'PU': 'Popup',
    'FB': 'FlyBall',
    'GB': 'GroundBall',
    'LD': 'LineDrive',
    'UI': 'Unidentified',
}

MPH_PER_FEET_PER_SECOND = 0.681818

def est_poly_at_t(t, poly_list):
    total = 0
    for i, p in enumerate(poly_list):
        total += p * np.power(t, i)
    return total

def find_t_for_poly_crossing(poly_list, crossing):
    def min_fun(t):
        return np.power(est_poly_at_t(t, poly_list) - crossing, 2)
    res = scipy.optimize.minimize(min_fun, [0])
    return res.x[0]

def add_extras_to_row(row):
    # Create new keys with a "fs_" prefix to preserve original values.
    old_keys = list(row.keys())
    for k in old_keys:
        new_k = ''.join(filter(lambda x: x in string.printable, k))
        row['fs_' + new_k] = row[k]
        del row[k]
    # Remap columns using the mapping.
    for key, val in FS_TO_YT_MAPPING.items():
        if 'fs_' + key in row:
            row[val] = row['fs_' + key].strip()
    if 'fs_Batter' in row:
        row['Batter'] = row['fs_Batter'].split('(')[0].strip()
    if 'fs_Batter_Hand' in row:
        batter_hand = row['fs_Batter_Hand'].strip()
        if batter_hand == 'R':
            row['BatterSide'] = 'Right'
        elif batter_hand == 'L':
            row['BatterSide'] = 'Left'
        elif batter_hand == 'S':
            row['BatterSide'] = 'Switch'
        else:
            row['BatterSide'] = 'Unknown'
    if 'fs_Batted_Ball_Type' in row:
        bb_type = row['fs_Batted_Ball_Type'].strip()
        row['HitType'] = HIT_TYPE_MAPPING.get(bb_type, bb_type)
    try:
        x_val = float(row.get('yt_GroundLocationX', 0))
        y_val = float(row.get('yt_GroundLocationY', 0))
        bearing = np.rad2deg(np.arctan2(x_val, y_val))
        row["Bearing"] = f"{bearing:.2f}"
    except (ValueError, TypeError):
        row["Bearing"] = "-"
    try:
        hit_poly_x = [float(r) for r in row.get('fs_Hit_Poly_X', '').split(';') if r]
        hit_poly_y = [float(r) for r in row.get('fs_Hit_Poly_Y', '').split(';') if r]
        hit_poly_z = [float(r) for r in row.get('fs_Hit_Poly_Z', '').split(';') if r]
        if hit_poly_x and hit_poly_y and hit_poly_z:
            row["yt_HitLocationX"] = hit_poly_x[0]
            row["yt_HitLocationY"] = hit_poly_y[0]
            row["yt_HitLocationZ"] = hit_poly_z[0]
            if len(hit_poly_x) > 1 and len(hit_poly_y) > 1 and len(hit_poly_z) > 1:
                row["yt_HitVelocityX"] = f"{hit_poly_x[1] * MPH_PER_FEET_PER_SECOND:.2f}"
                row["yt_HitVelocityY"] = f"{hit_poly_y[1] * MPH_PER_FEET_PER_SECOND:.2f}"
                row["yt_HitVelocityZ"] = f"{hit_poly_z[1] * MPH_PER_FEET_PER_SECOND:.2f}"
            plate_time = find_t_for_poly_crossing(hit_poly_y, 17./12.)
            row["PlateLocHeight"] = f"{est_poly_at_t(plate_time, hit_poly_z):.2f}"
            row["PlateLocSide"] = f"{-1.0 * est_poly_at_t(plate_time, hit_poly_x):.2f}"
    except Exception:
        row["yt_HitLocationX"] = '-'
        row["yt_HitLocationY"] = '-'
        row["yt_HitLocationZ"] = '-'
        row["yt_HitVelocityX"] = '-'
        row["yt_HitVelocityY"] = '-'
        row["yt_HitVelocityZ"] = '-'
        row["PlateLocHeight"] = '-'
        row["PlateLocSide"] = '-'

def convert_flightscope_csv(file, event_date):
    """
    Converts an uploaded Flightscope CSV file to the YT format.
    Adds the event_date to the "Date" column and sequentially assigns "PitchNo" for rows with a Batter.
    """
    file.seek(0)
    lines = file.read().decode('utf-8').splitlines()
    new_lines = []
    found_header = False
    for line in lines:
        if not found_header and "Batter_ID" in line:
            found_header = True
        if found_header:
            new_lines.append(line)
    reader = csv.DictReader(new_lines)
    rows = list(reader)
    for row in rows:
        add_extras_to_row(row)
    pitch_no_counter = 1
    for row in rows:
        if row.get("Batter", "").strip():
            row["Date"] = str(event_date)
            row["PitchNo"] = pitch_no_counter
            pitch_no_counter += 1
    output_str = io.StringIO()
    writer = csv.DictWriter(output_str, fieldnames=yt_header_list(), extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    csv_text = output_str.getvalue()
    output_bytes = io.BytesIO(csv_text.encode('utf-8'))
    output_bytes.seek(0)
    return output_bytes



# --- Blast CSV Merge Functionality ---

def merge_blast_csvs(file_name_pairs):
    """
    Receives a list of tuples (file, name) for Blast CSV files and merges them.
    Each file gets a new "Name" column set to the provided name.
    """
    df_list = []
    for file, name in file_name_pairs:
        df = pd.read_csv(file)
        df['Name'] = name
        df_list.append(df)
    if df_list:
        merged_df = pd.concat(df_list, ignore_index=True)
        return merged_df
    else:
        return pd.DataFrame()

def create_database_section(player_db, doc_width):
    """
    Creates a table flowable displaying the player database information with column widths
    calculated based on the content.
    
    Displays the columns: Name, Age, Class, High School, Position, Bats, Throws.
    """
    # Create a copy and rename columns.
    filtered_db = player_db.copy().rename(columns={
        "BattingHandedness": "Bats", 
        "ThrowingHandedness": "Throws"
    })
    
    columns_to_show = ["Name", "Age", "Class", "High School", "Position", "Bats", "Throws"]
    # Convert to a list of lists
    data = [filtered_db[columns_to_show].columns.tolist()] + filtered_db[columns_to_show].values.tolist()
    
    # Calculate the auto column widths using our helper function.
    col_widths = auto_col_widths(data, font_name='Helvetica', font_size=12, padding=8)
    
    # If the sum of widths is larger than the available width, you might scale them down proportionally:
    total_width = sum(col_widths)
    if total_width > doc_width:
        scale_factor = doc_width / total_width
        col_widths = [w * scale_factor for w in col_widths]
    
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors
    db_table = Table(data, colWidths=col_widths, repeatRows=1)
    db_table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), '#4B4B4B'),
    ('TEXTCOLOR', (0,0), (-1,0), '#FFFFFF'),
    ('BACKGROUND', (0,1), (-1,-1), '#C0C0C0'),
    ('TEXTCOLOR', (0,1), (-1,-1), '#000000'),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,0), 10),  # Header font size lowered from 14 to 10
    ('FONTSIZE', (0,1), (-1,-1), 12),
    ('BOTTOMPADDING', (0,0), (-1,0), 8),
    ('GRID', (0,0), (-1,-1), 0.5, colors.black)
    ]))
    return db_table




    """
    Generates a PNG image of a 2D baseball field (Marlins Park dimensions) with hit locations overlaid.
    
    Assumes:
      - "yt_HitLocationX" and "yt_HitLocationY" are normalized between 0 and 1.
    
    Returns:
      A BytesIO buffer containing the PNG image.
    """
    fig, ax = plt.subplots(figsize=(8,6))
    extent = [-330, 341, 0, 407]
    draw_baseball_field_2D_marlins(ax, extent=extent)
    
    # Extract hit location data.
    x = pd.to_numeric(merged_df["yt_HitLocationX"], errors='coerce').dropna()
    y = pd.to_numeric(merged_df["yt_HitLocationY"], errors='coerce').dropna()
    
    # Scale normalized values:
    # For x: 0 -> -330, 1 -> 341
    # For y: 0 -> 0, 1 -> 407
    x_scaled = x * (extent[1] - extent[0]) + extent[0]  # x * 671 - 330
    y_scaled = y * extent[3]  # y * 407
    
    # Overlay hit locations as red dots with black edges.
    ax.scatter(x_scaled, y_scaled, color='red', edgecolor='black', zorder=2, alpha=0.8)
    ax.set_title("Batted Ball Locations on Marlins Field")
    
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='PNG', bbox_inches='tight')
    plt.close(fig)
    img_buffer.seek(0)
    return img_buffer
# --- Report Generation Functions ---
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
        st.write(f"### Preview of {file.name}")
        st.dataframe(df.head())
        return df
    return None
def generate_flightscope_pdf(flightscope_df, player_db):

    """
    Generates a one-page PDF report for Flightscope data including:
      - A header section.
      - A player database section (only players present in the CSV).
      - A summary section.
      - A section for batted ball locations on a drawn 2D baseball field.
    Returns a BytesIO object containing the PDF.
    """
    import io
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    
    # Merge flightscope data with player database.
    merged_df = flightscope_df.merge(player_db, left_on="Batter", right_on="Name", how="left")
    players_in_csv = merged_df["Batter"].unique()
    filtered_player_db = player_db[player_db["Name"].isin(players_in_csv)]
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=30, leftMargin=30,
                            topMargin=30, bottomMargin=18)
    available_width = doc.width  # Use available width for proportional sizing.
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleCustom', fontSize=24, leading=28, alignment=1, spaceAfter=10))
    styles.add(ParagraphStyle(name='HeadingCustom', fontSize=16, leading=20, spaceAfter=8))
    styles.add(ParagraphStyle(name='NormalCustom', fontSize=10, leading=12, spaceAfter=6))
    
    flowables = []
    
    # Add Header (only once).
    header_flowable = create_report_header("Flightscope Report", "Player Database Information")
    flowables.append(header_flowable)
    flowables.append(Spacer(1, 12))
    
    # Add the filtered player database section.
    db_flowable = create_database_section(filtered_player_db, available_width)
    flowables.append(db_flowable)
    flowables.append(Spacer(1, 12))
    
    # Add Summary section.
    # (Removed duplicate "Flightscope Report" title here.)
    flowables.append(Paragraph(f"Total Merged Records: {len(merged_df)}", styles['NormalCustom']))
    flowables.append(Spacer(1, 6))
    
    # Batted Ball Locations on Field section.
    if "yt_HitLocationX" in merged_df.columns and "yt_HitLocationY" in merged_df.columns:
        flowables.append(Paragraph("Batted Ball Locations on Field", styles['HeadingCustom']))
        img_buffer = generate_2d_field_with_hits_marlins(merged_df)
        flowables.append(Image(img_buffer, width=400, height=300))
        flowables.append(Spacer(1, 6))
    
    available_width = doc.width
    header_flowable = create_header_without_logo(available_width, styles)
    flowables.append(header_flowable)
    flowables.append(Spacer(1, 12))


    # Build the PDF.
    doc.build(flowables)
    buffer.seek(0)
    return buffer

def generate_combined_pdf_report(fs_data, blast_data, forcedecks_data, strength_data, running_speed_data, core_strength_data, throwing_velocities_data, player_db):
    import io
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=30, leftMargin=30,
                            topMargin=30, bottomMargin=18)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleCustom', fontSize=24, leading=28, alignment=1, spaceAfter=10))
    flowables = []
    
    # Header Section (e.g., without logo)
    header = Paragraph("Combined PDF Report", styles['TitleCustom'])
    flowables.append(header)
    flowables.append(Spacer(1, 12))
    
    # Add a section for each CSV
    # Example: Flightscope Section
    fs_section = Paragraph(f"Flightscope Report - Total Records: {len(fs_data)}", styles['Normal'])
    flowables.append(fs_section)
    # ... add charts, tables, etc. for flightscope data
    
    # Example: Blast Section
    blast_section = Paragraph(f"Blast Report - Total Records: {len(blast_data)}", styles['Normal'])
    flowables.append(blast_section)
    # ... add charts, tables, etc.
    
    # Continue similarly for other CSVs...
    
    # Add Player Database Section (filter if needed)
    players_in_csv = fs_data["Batter"].unique()  # adjust if needed
    filtered_db = player_db[player_db["Name"].isin(players_in_csv)]
    db_section = create_database_section(filtered_db, doc.width)
    flowables.append(db_section)
    
    # Build the combined PDF
    doc.build(flowables)
    buffer.seek(0)
    return buffer
# ================================
# Player Database Persistence
# ================================
DATABASE_FILENAME = "player_database.csv"
if 'player_db' not in st.session_state:
    if os.path.exists(DATABASE_FILENAME):
        st.session_state['player_db'] = pd.read_csv(DATABASE_FILENAME)
    else:
        st.session_state['player_db'] = pd.DataFrame(
            columns=["Name", "Age", "Class", "High School", "Height", "Weight", "Position", "BattingHandedness", "ThrowingHandedness"]
        )

# ================================
# Streamlit App Tabs
# ================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Flightscope CSV Conversion",
    "Blast CSV Merge",
    "Player Database",
    "Report Generation"
])
# Tab1 - Flightscope CSV Conversion, Tab2 - Blast CSV Merge, Tab3 - Player Database, Tab4 - Report Generation

with tab1:
    st.header("Flightscope CSV Conversion")
    st.info("Convert your Flightscope CSV to the YT format.")
    event_date = st.date_input("Select Event Date")
    flightscope_upload = st.file_uploader("Convert Flightscope CSV to YT Format CSV", type=["csv"], key="flightscope_conversion")
    if flightscope_upload is not None:
        converted_csv = convert_flightscope_csv(flightscope_upload, event_date)
        st.download_button("Download Converted CSV", data=converted_csv, file_name="converted_flightscope.csv", mime="text/csv")
        st.success("Conversion complete!")
with tab2:
    st.header("Blast CSV Merge")
    st.info("Upload multiple Blast CSV files. For each file, enter the Name to add as a new column before merging.")
    blast_files = st.file_uploader("Upload Blast CSV Files", type=["csv"], key="blast_merge", accept_multiple_files=True)
    file_name_pairs = []
    if blast_files:
        st.write("### Provide a name for each Blast CSV file:")
        for i, file in enumerate(blast_files):
            default_name = "Blast"
            name = st.text_input(f"Name for file {i+1} ({file.name})", value=default_name, key=f"blast_name_{i}")
            file_name_pairs.append((file, name))
        if st.button("Merge Blast CSV Files"):
            merged_blast_df = merge_blast_csvs(file_name_pairs)
            st.write("### Merged Blast CSV Preview")
            st.dataframe(merged_blast_df.head())
            csv_blast = merged_blast_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Merged Blast CSV", data=csv_blast, file_name="merged_blast.csv", mime="text/csv")
with tab3:
    st.header("Player Database")
    st.info("Add, edit, or delete players. The database is saved persistently.")

    # Define the database filename and initialize if necessary.
    DATABASE_FILENAME = "player_database.csv"
    if "player_db" not in st.session_state:
        if os.path.exists(DATABASE_FILENAME):
            st.session_state.player_db = pd.read_csv(DATABASE_FILENAME)
        else:
            st.session_state.player_db = pd.DataFrame(columns=[
                "Name", "Age", "Class", "High School", "Height", "Weight",
                "Position", "BattingHandedness", "ThrowingHandedness"
            ])

    # --- Add New Player Form ---
    with st.form("player_form", clear_on_submit=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=0, max_value=100, step=1)
        with col2:
            pclass = st.text_input("Class")
            high_school = st.text_input("High School")
        with col3:
            height = st.number_input("Height (in inches)", min_value=0, max_value=120, step=1)
            weight = st.number_input("Weight (lbs)", min_value=0, max_value=500, step=1)
        with col4:
            position = st.text_input("Position")
            batting_hand = st.selectbox("Batting Handedness", options=["Left", "Right", "Switch"])
            throwing_hand = st.selectbox("Throwing Handedness", options=["Left", "Right"])
        
        submitted = st.form_submit_button("Add Player")
        if submitted:
            new_player = {
                "Name": name,
                "Age": age,
                "Class": pclass,
                "High School": high_school,
                "Height": height,
                "Weight": weight,
                "Position": position,
                "BattingHandedness": batting_hand,
                "ThrowingHandedness": throwing_hand,
            }
            new_player_df = pd.DataFrame([new_player])
            st.session_state.player_db = pd.concat([st.session_state.player_db, new_player_df], ignore_index=True)
            st.session_state.player_db.to_csv(DATABASE_FILENAME, index=False)
            st.success(f"Player '{name}' added!")

    # Display the current player database (only once).
    st.write("### Current Player Database")
    st.dataframe(st.session_state.player_db)

    # --- Edit / Delete Player Section ---
    st.write("### Edit / Delete Player")
    if not st.session_state.player_db.empty:
        player_options = st.session_state.player_db.index.tolist()
        selected_index = st.selectbox(
            "Select a player to edit or delete",
            player_options,
            format_func=lambda x: f"{x}: {st.session_state.player_db.loc[x, 'Name']}"
        )
        selected_player = st.session_state.player_db.loc[selected_index]
        
        with st.form("edit_player_form", clear_on_submit=False):
            new_name = st.text_input("Name", value=selected_player["Name"])
            new_age = st.number_input("Age", min_value=0, max_value=100, step=1, value=int(selected_player["Age"]))
            new_class = st.text_input("Class", value=selected_player["Class"])
            new_high_school = st.text_input("High School", value=selected_player["High School"])
            new_height = st.number_input("Height (in inches)", min_value=0, max_value=120, step=1, value=int(selected_player["Height"]))
            new_weight = st.number_input("Weight (lbs)", min_value=0, max_value=500, step=1, value=int(selected_player["Weight"]))
            new_position = st.text_input("Position", value=selected_player["Position"])
            
            # Precompute default indices for handedness.
            batting_default = selected_player.get("BattingHandedness", "Left")
            if batting_default not in ["Left", "Right", "Switch"]:
                batting_default = "Left"
            batting_index = ["Left", "Right", "Switch"].index(batting_default)
            
            throwing_default = selected_player.get("ThrowingHandedness", "Right")
            if throwing_default not in ["Left", "Right"]:
                throwing_default = "Right"
            throwing_index = ["Left", "Right"].index(throwing_default)
            
            new_batting_hand = st.selectbox(
                "Batting Handedness", 
                options=["Left", "Right", "Switch"],
                index=batting_index
            )
            new_throwing_hand = st.selectbox(
                "Throwing Handedness", 
                options=["Left", "Right"],
                index=throwing_index
            )
            
            col_update, col_delete = st.columns(2)
            update_submitted = col_update.form_submit_button("Update Player")
            delete_submitted = col_delete.form_submit_button("Delete Player")
        
        if update_submitted:
            st.session_state.player_db.loc[selected_index, "Name"] = new_name
            st.session_state.player_db.loc[selected_index, "Age"] = new_age
            st.session_state.player_db.loc[selected_index, "Class"] = new_class
            st.session_state.player_db.loc[selected_index, "High School"] = new_high_school
            st.session_state.player_db.loc[selected_index, "Height"] = new_height
            st.session_state.player_db.loc[selected_index, "Weight"] = new_weight
            st.session_state.player_db.loc[selected_index, "Position"] = new_position
            st.session_state.player_db.loc[selected_index, "BattingHandedness"] = new_batting_hand
            st.session_state.player_db.loc[selected_index, "ThrowingHandedness"] = new_throwing_hand
            st.session_state.player_db.to_csv(DATABASE_FILENAME, index=False)
            st.success("Player updated!")
        
        if delete_submitted:
            st.session_state.player_db = st.session_state.player_db.drop(index=selected_index).reset_index(drop=True)
            st.session_state.player_db.to_csv(DATABASE_FILENAME, index=False)
            st.success("Player deleted!")
    else:
        st.write("No players to edit.")

 # Tab 4: Performance Report Generation
with tab4:
    st.header(" Report Generation")
    st.sidebar.header("Upload Your Data Files for Report")
    
    flightscope_file = st.sidebar.file_uploader("Upload Flightscope CSV", type=["csv"], key="flightscope_report")
    blast_file = st.sidebar.file_uploader("Upload Blast Motion CSV", type=["csv"], key="blast")
    forcedecks_file = st.sidebar.file_uploader("Upload ForceDecks CSV", type=["csv"], key="forcedecks")
    strength_file = st.sidebar.file_uploader("Upload Strength Training CSV", type=["csv"], key="strength")
    running_speed_file = st.sidebar.file_uploader("Upload Running Speed CSV", type=["csv"], key="running_speed")
    core_strength_file = st.sidebar.file_uploader("Upload Core Strength CSV", type=["csv"], key="core_strength")
    throwing_velocities_file = st.sidebar.file_uploader("Upload Throwing Velocities CSV", type=["csv"], key="throwing_velocities")
    
    flightscope_data = load_data(flightscope_file)
    blast_data = load_data(blast_file)
    forcedecks_data = load_data(forcedecks_file)
    strength_data = load_data(strength_file)
    running_speed_data = load_data(running_speed_file)
    core_strength_data = load_data(core_strength_file)
    throwing_velocities_data = load_data(throwing_velocities_file)
    
    if all(data is not None for data in [flightscope_data, blast_data, forcedecks_data, strength_data, running_speed_data, core_strength_data, throwing_velocities_data]):
        st.write("## Merged Dataset Preview")
        merged_df = pd.concat(
            [flightscope_data, blast_data, forcedecks_data, strength_data, running_speed_data, core_strength_data, throwing_velocities_data],
            axis=0, ignore_index=True
        )
        st.dataframe(merged_df.head())
        
        csv_merged = merged_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Merged CSV", data=csv_merged, file_name="merged_athlete_data.csv", mime='text/csv')
        
        if st.button("Generate On-Screen Performance Report"):
            st.header("Performance Report")
            if "Exit_Speed" in merged_df.columns:
                st.subheader("Exit Speed Distribution")
                fig, ax = plt.subplots()
                data = pd.to_numeric(merged_df["Exit_Speed"], errors='coerce').dropna()
                ax.hist(data, bins=20, color='blue', alpha=0.7)
                ax.axvline(data.mean(), color='red', linestyle='dashed', linewidth=1,
                           label=f'Mean: {data.mean():.2f}')
                ax.legend()
                ax.set_xlabel("Exit Speed (mph)")
                ax.set_ylabel("Count")
                ax.set_title("Distribution of Exit Speeds")
                st.pyplot(fig)
            if "Bat Speed (mph)" in merged_df.columns and "Rotational Acceleration (g)" in merged_df.columns:
                st.subheader("Bat Speed vs Rotational Acceleration")
                fig2, ax2 = plt.subplots()
                x = merged_df["Bat Speed (mph)"]
                y = merged_df["Rotational Acceleration (g)"]
                ax2.scatter(x, y, alpha=0.7, color='red')
                if len(x.dropna()) > 1:
                    m, b = np.polyfit(x.dropna(), y.dropna(), 1)
                    ax2.plot(x, m*x + b, color='blue', linestyle='--', linewidth=2,
                             label=f'Trend: y={m:.2f}x+{b:.2f}')
                    ax2.legend()
                ax2.set_xlabel("Bat Speed (mph)")
                ax2.set_ylabel("Rotational Acceleration (g)")
                ax2.set_title("Bat Speed vs Rotational Acceleration")
                st.pyplot(fig2)
            report_text = "Athlete Performance Report\n" + "="*30 + "\n\n"
            report_text += f"Total Records: {len(merged_df)}\n"
            report_text += f"Available Features: {', '.join(merged_df.columns)}\n\n"
            if "Exit_Speed" in merged_df.columns:
                exit_speed = pd.to_numeric(merged_df["Exit_Speed"], errors='coerce').dropna()
                report_text += "Exit Speed Statistics:\n"
                report_text += f"  Mean: {exit_speed.mean():.2f} mph\n"
                report_text += f"  Median: {exit_speed.median():.2f} mph\n"
                report_text += f"  Min: {exit_speed.min():.2f} mph\n"
                report_text += f"  Max: {exit_speed.max():.2f} mph\n"
            st.text(report_text)
        
        if st.button("Generate PDF Report"):
            buffer_pdf = BytesIO()
            c = canvas.Canvas(buffer_pdf, pagesize=letter)
            width, height = letter
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 50, "Athlete Performance Report")
            c.setFont("Helvetica", 12)
            text = c.beginText(50, height - 80)
            text.textLine(f"Total Records: {len(merged_df)}")
            text.textLine(f"Available Features: {', '.join(merged_df.columns)}")
            if "Exit_Speed" in merged_df.columns:
                exit_speed = pd.to_numeric(merged_df["Exit_Speed"], errors='coerce').dropna()
                text.textLine("")
                text.textLine("Exit Speed Statistics:")
                text.textLine(f"  Mean: {exit_speed.mean():.2f} mph")
                text.textLine(f"  Median: {exit_speed.median():.2f} mph")
                text.textLine(f"  Min: {exit_speed.min():.2f} mph")
                text.textLine(f"  Max: {exit_speed.max():.2f} mph")
            c.drawText(text)
            if "Exit_Speed" in merged_df.columns:
                fig, ax = plt.subplots()
                data = pd.to_numeric(merged_df["Exit_Speed"], errors='coerce').dropna()
                ax.hist(data, bins=20, color='blue', alpha=0.7)
                ax.axvline(data.mean(), color='red', linestyle='dashed', linewidth=1,
                           label=f'Mean: {data.mean():.2f}')
                ax.legend()
                ax.set_xlabel("Exit Speed (mph)")
                ax.set_ylabel("Count")
                ax.set_title("Distribution of Exit Speeds")
                buf_hist = BytesIO()
                fig.savefig(buf_hist, format="png")
                buf_hist.seek(0)
                img_hist = ImageReader(buf_hist)
                c.showPage()
                c.drawImage(img_hist, 50, height/2 - 150, width=500, height=300)
            if "Bat Speed (mph)" in merged_df.columns and "Rotational Acceleration (g)" in merged_df.columns:
                fig2, ax2 = plt.subplots()
                x = merged_df["Bat Speed (mph)"]
                y = merged_df["Rotational Acceleration (g)"]
                ax2.scatter(x, y, alpha=0.7, color='red')
                if len(x.dropna()) > 1:
                    m, b = np.polyfit(x.dropna(), y.dropna(), 1)
                    ax2.plot(x, m*x + b, color='blue', linestyle='--', linewidth=2,
                             label=f'Trend: y={m:.2f}x+{b:.2f}')
                    ax2.legend()
                ax2.set_xlabel("Bat Speed (mph)")
                ax2.set_ylabel("Rotational Acceleration (g)")
                ax2.set_title("Bat Speed vs Rotational Acceleration")
                buf_scatter = BytesIO()
                fig2.savefig(buf_scatter, format="png")
                buf_scatter.seek(0)
                img_scatter = ImageReader(buf_scatter)
                c.showPage()
                c.drawImage(img_scatter, 50, height/2 - 150, width=500, height=300)
            c.save()
            buffer_pdf.seek(0)
            st.download_button(label="Download PDF Report", data=buffer_pdf, file_name="performance_report.pdf", mime="application/pdf")
        
    if flightscope_data is not None:
     st.write("## Flightscope Data Report")
    if st.button("Generate Flightscope PDF Report"):
        pdf_buffer = generate_flightscope_pdf(flightscope_data, st.session_state.player_db)
        st.download_button("Download Flightscope PDF", data=pdf_buffer, file_name="flightscope_report.pdf", mime="application/pdf")
        combined_pdf_buffer = generate_combined_pdf_report(flightscope_file, blast_data, forcedecks_data, strength_data, running_speed_data, core_strength_data, throwing_velocities_data, st.session_state.player_db)
        st.download_button("Download Combined PDF Report", data=combined_pdf_buffer, file_name="combined_report.pdf", mime="application/pdf")


# -------- CSV Templates Download Section --------
st.sidebar.header("Download Data Templates")
def create_template(file_name, columns):
    template_df = pd.DataFrame(columns=columns)
    csv_data = template_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(label=f"Download {file_name} Template",
                               data=csv_data,
                               file_name=file_name,
                               mime='text/csv')

running_speed_columns = ["Player Name", "30yd Time", "60yd Time", "5-5-10 Shuttle Time"]
core_strength_columns = ["Player Name", "Core Strength Measurement"]
throwing_velocities_columns = ["Player Name", "Positional Throw Velocity", "Pulldown Velocity", "FB Velocity", "SL Velocity", "CB Velocity", "CH Velocity"]

create_template("running_speed_template.csv", running_speed_columns)
create_template("core_strength_template.csv", core_strength_columns)
create_template("throwing_velocities_template.csv", throwing_velocities_columns)



