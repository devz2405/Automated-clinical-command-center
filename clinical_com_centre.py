#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import sqlalchemy as sa
from sqlalchemy import text
from faker import Faker
import random
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

fake = Faker()

# --- CONFIGURATION ---
NUM_PATIENTS = 500
RECORDS_PER_PATIENT = 12  # 12 readings (one every 15 minutes = 3 hours total)
DB_URL = 'postgresql+psycopg2://postgres:rootd@localhost:5432/healthcare'

def get_or_create_patients():
    """Ensures we use the same 500 patients every day."""
    if os.path.exists("patient_dim.csv"):
        print("Loading existing patient cohort...")
        return pd.read_csv("patient_dim.csv")
    else:
        print("Generating new patient cohort (1,000 patients)...")
        patients = []
        for i in range(NUM_PATIENTS):
            patients.append({
                "patient_id": f"P{500 + i}",
                "name": fake.name(),
                "age": random.randint(18, 90),
                "gender": random.choice(["Male", "Female", "Other"]),
                "blood_type": random.choice(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]),
                "condition": random.choice(["Post-Op", "Critical Care", "Observation", "Stable"])
            })
        df = pd.DataFrame(patients)
        df.to_csv("patient_dim.csv", index=False)
        return df

def generate_fresh_vitals(df_patients):
    """Generates vitals for the last 3 hours for the current cohort."""
    print(f"Generating {RECORDS_PER_PATIENT} hours of vitals for each patient...")
    vitals_data = []
    # Start time is 3 hours ago, ending NOW
    start_time = datetime.now() - timedelta(hours=3)

    for p_id in df_patients['patient_id']:
        base_hr = random.randint(65, 85)
        base_sbp = random.randint(110, 130)

        for h in range(RECORDS_PER_PATIENT):
            hr_noise = np.random.normal(0, 5)
            sbp_noise = np.random.normal(0, 10)

            # 5% chance of a critical event
            if random.random() < 0.05:
                hr_noise += 40 
                sbp_noise += 50

            vitals_data.append({
                "patient_id": p_id,
                "timestamp": start_time + timedelta(minutes=h*15),
                "heart_rate": round(base_hr + hr_noise, 1),
                "systolic_bp": round(base_sbp + sbp_noise, 1),
                "spo2_percent": max(min(round(np.random.normal(97, 2), 1), 100), 70),
                "temp_c": round(np.random.normal(37, 0.5), 1)
            })
    return pd.DataFrame(vitals_data)

def calculate_vitals_trend(df):
    """Predicts Heart Rate slope/trend."""
    trends = []
    for p_id in df['patient_id'].unique():
        patient_data = df[df['patient_id'] == p_id].sort_values('timestamp')
        X = np.array(range(len(patient_data))).reshape(-1, 1)
        y = patient_data['heart_rate'].values

        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        prediction_24h = model.predict([[len(patient_data) + 24]])[0]

        trends.append({
            "patient_id": p_id,
            "hr_slope": round(slope, 3),
            "predicted_hr_24h": round(prediction_24h, 1),
            "trend_status": "Rising" if slope > 0.5 else "Falling" if slope < -0.5 else "Stable"
        })
    return pd.DataFrame(trends)

def run_healthcare_pipeline():
    print(f"\n[{datetime.now()}] --- STARTING FULL PIPELINE ---")

    # 1. DATA GENERATION / INGESTION
    df_patients = get_or_create_patients()
    df_vitals = generate_fresh_vitals(df_patients)

    # 2. AUTOMATED CLEANING
    df_vitals['timestamp'] = pd.to_datetime(df_vitals['timestamp'])
    df_vitals['heart_rate'] = df_vitals.groupby('patient_id')['heart_rate'].transform(lambda x: x.fillna(x.median()))
    df_vitals['spo2_percent'] = df_vitals['spo2_percent'].clip(lower=70, upper=100)

    # 3. INTELLIGENT RISK MODELLING
    df_vitals['is_alert'] = (
        (df_vitals['heart_rate'] > 115) | 
        (df_vitals['spo2_percent'] < 90) | 
        (df_vitals['systolic_bp'] > 160)
    ).astype(int)

    # 4. FEATURE ENGINEERING
    df_vitals = df_vitals.sort_values(['patient_id', 'timestamp'])
    print("Calculating Predictive Trends...")
    df_trends = calculate_vitals_trend(df_vitals)

    # 5. INTEGRATION
    df_silver = pd.merge(df_vitals, df_patients, on="patient_id", how="inner")
    df_final = pd.merge(df_silver, df_trends, on="patient_id", how="left")

    # # 6. POSTGRESQL SYNC
    # engine = sa.create_engine(DB_URL)

    # try:
    #     with engine.begin() as conn:
    #         print("Syncing to PostgreSQL (Truncate & Append)...")
    #         conn.execute(text("TRUNCATE TABLE vitals_processed_fact CASCADE"))
    #         df_final.to_sql('vitals_processed_fact', con=conn, if_exists='append', index=False)

    #         print("Refreshing Gold View...")
    #         view_sql = """
    #         CREATE OR REPLACE VIEW public.gold_clinical_command_center AS
    #         WITH latestvitals AS (
    #             SELECT 
    #                 patient_id, name, age, condition, heart_rate, spo2_percent, hr_slope, is_alert, "timestamp",
    #                 ROW_NUMBER() OVER (PARTITION BY patient_id ORDER BY "timestamp" DESC) AS recency_rank
    #             FROM vitals_processed_fact
    #         )
    #         SELECT 
    #             patient_id, name, age, condition,
    #             heart_rate AS current_hr,
    #             spo2_percent AS current_spo2,
    #             hr_slope,
    #             CASE
    #                 WHEN hr_slope > 1.5 OR (heart_rate > 120 AND is_alert = 1) THEN 'CRITICAL'
    #                 WHEN hr_slope > 0.5 OR is_alert = 1 THEN 'WARNING'
    #                 ELSE 'STABLE'
    #             END AS triage_priority,
    #             "timestamp" AS last_updated
    #         FROM latestvitals
    #         WHERE recency_rank = 1;
    #         """
    #         conn.execute(text(view_sql))

    #     print(f"[{datetime.now()}] SUCCESS: Alerts Triggered: {df_final['is_alert'].sum()}")

    # except Exception as e:
    #     print(f"Database Error: {e}")
# 6. POSTGRESQL SYNC (Full Integration with History & View)
    engine = sa.create_engine(DB_URL)

    try:
        with engine.begin() as conn:
            # A. ARCHIVE TO HISTORY (Cumulative - No Truncate)
            print("Archiving data to vitals_history...")
            # We use 'if_exists=append' so data from today is added to data from yesterday
            df_final.to_sql('vitals_history', con=conn, if_exists='append', index=False)

            # B. SYNC TO LIVE TABLE (24-Hour Window only)
            print("Syncing to vitals_processed_fact (Truncate & Append)...")
            conn.execute(text("TRUNCATE TABLE vitals_processed_fact CASCADE"))
            df_final.to_sql('vitals_processed_fact', con=conn, if_exists='append', index=False)

            # C. REFRESH THE GOLD VIEW LOGIC
            print("Refreshing Gold View...")
            view_sql = """
            CREATE OR REPLACE VIEW public.gold_clinical_command_center AS
            WITH latestvitals AS (
                SELECT 
                    patient_id, name, age, condition, heart_rate, spo2_percent, hr_slope, is_alert, "timestamp",
                    ROW_NUMBER() OVER (PARTITION BY patient_id ORDER BY "timestamp" DESC) AS recency_rank
                FROM vitals_processed_fact
            )
            SELECT 
                patient_id, name, age, condition,
                heart_rate AS current_hr,
                spo2_percent AS current_spo2,
                hr_slope,
                CASE
                    WHEN hr_slope > 1.5 OR (heart_rate > 120 AND is_alert = 1) THEN 'CRITICAL'
                    WHEN hr_slope > 0.5 OR is_alert = 1 THEN 'WARNING'
                    ELSE 'STABLE'
                END AS triage_priority,
                "timestamp" AS last_updated
            FROM latestvitals
            WHERE recency_rank = 1;
            """
            conn.execute(text(view_sql))

        print(f"[{datetime.now()}] SUCCESS: History archived, Live table synced, and View refreshed.")
        print(f"Alerts Triggered in current batch: {df_final['is_alert'].sum()}")

    except Exception as e:
        print(f"Database Error: {e}")

if __name__ == "__main__":
    run_healthcare_pipeline()

import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

# 1. Setup Connection (Connects to existing pipeline)

# def get_data():
#     # Streamlit pulls these from the secrets.toml file automatically
#     pg = st.secrets["postgres"]
    
#     conn_str = f"postgresql://{pg['user']}:{pg['password']}@{pg['host']}:{pg['port']}/{pg['database']}"
#     engine = create_engine(conn_str)
    
#     query = "SELECT * FROM gold_clinical_command_center"
#     return pd.read_sql(query, engine)

# # 2. Sidebar & KPI Header
# st.title("🚀 Automated Clinical Command Center")
# df = get_data()

# col1, col2, col3 = st.columns(3)
# col1.metric("Total Patients", len(df))
# col2.metric("Critical Alerts", len(df[df['triage_priority'] == 'CRITICAL']))
# col3.metric("System Status", "Live Sync Active")

# # 3. The Triage Logic (Recreating your DAX/SQL logic)
# st.subheader("⚠️ Patient Triage Monitor")
# # Styling the dataframe like a professional dashboard
# st.dataframe(df.style.applymap(lambda x: 'background-color: #ff4b4b' if x == 'CRITICAL' else '', subset=['triage_priority']))

# # 4. Interactive Trend Analysis (Plotly)
# st.subheader("📈 Patient Vital Trends (Heart Rate Slope)")
# selected_patient = st.selectbox("Select Patient to Inspect", df['patient_id'].unique())
# patient_data = df[df['patient_id'] == selected_patient]

# fig = px.line(patient_data, x='timestamp', y='heart_rate', 
#               title=f"Predictive HR Slope for Patient {selected_patient}",
#               template="plotly_dark")
# st.plotly_chart(fig, use_container_width=True)

# # 5. Pipeline Log (Shows off your automation)
# with st.expander("View Backend Automation Pipeline Status"):
#     st.write("Current Strategy: Truncate and Append")
#     st.write("Model: Linear Regression (hr_slope calculation)")


# --- STREAMLIT DASHBOARD SECTION ---

def get_data():
    """Smart data fetcher: Database first, CSV fallback for Demo Mode."""
    try:
        # Try Cloud/Local Database first
        pg = st.secrets["postgres"]
        conn_str = f"postgresql://{pg['user']}:{pg['password']}@{pg['host']}:{pg['port']}/{pg['database']}"
        engine = sa.create_engine(conn_str)
        return pd.read_sql("SELECT * FROM gold_clinical_command_center", engine)
    except Exception:
        # FALLBACK: If DB fails, load the CSV and simulate the 'Gold View'
        st.sidebar.warning("⚠️ Running in Demo Mode (Local CSV)")
        df_demo = pd.read_csv("patient_dim.csv")
        
        # Add some mock data so the table isn't empty in the demo
        df_demo['current_hr'] = [random.randint(70, 110) for _ in range(len(df_demo))]
        df_demo['current_spo2'] = [random.randint(92, 99) for _ in range(len(df_demo))]
        df_demo['hr_slope'] = [round(random.uniform(-1, 2), 2) for _ in range(len(df_demo))]
        df_demo['triage_priority'] = df_demo['hr_slope'].apply(lambda x: 'CRITICAL' if x > 1.5 else 'STABLE')
        df_demo['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M")
        return df_demo

def get_history(patient_id):
    """Smart history fetcher: Database first, Simulated history fallback."""
    try:
        pg = st.secrets["postgres"]
        conn_str = f"postgresql://{pg['user']}:{pg['password']}@{pg['host']}:{pg['port']}/{pg['database']}"
        engine = sa.create_engine(conn_str)
        query = f"SELECT timestamp, heart_rate FROM vitals_history WHERE patient_id = '{patient_id}' ORDER BY timestamp ASC"
        return pd.read_sql(query, engine)
    except Exception:
        # FALLBACK: Generate 12 random points so the line chart shows up
        start_time = datetime.now() - timedelta(hours=3)
        mock_hist = []
        base = random.randint(70, 90)
        for i in range(12):
            mock_hist.append({
                "timestamp": start_time + timedelta(minutes=i*15),
                "heart_rate": base + random.randint(-5, 5)
            })
        return pd.DataFrame(mock_hist)

if __name__ == "__main__":
    st.markdown("""
   <style>
    /* Change the background to a deep clinical blue */
    .stApp {
        background-color: #0E1117;
    }

    /* BRIGHTEN HEADINGS: Make Titles and Subheaders Pure White and Bold */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        text-shadow: 0 0 2px rgba(255, 255, 255, 0.2);
    }

    /* BRIGHTEN TEXT: Make standard labels and paragraph text a bright silver */
    p, label, .stMarkdown {
        color: #E0E0E0 !important;
    }

    /* Make metrics look like a patient monitor */
    [data-testid="stMetricValue"] {
        font-family: 'Courier New', monospace;
        color: #00FF00; /* Neon Green */
        text-shadow: 0 0 5px #00FF00;
    }

    /* Style the table to look like an EMR interface */
    .styled-table {
        border-collapse: collapse;
        font-size: 0.9em;
        font-family: sans-serif;
        min-width: 400px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    }
</style>
    """, unsafe_allow_html=True)
    st.title("🚀 Clinical Command Center")

    # 1. Manual Refresh Button (Instead of auto-refreshing on every click)
    if st.sidebar.button("Run New Pipeline Sync"):
        run_healthcare_pipeline()
        st.sidebar.success("Pipeline Updated!")

    try:
        df = get_data()
        
        # 2. Display Metrics
        col1, col2 = st.columns(2)
        col1.metric("Total Patients", len(df))
        col2.metric("Critical Alerts", len(df[df['triage_priority'] == 'CRITICAL']))

        # 3. Triage Table
        st.subheader("⚠️ Patient Triage Monitor")
        st.dataframe(df.style.applymap(
            lambda x: 'background-color: #ff4b4b' if x == 'CRITICAL' else '', 
            subset=['triage_priority']
        ), use_container_width=True)

        # 4. FIXED Trend Analysis
        st.subheader("📈 Patient Vital Trends")
        selected_p = st.selectbox("Select Patient to Inspect", df['patient_id'].unique())
        
        # We fetch the history for the SPECIFIC patient so the chart isn't empty
        hist_df = get_history(selected_p)
        
        if not hist_df.empty:
            fig = px.line(hist_df, x='timestamp', y='heart_rate', 
                          title=f"Heart Rate History: {selected_p}",
                          template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical data found for this patient.")

    except Exception as e:
        st.error(f"Waiting for initial data... Error: {e}")
