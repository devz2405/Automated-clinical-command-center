# 🏥 Automated Clinical Command Center & Predictive Triage Pipeline

## 🌟 Project Overview
This project is a full-stack Healthcare Data Engineering and Analytics solution. It automates the ingestion, processing, and visualization of patient vitals to provide a real-time **Clinical Command Center**. 

The system doesn't just show current stats; it uses **Machine Learning (Linear Regression)** to predict patient deterioration by calculating the "Heart Rate Slope" over a rolling 3-hour window.

## 🚀 Key Features
- **Automated ETL Pipeline:** A Python-based pipeline that generates synthetic patient data, cleans it using Pandas, and syncs it to a PostgreSQL database.
- **Predictive Risk Modeling:** Implements `scikit-learn` to identify patients with rising heart rates, even if they are currently within "normal" ranges.
- **Live Medical Dashboard:** A Streamlit web application designed with a high-contrast "Medical Console" aesthetic for clinical glanceability.
- **Dynamic Triage:** Automatically categorizes patients into **CRITICAL**, **WARNING**, or **STABLE** based on predictive slopes and real-time alert triggers (SpO2 < 90%, HR > 115).

## 🛠️ Tech Stack
- **Language:** Python 3.13
- **Database:** PostgreSQL (Relational Data Modeling)
- **Libraries:** Pandas, NumPy, SQLAlchemy, Scikit-Learn
- **Visualization:** Streamlit, Plotly Express
- **DevOps:** Streamlit Secrets Management, .gitignore security protocols

## 📂 Project Structure
- `clinical_com_centre.py`: The core engine containing both the ETL pipeline and the Streamlit UI.
- `patient_dim.csv`: Persistent dimension table for the 500-patient cohort.
- `.streamlit/secrets.toml`: (Excluded from Git) Secure database credentials.
- `requirements.txt`: List of dependencies for easy deployment.

## 📊 Logic & Methodology
1. **Data Layer:** Uses a *Truncate and Append* strategy to maintain a high-performance "Live" table while archiving all data to a cumulative history table.
2. **Analytics Layer:** A SQL Gold View (`gold_clinical_command_center`) handles the final business logic, ranking patient recency and assigning triage priorities.
3. **UI Layer:** Custom CSS injection creates a dark-mode clinical interface, prioritizing "CRITICAL" status alerts through visual highlighting.

## 📈 Dashboard Preview
- **Top Metrics:** Total Patients, Active Alerts, System Sync Status.
- **Triage Monitor:** Color-coded patient list sorted by severity.
- **Trend Analysis:** Interactive Plotly charts showing the Heart Rate Slope for specific patients.

---
**Developer:** Devi Sasi  
**Role:** Clinical Data & Validation Analyst  
**Focus:** Healthcare Automation & Data Integrity
