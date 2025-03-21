import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from datetime import datetime

# Page Config
st.set_page_config(page_title="HealthSense", page_icon="🩺", layout="wide")

# Custom CSS
st.markdown("""
<style>
body {
    font-family: 'Roboto', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    color: #ffffff;
}
.sidebar .sidebar-content {
    background: #203a43;
    color: #ffffff;
    padding: 20px;
    border-radius: 10px;
}
.stButton>button {
    background: linear-gradient(90deg, #ff6b6b 0%, #ff8e53 100%);
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 25px;
    font-weight: bold;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #ff8e53 0%, #ff6b6b 100%);
    transform: scale(1.05);
}
.stNumberInput input, .stSelectbox select, .stMultiselect div {
    background-color: #ffffff;
    color: #333;
    border-radius: 8px;
    padding: 8px;
    border: 1px solid #ddd;
}
.result-box {
    background: #ffffff;
    color: #333;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    margin: 20px 0;
}
.header {
    text-align: center;
    font-size: 48px;
    font-weight: 700;
    color: #ffffff;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    margin-bottom: 30px;
}
.subheader {
    color: #ff8e53;
    font-size: 24px;
    font-weight: 600;
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
.fade-in {
    animation: fadeIn 1s ease-in;
}
.recommendation-box {
    background: #f9f9f9;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    color: #333;
}
.recommendation-box details summary {
    font-weight: bold;
    color: #ff6b6b;
    cursor: pointer;
}
.recommendation-box details p {
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Load Models
def load_model(file_path):
    try:
        return pickle.load(open(file_path, 'rb'))
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

disease_models = {d: {m: load_model(f'Models/{d}_{m}.sav') for m in ['svm', 'lr', 'rf']}
                 for d in ['diabetes', 'heart', 'ckd', 'parkinsons', 'lung_cancer', 'liver', 'stroke',
                           'alzheimers', 'pneumonia', 'thyroid', 'breast_cancer']}

# Disease Configurations (for input matching)
disease_configs = {
    'diabetes': {'features': ['Symptom_0', 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']},
    'heart': {'features': ['Symptom_0', 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']},
    'ckd': {'features': ['Symptom_0', 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'bp', 'albumin', 'creatinine', 'hemoglobin', 'urea']},
    'parkinsons': {'features': ['Symptom_0', 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'fo', 'fhi', 'flo', 'jitter', 'shimmer', 'nhr', 'hnr']},
    'lung_cancer': {'features': ['Symptom_0', 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'age', 'gender', 'smoking', 'yellow_fingers', 'anxiety', 'chronic_disease', 'coughing']},
    'liver': {'features': ['Symptom_0', 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'age', 'gender', 'total_bilirubin', 'albumin', 'alk_phos', 'sgpt']},
    'stroke': {'features': ['Symptom_0', 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status']},
    'alzheimers': {'features': ['Symptom_0', 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'age', 'mmse', 'cdr', 'education', 'memory_score']},
    'pneumonia': {'features': ['Symptom_0', 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'age', 'fever', 'cough_severity', 'resp_rate', 'o2_saturation']},
    'thyroid': {'features': ['Symptom_0', 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'age', 'tsh', 't3', 'tt4', 'gender']},
    'breast_cancer': {'features': ['Symptom_0', 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']}
}

# Input Function
def display_input(label, tooltip, key, min_value=0, max_value=None, is_int=False):
    if is_int:
        return st.number_input(label, key=key, help=tooltip, min_value=min_value, max_value=max_value, step=1, value=min_value, format="%d")
    return st.number_input(label, key=key, help=tooltip, min_value=float(min_value), max_value=float(max_value) if max_value else None, step=0.1, value=float(min_value))

# Prediction Function
def predict(models, input_data_df):
    results = {}
    for name, model in models.items():
        if model:
            pred = model.predict(input_data_df)[0]
            prob = model.predict_proba(input_data_df)[0].max() * 100
            results[name] = (pred, prob)
    return results

# Disease Name Mapping
disease_name_map = {
    "Diabetes": "diabetes",
    "Heart Disease": "heart",
    "Chronic Kidney Disease": "ckd",
    "Parkinson's Disease": "parkinsons",
    "Lung Cancer": "lung_cancer",
    "Liver Disease": "liver",
    "Stroke": "stroke",
    "Alzheimer's Disease": "alzheimers",
    "Pneumonia": "pneumonia",
    "Thyroid Disease": "thyroid",
    "Breast Cancer": "breast_cancer"
}

# Health Recommendations (Enhanced with predicted values)
def get_recommendations(disease, results, inputs):
    recommendations = []
    max_confidence = max(prob for _, prob in results.values())
    disease_key = disease_name_map[disease]
    expected_features = disease_configs[disease_key]['features']
    input_df = pd.DataFrame([inputs], columns=expected_features)

    if disease == "Diabetes" and max_confidence > 50:
        glucose = input_df['Glucose'].iloc[0]
        bmi = input_df['BMI'].iloc[0]
        recommendations.append(f"Monitor blood sugar levels regularly (current Glucose: {glucose} mg/dL).")
        if glucose > 150:
            recommendations.append("Adopt a low-sugar diet to manage high glucose levels.")
        if bmi > 30:
            recommendations.append(f"Consider a weight loss plan (current BMI: {bmi} is in obese range).")
        recommendations.append("Engage in 30 minutes of exercise daily.")
        recommendations.append("Consult a doctor for a fasting glucose test.")
        recommendations.append("Avoid processed carbohydrates to stabilize blood sugar.")
    elif disease == "Heart Disease" and max_confidence > 50:
        trestbps = input_df['trestbps'].iloc[0]
        chol = input_df['chol'].iloc[0]
        recommendations.append(f"Check blood pressure and cholesterol levels (BP: {trestbps} mmHg, Chol: {chol} mg/dL).")
        if trestbps > 120:
            recommendations.append("Reduce salt intake to lower high blood pressure.")
        if chol > 200:
            recommendations.append("Follow a low-cholesterol diet (e.g., less fatty foods).")
        recommendations.append("Avoid smoking and manage stress.")
        recommendations.append("Schedule a cardiovascular check-up.")
        recommendations.append("Incorporate heart-healthy foods like oats and nuts.")
    elif disease == "Chronic Kidney Disease" and max_confidence > 50:
        creatinine = input_df['creatinine'].iloc[0]
        recommendations.append(f"Stay hydrated and monitor kidney function (current Creatinine: {creatinine} mg/dL).")
        if creatinine > 1.2:
            recommendations.append("Seek medical advice for elevated creatinine levels.")
        recommendations.append("Reduce protein intake if advised by a doctor.")
        recommendations.append("Get regular kidney function tests.")
        recommendations.append("Avoid over-the-counter painkillers like ibuprofen.")
    elif disease == "Parkinson's Disease" and max_confidence > 50:
        recommendations.append("Seek neurological evaluation for motor symptoms.")
        recommendations.append("Consider physical therapy to improve mobility.")
        recommendations.append("Monitor for tremors and consult a specialist.")
        recommendations.append("Join a support group for Parkinson’s patients.")
        recommendations.append("Maintain a balanced diet to support nerve health.")
    elif disease == "Lung Cancer" and max_confidence > 50:
        smoking = input_df['smoking'].iloc[0]
        recommendations.append("Schedule a chest X-ray or CT scan for early detection.")
        if smoking == 1:
            recommendations.append("Quit smoking immediately to reduce lung cancer risk.")
        recommendations.append("Avoid exposure to secondhand smoke.")
        recommendations.append("Consult a pulmonologist.")
        recommendations.append("Increase intake of antioxidants (e.g., fruits and vegetables).")
    elif disease == "Liver Disease" and max_confidence > 50:
        total_bilirubin = input_df['total_bilirubin'].iloc[0]
        recommendations.append(f"Limit alcohol consumption (current Bilirubin: {total_bilirubin} mg/dL).")
        if total_bilirubin > 1.2:
            recommendations.append("Get a liver function test due to elevated bilirubin.")
        recommendations.append("Maintain a balanced diet rich in vegetables.")
        recommendations.append("Avoid hepatotoxic substances.")
        recommendations.append("Consider a detox plan under medical supervision.")
    elif disease == "Stroke" and max_confidence > 50:
        hypertension = input_df['hypertension'].iloc[0]
        recommendations.append("Control blood pressure and cholesterol levels.")
        if hypertension == 1:
            recommendations.append("Monitor blood pressure regularly (currently high).")
        recommendations.append("Exercise 30 minutes daily to improve circulation.")
        recommendations.append("Seek immediate help if you experience sudden weakness.")
        recommendations.append("Reduce saturated fat intake to prevent plaque buildup.")
    elif disease == "Alzheimer's Disease" and max_confidence > 50:
        mmse = input_df['mmse'].iloc[0]
        recommendations.append(f"Engage in mental exercises (current MMSE: {mmse}).")
        if mmse < 24:
            recommendations.append("Consult a neurologist for memory assessment due to low MMSE.")
        recommendations.append("Maintain a healthy diet (e.g., Mediterranean diet).")
        recommendations.append("Stay socially active.")
        recommendations.append("Get adequate sleep to support brain health.")
    elif disease == "Pneumonia" and max_confidence > 50:
        fever = input_df['fever'].iloc[0]
        recommendations.append(f"Rest and stay hydrated (current Fever: {fever}°C).")
        if fever > 38:
            recommendations.append("See a doctor for possible antibiotics due to high fever.")
        recommendations.append("Avoid crowded places to prevent worsening.")
        recommendations.append("Monitor oxygen levels.")
        recommendations.append("Use a humidifier to ease breathing.")
    elif disease == "Thyroid Disease" and max_confidence > 50:
        tsh = input_df['tsh'].iloc[0]
        recommendations.append(f"Get a thyroid function test (current TSH: {tsh} mIU/L).")
        if tsh > 4.0 or tsh < 0.4:
            recommendations.append("Consult an endocrinologist for abnormal TSH levels.")
        recommendations.append("Ensure adequate iodine intake.")
        recommendations.append("Avoid stress to support thyroid health.")
        recommendations.append("Consider selenium-rich foods like Brazil nuts.")
    elif disease == "Breast Cancer" and max_confidence > 50:
        radius_mean = input_df['radius_mean'].iloc[0]
        recommendations.append(f"Schedule a mammogram or clinical exam (current Radius Mean: {radius_mean} mm).")
        if radius_mean > 15:
            recommendations.append("Monitor for lumps or changes due to abnormal tumor size.")
        recommendations.append("Perform regular self-exams.")
        recommendations.append("Consult an oncologist.")
        recommendations.append("Maintain a healthy weight to reduce risk.")
    
    if max_confidence < 50:
        recommendations.append("Your risk is low, but maintain a healthy lifestyle with regular check-ups.")
    
    return recommendations

# Sidebar Menu
with st.sidebar:
    try:
        st.image("logo.png", width=120)  # Use relative path
    except Exception as e:
        st.warning("Logo file 'logo.png' not found. Please add it to the project directory.")
        st.image("https://cdn-icons-png.flaticon.com/512/3094/3094858.png", width=120)  # Fallback image
    st.markdown("<h2 style='color: #ff8e53;'>HealthSense</h2>", unsafe_allow_html=True)
    page = option_menu("Menu", ["Diagnosis", "Symptom Checker", "Health Dashboard", "History", "About"],
                       icons=['stethoscope', 'search', 'dashboard', 'clock', 'info-circle'],
                       menu_icon="cast", default_index=0, styles={
        "container": {"padding": "0!important", "background-color": "#203a43"},
        "icon": {"color": "#ff8e53", "font-size": "20px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#ff6b6b"},
        "nav-link-selected": {"background-color": "#ff8e53"}
    })

# Diagnosis Page
if page == "Diagnosis":
    st.markdown("<div class='header fade-in'>HealthSense: Your Health Monitor</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        disease = st.selectbox("Select Disease", list(disease_name_map.keys()), key="disease_select")
    with col2:
        compare_models = st.checkbox("Compare Models", value=True)
    
    st.markdown("<div class='subheader'>Enter Patient Data</div>", unsafe_allow_html=True)
    input_cols = st.columns(3)
    inputs = []

    if disease == "Diabetes":
        with input_cols[0]:
            Pregnancies = display_input("Pregnancies", "Times pregnant", "preg", max_value=20, is_int=True)
            Glucose = display_input("Glucose (mg/dL)", "Blood sugar", "gluc", max_value=300)
        with input_cols[1]:
            BloodPressure = display_input("BP (mmHg)", "Systolic BP", "bp_d", max_value=200)
            SkinThickness = display_input("Skin Thickness (mm)", "Triceps", "skin", max_value=100)
            Insulin = display_input("Insulin (mu U/ml)", "Insulin level", "ins", max_value=500)
        with input_cols[2]:
            BMI = display_input("BMI", "Body Mass Index", "bmi_d", max_value=70)
            DPF = display_input("Diabetes Pedigree", "Genetic factor", "dpf", max_value=3)
            Age = display_input("Age (years)", "Patient age", "age_d", max_value=120, is_int=True)
        inputs = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age, 5, 5, 5, 5, 5]

    elif disease == "Heart Disease":
        with input_cols[0]:
            age = display_input("Age (years)", "Patient age", "age_h", max_value=120, is_int=True)
            sex = display_input("Sex (1=M, 0=F)", "Gender", "sex", max_value=1, is_int=True)
            cp = display_input("Chest Pain (0-3)", "Pain type", "cp", max_value=3, is_int=True)
            trestbps = display_input("Resting BP (mmHg)", "Systolic BP", "trestbps", max_value=200)
        with input_cols[1]:
            chol = display_input("Cholesterol (mg/dL)", "Serum", "chol", max_value=600)
            fbs = display_input("Fasting BS > 120 (1=Y, 0=N)", "Blood sugar", "fbs", max_value=1, is_int=True)
            restecg = display_input("Resting ECG (0-2)", "ECG", "restecg", max_value=2, is_int=True)
            thalach = display_input("Max HR (bpm)", "Peak HR", "thalach", max_value=220)
        with input_cols[2]:
            exang = display_input("Exercise Angina (1=Y, 0=N)", "Angina", "exang", max_value=1, is_int=True)
            oldpeak = display_input("ST Depression", "Exercise", "oldpeak", max_value=10)
            slope = display_input("ST Slope (0-2)", "Slope", "slope", max_value=2, is_int=True)
            ca = display_input("Vessels (0-3)", "Fluoroscopy", "ca", max_value=3, is_int=True)
            thal = display_input("Thalassemia (0-2)", "Thal", "thal", max_value=2, is_int=True)
        inputs = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, 5, 5, 5, 5, 5]

    elif disease == "Chronic Kidney Disease":
        with input_cols[0]:
            bp = display_input("BP (mmHg)", "Systolic", "bp_ckd", max_value=200)
            albumin = display_input("Albumin (0-5)", "Protein", "alb", max_value=5, is_int=True)
        with input_cols[1]:
            creatinine = display_input("Creatinine (mg/dL)", "Kidney", "creat", max_value=15)
            hemoglobin = display_input("Hemoglobin (g/dL)", "Blood", "hemo", max_value=20)
        with input_cols[2]:
            urea = display_input("Urea (mg/dL)", "Kidney waste", "urea", max_value=50)
        inputs = [bp, albumin, creatinine, hemoglobin, urea, 5, 5, 5, 5, 5]

    elif disease == "Parkinson's Disease":
        with input_cols[0]:
            fo = display_input("Fo (Hz)", "Fundamental freq", "fo", max_value=500)
            fhi = display_input("Fhi (Hz)", "Max freq", "fhi", max_value=600)
        with input_cols[1]:
            flo = display_input("Flo (Hz)", "Min freq", "flo", max_value=300)
            jitter = display_input("Jitter (%)", "Freq variation", "jitter", max_value=1)
        with input_cols[2]:
            shimmer = display_input("Shimmer (%)", "Amp variation", "shimmer", max_value=1)
            nhr = display_input("NHR", "Noise-to-harmonics", "nhr", max_value=1)
            hnr = display_input("HNR", "Harmonics-to-noise", "hnr", max_value=50)
        inputs = [fo, fhi, flo, jitter, shimmer, nhr, hnr, 5, 5, 5, 5, 5]

    elif disease == "Lung Cancer":
        with input_cols[0]:
            age = display_input("Age (years)", "Patient age", "age_l", max_value=120, is_int=True)
            gender = display_input("Gender (1=M, 0=F)", "Gender", "gender", max_value=1, is_int=True)
            smoking = display_input("Smoking (1=Y, 0=N)", "Smoking", "smoke", max_value=1, is_int=True)
        with input_cols[1]:
            yellow_fingers = display_input("Yellow Fingers (1=Y, 0=N)", "Discoloration", "yellow", max_value=1, is_int=True)
            anxiety = display_input("Anxiety (1=Y, 0=N)", "Anxiety", "anx", max_value=1, is_int=True)
        with input_cols[2]:
            chronic_disease = display_input("Chronic Disease (1=Y, 0=N)", "Other conditions", "chronic", max_value=1, is_int=True)
            coughing = display_input("Coughing (1=Y, 0=N)", "Cough", "cough", max_value=1, is_int=True)
        inputs = [age, gender, smoking, yellow_fingers, anxiety, chronic_disease, coughing, 5, 5, 5, 5, 5]

    elif disease == "Liver Disease":
        with input_cols[0]:
            age = display_input("Age (years)", "Patient age", "age_liv", max_value=120, is_int=True)
            gender = display_input("Gender (1=M, 0=F)", "Gender", "gen_liv", max_value=1, is_int=True)
        with input_cols[1]:
            total_bilirubin = display_input("Total Bilirubin (mg/dL)", "Liver pigment", "bilirubin", max_value=10)
            albumin = display_input("Albumin (g/dL)", "Protein", "alb_liv", max_value=6)
        with input_cols[2]:
            alk_phos = display_input("Alk Phosphatase (U/L)", "Enzyme", "alk", max_value=300)
            sgpt = display_input("SGPT (U/L)", "Liver enzyme", "sgpt", max_value=100)
        inputs = [age, gender, total_bilirubin, albumin, alk_phos, sgpt, 5, 5, 5, 5, 5]

    elif disease == "Stroke":
        with input_cols[0]:
            age = display_input("Age (years)", "Patient age", "age_s", max_value=120, is_int=True)
            hypertension = display_input("Hypertension (1=Y, 0=N)", "High BP", "hyper", max_value=1, is_int=True)
        with input_cols[1]:
            heart_disease = display_input("Heart Disease (1=Y, 0=N)", "Heart condition", "heart_s", max_value=1, is_int=True)
            avg_glucose_level = display_input("Avg Glucose (mg/dL)", "Blood sugar", "gluc_s", max_value=300)
        with input_cols[2]:
            bmi = display_input("BMI", "Body Mass Index", "bmi_s", max_value=70)
            smoking_status = display_input("Smoking (0-2)", "0=Never, 1=Former, 2=Current", "smoke_s", max_value=2, is_int=True)
        inputs = [age, hypertension, heart_disease, avg_glucose_level, bmi, smoking_status, 5, 5, 5, 5, 5]

    elif disease == "Alzheimer's Disease":
        with input_cols[0]:
            age = display_input("Age (years)", "Patient age", "age_a", max_value=120, is_int=True)
            mmse = display_input("MMSE (0-30)", "Mental status", "mmse", max_value=30, is_int=True)
        with input_cols[1]:
            cdr = display_input("CDR (0-3)", "Dementia rating", "cdr", max_value=3)
            education = display_input("Education (years)", "Years of education", "edu", max_value=20, is_int=True)
        with input_cols[2]:
            memory_score = display_input("Memory Score (0-10)", "Memory test", "mem", max_value=10)
        inputs = [age, mmse, cdr, education, memory_score, 5, 5, 5, 5, 5]

    elif disease == "Pneumonia":
        with input_cols[0]:
            age = display_input("Age (years)", "Patient age", "age_p", max_value=120, is_int=True)
            fever = display_input("Fever (°C)", "Body temp", "fever", min_value=35, max_value=42)
        with input_cols[1]:
            cough_severity = display_input("Cough Severity (0-3)", "Cough intensity", "cough_s", max_value=3, is_int=True)
            resp_rate = display_input("Resp Rate (breaths/min)", "Breathing rate", "resp", max_value=60, is_int=True)
        with input_cols[2]:
            o2_saturation = display_input("O2 Saturation (%)", "Oxygen level", "o2", max_value=100)
        inputs = [age, fever, cough_severity, resp_rate, o2_saturation, 5, 5, 5, 5, 5]

    elif disease == "Thyroid Disease":
        with input_cols[0]:
            age = display_input("Age (years)", "Patient age", "age_t", max_value=120, is_int=True)
            tsh = display_input("TSH (mIU/L)", "Thyroid hormone", "tsh", max_value=10)
        with input_cols[1]:
            t3 = display_input("T3 (ng/dL)", "Triiodothyronine", "t3", max_value=3)
            tt4 = display_input("TT4 (µg/dL)", "Total thyroxine", "tt4", max_value=150)
        with input_cols[2]:
            gender = display_input("Gender (1=M, 0=F)", "Gender", "gen_t", max_value=1, is_int=True)
        inputs = [age, tsh, t3, tt4, gender, 5, 5, 5, 5, 5]

    elif disease == "Breast Cancer":
        with input_cols[0]:
            radius_mean = display_input("Radius Mean (mm)", "Tumor size", "rad", max_value=28)
            texture_mean = display_input("Texture Mean", "Tumor texture", "tex", max_value=40)
        with input_cols[1]:
            perimeter_mean = display_input("Perimeter Mean (mm)", "Tumor perimeter", "per", max_value=190)
            area_mean = display_input("Area Mean (mm²)", "Tumor area", "area", max_value=2500)
        with input_cols[2]:
            smoothness_mean = display_input("Smoothness Mean", "Tumor smoothness", "smooth", max_value=0.15)
        inputs = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, 5, 5, 5, 5, 5]

    # Convert inputs to DataFrame with feature names
    disease_key = disease_name_map[disease]
    expected_features = disease_configs[disease_key]['features']
    input_df = pd.DataFrame([inputs], columns=expected_features)

    if st.button("Predict Now"):
        with st.spinner("Analyzing Health Data..."):
            models = disease_models[disease_key]
            results = predict(models, input_df)
            # Store for dashboard and history (single entry with all models)
            st.session_state.latest_inputs = inputs
            st.session_state.latest_disease = disease
            st.session_state.latest_results = results
            if 'history' not in st.session_state:
                st.session_state.history = []
            # Store one entry with all model results
            history_entry = [disease, ", ".join([f"{m.upper()}: {p if p == 1 else 'Negative'} ({prob:.2f}%)" for m, (p, prob) in results.items()]),
                             datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            st.session_state.history.append(history_entry)
        
        st.markdown("<div class='result-box fade-in'>", unsafe_allow_html=True)
        st.subheader("Diagnosis Results")
        if compare_models:
            df = pd.DataFrame([(m.upper(), "Positive" if p == 1 else "Negative", prob) for m, (p, prob) in results.items()],
                              columns=["Model", "Prediction", "Confidence"])
            st.dataframe(df)
            fig = px.bar(df, x="Model", y="Confidence", color="Prediction", title="Model Confidence Comparison",
                         color_discrete_map={"Positive": "#ff6b6b", "Negative": "#00cc99"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            pred, prob = results['rf']
            st.write(f"**Result**: {'Positive' if pred == 1 else 'Negative'} (Confidence: {prob:.2f}%)")
        st.write("**Disclaimer**: Consult a healthcare professional for accurate diagnosis.")
        st.markdown("</div>", unsafe_allow_html=True)

# Symptom Checker Page
elif page == "Symptom Checker":
    st.markdown("<div class='header fade-in'>Symptom Analyzer</div>", unsafe_allow_html=True)
    symptoms = st.multiselect("Select Symptoms", ["Fever", "Cough", "Fatigue", "Shortness of Breath", "Chest Pain", "Tremors",
                                                  "Yellow Skin", "High BP", "Memory Loss", "Headache", "Weight Gain"],
                              key="symptoms")
    if st.button("Analyze Symptoms"):
        possible_conditions = {
            "Fever": ["Pneumonia", "Thyroid Disease"], "Cough": ["Lung Cancer", "Pneumonia"],
            "Fatigue": ["Heart Disease", "CKD", "Liver Disease"], "Shortness of Breath": ["Heart Disease", "Lung Cancer"],
            "Chest Pain": ["Heart Disease"], "Tremors": ["Parkinson's Disease"], "Yellow Skin": ["Liver Disease"],
            "High BP": ["Stroke", "CKD"], "Memory Loss": ["Alzheimer's Disease"], "Headache": ["Stroke"],
            "Weight Gain": ["Thyroid Disease"]
        }
        matches = set()
        for sym in symptoms:
            matches.update(possible_conditions.get(sym, []))
        st.markdown("<div class='result-box fade-in'>", unsafe_allow_html=True)
        st.subheader("Possible Conditions")
        if matches:
            st.write(", ".join(matches))
        else:
            st.write("No matches found. Seek medical advice.")
        st.markdown("</div>", unsafe_allow_html=True)

# Health Dashboard Page
elif page == "Health Dashboard":
    st.markdown("<div class='header fade-in'>Health Dashboard</div>", unsafe_allow_html=True)
    st.write("Overview of your latest health predictions and recommendations.")

    if 'latest_results' not in st.session_state or not st.session_state.latest_results:
        st.warning("No recent prediction data available. Run a diagnosis first.")
    else:
        latest_disease = st.session_state.latest_disease
        latest_results = st.session_state.latest_results
        latest_inputs = st.session_state.latest_inputs
        st.markdown(f"<div class='subheader'>Latest Prediction: {latest_disease}</div>", unsafe_allow_html=True)

        # Display latest results
        df_latest = pd.DataFrame([(m.upper(), "Positive" if p == 1 else "Negative", prob) for m, (p, prob) in latest_results.items()],
                                 columns=["Model", "Prediction", "Confidence"])
        st.dataframe(df_latest)

        # Gauge chart for highest confidence
        max_confidence = max(prob for _, prob in latest_results.values())
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=max_confidence,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Max Confidence (%)"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#ff6b6b"},
                   'steps': [
                       {'range': [0, 50], 'color': "#00cc99"},
                       {'range': [50, 100], 'color': "#ff6b6b"}],
                   'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 70}}))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Health Recommendations
        st.markdown("<div class='subheader'>Health Recommendations</div>", unsafe_allow_html=True)
        recommendations = get_recommendations(latest_disease, latest_results, latest_inputs)
        if recommendations:
            for rec in recommendations:
                st.markdown(f"<div class='recommendation-box'><details><summary>{rec}</summary><p>Consult a healthcare provider for personalized advice.</p></details></div>", unsafe_allow_html=True)
        else:
            st.write("No specific recommendations at this time. Maintain a healthy lifestyle.")

        # Historical Summary
        if 'history' in st.session_state and st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history, columns=["Disease", "Model Predictions", "Date"])
            st.markdown("<div class='subheader'>Prediction History Summary</div>", unsafe_allow_html=True)
            st.dataframe(history_df.tail(5))  # Show last 5 entries
        else:
            st.write("No history data available yet.")

# History Page
elif page == "History":
    st.markdown("<div class='header fade-in'>Prediction History</div>", unsafe_allow_html=True)
    if 'history' not in st.session_state:
        st.session_state.history = []
    if st.session_state.history:
        # Ensure unique entries by checking existing dates
        unique_history = []
        seen_dates = set()
        for entry in st.session_state.history:
            if entry[2] not in seen_dates:
                unique_history.append(entry)
                seen_dates.add(entry[2])
        st.dataframe(pd.DataFrame(unique_history, columns=["Disease", "Model Predictions", "Date"]))
    else:
        st.write("No predictions recorded yet.")

# About Page
elif page == "About":
    st.markdown("<div class='header fade-in'>About HealthSense</div>", unsafe_allow_html=True)
    st.write("""
    HealthSense is an AI-powered medical diagnosis tool featuring:
    - Prediction for 11 diseases using SVM, Logistic Regression, and Random Forest.
    - Interactive visualizations, symptom analysis, and health dashboard with recommendations.
    - Modern, engaging interface.
    
    **Note**: For educational purposes only—not a replacement for medical professionals.
    """)