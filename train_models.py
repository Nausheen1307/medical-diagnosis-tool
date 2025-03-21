import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('Models', exist_ok=True)

# Dataset Generators
def create_diabetes_data(n_samples=3000):
    data = {**{f'Symptom_{i}': np.random.uniform(0, 10, n_samples) for i in range(5)},
            'Pregnancies': np.random.randint(0, 15, n_samples), 'Glucose': np.random.uniform(50, 200, n_samples),
            'BloodPressure': np.random.uniform(50, 120, n_samples), 'SkinThickness': np.random.uniform(0, 50, n_samples),
            'Insulin': np.random.uniform(0, 300, n_samples), 'BMI': np.random.uniform(15, 50, n_samples),
            'DiabetesPedigreeFunction': np.random.uniform(0, 2.5, n_samples), 'Age': np.random.randint(18, 90, n_samples),
            'Outcome': np.random.choice([0, 1], n_samples, p=[0.65, 0.35])}
    pd.DataFrame(data).to_csv('data/diabetes.csv', index=False)

def create_heart_data(n_samples=3000):
    data = {**{f'Symptom_{i}': np.random.uniform(0, 10, n_samples) for i in range(5)},
            'age': np.random.randint(20, 80, n_samples), 'sex': np.random.randint(0, 2, n_samples),
            'cp': np.random.randint(0, 4, n_samples), 'trestbps': np.random.uniform(90, 200, n_samples),
            'chol': np.random.uniform(100, 400, n_samples), 'fbs': np.random.randint(0, 2, n_samples),
            'restecg': np.random.randint(0, 3, n_samples), 'thalach': np.random.uniform(70, 200, n_samples),
            'exang': np.random.randint(0, 2, n_samples), 'oldpeak': np.random.uniform(0, 6, n_samples),
            'slope': np.random.randint(0, 3, n_samples), 'ca': np.random.randint(0, 4, n_samples),
            'thal': np.random.randint(0, 3, n_samples), 'target': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])}
    pd.DataFrame(data).to_csv('data/heart.csv', index=False)

def create_ckd_data(n_samples=3000):
    data = {**{f'Symptom_{i}': np.random.uniform(0, 10, n_samples) for i in range(5)},
            'bp': np.random.uniform(50, 180, n_samples), 'albumin': np.random.randint(0, 6, n_samples),
            'creatinine': np.random.uniform(0.5, 10, n_samples), 'hemoglobin': np.random.uniform(10, 18, n_samples),
            'urea': np.random.uniform(10, 50, n_samples), 'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])}
    pd.DataFrame(data).to_csv('data/ckd.csv', index=False)

def create_parkinsons_data(n_samples=3000):
    data = {**{f'Symptom_{i}': np.random.uniform(0, 10, n_samples) for i in range(5)},
            'fo': np.random.uniform(80, 300, n_samples), 'fhi': np.random.uniform(100, 500, n_samples),
            'flo': np.random.uniform(60, 250, n_samples), 'jitter': np.random.uniform(0, 0.1, n_samples),
            'shimmer': np.random.uniform(0, 0.1, n_samples), 'nhr': np.random.uniform(0, 0.5, n_samples),
            'hnr': np.random.uniform(10, 40, n_samples), 'target': np.random.choice([0, 1], n_samples, p=[0.75, 0.25])}
    pd.DataFrame(data).to_csv('data/parkinsons.csv', index=False)

def create_lung_cancer_data(n_samples=3000):
    data = {**{f'Symptom_{i}': np.random.uniform(0, 10, n_samples) for i in range(5)},
            'age': np.random.randint(20, 80, n_samples), 'gender': np.random.randint(0, 2, n_samples),
            'smoking': np.random.randint(0, 2, n_samples), 'yellow_fingers': np.random.randint(0, 2, n_samples),
            'anxiety': np.random.randint(0, 2, n_samples), 'chronic_disease': np.random.randint(0, 2, n_samples),
            'coughing': np.random.randint(0, 2, n_samples), 'target': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])}
    pd.DataFrame(data).to_csv('data/lung_cancer.csv', index=False)

def create_liver_data(n_samples=3000):
    data = {**{f'Symptom_{i}': np.random.uniform(0, 10, n_samples) for i in range(5)},
            'age': np.random.randint(20, 80, n_samples), 'gender': np.random.randint(0, 2, n_samples),
            'total_bilirubin': np.random.uniform(0.1, 5, n_samples), 'albumin': np.random.uniform(2, 5, n_samples),
            'alk_phos': np.random.uniform(20, 200, n_samples), 'sgpt': np.random.uniform(10, 100, n_samples),
            'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])}
    pd.DataFrame(data).to_csv('data/liver.csv', index=False)

def create_stroke_data(n_samples=3000):
    data = {**{f'Symptom_{i}': np.random.uniform(0, 10, n_samples) for i in range(5)},
            'age': np.random.randint(20, 90, n_samples), 'hypertension': np.random.randint(0, 2, n_samples),
            'heart_disease': np.random.randint(0, 2, n_samples), 'avg_glucose_level': np.random.uniform(50, 250, n_samples),
            'bmi': np.random.uniform(15, 50, n_samples), 'smoking_status': np.random.randint(0, 3, n_samples),
            'target': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])}
    pd.DataFrame(data).to_csv('data/stroke.csv', index=False)

def create_alzheimers_data(n_samples=3000):
    data = {**{f'Symptom_{i}': np.random.uniform(0, 10, n_samples) for i in range(5)},
            'age': np.random.randint(50, 100, n_samples), 'mmse': np.random.randint(0, 30, n_samples),
            'cdr': np.random.uniform(0, 3, n_samples), 'education': np.random.randint(0, 20, n_samples),
            'memory_score': np.random.uniform(0, 10, n_samples), 'target': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])}
    pd.DataFrame(data).to_csv('data/alzheimers.csv', index=False)

def create_pneumonia_data(n_samples=3000):
    data = {**{f'Symptom_{i}': np.random.uniform(0, 10, n_samples) for i in range(5)},
            'age': np.random.randint(0, 90, n_samples), 'fever': np.random.uniform(36, 42, n_samples),
            'cough_severity': np.random.randint(0, 4, n_samples), 'resp_rate': np.random.randint(12, 40, n_samples),
            'o2_saturation': np.random.uniform(80, 100, n_samples), 'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])}
    pd.DataFrame(data).to_csv('data/pneumonia.csv', index=False)

def create_thyroid_data(n_samples=3000):
    data = {**{f'Symptom_{i}': np.random.uniform(0, 10, n_samples) for i in range(5)},
            'age': np.random.randint(20, 80, n_samples), 'tsh': np.random.uniform(0.1, 10, n_samples),
            't3': np.random.uniform(0.5, 3, n_samples), 'tt4': np.random.uniform(50, 150, n_samples),
            'gender': np.random.randint(0, 2, n_samples), 'target': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])}
    pd.DataFrame(data).to_csv('data/thyroid.csv', index=False)

def create_breast_cancer_data(n_samples=3000):
    data = {**{f'Symptom_{i}': np.random.uniform(0, 10, n_samples) for i in range(5)},
            'radius_mean': np.random.uniform(6, 28, n_samples), 'texture_mean': np.random.uniform(10, 40, n_samples),
            'perimeter_mean': np.random.uniform(40, 190, n_samples), 'area_mean': np.random.uniform(150, 2500, n_samples),
            'smoothness_mean': np.random.uniform(0.05, 0.15, n_samples), 'target': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])}
    pd.DataFrame(data).to_csv('data/breast_cancer.csv', index=False)

# Training Function
def train_and_save_model(data_path, feature_cols, target_col, model_type, save_path):
    df = pd.read_csv(data_path)
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'svm':
        model = SVC(probability=True)
    elif model_type == 'lr':
        model = LogisticRegression(max_iter=1000)
    else:  # rf
        model = RandomForestClassifier(n_estimators=100)
    
    model.fit(X_train, y_train)
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"{model_type.upper()} Accuracy for {os.path.basename(data_path)}: {model.score(X_test, y_test):.2f}")

# Main Execution
if __name__ == "__main__":
    # Generate all datasets
    dataset_functions = [
        create_diabetes_data, create_heart_data, create_ckd_data, create_parkinsons_data,
        create_lung_cancer_data, create_liver_data, create_stroke_data, create_alzheimers_data,
        create_pneumonia_data, create_thyroid_data, create_breast_cancer_data
    ]
    for func in dataset_functions:
        func()
    
    # Define disease configurations after datasets exist
    disease_configs = {
        'diabetes': {'data': 'data/diabetes.csv', 'features': [col for col in pd.read_csv('data/diabetes.csv').columns if col != 'Outcome'], 'target': 'Outcome'},
        'heart': {'data': 'data/heart.csv', 'features': [col for col in pd.read_csv('data/heart.csv').columns if col != 'target'], 'target': 'target'},
        'ckd': {'data': 'data/ckd.csv', 'features': [col for col in pd.read_csv('data/ckd.csv').columns if col != 'target'], 'target': 'target'},
        'parkinsons': {'data': 'data/parkinsons.csv', 'features': [col for col in pd.read_csv('data/parkinsons.csv').columns if col != 'target'], 'target': 'target'},
        'lung_cancer': {'data': 'data/lung_cancer.csv', 'features': [col for col in pd.read_csv('data/lung_cancer.csv').columns if col != 'target'], 'target': 'target'},
        'liver': {'data': 'data/liver.csv', 'features': [col for col in pd.read_csv('data/liver.csv').columns if col != 'target'], 'target': 'target'},
        'stroke': {'data': 'data/stroke.csv', 'features': [col for col in pd.read_csv('data/stroke.csv').columns if col != 'target'], 'target': 'target'},
        'alzheimers': {'data': 'data/alzheimers.csv', 'features': [col for col in pd.read_csv('data/alzheimers.csv').columns if col != 'target'], 'target': 'target'},
        'pneumonia': {'data': 'data/pneumonia.csv', 'features': [col for col in pd.read_csv('data/pneumonia.csv').columns if col != 'target'], 'target': 'target'},
        'thyroid': {'data': 'data/thyroid.csv', 'features': [col for col in pd.read_csv('data/thyroid.csv').columns if col != 'target'], 'target': 'target'},
        'breast_cancer': {'data': 'data/breast_cancer.csv', 'features': [col for col in pd.read_csv('data/breast_cancer.csv').columns if col != 'target'], 'target': 'target'}
    }
    
    # Train models
    for disease, config in disease_configs.items():
        for model_type in ['svm', 'lr', 'rf']:
            save_path = f'Models/{disease}_{model_type}.sav'
            train_and_save_model(config['data'], config['features'], config['target'], model_type, save_path)