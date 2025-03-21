 # 🚀 HealthSense - Your Health Monitor

HealthSense is an **AI-powered** 🧠 health diagnosis tool built with **Streamlit** and **Machine Learning** 🤖. It predicts the likelihood of **11 diseases** (e.g., **Diabetes, Heart Disease, Chronic Kidney Disease**) using pre-trained models (**SVM, Logistic Regression, Random Forest**) and provides **personalized health recommendations**. With a sleek **Glassmorphism UI**, interactive visualizations 📊, and a **symptom checker**, MediPredict makes health insights accessible and engaging! 💡

---

## ✨ Features

✅ **Disease Prediction**: Predicts **11 diseases** with **confidence scores** using **SVM, Logistic Regression, and Random Forest** models.  
✅ **Symptom Checker**: Input your symptoms to identify possible conditions.  
✅ **Health Dashboard**: View **prediction results**, **confidence charts** (bar charts 📊, gauge charts ⏳), and **historical data**.  
✅ **Personalized Recommendations**: Get **tailored health advice** based on prediction outcomes.  
✅ **Modern UI** 🎨: **Glassmorphism** design, smooth animations, and a user-friendly interface.  
✅ **History Tracking** 📜: Stores your **prediction history** for future reference.  

---

## 📥 Installation

Follow these steps to set up and run **HealthSense** locally:

### 1️⃣ Clone the Repository:
```bash
 git clone https://github.com/Atul-kr07/medical-app.git
 cd medical-app
```

### 2️⃣ Install Dependencies:
Ensure you have **Python 3.13** installed, then run:
```bash
 pip install streamlit scikit-learn pandas numpy plotly streamlit-option-menu
```

### 3️⃣ Run the Application:
```bash
 streamlit run app.py
```

### 4️⃣ Access the App:
Open your browser and visit: **[http://localhost:8501](http://localhost:8502)** 🌐

---

## 🎯 Usage

🔹 **Diagnosis**: Select a disease (e.g., **Diabetes**), enter patient data (e.g., **Glucose, BMI**), and click **"Predict Now"** to view results.  
🔹 **Symptom Checker**: Input symptoms to identify potential conditions.  
🔹 **Health Dashboard**: View **prediction history**, **confidence charts**, and **health recommendations**.  
🔹 **History**: Review past predictions with timestamps.  
🔹 **About**: Learn more about the project and its purpose.  

---

## 📁 Project Structure
```bash
 medical-app/
│
├── app.py                  # Main application file
├── Models/                 # Directory for pre-trained model files (.sav)
├── logo.png                # Custom logo (optional)
├── screenshots/            # Directory for screenshots (optional)
├── .gitignore              # Git ignore file
└── README.md               # Project documentation.
```

---

## 🏗️ System Architecture

💻 **User Interface (Frontend)**: Built with **Streamlit**, featuring a sidebar menu, disease selection, input fields, and **Plotly visualizations**.  
⚙️ **Processing Layer**: Uses **scikit-learn** models for predictions and generates **customized recommendations**.  
📂 **Data Layer**: Stores **pre-trained models (.sav files)** and manages **session state** for **history tracking**.  

---

## 🛠️ Technologies Used

🔹 **Python 3.13** 🐍 - Core programming language.  
🔹 **Streamlit** 🎛️ - Web application framework.  
🔹 **Scikit-learn** 🤖 - Machine learning model predictions.  
🔹 **Pandas & NumPy** 📊 - Data manipulation & processing.  
🔹 **Plotly** 📈 - Interactive visualizations.  
🔹 **Streamlit Option Menu** 📂 - Custom sidebar navigation.

---

## ⚠️ Disclaimer

⚠️ **HealthSense is for educational purposes only!** It is **not a substitute** for professional **medical advice, diagnosis, or treatment**. Always consult a **healthcare professional** for **accurate diagnosis** and medical guidance. 🏥

---

## 📩 Contact

For any **questions** or **feedback**, feel free to reach out at 📧 **jilaninausheen13@gmail.com**

🌟 If you find this project useful, don’t forget to **star ⭐ the repository** on GitHub!

---

🚀 Stay **healthy & informed** with **HealthSense**! 💙


 
