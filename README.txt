# Crime Pattern Detection and Prediction in Greater Manchester

This project is part of my MSc dissertation at **Manchester Metropolitan University**. It applies **machine learning (ML), deep learning (DL), and forecasting models** to analyse and predict crime trends in Greater Manchester. An **interactive Streamlit dashboard** is developed to make insights accessible for both technical and non-technical users.  

---

## 🚀 Project Overview
- **Goal:** Detect, classify, and predict crime patterns using advanced data-driven approaches.  
- **Data Source:** Dataset available here: [📂 Download Dataset](https://stummuac-my.sharepoint.com/:f:/g/personal/24836279_stu_mmu_ac_uk/EjkBEu44WtJAqYaHHkLjWggBnrSyGF2UVvzUmjAFxyUVeQ?e=7G6gra)  
- **Output:**  
  - Comparative evaluation of ML and DL models.  
  - Forecasting of crime trends using Prophet.  
  - Interactive dashboard with filtering, visualisation, and automated PDF reports.  

---

## 🧰 Tools and Technologies
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, PyTorch, Prophet, GeoPandas, Folium, Matplotlib, Seaborn  
- **Frameworks:** Streamlit (dashboard), FPDF (automated reporting)  
- **Models Implemented:**
  - Classical ML: Logistic Regression, Random Forest, Bagging, XGBoost  
  - Deep Learning: LSTM, CNN, Transformer  
  - Forecasting: ARIMA, Prophet  

---

## 📊 Key Results
- **Best performing model:** XGBoost (Accuracy = 0.92, ROC-AUC = 0.978).  
- **Deep learning models** (LSTM, CNN, Transformer) captured temporal/spatial patterns but required higher computation.  
- **Prophet forecasting** revealed long-term and seasonal crime trends, including the impact of COVID-19.  
- **Dashboard** allows dynamic filtering, spatio-temporal visualisation, co-occurrence networks, and PDF export.  

---

## 🖥️ Dashboard Features
- Explore **temporal and spatial crime trends**.  
- Generate **forecasts** of crime rates.  
- Visualise **offence co-occurrence networks**.  
- Export findings as **PDF reports**.  

---

## ⚡ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/crime-prediction-dashboard.git
   cd crime-prediction-dashboard
   
## 📄 Run the dashboard
    streamlit run dashboard.py

## 📖 Reference
This project is based on my MSc dissertation:
 - Javaid, S. (2025). An Automated, Data-driven Approach to Crime Pattern Detection and Prediction. Manchester Metropolitan University.
