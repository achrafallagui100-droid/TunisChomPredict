# TunisChomPredict 🇹🇳

**TunisChomPredict** is a comprehensive Data Science and Machine Learning web application designed to predict and analyze graduate unemployment risks across different governorates in Tunisia. The app leverages historical unemployment data, synthesized risk indicators (ChomScore), and advanced algorithms (XGBoost Regressor & Classifier) to provide actionable insights for stakeholders, policymakers, and job seekers.

![App Dashboard Screenshot](https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/images/streamlit-logo.png) <!-- Update with your own screenshot -->

## 🚀 Features
- **Unemployment Rate Prediction:** Uses an XGBoost Regressor to predict unemployment percentages based on demographic, geographic, and temporal features.
- **Risk Assessment:** Uses an XGBoost Classifier to categorize unemployment risk into Low, Moderate, or Critical levels.
- **ChomScore Dashboard:** A tailored metric evaluating the difficulty of the job market for each region.
- **Interactive Visualizations:** Historical trends, feature importance, and 2026 quarterly forecasts powered by Plotly.
- **Regional Risk Overview Map:** A choropleth bubble map that interactively displays risk overview.

## 🛠️ Tech Stack
- Frontend: [Streamlit](https://streamlit.io/)
- Data Manipulation: `pandas`, `numpy`
- Machine Learning models: `xgboost`, `scikit-learn` (deployed via `pickle`)
- Data Visualization: `plotly`

## 🗂️ Project Structure
```text
TunisChomPredict/
│
├── app.py                                   # Main Streamlit dashboard application
├── TunisChomPredict_Notebook.ipynb          # Jupyter notebook for EDAs and model training
├── requirements.txt                         # Dependencies
├── tunisia_unemployment_4datasets_augmented.csv # Cleaned & augmented dataset
│
├── models/                                  # Pretrained ML models & data lookups
│   ├── xgboost_regressor.pkl
│   ├── xgboost_classifier.pkl
│   ├── label_encoder_gouvernorat.pkl
│   └── chomscore_table.csv
│
└── dataset with out merge/                  # Raw dataset prior to processing
```

## ⚙️ Installation & Usage

### 1. Clone or Download the repository
```bash
git clone https://github.com/your-username/TunisChomPredict.git
cd TunisChomPredict
```

### 2. Install Dependencies
Make sure you have Python 3.8+ installed. Install the requirements via pip:
```bash
pip install -r requirements.txt
```

### 3. Run the App
To start the Streamlit server locally, execute:
```bash
streamlit run app.py
```

## ✨ Contributors
- **Student:** Achref Allegui | GR5 DS

## 📄 License
This project is for educational and research purposes.
