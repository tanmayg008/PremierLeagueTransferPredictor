# Premier League Transfer Value Predictor ⚽

A machine learning project that explores how player statistics, age, and other performance-related features can be used to predict the transfer market value of Premier League football players.

This is a personal learning project focused on developing practical machine learning skills and applying data science to football analytics.

## 🎯 Project Objectives

- Explore and preprocess football player datasets.
- Perform feature engineering to prepare data for machine learning.
- Train regression models to estimate player transfer values.
- Evaluate model performance using appropriate regression metrics.
- Investigate feature importance and improve model performance through hyperparameter tuning.

## 🛠️ Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib
- **Development Environment:** VS Code, Python virtual environment
- **Version Control:** Git and GitHub

## 📁 Project Structure

```text
PremierLeagueTransferPredictor/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── inspect_dataset.py
│   ├── load_dataset.py
│   ├── clean_dataset.py
│   ├── select_feature.py
│   ├── feature_engineering.py
│   ├── merge_transfer_values.py
│   ├── prepare_ml_data.py
│   ├── train_model.py
│   ├── improve_random_forest.py
│   └── hyperparameter_tuning.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Project Workflow

1. **Data Inspection:** Understand dataset structure, columns, and missing values.
2. **Data Cleaning:** Handle missing values and prepare consistent datasets.
3. **Feature Engineering:** Construct useful features for predicting transfer value.
4. **Dataset Preparation:** Prepare the machine learning dataset.
5. **Model Training:** Train regression models using Scikit-learn.
6. **Model Evaluation:** Measure performance using MAE, RMSE, and R².
7. **Hyperparameter Tuning:** Optimize the Random Forest model using cross-validation.
8. **Exploratory Data Analysis:** Investigate distributions, relationships, and feature importance.

## 🤖 Current Model Results

The current best-tuned Random Forest configuration was selected through hyperparameter tuning.

| Metric | Result |
|---|---:|
| Best cross-validation R² | 0.718 |
| Test R² | 0.532 |
| Test MAE | €11,707,714 |
| Test RMSE | €21,823,063 |

### Best Hyperparameters

- `n_estimators`: 200
- `max_depth`: 10
- `min_samples_split`: 2
- `min_samples_leaf`: 2

These results are from the current experimental dataset and model. They do not guarantee accurate predictions for every player or represent a validated real-world transfer valuation system.

The difference between cross-validation and test performance is an area for further investigation.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or a compatible version for the dependencies
- Git
- VS Code or another Python IDE

### 1. Clone the repository

```bash
git clone https://github.com/tanmayg008/PremierLeagueTransferPredictor.git
cd PremierLeagueTransferPredictor
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run a script

For example, to inspect the dataset:

```powershell
python src/inspect_dataset.py
```

Run scripts according to their dependencies and the workflow described above. The complete project pipeline is still under development.

## 📊 Current Status

- [x] Initial data inspection and preprocessing
- [x] Feature engineering and dataset preparation
- [x] Random Forest model training
- [x] Hyperparameter tuning and model evaluation
- [ ] Further exploratory data analysis
- [ ] Systematic comparison of multiple regression models
- [ ] Improved validation and error analysis
- [ ] Model packaging and prediction interface

## 🔬 Future Improvements

- Compare Random Forest with Linear Regression, Gradient Boosting, and other regression algorithms.
- Investigate data leakage and improve validation methodology.
- Analyze prediction errors across player positions, ages, and value ranges.
- Improve feature selection and model interpretability.
- Develop a reusable prediction pipeline.

## 👨‍💻 Author

**Tanmay Gosavi**

Artificial Intelligence and Data Science student interested in machine learning, data science, and sports analytics.

## ⚠️ Disclaimer

This project is for educational and experimental purposes. Predictions are model estimates, not official transfer valuations or financial advice. Ensure that source datasets and derived data can be redistributed before sharing them publicly.