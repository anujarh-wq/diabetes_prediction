# Diabetes Risk Prediction (Decision Tree)

A machine learning project that predicts whether a patient is likely to have diabetes from eight clinical measurements, using a Decision Tree classifier. It includes a multi-page Streamlit dashboard ("DiabetesAI") for risk assessment, analytics, and prediction logging.

> **Disclaimer:** This is an educational project, not a medical device. It must not be used for diagnosis or treatment decisions. Consult a qualified healthcare professional.

## Project Structure

```
.
├── diabetes.csv                              # Dataset (768 patients)
├── Day 58_Decision Tree Diabetes Data.ipynb  # Training, tuning, evaluation
├── app.py                                    # Streamlit dashboard
├── diabetes_model.pkl                        # Trained Decision Tree model
├── diabetes_features.pkl                     # Ordered list of 8 input features
├── prediction_history.csv                    # Log of predictions made in the app
└── README.md
```

## Dataset

`diabetes.csv` has 768 records and 8 numeric features plus a binary target. The data matches the layout of the Pima Indians Diabetes dataset. There are no null values, though some columns use `0` for missing measurements (e.g., Insulin, SkinThickness).

| Column | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure |
| `SkinThickness` | Triceps skinfold thickness |
| `Insulin` | Serum insulin |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Genetic diabetes risk score based on family history |
| `Age` | Age in years |
| `Outcome` | **Target**: 1 = diabetic, 0 = non-diabetic |

Class balance: 500 non-diabetic, 268 diabetic.

## Methodology

1. **Split**: 80% train / 20% test (`random_state=42`), giving 614 training and 154 test rows.
2. **Baseline**: an unconstrained `DecisionTreeClassifier` (gini).
3. **Tuning**: `GridSearchCV` (5-fold) over `criterion`, `max_depth` (2–9), `min_samples_split` (2–14), and `min_samples_leaf` (2–15).
4. **Final model**: `DecisionTreeClassifier(max_depth=7, min_samples_leaf=15, min_samples_split=2)`, saved to `diabetes_model.pkl`.

No feature scaling is applied, since decision trees do not need it.

## Results

| Model | Train accuracy | Test accuracy |
|---|---|---|
| Baseline (unpruned tree) | 100.0% | 75.3% |
| Tuned tree (depth 7, min leaf 15) | 83.7% | 72.7% |

**Tuned model, test set (154 rows)**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 (non-diabetic) | 0.83 | 0.73 | 0.77 |
| 1 (diabetic) | 0.60 | 0.73 | 0.66 |

Confusion matrix: `[[72, 27], [15, 40]]`

The unpruned tree overfits heavily (perfect training score). Tuning narrows the train–test gap, but test accuracy does not improve on this split, so results should be read as roughly 73–75%.

## Streamlit Dashboard

`app.py` provides eight pages:

- **Overview**: project summary and feature scope
- **Prediction Engine**: input form for the 8 measurements, with probability, risk category (Low / Moderate / High / Very High), gauge chart, and a downloadable CSV report
- **Analytics Hub**: summary statistics and charts drawn from logged predictions
- **Prediction History**: browse and download every prediction made (stored in `prediction_history.csv`)
- **Dataset Explorer**: inspect the training dataset
- **Model Architecture**: model configuration details
- **Clinical Insights** and **About**: background information and the disclaimer

## Getting Started

### Prerequisites

- Python 3.9+

### Installation

```bash
pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn jupyter
```

> For reliable pickle loading, use the same scikit-learn version the model was trained with.

### Run the dashboard

```bash
streamlit run app.py
```

Keep `diabetes_model.pkl`, `diabetes_features.pkl`, and `diabetes.csv` in the same folder as `app.py`. `prediction_history.csv` is created automatically if missing.

### Retrain the model

Run all cells in the notebook; it regenerates both `.pkl` files.

## Limitations & Future Improvements

- **Small dataset** (768 rows) with a single train/test split, so accuracy estimates are noisy. Use repeated or stratified cross-validation for more reliable numbers.
- **Zeros as missing values**: `Insulin`, `SkinThickness`, `BloodPressure`, and `Glucose` contain zeros that likely represent missing data. Imputing them (e.g., median) may improve the model.
- **Moderate recall and precision** on the diabetic class (0.73 / 0.60): expect false positives and missed cases.
- **Prediction history** in `prediction_history.csv` contains test entries from app usage; clear it before sharing or deploying, especially if real patient data could be entered.
- Try ensemble models (Random Forest, Gradient Boosting, XGBoost), `class_weight='balanced'`, and probability calibration.
- Add a `requirements.txt` with pinned versions.

## Tech Stack

Python · pandas · NumPy · scikit-learn · Plotly · Matplotlib · Seaborn · Streamlit
