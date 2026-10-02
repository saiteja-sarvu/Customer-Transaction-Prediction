# Customer Transaction Prediction

A machine learning project that predicts whether a customer will make a transaction using anonymized customer features. The project is implemented in Python using Jupyter Notebook.

## Project Overview

This project explores customer transaction data, performs exploratory data analysis (EDA), trains classification models, evaluates their performance, and generates transaction probability predictions for the provided test dataset.

**Problem type:** Binary classification  
**Target:** `target` (`0` = no transaction, `1` = transaction)  
**Notebook:** `notebooks/customer_transaction_prediction.ipynb`

## Dataset

The project uses the Santander Customer Transaction Prediction dataset format.

The local training file used in this project contains **20,000 rows and 202 columns**:
- `ID_code`: customer record identifier
- `target`: binary target
- `var_0` through `var_199`: 200 numerical input features

The local test file contains **20,000 rows and 201 columns** (the identifier and 200 features). The sample submission file contains 20,000 rows and two columns.

> Note: These are the dimensions of the local files used in this project. They differ from the full original Kaggle competition dataset.

### Data files

Place the downloaded files in `data/raw/`:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

The original files are kept in the `raw` directory. The `processed` directory is available for any derived datasets.

## Project Structure

```text
CUSTOMER_TRANSACTION_PREDICTION/
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   └── processed/
├── models/
│   └── logistic_regression.joblib
├── notebooks/
│   └── customer_transaction_prediction.ipynb
├── outputs/
│   └── customer_transaction_submission.csv
└── README.md
```

## Technologies Used

- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- joblib

## Workflow

1. **Load data:** Read the training, test, and sample submission CSV files with pandas.
2. **Explore data:** Inspect dimensions, column types, descriptive statistics, missing values, duplicate rows, duplicate IDs, and target distribution.
3. **Prepare features:** Remove `ID_code` and `target` from the training input features. Keep `target` as the prediction label.
4. **Split data:** Use an 80/20 train-validation split with `random_state=42` and stratification.
5. **Train models:** Train Logistic Regression with feature scaling and a Random Forest classifier.
6. **Evaluate models:** Compare accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC. Examine the confusion matrix and classification report.
7. **Explore thresholds:** Compare precision, recall, and F1-score at several probability thresholds.
8. **Train final model:** Fit the Logistic Regression pipeline on the full local training dataset.
9. **Generate predictions:** Predict transaction probabilities for the test dataset.
10. **Save artifacts:** Save the trained model and a CSV containing test IDs and predicted probabilities.

## Exploratory Data Analysis

The local training data checks produced the following results:

| Check | Result |
|---|---:|
| Training rows | 20,000 |
| Training columns | 202 |
| Input features | 200 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Duplicate `ID_code` values | 0 |
| Target 0 | 17,990 (89.95%) |
| Target 1 | 2,010 (10.05%) |

The target is imbalanced, with approximately 10% positive examples. Therefore, accuracy alone is not sufficient to assess model performance.

## Model Results

The following results were obtained on the 20% validation split using the default 0.5 prediction threshold.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.7834 | 0.9014 |
| Precision | 0.2865 | 0.5230 |
| Recall | 0.7754 | 0.2117 |
| F1-score | 0.4184 | 0.3014 |
| ROC-AUC | 0.8599 | 0.8090 |
| PR-AUC | 0.5004 | 0.3682 |

### Interpretation

- Logistic Regression achieved higher recall, F1-score, ROC-AUC, and PR-AUC on this validation split.
- Random Forest achieved higher accuracy and precision at the default threshold, but identified a smaller share of positive cases.
- Because the target is imbalanced, accuracy should be interpreted alongside recall, precision, F1-score, ROC-AUC, and PR-AUC.
- Threshold experiments showed that changing the classification threshold changes the precision-recall trade-off. The threshold should be selected based on the intended use case and evaluated without using the final test labels.

These are validation results from one split, not a guarantee of performance on new data.

## How to Run

### 1. Clone or download the project

```bash
git clone <your-github-repository-url>
cd CUSTOMER_TRANSACTION_PREDICTION
```

If you have not uploaded the project to GitHub yet, open the project folder directly in VS Code.

### 2. Create and activate a virtual environment (optional but recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel joblib
```

### 4. Add the dataset

Place `train.csv`, `test.csv`, and `sample_submission.csv` inside `data/raw/`.

### 5. Open Jupyter Notebook

From the project root, run:

```bash
jupyter notebook
```

Open `notebooks/customer_transaction_prediction.ipynb` and run the cells in order.

In VS Code, you can also open the notebook directly and select the Python kernel for your environment.

## Output Files

- `models/logistic_regression.joblib`: the fitted Logistic Regression pipeline, including its scaler.
- `outputs/customer_transaction_submission.csv`: test IDs and predicted transaction probabilities.

The submission CSV uses the columns `ID_code` and `target`. The `target` column contains probabilities rather than thresholded class labels, which is appropriate for ROC-AUC-based competition submissions.

## Limitations and Future Improvements

- The reported evaluation uses a single stratified train-validation split. Repeated stratified cross-validation can provide a more robust estimate.
- Threshold selection should be based on a clearly defined business objective, such as minimizing missed transactions or controlling false positives.
- Additional models and hyperparameter tuning can be explored.
- Model calibration and feature-importance analysis may provide further insight.
- Final performance should be assessed on a genuinely unseen test set or through the competition's evaluation process.

## Author

**Saiteja Sarvu**

---

This project is for learning and demonstrating a complete introductory classification workflow.
# Customer-Transaction-Prediction
