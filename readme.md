# Vietnamese Real Estate Analysis & Price Prediction
**Course:** IT4142E – Introduction to Data Science | **School:** SoICT, HUST

An end-to-end Data Science pipeline to crawl, standardize, model, and predict residential real estate prices across Vietnam with zero data leakage and reproducible grouped evaluation.

---

## 1. Project Overview & Pipeline Highlights

- **Multi-Source Crawling:** Scrapes real estate listings across 50+ Vietnamese provinces from `batdongsan.com.vn` (Selenium with anti-ban mechanisms) and `muaban.net`.
- **Auditable Preprocessing:** Reconstructs canonical datasets from raw CSVs with full traceability, deduplication tracking, URL conflict isolation, and strict domain validation.
- **Zero Data Leakage:** Eliminates target-derived features (`price_per_m2` banned from $X$), fits imputers/scalers strictly within cross-validation training folds.
- **Grouped Cross-Validation:** Clusters identical/similar listings by spatial-structural fingerprints (`candidate_group_id`) to ensure listings of the same property never leak across train and test folds.
- **End-to-End Scikit-Learn Pipeline:** Features `ColumnTransformer`, `SimpleImputer(add_indicator=True)` capturing MNAR (Missing Not At Random) patterns, `OneHotEncoder(handle_unknown='ignore')`, and `TransformedTargetRegressor` for automatic log-scale training and original-scale inference.
- **Interactive Streamlit Application:** Provides an interactive 4-tab dashboard for data inspection, EDA distributions, model evaluation metrics, and single/batch valuation.

---

## 2. Directory Structure

```text
Real-Estate-Analysis/
├── data/
│   ├── raw/                        # Immutable raw scraped listings (batdongsan, muaban)
│   └── processed/                  # Auditable cleaned datasets, train/test splits & manifests
│       ├── canonical_all.csv       # Complete canonical dataset (10,173 listings)
│       ├── X_train.csv, y_train.csv# Clean training set (7,728 listings)
│       ├── X_test.csv, y_test.csv  # Holdout test set (1,966 listings)
│       ├── split_manifest.csv      # Candidate group assignments for grouped CV
│       └── quality_report.json     # Data audit summary, SHA256 hashes & statistics
├── data_preprocessed/
│   ├── rebuild_dataset.py          # Deterministic ETL pipeline from raw CSVs
│   ├── test_rebuild.py             # Unit tests for text parsers & split integrity
│   └── README.md                   # Preprocessing specifications & data rules
├── data_scraping/
│   ├── scraper.py                  # Selenium dynamic crawler for batdongsan.com.vn
│   └── utils.py                    # Crawler utility functions
├── EDA_and_Data_Visualization/
│   └── eda_data_visualize.ipynb    # Exploratory Data Analysis notebook
├── modelling/
│   ├── model.joblib                # Serialized best Scikit-learn Pipeline artifact
│   ├── metadata.json               # Model specs, feature names, package versions & data hashes
│   └── selection.json              # Automated CV selection audit log
├── reports/
│   ├── metrics.csv                 # Comparative metrics (MAE, RMSE, R²) on original price scale
│   ├── cv_folds.csv                # 5-fold grouped split mapping
│   ├── segment_errors.csv          # Error breakdown by housing category, city & price band
│   ├── largest_errors.csv          # Top 30 worst predictions for diagnostic analysis
│   ├── test_diagnostics.png        # Actual vs. Predicted & Residual scatter plots
│   └── KET_QUA_THUC_NGHIEM.md      # Detailed experimental results & empirical findings
├── tests/
│   ├── test_crawler_csv.py         # Tests for CSV writing integrity & schema consistency
│   ├── test_pipeline.py            # Tests for input validation, train-only imputation & artifact parity
│   └── test_streamlit.py           # Streamlit app automated UI tests
├── ml_pipeline.py                  # Scikit-learn Pipeline architecture & preprocessing definition
├── train.py                        # Model training, grouped GridSearchCV & holdout evaluation
├── run_pipeline.py                 # One-click master script to run ETL, train & all tests
├── streamlit_app.py                # Streamlit interactive web application
└── requirements.txt                # Pinned dependency requirements
```

---

## 3. Quickstart & Installation

### Step 1: Environment Setup
```bash
# Clone the repository
git clone git@github.com:lhd10fls/Real-Estate-Analysis.git
cd Real-Estate-Analysis

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Master Pipeline (One Command)
To rebuild the dataset from raw files, run data tests, train all models with GroupKFold CV, evaluate holdout metrics, and execute all tests:
```bash
python run_pipeline.py
```

### Step 3: Launch Interactive Web Application
```bash
streamlit run streamlit_app.py
```
The app features:
1. **Dữ liệu (Data):** Filterable table of verified listings with CSV export.
2. **Phân tích (EDA):** Interactive price distributions and property type breakdowns.
3. **Đánh giá (Evaluation):** Holdout performance metrics, residual plots, and segment error analysis.
4. **Dự đoán (Predict):** Real-time single-property valuation and batch prediction on filtered datasets.

---

## 4. Individual Pipeline Steps

If you prefer executing stages individually:

```bash
# 1. Rebuild datasets from raw data
python data_preprocessed/rebuild_dataset.py

# 2. Run data parsing tests
python -m unittest discover -s data_preprocessed -p test_rebuild.py -v

# 3. Train models and generate reports
python train.py

# 4. Run full test suite
python -m unittest discover -s tests -v
```

---

## 5. Experimental Results (Evaluated on Raw Price Scale - Billion VND)

All models are evaluated on the held-out test set ($N=1{,}966$) across the original currency scale (Billion VND):

| Model | 5-Fold CV MAE (tỷ VND) | Test MAE (tỷ VND) | Test RMSE (tỷ VND) | Test $R^2$ (Giá gốc) | Median Absolute Error (tỷ) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Median** | $15.00 \pm 0.87$ | $15.47$ | $39.87$ | $-0.090$ | $5.525$ |
| **Ridge Regression** | $20.40 \pm 11.02$ | $13.43$ | $94.41$ | $-5.112$ | $2.759$ |
| **Random Forest (Selected)** | $\mathbf{7.43 \pm 0.52}$ | $\mathbf{7.06}$ | $\mathbf{21.92}$ | $\mathbf{0.670}$ | $\mathbf{1.713}$ |

### Key Findings:
- **Random Forest** achieved the lowest CV MAE and was automatically selected as the deployment model. It reduces MAE by **$54.4\%$** compared to the baseline, with a median error of only **$1.71$ billion VND**.
- **Segment Breakdown:** Prediction accuracy is highest on **Apartments** (Median Error: $0.78$B) and **Alley Houses** (Median Error: $1.01$B), while luxury villas and street-front properties exhibit higher variance due to extreme right-skewed valuations.
