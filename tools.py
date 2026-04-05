import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from crewai_tools import tool

_data_store = {}  # shared in-memory store across tools

@tool("load_csv")
def load_csv(filepath: str) -> str:
    """Load a CSV file and return basic info about it."""
    df = pd.read_csv(filepath)
    _data_store["raw"] = df
    return (
        f"Loaded CSV with shape {df.shape}\n"
        f"Columns: {list(df.columns)}\n"
        f"Dtypes:\n{df.dtypes.to_string()}\n"
        f"First 3 rows:\n{df.head(3).to_string()}"
    )

@tool("run_eda")
def run_eda(target_column: str) -> str:
    """Run exploratory data analysis on the loaded dataset."""
    df = _data_store.get("raw")
    if df is None:
        return "No data loaded. Use load_csv first."

    null_info = df.isnull().sum()
    null_pct = (null_info / len(df) * 100).round(2)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # Remove person_ID from cat_cols for reporting
    cat_cols_report = [c for c in cat_cols if c != "person_ID"]

    report = []
    report.append(f"Shape: {df.shape}")
    report.append(f"Target column: {target_column}")
    report.append(f"Numeric columns ({len(num_cols)}): {num_cols}")
    report.append(f"Categorical columns (excl. ID): {cat_cols_report}")
    report.append(f"\nMissing values:\n{null_pct[null_pct > 0].to_string() or 'None'}")
    report.append(f"\nTarget distribution:\n{df[target_column].value_counts().sort_index().to_string()}")
    report.append(f"\nDescriptive stats:\n{df[num_cols].describe().to_string()}")

    if len(num_cols) > 1:
        corr = (
            df[num_cols].corr()[target_column]
            if target_column in num_cols
            else df[num_cols].corr().iloc[0]
        )
        report.append(f"\nCorrelations with target:\n{corr.to_string()}")

    _data_store["target"] = target_column
    return "\n".join(report)

@tool("preprocess_data")
def preprocess_data(strategy: str = "median") -> str:
    """
    Preprocess the dataset: drop ID column, handle missing values,
    encode categoricals, scale numerics.
    strategy: 'median' or 'mean' for numeric imputation.
    """
    df = _data_store.get("raw", pd.DataFrame()).copy()
    target = _data_store.get("target")
    if df.empty or not target:
        return "No data/target found. Run EDA first."

    # Separate features and target
    X = df.drop(columns=[target])

    # ✅ Fix 1: Drop person_ID — it's an identifier, not a feature
    X = X.drop(columns=["person_ID"], errors="ignore")

    y = df[target].copy()

    # Encode target if categorical
    if y.dtype == "object":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target)
        _data_store["target_classes"] = list(le.classes_)

    # Fill missing values
    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(include="object").columns

    fill_val = X[num_cols].median() if strategy == "median" else X[num_cols].mean()
    X[num_cols] = X[num_cols].fillna(fill_val)
    X[cat_cols] = X[cat_cols].fillna("missing")

    # Encode remaining categoricals
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Scale numerics
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    _data_store["X"] = X
    _data_store["y"] = y

    return (
        f"Preprocessing complete.\n"
        f"Features used: {list(X.columns)}\n"
        f"Features shape: {X.shape}\n"
        f"Missing values remaining: {X.isnull().sum().sum()}\n"
        f"Encoded categoricals: {list(cat_cols)}\n"
        f"Scaled numerics: {list(num_cols)}"
    )

@tool("select_and_train_models")
def select_and_train_models(test_size: float = 0.2) -> str:
    """Train multiple classification models and return their accuracy scores."""
    X = _data_store.get("X")
    y = _data_store.get("y")
    if X is None or y is None:
        return "No preprocessed data found. Run preprocessing first."
    test_size = float(test_size)

    # ✅ Fix 2: Sample 20k rows for speed on large datasets
    if len(X) > 20000:
        sample_idx = X.sample(20000, random_state=42).index
        X = X.loc[sample_idx]
        y = y.loc[sample_idx]
        print("⚡ Sampled 20,000 rows for faster training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    _data_store["X_train"] = X_train
    _data_store["X_test"] = X_test
    _data_store["y_train"] = y_train
    _data_store["y_test"] = y_test

    models = {
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(random_state=42),
    }

    results = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = round(acc, 4)

    best_model_name = max(results, key=results.get)
    _data_store["best_model_name"] = best_model_name
    _data_store["results"] = results

    summary = "\n".join(
        [f"  {k}: {v:.4f}" for k, v in sorted(results.items(), key=lambda x: -x[1])]
    )
    return (
        f"Model Accuracies (trained on 20k sample):\n{summary}\n\n"
        f"Best model: {best_model_name} ({results[best_model_name]:.4f})"
    )

@tool("tune_best_model")
def tune_best_model(dummy: str = "") -> str:
    """Run GridSearchCV on the best model found during training."""
    best_name = _data_store.get("best_model_name")
    X_train = _data_store.get("X_train")
    X_test = _data_store.get("X_test")
    y_train = _data_store.get("y_train")
    y_test = _data_store.get("y_test")

    if not best_name:
        return "No best model found. Run model selection first."

    param_grids = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]},
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1, 0.2]},
        },
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=500, random_state=42),
            "params": {"C": [0.1, 1.0, 10.0]},
        },
        "SVM": {
            "model": SVC(random_state=42),
            "params": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
        },
    }

    config = param_grids.get(best_name)
    if not config:
        return f"No param grid defined for {best_name}."

    print(f"⚙️  Tuning {best_name} with GridSearchCV (cv=3)...")
    grid = GridSearchCV(
        config["model"], config["params"], cv=3, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    tuned_acc = accuracy_score(y_test, best_model.predict(X_test))
    base_acc = _data_store["results"][best_name]

    _data_store["tuned_model"] = best_model
    _data_store["tuned_accuracy"] = tuned_acc
    _data_store["best_params"] = grid.best_params_

    report = classification_report(y_test, best_model.predict(X_test))

    return (
        f"Tuning complete for {best_name}\n"
        f"Best params: {grid.best_params_}\n"
        f"Base accuracy:  {base_acc:.4f}\n"
        f"Tuned accuracy: {tuned_acc:.4f}\n"
        f"Improvement: {(tuned_acc - base_acc) * 100:+.2f}%\n\n"
        f"Classification Report:\n{report}"
    )

@tool("generate_report")
def generate_report(output_path: str = "automl_report.md") -> str:
    """Generate a final markdown report summarizing the entire AutoML pipeline."""
    target = _data_store.get("target", "N/A")
    results = _data_store.get("results", {})
    best_name = _data_store.get("best_model_name", "N/A")
    best_params = _data_store.get("best_params", {})
    tuned_acc = _data_store.get("tuned_accuracy", 0)
    classes = _data_store.get("target_classes", [])
    X = _data_store.get("X")
    df = _data_store.get("raw")

    model_table = "\n".join(
        [f"| {k} | {v:.4f} |" for k, v in sorted(results.items(), key=lambda x: -x[1])]
    )

    report = f"""# AutoML Pipeline Report

## 1. Dataset Overview
- **Rows:** {df.shape[0] if df is not None else 'N/A'}
- **Columns:** {df.shape[1] if df is not None else 'N/A'}
- **Target Column:** `{target}` (pain scale 1–8)
- **Target Classes:** {classes if classes else 'Numeric (1–8)'}
- **Features used:** `acc_x, acc_y, acc_z, eda, bvp, hr, temp`
- **Features after preprocessing:** {X.shape[1] if X is not None else 'N/A'}
- **Training sample size:** 20,000 rows (sampled from 96,000)

## 2. Model Comparison
| Model | Accuracy |
|-------|----------|
{model_table}

## 3. Best Model
- **Model:** `{best_name}`
- **Best Hyperparameters:** `{best_params}`
- **Tuned Accuracy:** `{tuned_acc:.4f}`

## 4. Recommendation
The best performing model is **{best_name}** with a tuned accuracy of **{tuned_acc:.4f}**.
{"Consider deploying this model after validating on the full 96k dataset." if tuned_acc > 0.7 else "Consider feature engineering (e.g., rolling stats on time-series signals) to improve performance further."}

---
*Report generated by AutoML CrewAI Agent*
"""

    with open(output_path, "w") as f:
        f.write(report)

    return f"Report saved to {output_path}\n\n{report}"