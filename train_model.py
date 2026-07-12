"""Compare baseline models, evaluate transportability, and save all results."""
from pathlib import Path
import json
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, LeaveOneGroupOut
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, brier_score_loss
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

DATA = Path("data/processed/heart_disease_analysis_ready.csv")
MODEL_OUTPUT = Path("models/logistic_regression_pipeline.joblib")
SUMMARY_OUTPUT = Path("models/modeling_summary.json")
COMPARISON_OUTPUT = Path("reports/model_comparison.csv")
TRANSPORT_OUTPUT = Path("reports/leave_one_dataset_out.csv")

NUMERIC = ["age", "trestbps", "chol", "thalch", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
FEATURES = NUMERIC + CATEGORICAL

def specificity(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(tn / (tn + fp)) if tn + fp else 0.0

def build_pipeline(model) -> Pipeline:
    prep = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler())
        ]), NUMERIC),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), CATEGORICAL)
    ])
    return Pipeline([("preprocessor", prep), ("classifier", model)])

def main() -> None:
    df = pd.read_csv(DATA)
    X = df[FEATURES]
    y = df["disease_present"].eq("Yes").astype(int)
    groups = df["dataset"].astype(str)
    strata = groups + "|" + y.astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=strata
    )

    models = {
        "Dummy baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", solver="liblinear", random_state=42
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, class_weight="balanced",
            random_state=42, n_jobs=1
        ),
        "Gradient boosting": GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.05, max_depth=2, random_state=42
        ),
    }

    scoring = {
        "roc_auc": "roc_auc", "pr_auc": "average_precision",
        "balanced_accuracy": "balanced_accuracy", "f1": "f1",
        "recall": "recall", "precision": "precision",
        "neg_brier": "neg_brier_score",
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    fitted = {}
    for name, estimator in models.items():
        pipe = build_pipeline(estimator)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe

        probability = pipe.predict_proba(X_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()

        rows.append({
            "model": name,
            "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "cv_roc_auc_sd": float(scores["test_roc_auc"].std()),
            "cv_pr_auc_mean": float(scores["test_pr_auc"].mean()),
            "cv_pr_auc_sd": float(scores["test_pr_auc"].std()),
            "cv_balanced_accuracy_mean": float(scores["test_balanced_accuracy"].mean()),
            "cv_f1_mean": float(scores["test_f1"].mean()),
            "cv_recall_mean": float(scores["test_recall"].mean()),
            "cv_precision_mean": float(scores["test_precision"].mean()),
            "cv_brier_mean": float(-scores["test_neg_brier"].mean()),
            "holdout_roc_auc": float(roc_auc_score(y_test, probability)),
            "holdout_pr_auc": float(average_precision_score(y_test, probability)),
            "holdout_balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
            "holdout_accuracy": float(accuracy_score(y_test, prediction)),
            "holdout_sensitivity": float(recall_score(y_test, prediction)),
            "holdout_specificity": specificity(y_test, prediction),
            "holdout_precision": float(precision_score(y_test, prediction, zero_division=0)),
            "holdout_f1": float(f1_score(y_test, prediction)),
            "holdout_brier": float(brier_score_loss(y_test, probability)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(COMPARISON_OUTPUT, index=False)

    best = comparison.sort_values("cv_pr_auc_mean", ascending=False).iloc[0]
    logistic = comparison.loc[comparison["model"].eq("Logistic regression")].iloc[0]
    selected_name = str(best["model"])
    if selected_name != "Logistic regression" and (
        best["cv_pr_auc_mean"] - logistic["cv_pr_auc_mean"] <= 0.02
    ):
        selected_name = "Logistic regression"

    selected_model = fitted[selected_name]
    dump(selected_model, MODEL_OUTPUT)

    logo_rows = []
    logo = LeaveOneGroupOut()
    for train_index, test_index in logo.split(X, y, groups):
        held_out = groups.iloc[test_index].iloc[0]
        pipe = build_pipeline(LogisticRegression(
            max_iter=3000, class_weight="balanced", solver="liblinear", random_state=42
        ))
        pipe.fit(X.iloc[train_index], y.iloc[train_index])
        probability = pipe.predict_proba(X.iloc[test_index])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        y_group = y.iloc[test_index]

        logo_rows.append({
            "held_out_dataset": held_out,
            "n": int(len(test_index)),
            "disease_prevalence": float(y_group.mean()),
            "roc_auc": float(roc_auc_score(y_group, probability)),
            "pr_auc": float(average_precision_score(y_group, probability)),
            "balanced_accuracy": float(balanced_accuracy_score(y_group, prediction)),
            "sensitivity": float(recall_score(y_group, prediction, zero_division=0)),
            "specificity": specificity(y_group, prediction),
        })

    pd.DataFrame(logo_rows).to_csv(TRANSPORT_OUTPUT, index=False)

    selected_row = comparison.loc[comparison["model"].eq(selected_name)].iloc[0].to_dict()
    summary = {
        "selected_model": selected_name,
        "selection_rule": "Highest cross-validated PR-AUC; prefer logistic regression if within 0.02.",
        "features": FEATURES,
        "excluded_from_predictors": [
            "id", "num", "disease_present", "dataset", "ca", "thal", "audit columns"
        ],
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "selected_model_results": selected_row,
        "all_model_results": comparison.to_dict(orient="records"),
        "leave_one_dataset_out": logo_rows,
        "threshold": 0.50,
        "clinical_deployment_approved": False,
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
