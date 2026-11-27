
# ==========================================================
# TOURISM PACKAGE PREDICTION - MODEL BUILDING
# ==========================================================

# ==========================================================
# 1. Import required libraries
# ==========================================================

import json
import joblib
import pandas as pd
import numpy as np

from pathlib import Path

import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ==========================================================
# 2. Project configuration
# ==========================================================

if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parent.parent
else:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "processed_data"
MODEL_DIR = BASE_DIR / "models"
MLFLOW_DIR = BASE_DIR / "mlruns"

TARGET_COLUMN = "ProdTaken"
RANDOM_STATE = 42

# Create output directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 3. MLflow experiment configuration
# ==========================================================

def configure_mlflow():

    mlflow_db = MLFLOW_DIR / "mlflow.db"

    artifact_dir = MLFLOW_DIR / "artifacts"

    artifact_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(
        f"sqlite:///{mlflow_db.as_posix()}"
    )

    experiment_name = "Tourism Package Prediction"

    experiment = mlflow.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:

        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_dir.as_uri()
        )

    mlflow.set_experiment(experiment_name)

    print("MLflow Tracking URI:",
          mlflow.get_tracking_uri())

    print("MLflow Experiment:", experiment_name)


# ==========================================================
# 4. Load prepared datasets
# ==========================================================

def load_data():

    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    if not train_path.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {train_path}"
        )

    if not test_path.is_file():
        raise FileNotFoundError(
            f"Testing dataset not found: {test_path}"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    print("Training data:", X_train.shape)
    print("Testing data:", X_test.shape)

    print("\nTraining target distribution:")
    print(y_train.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test


# ==========================================================
# 5. Build preprocessing pipeline
# ==========================================================

def build_preprocessor(X_train):

    numerical_features = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# ==========================================================
# 6. Define models and hyperparameter grids
# ==========================================================

def get_models():

    models = {

        "RandomForest": {

            "model": RandomForestClassifier(
                random_state=RANDOM_STATE
            ),

            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 5],
                "classifier__min_samples_leaf": [1, 2],
                "classifier__class_weight": [
                    None, "balanced"
                ]
            }
        },

        "GradientBoosting": {

            "model": GradientBoostingClassifier(
                random_state=RANDOM_STATE
            ),

            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 5],
                "classifier__subsample": [0.8, 1.0]
            }
        }
    }

    return models


# ==========================================================
# 7. Evaluate model
# ==========================================================

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {

        "accuracy": accuracy_score(
            y_test, y_pred
        ),

        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),

        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),

        "f1_score": f1_score(
            y_test, y_pred, zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_test, y_prob
        )
    }

    print("\nClassification Report:")

    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


# ==========================================================
# 8. Reusable MLflow model logging function
# ==========================================================

def log_mlflow_model(model, X_sample, model_name="model"):

    signature = infer_signature(
        X_sample,
        model.predict(X_sample)
    )

    return mlflow.sklearn.log_model(
        sk_model=model,
        name=model_name,
        serialization_format="cloudpickle",
        signature=signature,
        input_example=X_sample
    )


# ==========================================================
# 9. Train models and track experiments
# ==========================================================

def train_models(X_train, X_test, y_train, y_test):

    preprocessor = build_preprocessor(X_train)

    models = get_models()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    best_model = None
    best_model_name = None
    best_score = -np.inf
    best_params = None

    model_results = []

    for model_name, config in models.items():

        print(f"\nTraining {model_name}...")

        with mlflow.start_run(run_name=model_name):

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", config["model"])
                ]
            )

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=config["params"],
                scoring="f1",
                cv=cv,
                n_jobs=-1,
                verbose=1,
                refit=True
            )

            grid_search.fit(X_train, y_train)

            trained_model = grid_search.best_estimator_

            cv_score = grid_search.best_score_

            # Log model configuration
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("scoring", "f1")
            mlflow.log_param("random_state", RANDOM_STATE)

            # Log best hyperparameters
            mlflow.log_params(grid_search.best_params_)

            # Log cross-validation performance
            mlflow.log_metric("cv_f1_score", cv_score)

            # Log trained model
            log_mlflow_model(
                trained_model,
                X_train.head(5),
                "model"
            )

            # Log all GridSearchCV results
            cv_results = pd.DataFrame(
                grid_search.cv_results_
            )

            mlflow.log_text(
                cv_results.to_csv(index=False),
                "grid_search_results.csv"
            )

            print("Best Parameters:")
            print(grid_search.best_params_)

            print(f"CV F1 Score: {cv_score:.4f}")

            model_results.append({
                "model": model_name,
                "cv_f1_score": cv_score
            })

            # Select best model using CV F1-score
            if cv_score > best_score:

                best_score = cv_score
                best_model = trained_model
                best_model_name = model_name
                best_params = grid_search.best_params_

    results_df = pd.DataFrame(model_results)

    results_df = results_df.sort_values(
        by="cv_f1_score",
        ascending=False
    )

    print("\nModel Comparison:")
    print(results_df.to_string(index=False))

    results_df.to_csv(
        MODEL_DIR / "model_comparison.csv",
        index=False
    )

    return (
        best_model,
        best_model_name,
        best_score,
        best_params
    )


# ==========================================================
# 10. Evaluate and save best model
# ==========================================================

def save_best_model(
    best_model,
    best_model_name,
    best_score,
    best_params,
    X_test,
    y_test
):

    with mlflow.start_run(run_name="Best_Model_Evaluation"):

        metrics = evaluate_model(
            best_model,
            X_test,
            y_test
        )

        mlflow.log_param(
            "selected_model",
            best_model_name
        )

        mlflow.log_params(best_params)

        mlflow.log_metric(
            "cv_f1_score",
            best_score
        )

        mlflow.log_metrics(metrics)

        # Save complete preprocessing and model pipeline
        model_path = MODEL_DIR / "best_model.joblib"

        joblib.dump(best_model, model_path)

        mlflow.log_artifact(str(model_path))

        # Log best model to MLflow
        log_mlflow_model(
            best_model,
            X_test.head(5),
            "best_model"
        )

        # Save model metadata
        metadata = {

            "model_name": best_model_name,

            "cv_f1_score": float(best_score),

            "best_parameters": best_params,

            "test_metrics": {
                key: float(value)
                for key, value in metrics.items()
            }
        }

        metadata_path = MODEL_DIR / "model_metadata.json"

        with open(metadata_path, "w") as file:

            json.dump(metadata, file, indent=4)

        mlflow.log_artifact(str(metadata_path))

        mlflow.log_artifact(
            str(MODEL_DIR / "model_comparison.csv")
        )

        print("\nBest Model:", best_model_name)

        print("Best CV F1 Score:", best_score)

        print("\nTest Metrics:")

        for metric, value in metrics.items():

            print(f"{metric}: {value:.4f}")

        print("\nModel saved successfully:", model_path)


# ==========================================================
# 11. Main execution function
# ==========================================================

def main():

    print("=" * 60)

    print("TOURISM PACKAGE PREDICTION - MODEL BUILDING")

    print("=" * 60)

    configure_mlflow()

    X_train, X_test, y_train, y_test = load_data()

    (
        best_model,
        best_model_name,
        best_score,
        best_params
    ) = train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    save_best_model(
        best_model,
        best_model_name,
        best_score,
        best_params,
        X_test,
        y_test
    )

    print("\nModel building completed successfully.")


# ==========================================================
# 12. Entry point
# ==========================================================

if __name__ == "__main__":
    main()
