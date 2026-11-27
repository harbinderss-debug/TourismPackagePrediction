
import pandas as pd
from pathlib import Path

# Define project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Define dataset path
DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"

# Define expected dataset columns
EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]


def validate_dataset(data_path):
    """
    Validate the registered tourism dataset.

    Checks:
        1. Dataset file existence
        2. Dataset is not empty
        3. Expected columns are available
        4. Target variable contains valid values
        5. Print dataset summary
    """

    print("=" * 60)
    print("DATA REGISTRATION AND VALIDATION")
    print("=" * 60)

    # 1. Check whether the dataset exists
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    print(f"\nDataset found: {data_path}")

    # 2. Load the dataset
    df = pd.read_csv(data_path)

    # Check whether the dataset is empty
    if df.empty:
        raise ValueError("The dataset is empty.")

    print("Dataset loaded successfully.")

    # 3. Validate expected columns
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_columns = set(df.columns) - set(EXPECTED_COLUMNS)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    print("Column validation passed.")

    if extra_columns:
        print(
            f"Warning: Unexpected columns found: "
            f"{sorted(extra_columns)}"
        )

    # 4. Validate target variable
    if df["ProdTaken"].isna().any():
        raise ValueError(
            "Target variable contains missing values."
        )

    valid_target_values = {0, 1}
    actual_target_values = set(df["ProdTaken"].unique())

    if not actual_target_values.issubset(valid_target_values):
        raise ValueError(
            f"Invalid target values: {actual_target_values}"
        )

    if actual_target_values != valid_target_values:
        raise ValueError(
            "Dataset must contain both target classes."
        )

    print("Target variable validation passed.")

    # 5. Print dataset summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Number of Rows       : {df.shape[0]}")
    print(f"Number of Columns    : {df.shape[1]}")
    print(f"Total Missing Values : {df.isna().sum().sum()}")
    print(f"Duplicate Records    : {df.duplicated().sum()}")

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values by Column:")
    print(df.isna().sum())

    print("\nTarget Distribution:")
    print(df["ProdTaken"].value_counts().sort_index())

    print("\nTarget Distribution (%):")
    print(
        df["ProdTaken"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\n" + "=" * 60)
    print("DATA REGISTRATION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return df


# Main entry point
if __name__ == "__main__":
    validate_dataset(DATA_PATH)
