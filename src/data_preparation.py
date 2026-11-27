
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split

from data_registration import validate_dataset


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"

PROCESSED_DIR = PROJECT_DIR / "processed_data"

TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

TARGET_COLUMN = "ProdTaken"

TEST_SIZE = 0.20
RANDOM_STATE = 42

DROP_COLUMNS = [
    "CustomerID",
    "DurationOfPitch",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "ProductPitched",
    "TypeofContact"
]

FEATURE_COLUMNS = [
    "Age",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]


# --------------------------------------------------
# 2. Load Dataset
# --------------------------------------------------

def load_dataset(data_path):

    print("\nLoading registered dataset...")

    df = validate_dataset(data_path)

    print("Original Dataset Shape:", df.shape)

    return df


# --------------------------------------------------
# 3. Clean Dataset
# --------------------------------------------------

def clean_dataset(df):

    print("\nStarting data cleaning...")

    df = df.copy()

    original_rows = len(df)

    # Standardize categorical values
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:

        df[column] = df[column].str.strip()

        df[column] = df[column].replace(
            "", np.nan
        )

    # Correct inconsistent gender values
    df["Gender"] = df["Gender"].replace({
        "Fe Male": "Female",
        "Fe male": "Female",
        "female": "Female",
        "male": "Male"
    })

    # Standardize marital status
    df["MaritalStatus"] = df[
        "MaritalStatus"
    ].replace({
        "Unmarried": "Single"
    })

    # Remove duplicate records
    df.drop_duplicates(inplace=True)

    print(
        "Duplicate Records Removed:",
        original_rows - len(df)
    )

    # Ensure customer identifiers are valid
    if df["CustomerID"].isna().any():

        raise ValueError(
            "Missing CustomerID values found."
        )

    # Prevent customer overlap between train and test
    if df["CustomerID"].duplicated().any():

        raise ValueError(
            "Repeated CustomerIDs found. "
            "Investigate before splitting the dataset."
        )

    # Validate target column
    if df[TARGET_COLUMN].isna().any():

        raise ValueError(
            "Target column contains missing values."
        )

    if set(df[TARGET_COLUMN].unique()) != {0, 1}:

        raise ValueError(
            "Target must contain both binary classes."
        )

    # Remove unnecessary columns
    df.drop(columns=DROP_COLUMNS, inplace=True)

    # Retain only approved features and target
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]]

    print("Removed Columns:", DROP_COLUMNS)

    # Report missing values
    # Imputation will be performed during model training
    print("\nMissing Values After Cleaning:")
    print(df.isna().sum())

    print("\nCleaned Dataset Shape:", df.shape)

    return df


# --------------------------------------------------
# 4. Split Dataset
# --------------------------------------------------

def split_dataset(df):

    print("\nSplitting dataset...")

    X = df.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    train_df = pd.concat(
        [X_train, y_train],
        axis=1
    )

    test_df = pd.concat(
        [X_test, y_test],
        axis=1
    )

    print("Training Dataset Shape:", train_df.shape)
    print("Testing Dataset Shape:", test_df.shape)

    print("\nTraining Target Distribution (%):")

    print(
        y_train.value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nTesting Target Distribution (%):")

    print(
        y_test.value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    return train_df, test_df


# --------------------------------------------------
# 5. Save Prepared Datasets
# --------------------------------------------------

def save_datasets(train_df, test_df):

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        TRAIN_PATH,
        index=False
    )

    test_df.to_csv(
        TEST_PATH,
        index=False
    )

    print("\nTraining Dataset Saved:", TRAIN_PATH)

    print("Testing Dataset Saved:", TEST_PATH)


# --------------------------------------------------
# 6. Main Function
# --------------------------------------------------

def main():

    print("=" * 60)
    print("DATA PREPARATION PIPELINE")
    print("=" * 60)

    # Load registered dataset
    df = load_dataset(DATA_PATH)

    # Clean dataset
    df_clean = clean_dataset(df)

    # Split dataset
    train_df, test_df = split_dataset(df_clean)

    # Save datasets
    save_datasets(train_df, test_df)

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":

    main()
