"""
FIFA 19 Player Rating Prediction — Linear Regression
------------------------------------------------------

Predicts a player's FIFA 19 `Overall` rating from their attribute data
(age, potential, skill ratings, physical attributes, etc.) using Linear
Regression, and ranks which attributes matter most via permutation
importance.

This closely follows the original notebook's cleaning and feature
engineering steps (dropping the 26 position-specific rating columns,
encoding Real Face / Preferred Foot / simplified Position / "major
nation" / split Work Rate, then one-hot encoding the rest), re-run here
against the full dataset with a fixed random seed so the results below
are reproducible.

Note: `eli5.PermutationImportance` (used in the original notebook) is no
longer maintained and doesn't install cleanly on modern scikit-learn, so
this script uses `sklearn.inspection.permutation_importance`, which
measures the same thing (drop in R^2 when a feature is shuffled).

Usage:
    pip install -r requirements.txt
    python train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42


def load_and_clean(path="data.csv"):
    df = pd.read_csv(path)

    # Rows with a missing Height are missing almost every physical/skill
    # column too (verified: same row indices as missing Weight) — drop them.
    missing_height = df[df["Height"].isnull()].index.tolist()
    df = df.drop(df.index[missing_height])

    return df


def engineer_features(df):
    # Drop the 26 position-specific overall ratings (LS, ST, RS, ... RB) —
    # these are themselves derived from the base skill attributes and would
    # leak information about Overall.
    drop_cols = df.columns[28:54]
    df = df.drop(drop_cols, axis=1)
    df = df.drop(
        [
            "Unnamed: 0", "ID", "Photo", "Flag", "Club Logo", "Jersey Number",
            "Joined", "Special", "Loaned From", "Body Type", "Release Clause",
            "Weight", "Height", "Contract Valid Until", "Wage", "Value",
            "Name", "Club",
        ],
        axis=1,
    )
    df = df.dropna()

    def face_to_num(row):
        return 1 if row["Real Face"] == "Yes" else 0

    def right_footed(row):
        return 1 if row["Preferred Foot"] == "Right" else 0

    def simple_position(row):
        pos = row["Position"]
        if pos == "GK":
            return "GK"
        if pos in ("RB", "LB", "CB", "LCB", "RCB", "RWB", "LWB"):
            return "DF"
        if pos in ("LDM", "CDM", "RDM"):
            return "DM"
        if pos in ("LM", "LCM", "CM", "RCM", "RM"):
            return "MF"
        if pos in ("LAM", "CAM", "RAM", "LW", "RW"):
            return "AM"
        if pos in ("RS", "ST", "LS", "CF", "LF", "RF"):
            return "ST"
        return pos

    nat_counts = df.Nationality.value_counts()
    nat_list = nat_counts[nat_counts > 250].index.tolist()

    def major_nation(row):
        return 1 if row.Nationality in nat_list else 0

    df1 = df.copy()
    df1["Real_Face"] = df1.apply(face_to_num, axis=1)
    df1["Right_Foot"] = df1.apply(right_footed, axis=1)
    df1["Simple_Position"] = df1.apply(simple_position, axis=1)
    df1["Major_Nation"] = df1.apply(major_nation, axis=1)

    tempwork = df1["Work Rate"].str.split("/ ", n=1, expand=True)
    df1["WorkRate1"] = tempwork[0]
    df1["WorkRate2"] = tempwork[1]

    df1 = df1.drop(
        ["Work Rate", "Preferred Foot", "Real Face", "Position", "Nationality"],
        axis=1,
    )
    return df1


def main():
    df = load_and_clean()
    df1 = engineer_features(df)

    target = df1.Overall
    features = df1.drop(["Overall"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE
    )
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    # align columns in case a category only appears in one split
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Test rows: {len(X_test)}, Train rows: {len(X_train)}, Features: {X_train.shape[1]}")
    print(f"r2 score: {r2}")
    print(f"RMSE: {rmse}")

    print("\nComputing permutation importance (this can take a minute)...")
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    importances = pd.Series(result.importances_mean, index=X_test.columns)
    top15 = importances.sort_values(ascending=False).head(15)
    print("\nTop 15 features by permutation importance (mean drop in R^2 when shuffled):")
    print(top15.to_string())

    top15.to_csv("permutation_importance.csv", header=["importance"])
    with open("results.txt", "w") as f:
        f.write(f"r2 score: {r2}\nRMSE: {rmse}\n")
        f.write(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}, Features: {X_train.shape[1]}\n")


if __name__ == "__main__":
    main()
"""
FIFA 19 Player Rating Prediction — Linear Regression
------------------------------------------------------

Predicts a player's FIFA 19 `Overall` rating from their attribute data
(age, potential, skill ratings, physical attributes, etc.) using Linear
Regression, and ranks which attributes matter most via permutation
importance.

This closely follows the original notebook's cleaning and feature
engineering steps (dropping the 26 position-specific rating columns,
encoding Real Face / Preferred Foot / simplified Position / "major
nation" / split Work Rate, then one-hot encoding the rest), re-run here
against the full dataset with a fixed random seed so the results below
are reproducible.

Note: `eli5.PermutationImportance` (used in the original notebook) is no
longer maintained and doesn't install cleanly on modern scikit-learn, so
this script uses `sklearn.inspection.permutation_importance`, which
measures the same thing (drop in R^2 when a feature is shuffled).

Usage:
    pip install -r requirements.txt
    python train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42


def load_and_clean(path="data.csv"):
    df = pd.read_csv(path)

    # Rows with a missing Height are missing almost every physical/skill
    # column too (verified: same row indices as missing Weight) — drop them.
    missing_height = df[df["Height"].isnull()].index.tolist()
    df = df.drop(df.index[missing_height])

    return df


def engineer_features(df):
    # Drop the 26 position-specific overall ratings (LS, ST, RS, ... RB) —
    # these are themselves derived from the base skill attributes and would
    # leak information about Overall.
    drop_cols = df.columns[28:54]
    df = df.drop(drop_cols, axis=1)
    df = df.drop(
        [
            "Unnamed: 0", "ID", "Photo", "Flag", "Club Logo", "Jersey Number",
            "Joined", "Special", "Loaned From", "Body Type", "Release Clause",
            "Weight", "Height", "Contract Valid Until", "Wage", "Value",
            "Name", "Club",
        ],
        axis=1,
    )
    df = df.dropna()

    def face_to_num(row):
        return 1 if row["Real Face"] == "Yes" else 0

    def right_footed(row):
        return 1 if row["Preferred Foot"] == "Right" else 0

    def simple_position(row):
        pos = row["Position"]
        if pos == "GK":
            return "GK"
        if pos in ("RB", "LB", "CB", "LCB", "RCB", "RWB", "LWB"):
            return "DF"
        if pos in ("LDM", "CDM", "RDM"):
            return "DM"
        if pos in ("LM", "LCM", "CM", "RCM", "RM"):
            return "MF"
        if pos in ("LAM", "CAM", "RAM", "LW", "RW"):
            return "AM"
        if pos in ("RS", "ST", "LS", "CF", "LF", "RF"):
            return "ST"
        return pos

    nat_counts = df.Nationality.value_counts()
    nat_list = nat_counts[nat_counts > 250].index.tolist()

    def major_nation(row):
        return 1 if row.Nationality in nat_list else 0

    df1 = df.copy()
    df1["Real_Face"] = df1.apply(face_to_num, axis=1)
    df1["Right_Foot"] = df1.apply(right_footed, axis=1)
    df1["Simple_Position"] = df1.apply(simple_position, axis=1)
    df1["Major_Nation"] = df1.apply(major_nation, axis=1)

    tempwork = df1["Work Rate"].str.split("/ ", n=1, expand=True)
    df1["WorkRate1"] = tempwork[0]
    df1["WorkRate2"] = tempwork[1]

    df1 = df1.drop(
        ["Work Rate", "Preferred Foot", "Real Face", "Position", "Nationality"],
        axis=1,
    )
    return df1


def main():
    df = load_and_clean()
    df1 = engineer_features(df)

    target = df1.Overall
    features = df1.drop(["Overall"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE
    )
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    # align columns in case a category only appears in one split
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Test rows: {len(X_test)}, Train rows: {len(X_train)}, Features: {X_train.shape[1]}")
    print(f"r2 score: {r2}")
    print(f"RMSE: {rmse}")

    print("\nComputing permutation importance (this can take a minute)...")
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    importances = pd.Series(result.importances_mean, index=X_test.columns)
    top15 = importances.sort_values(ascending=False).head(15)
    print("\nTop 15 features by permutation importance (mean drop in R^2 when shuffled):")
    print(top15.to_string())

    top15.to_csv("permutation_importance.csv", header=["importance"])
    with open("results.txt", "w") as f:
        f.write(f"r2 score: {r2}\nRMSE: {rmse}\n")
        f.write(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}, Features: {X_train.shape[1]}\n")


if __name__ == "__main__":
    main()
