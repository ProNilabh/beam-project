import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

FEATURE_NAMES = [
    "Relative_Compactness",       # X1
    "Surface_Area",               # X2
    "Wall_Area",                  # X3
    "Roof_Area",                  # X4
    "Overall_Height",             # X5
    "Orientation",                # X6
    "Glazing_Area",               # X7
    "Glazing_Area_Distribution",  # X8
]

TARGET_NAMES = ["Heating_Load", "Cooling_Load"]  # Y1, Y2


def load_data(filepath: str = None) -> pd.DataFrame:
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "..", "data", "ENB2012_data.csv")

    df = pd.read_csv(filepath)

    rename_map = {f"X{i+1}": FEATURE_NAMES[i] for i in range(8)}
    rename_map["Y1"] = "Heating_Load"
    rename_map["Y2"] = "Cooling_Load"
    df = df.rename(columns=rename_map)

    return df


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
):
    X = df[FEATURE_NAMES].values
    y = df[TARGET_NAMES].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
