"""Shared feature specification for model training and inference.

train.py and prediction_tools.py must agree on the feature order and the
categorical encodings, otherwise the model is fed skewed inputs at predict
time. Both import from here — do not redefine encodings elsewhere.
"""

# Ordered feature columns as they appear in the dataset / Glue schema.
FEATURE_COLUMNS = [
    "width_um",
    "height_nm",
    "wavelength_nm",
    "polarization",
    "deposition_method",
    "etch_method",
    "cladding_material",
    "anneal_temp_C",
    "anneal_hours",
]

TARGET_COLUMN = "propagation_loss_dB_cm"

# Fixed integer encodings for categorical columns. Unknown values map to 0.
CATEGORICAL_ENCODINGS = {
    "polarization": {"TE": 0, "TM": 1},
    "deposition_method": {"LPCVD": 0, "PECVD": 1, "HDPCVD": 2},
    "etch_method": {"RIE": 0, "ICP-RIE": 1, "Wet": 2},
    "cladding_material": {"SiO2": 0, "Air": 1, "SiN": 2},
}


def encode_categorical(column: str, value: str) -> int:
    """Encode a categorical value with the shared fixed mapping."""
    return CATEGORICAL_ENCODINGS[column].get(str(value).strip(), 0)


def prepare_features_and_target(df):
    """Canonicalize column names and encode categorical features for training/evaluation.

    Returns `(X, y, missing_columns)` where `X` and `y` are `None` if any required
    columns are missing.
    """
    canonical = {c.lower(): c for c in FEATURE_COLUMNS + [TARGET_COLUMN]}
    df = df.rename(columns={c: canonical[c.lower()] for c in df.columns if c.lower() in canonical})

    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        return None, None, missing

    df = df.copy()
    for col, mapping in CATEGORICAL_ENCODINGS.items():
        df[col] = df[col].astype(str).str.strip().map(mapping).fillna(0).astype(int)

    return df[FEATURE_COLUMNS], df[TARGET_COLUMN], []
