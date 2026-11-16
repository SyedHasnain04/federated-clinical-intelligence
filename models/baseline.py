"""Tabular baseline models for 24-hour ICU landmark mortality prediction.

Provides:
- TabularPreprocessor (median imputation and standardization fit on train only)
- LogisticRegressionModel (PyTorch linear model)
- MLPModel (PyTorch 2-hidden-layer neural network)
Both models implement a unified interface (fit, predict_proba, get_parameters, set_parameters)
compatible with centralized benchmarking and Flower federated aggregation.
"""

from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data.mock_schema import get_feature_columns

# Authoritative canonical 51-feature schema
CANONICAL_FEATURES: List[str] = get_feature_columns()
CANONICAL_FEATURE_DIM: int = len(CANONICAL_FEATURES)
if (
    not isinstance(CANONICAL_FEATURES, list)
    or len(CANONICAL_FEATURES) != 51
    or len(set(CANONICAL_FEATURES)) != 51
    or not all(isinstance(c, str) for c in CANONICAL_FEATURES)
):
    raise RuntimeError(
        f"Canonical feature schema from get_feature_columns() must contain exactly "
        f"51 unique string feature names, got {len(CANONICAL_FEATURES)}."
    )

# Non-feature identifiers, landmark metadata, temporal grouping, demographics, and labels
EXCLUDED_COLS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "landmark_time",
    "anchor_year_group",
    "in_hospital_mortality",
    "first_careunit",
    "gender",
    "age",
]


def validate_canonical_features(feature_cols: Optional[List[str]]) -> List[str]:
    """Validates that feature_cols matches CANONICAL_FEATURES exactly in length, content, and order.

    Parameters
    ----------
    feature_cols : Optional[List[str]]
        List or tuple of feature column names, or None to use CANONICAL_FEATURES.

    Returns
    -------
    List[str]
        Validated list of canonical feature names in authoritative order.

    Raises
    ------
    TypeError
        If feature_cols is not None, list, or tuple, or contains non-string elements.
    ValueError
        If feature_cols is a subset, superset, duplicate, or reordered variant of CANONICAL_FEATURES.
    """
    if feature_cols is None:
        return list(CANONICAL_FEATURES)

    if not isinstance(feature_cols, (list, tuple)):
        raise TypeError(
            f"feature_cols must be a list or tuple of column names, got {type(feature_cols).__name__}."
        )

    if not all(isinstance(c, str) for c in feature_cols):
        raise TypeError("All column names in feature_cols must be strings.")

    feature_list = list(feature_cols)

    if len(feature_list) != len(CANONICAL_FEATURES):
        raise ValueError(
            f"Invalid feature schema: expected exactly {len(CANONICAL_FEATURES)} canonical features, "
            f"got {len(feature_list)}. Subsets and supersets are not permitted; "
            "this project requires the authoritative canonical 51-feature schema."
        )

    if len(feature_list) != len(set(feature_list)):
        raise ValueError(
            "Invalid feature schema: feature_cols contains duplicate column names. "
            "This project requires the canonical 51-feature schema without duplicates."
        )

    if feature_list != CANONICAL_FEATURES:
        missing = [c for c in CANONICAL_FEATURES if c not in feature_list]
        unexpected = [c for c in feature_list if c not in CANONICAL_FEATURES]
        if missing or unexpected:
            raise ValueError(
                f"Invalid feature schema: feature_cols does not match the canonical 51-feature schema. "
                f"Missing {len(missing)} feature(s): {missing}; Unexpected {len(unexpected)} feature(s): {unexpected}. "
                "This project requires the canonical 51-feature schema."
            )
        raise ValueError(
            "Invalid feature schema: feature_cols must match the exact canonical 51-feature ordering. "
            "Reordered feature lists are not permitted; this project requires the canonical feature order."
        )

    return feature_list


def validate_batch_size(batch_size: Any) -> int:
    """Validates and returns batch_size as a strictly positive Python integer.

    Parameters
    ----------
    batch_size : Any
        Requested batch size.

    Returns
    -------
    int
        Validated positive Python int.

    Raises
    ------
    TypeError
        If batch_size is a bool, np.bool_, or not an instance of (int, np.integer).
    ValueError
        If batch_size is less than or equal to 0.
    """
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)):
        raise TypeError(
            f"batch_size must be a positive integer, got {type(batch_size).__name__} ({batch_size!r})."
        )
    val = int(batch_size)
    if val <= 0:
        raise ValueError(
            f"batch_size must be a positive integer (> 0), got {batch_size!r}."
        )
    return val


def validate_binary_labels(y: Any, context: str = "Labels") -> np.ndarray:
    """Validates that y is a one-dimensional array-like containing only binary values {0, 1}.

    Parameters
    ----------
    y : Any
        Target labels (e.g. pandas Series, numpy array, list).
    context : str, optional
        Label context for error messages (default 'Labels').

    Returns
    -------
    np.ndarray
        Validated finite 1D float32 numpy array with values in {0.0, 1.0}.

    Raises
    ------
    TypeError
        If y is None, has non-numeric/non-boolean dtype, or contains unsupported types.
    ValueError
        If y is empty, non-1D, contains non-finite values (NaN, Inf), or contains values outside {0, 1}.
    """
    prefix = f"{context}: " if context else ""
    if y is None:
        raise TypeError(f"{prefix}labels cannot be None; a one-dimensional array-like of binary values is required.")

    if isinstance(y, torch.Tensor):
        arr = y.cpu().detach().numpy()
    elif isinstance(y, pd.Series):
        if not (pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y)):
            raise TypeError(f"{prefix}labels have non-numeric dtype '{y.dtype}'.")
        arr = y.to_numpy()
    elif isinstance(y, pd.DataFrame):
        raise ValueError(
            f"{prefix}labels must be 1-dimensional, got DataFrame with shape {y.shape}."
        )
    elif isinstance(y, np.ndarray):
        arr = y
    elif isinstance(y, (list, tuple)):
        if any(isinstance(elem, (str, bytes, dict, set)) for elem in y):
            raise TypeError(f"{prefix}labels must have numeric or boolean dtype, got string/object elements.")
        try:
            arr = np.asarray(y)
        except Exception as e:
            raise TypeError(f"{prefix}cannot convert labels to numpy array: {e}")
    else:
        raise TypeError(f"{prefix}labels must be a 1D array-like, got {type(y).__name__}.")

    if arr.dtype.kind not in {"b", "i", "u", "f"}:
        raise TypeError(f"{prefix}labels must have numeric or boolean dtype, got '{arr.dtype}'.")

    if arr.ndim != 1:
        raise ValueError(
            f"{prefix}labels must be 1-dimensional, got {arr.ndim} dimensions with shape {arr.shape}."
        )

    if arr.size == 0:
        raise ValueError(f"{prefix}labels array is empty; at least 1 observation is required.")

    labels_f32 = arr.astype(np.float32)

    if not np.all(np.isfinite(labels_f32)):
        raise ValueError(f"{prefix}labels contain non-finite values (NaN or Inf).")

    unique_vals = np.unique(labels_f32)
    if not np.all(np.isin(unique_vals, (0.0, 1.0))):
        invalid = [float(v) for v in unique_vals if v not in (0.0, 1.0)]
        raise ValueError(
            f"{prefix}labels must contain only binary values in {{0, 1}}, got invalid value(s): {invalid}."
        )

    return labels_f32


class TabularPreprocessor:
    """Preprocesses tabular EHR features with median imputation and standardization.

    Fitted strictly on training data to prevent information leakage from val/test sets
    or across federated clients. Enforces the canonical 51-feature schema.
    """

    def __init__(self, feature_cols: Optional[List[str]] = None):
        self.feature_cols: List[str] = validate_canonical_features(feature_cols)
        self.medians_: Optional[pd.Series] = None
        self.means_: Optional[pd.Series] = None
        self.stds_: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        """Fits median imputation and standardization statistics on training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame containing all required feature columns.

        Returns
        -------
        TabularPreprocessor
            Fitted preprocessor instance.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")

        if df.columns.duplicated().any():
            dups = df.columns[df.columns.duplicated()].unique().tolist()
            raise ValueError(f"Input DataFrame contains duplicate column names: {dups}.")

        if len(df) < 2:
            raise ValueError(
                f"Cannot fit preprocessor on fewer than 2 rows (got {len(df)}). "
                "At least 2 observations are required to compute sample variance."
            )

        # Enforce canonical 51-feature schema validation (cannot be bypassed by modifying self.feature_cols)
        self.feature_cols = validate_canonical_features(self.feature_cols)

        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Training DataFrame is missing {len(missing)} required feature column(s): {missing}."
            )

        # Select features in canonical order; ignore additional unrelated columns
        sub = df[self.feature_cols].copy()

        for col in self.feature_cols:
            if pd.api.types.is_bool_dtype(sub[col]) or not pd.api.types.is_numeric_dtype(sub[col]):
                raise TypeError(
                    f"Feature column '{col}' has non-numeric or boolean dtype '{sub[col].dtype}'. "
                    "All feature columns must contain numeric data."
                )
            if sub[col].isna().all():
                raise ValueError(
                    f"Feature column '{col}' is entirely null (contains no valid observations)."
                )
            # Check for infinite values (NaNs are handled by median imputation)
            if np.isinf(sub[col].dropna().to_numpy()).any():
                raise ValueError(
                    f"Feature column '{col}' contains non-finite values (Inf or -Inf) "
                    "which cannot be safely imputed."
                )

        self.medians_ = sub.median()
        imputed = sub.fillna(self.medians_)
        self.means_ = imputed.mean()
        self.stds_ = imputed.std().replace(0, 1.0)

        # Validate fitted statistics are strictly finite and non-empty
        if not np.all(np.isfinite(self.medians_.to_numpy(dtype=np.float64))):
            bad = self.medians_[~np.isfinite(self.medians_.to_numpy(dtype=np.float64))].index.tolist()
            raise ValueError(f"Fitted medians contain non-finite values for feature(s): {bad}.")

        if not np.all(np.isfinite(self.means_.to_numpy(dtype=np.float64))):
            bad = self.means_[~np.isfinite(self.means_.to_numpy(dtype=np.float64))].index.tolist()
            raise ValueError(f"Fitted means contain non-finite values for feature(s): {bad}.")

        if not np.all(np.isfinite(self.stds_.to_numpy(dtype=np.float64))):
            bad = self.stds_[~np.isfinite(self.stds_.to_numpy(dtype=np.float64))].index.tolist()
            raise ValueError(f"Fitted standard deviations contain non-finite values for feature(s): {bad}.")

        if (self.stds_ <= 0).any():
            bad = self.stds_[self.stds_ <= 0].index.tolist()
            raise ValueError(f"Fitted standard deviations contain non-positive values for feature(s): {bad}.")

        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transforms input DataFrame using fitted training statistics.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing all fitted feature columns.

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            Transformed feature matrix (float32) and optional binary target array (float32).
        """
        if (
            self.medians_ is None
            or self.means_ is None
            or self.stds_ is None
            or self.feature_cols is None
        ):
            raise RuntimeError("Preprocessor must be fit before calling transform.")

        # Enforce canonical 51-feature schema validation (cannot be bypassed by modifying self.feature_cols)
        self.feature_cols = validate_canonical_features(self.feature_cols)

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")

        if df.columns.duplicated().any():
            dups = df.columns[df.columns.duplicated()].unique().tolist()
            raise ValueError(f"Input DataFrame contains duplicate column names: {dups}.")

        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame is missing {len(missing)} required feature column(s): {missing}."
            )

        # Select features in the fitted canonical order; ignore additional unrelated columns
        sub = df[self.feature_cols].copy()

        for col in self.feature_cols:
            if pd.api.types.is_bool_dtype(sub[col]) or not pd.api.types.is_numeric_dtype(sub[col]):
                raise TypeError(
                    f"Feature column '{col}' has non-numeric or boolean dtype '{sub[col].dtype}'. "
                    "All feature columns must contain numeric data."
                )
            if np.isinf(sub[col].dropna().to_numpy()).any():
                raise ValueError(
                    f"Feature column '{col}' contains non-finite values (Inf or -Inf) "
                    "which cannot be safely imputed."
                )

        imputed = sub.fillna(self.medians_)
        standardized = (imputed - self.means_) / self.stds_
        X = standardized.values.astype(np.float32)

        if not np.all(np.isfinite(X)):
            raise ValueError("Transformed feature matrix contains non-finite values (NaN or Inf).")

        y = None
        if "in_hospital_mortality" in df.columns:
            y = validate_binary_labels(
                df["in_hospital_mortality"],
                context="Target column 'in_hospital_mortality'",
            )

        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.fit(df).transform(df)


def validate_fitted_preprocessor(
    preprocessor: Any,
    context: str = "Preprocessor",
) -> TabularPreprocessor:
    """Validates that a preprocessor exists, is a TabularPreprocessor, is fitted, and matches canonical schema.

    Parameters
    ----------
    preprocessor : Any
        Object to validate.
    context : str, optional
        Label for error messages (e.g. 'Server' or 'Client 0').

    Returns
    -------
    TabularPreprocessor
        Validated fitted preprocessor.

    Raises
    ------
    TypeError
        If preprocessor is None or not a TabularPreprocessor instance.
    RuntimeError
        If preprocessor has not been fit.
    ValueError
        If preprocessor.feature_cols does not match CANONICAL_FEATURES.
    """
    prefix = f"{context}: " if context else ""
    if preprocessor is None:
        raise TypeError(f"{prefix}preprocessor cannot be None; a fitted TabularPreprocessor is required.")

    if not isinstance(preprocessor, TabularPreprocessor):
        raise TypeError(
            f"{prefix}preprocessor must be an instance of TabularPreprocessor, got {type(preprocessor).__name__}."
        )

    if (
        preprocessor.medians_ is None
        or preprocessor.means_ is None
        or preprocessor.stds_ is None
        or preprocessor.feature_cols is None
    ):
        raise RuntimeError(f"{prefix}preprocessor has not been fit.")

    validate_canonical_features(preprocessor.feature_cols)

    return preprocessor


class BaseModel:
    """Base class providing get_parameters and set_parameters for Flower integration."""

    def __init__(self):
        self.net: Optional[nn.Module] = None

    def get_parameters(self) -> List[np.ndarray]:
        """Extracts model parameters as a list of NumPy arrays."""
        if self.net is None:
            raise RuntimeError("Network not initialized.")
        return [val.cpu().detach().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Sets model parameters from a list of NumPy arrays with strict validation.

        Parameters
        ----------
        parameters : List[np.ndarray]
            List or tuple of NumPy ndarrays corresponding to model state_dict entries.

        Raises
        ------
        RuntimeError
            If ``self.net`` has not been initialized.
        TypeError
            If ``parameters`` is None, not a list/tuple, contains elements that are not
            NumPy arrays, or contains arrays with non-numeric or incompatible dtypes.
        ValueError
            If the number of parameter arrays does not match the model's state_dict,
            if any array shape does not match the target parameter shape, or if any
            floating-point array contains non-finite values (NaN or Inf).
        """
        if self.net is None:
            raise RuntimeError("Network not initialized.")
        if parameters is None:
            raise TypeError("parameters cannot be None.")
        if not isinstance(parameters, (list, tuple)):
            raise TypeError(
                f"parameters must be a list or tuple of NumPy arrays, got {type(parameters).__name__}."
            )

        current_state = self.net.state_dict()
        expected_count = len(current_state)
        actual_count = len(parameters)

        if actual_count == 0 and expected_count > 0:
            raise ValueError(
                f"parameters cannot be empty; model expects {expected_count} parameter arrays."
            )
        if actual_count != expected_count:
            raise ValueError(
                f"Parameter count mismatch: model expects {expected_count} arrays, got {actual_count}."
            )

        new_state_dict = {}
        for idx, (key, target_tensor) in enumerate(current_state.items()):
            param = parameters[idx]
            if not isinstance(param, np.ndarray):
                raise TypeError(
                    f"Parameter at index {idx} ('{key}') must be a numpy.ndarray, "
                    f"got {type(param).__name__}."
                )

            if not (
                np.issubdtype(param.dtype, np.floating)
                or np.issubdtype(param.dtype, np.integer)
                or np.issubdtype(param.dtype, np.bool_)
            ):
                raise TypeError(
                    f"Parameter at index {idx} ('{key}') has non-numeric dtype '{param.dtype}'. "
                    "Expected a floating-point or integer numeric array."
                )

            if target_tensor.is_floating_point():
                if not np.issubdtype(param.dtype, np.floating):
                    raise TypeError(
                        f"Parameter at index {idx} ('{key}') has incompatible dtype '{param.dtype}'; "
                        f"expected floating-point array for {target_tensor.dtype} parameter."
                    )
                if not np.all(np.isfinite(param)):
                    raise ValueError(
                        f"Parameter at index {idx} ('{key}') contains non-finite values (NaN or Inf)."
                    )
            elif target_tensor.dtype in (
                torch.int64,
                torch.int32,
                torch.int16,
                torch.int8,
                torch.uint8,
            ):
                if not np.issubdtype(param.dtype, np.integer):
                    raise TypeError(
                        f"Parameter at index {idx} ('{key}') has incompatible dtype '{param.dtype}'; "
                        f"expected integer array for {target_tensor.dtype} parameter."
                    )
            elif target_tensor.dtype == torch.bool:
                if not np.issubdtype(param.dtype, np.bool_):
                    raise TypeError(
                        f"Parameter at index {idx} ('{key}') has incompatible dtype '{param.dtype}'; "
                        f"expected boolean array for {target_tensor.dtype} parameter."
                    )

            if param.shape != tuple(target_tensor.shape):
                raise ValueError(
                    f"Parameter at index {idx} ('{key}') shape mismatch: "
                    f"expected {tuple(target_tensor.shape)}, got {param.shape}."
                )

            new_state_dict[key] = torch.as_tensor(
                np.ascontiguousarray(param),
                dtype=target_tensor.dtype,
                device=target_tensor.device,
            )

        self.net.load_state_dict(new_state_dict, strict=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Computes predicted mortality probabilities."""
        if self.net is None:
            raise RuntimeError("Network not initialized.")
        self.net.eval()
        with torch.no_grad():
            tensor_X = torch.tensor(X, dtype=torch.float32)
            logits = self.net(tensor_X).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 15,
        lr: float = 1e-3,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ) -> "BaseModel":
        """Trains model weights using binary cross-entropy loss."""
        if self.net is None:
            raise RuntimeError("Network not initialized.")

        # Fail-fast validation before setting train mode, seed, or creating DataLoader
        validated_batch_size = validate_batch_size(batch_size)
        validated_y = validate_binary_labels(y, context="BaseModel.fit target 'y'")

        if X is None:
            raise TypeError("BaseModel.fit: feature matrix 'X' cannot be None.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=np.float32)
        elif isinstance(X, np.ndarray):
            X_arr = X
        elif isinstance(X, (list, tuple)):
            X_arr = np.asarray(X)
        elif isinstance(X, torch.Tensor):
            X_arr = X.cpu().detach().numpy()
        else:
            raise TypeError(f"BaseModel.fit: feature matrix 'X' must be an array-like, got {type(X).__name__}.")

        if X_arr.ndim != 2:
            raise ValueError(
                f"BaseModel.fit: feature matrix 'X' must be 2-dimensional (samples, features), "
                f"got {X_arr.ndim} dimensions with shape {X_arr.shape}."
            )

        if len(X_arr) == 0:
            raise ValueError(
                f"BaseModel.fit: training data cannot be empty (got {len(X_arr)} samples in X)."
            )

        if len(X_arr) != len(validated_y):
            raise ValueError(
                f"BaseModel.fit: sample count mismatch between X ({len(X_arr)} samples) and y ({len(validated_y)} samples)."
            )

        torch.manual_seed(seed)
        self.net.train()
        dataset = TensorDataset(torch.tensor(X_arr, dtype=torch.float32), torch.tensor(validated_y, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=validated_batch_size, shuffle=True)

        optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits = self.net(batch_X).squeeze(-1)
                loss = criterion(logits, batch_y)
                if torch.isnan(loss):
                    raise ValueError(f"Encountered NaN loss during training at epoch {epoch}")
                loss.backward()
                optimizer.step()

        return self


class LogisticRegressionModule(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier implemented in PyTorch."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.net = LogisticRegressionModule(input_dim)


class MLPModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, int] = (64, 32), dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dims[1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.drop1(self.relu1(self.fc1(x)))
        h2 = self.drop2(self.relu2(self.fc2(h1)))
        return self.out(h2)


class MLPModel(BaseModel):
    """Small 2-hidden-layer Multilayer Perceptron."""

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, int] = (64, 32), dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.net = MLPModule(input_dim, hidden_dims=hidden_dims, dropout=dropout)
