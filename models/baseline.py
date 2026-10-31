"""Tabular baseline models for 24-hour ICU landmark mortality prediction.

Provides:
- TabularPreprocessor (median imputation and standardization fit on train only)
- LogisticRegressionModel (PyTorch linear model)
- MLPModel (PyTorch 2-hidden-layer neural network)
Both models implement a unified interface (fit, predict_proba, get_parameters, set_parameters)
compatible with centralized benchmarking and Flower federated aggregation.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data.mock_schema import get_feature_columns

# Non-feature identifiers, landmark metadata, temporal grouping, and labels
EXCLUDED_COLS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "landmark_time",
    "anchor_year_group",
    "in_hospital_mortality",
    "first_careunit",
    "gender",
]


class TabularPreprocessor:
    """Preprocesses tabular EHR features with median imputation and standardization.

    Fitted strictly on training data to prevent information leakage from val/test sets
    or across federated clients.
    """

    def __init__(self, feature_cols: Optional[List[str]] = None):
        self.feature_cols = feature_cols
        self.medians_: Optional[pd.Series] = None
        self.means_: Optional[pd.Series] = None
        self.stds_: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        if self.feature_cols is None:
            # Auto-detect numerical feature columns
            candidates = get_feature_columns()
            self.feature_cols = [c for c in candidates if c in df.columns]
            if not self.feature_cols:
                # Fallback to all numerical columns excluding metadata
                self.feature_cols = [
                    c for c in df.columns
                    if c not in EXCLUDED_COLS and pd.api.types.is_numeric_dtype(df[c])
                ]

        sub = df[self.feature_cols]
        self.medians_ = sub.median()
        imputed = sub.fillna(self.medians_)
        self.means_ = imputed.mean()
        self.stds_ = imputed.std().replace(0, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.medians_ is None:
            raise RuntimeError("Preprocessor must be fit before calling transform.")

        sub = df[self.feature_cols].copy()
        imputed = sub.fillna(self.medians_)
        standardized = (imputed - self.means_) / self.stds_
        X = standardized.values.astype(np.float32)

        y = None
        if "in_hospital_mortality" in df.columns:
            y = df["in_hospital_mortality"].values.astype(np.float32)

        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.fit(df).transform(df)


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
        """Sets model parameters from a list of NumPy arrays."""
        if self.net is None:
            raise RuntimeError("Network not initialized.")
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)

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
        torch.manual_seed(seed)
        self.net.train()
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
