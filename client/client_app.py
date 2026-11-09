"""Client application for Flower federated learning using FedAvg.

Defines FLClient inheriting from flwr.client.NumPyClient with thread-isolated
random generators, client-specific preprocessing, and independent PyTorch models.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import flwr as fl
from flwr.common.typing import Scalar

from models.baseline import (
    BaseModel,
    LogisticRegressionModel,
    MLPModel,
    TabularPreprocessor,
)

logger = logging.getLogger(__name__)


class FLClient(fl.client.NumPyClient):
    """Federated learning client for tabular clinical mortality prediction.

    Each client encapsulates its own local dataset partition, independent
    preprocessor state, independent PyTorch model instance, and isolated
    RNG generators to guarantee thread safety during concurrent dispatch.
    """

    def __init__(
        self,
        client_id: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_type: str = "logistic_regression",
        batch_size: int = 32,
        base_seed: int = 42,
        preprocessor: Optional[TabularPreprocessor] = None,
    ):
        """Initializes client with independent local state and preprocessor.

        Parameters
        ----------
        client_id : int
            Unique identifier for the simulated hospital client (0 to 4).
        train_df : pd.DataFrame
            Client local training partition. Must be non-empty.
        val_df : pd.DataFrame
            Client local validation partition. Must be non-empty.
        model_type : str
            "logistic_regression" or "mlp".
        batch_size : int
            Local batch size for DataLoader (default: 32).
        base_seed : int
            Base seed used to derive isolated client-level RNG seeds.
        preprocessor : Optional[TabularPreprocessor]
            Shared preprocessor pre-fitted on the global training split.
            In federated execution, this must always be provided to guarantee identical
            feature coordinate scaling across all clients. The local fitting fallback
            is reserved strictly for isolated standalone client testing.
        """
        if len(train_df) == 0:
            raise ValueError(f"Client {client_id} training partition is empty.")
        if len(val_df) == 0:
            raise ValueError(f"Client {client_id} validation partition is empty.")

        self.client_id = client_id
        self.model_type = model_type
        self.batch_size = batch_size
        self.client_seed = int(base_seed * 1000 + client_id)

        # 1. Independent client random generators (ensures thread isolation)
        self.torch_generator = torch.Generator()
        self.torch_generator.manual_seed(self.client_seed)
        self.np_rng = np.random.default_rng(self.client_seed)

        # 2. Shared or fallback local preprocessor
        if preprocessor is not None:
            if (
                preprocessor.medians_ is None
                or preprocessor.means_ is None
                or preprocessor.stds_ is None
            ):
                raise RuntimeError(
                    f"Client {client_id} received a preprocessor that has not been fit."
                )
            self.preprocessor = preprocessor
        else:
            # Standalone fallback: fit locally on client train partition
            logger.warning(
                "Client %s instantiated without shared preprocessor; falling back to local fit (standalone only).",
                client_id,
            )
            self.preprocessor = TabularPreprocessor()
            self.preprocessor.fit(train_df)

        X_train, y_train = self.preprocessor.transform(train_df)
        X_val, y_val = self.preprocessor.transform(val_df)

        if y_train is None or y_val is None:
            raise ValueError(f"Client {client_id} partition is missing 'in_hospital_mortality' labels.")

        self.num_train_samples = len(X_train)
        self.num_val_samples = len(X_val)

        # 3. Local PyTorch DataLoaders with isolated generator
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=self.torch_generator,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        # 4. Independent model instance (strictly unshared between clients)
        input_dim = X_train.shape[1]
        if model_type == "logistic_regression":
            self.model: BaseModel = LogisticRegressionModel(input_dim=input_dim)
        elif model_type == "mlp":
            self.model = MLPModel(input_dim=input_dim, hidden_dims=(64, 32), dropout=0.2)
        else:
            raise ValueError(f"Unknown model_type: '{model_type}'. Choose 'logistic_regression' or 'mlp'.")

    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """Extracts model weights as a list of NumPy ndarrays."""
        return self.model.get_parameters()

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Trains the local model on the client partition using received global parameters.

        Parameters
        ----------
        parameters : List[np.ndarray]
            Current global model weights sent by the server.
        config : Dict[str, Scalar]
            Training configuration dictionary containing hyperparameter overrides.

        Returns
        -------
        Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
            Updated parameters, sample count, and dictionary of local training metrics.
        """
        self.model.set_parameters(parameters)

        local_epochs = int(config.get("local_epochs", 3))
        lr = float(config.get("lr", 1e-3))
        weight_decay = float(config.get("weight_decay", 1e-4))

        if self.model.net is None:
            raise RuntimeError(f"Client {self.client_id} model network is not initialized.")

        # Ensure train mode (activates Dropout layers if present)
        self.model.net.train()
        optimizer = optim.Adam(self.model.net.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        total_loss = 0.0
        total_batches = 0

        for epoch in range(local_epochs):
            for batch_X, batch_y in self.train_loader:
                optimizer.zero_grad()
                logits = self.model.net(batch_X).squeeze(-1)
                loss = criterion(logits, batch_y)

                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError(f"Encountered non-finite loss {loss.item()} on client {self.client_id} at epoch {epoch}")

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        return self.model.get_parameters(), self.num_train_samples, {"train_loss": float(avg_loss)}

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluates received global parameters on the client local validation partition.

        Parameters
        ----------
        parameters : List[np.ndarray]
            Global model weights to evaluate.
        config : Dict[str, Scalar]
            Evaluation configuration dictionary.

        Returns
        -------
        Tuple[float, int, Dict[str, Scalar]]:
            Evaluation loss (scalar float), sample count, and metrics dictionary.
        """
        self.model.set_parameters(parameters)

        if self.model.net is None:
            raise RuntimeError(f"Client {self.client_id} model network is not initialized.")

        # Ensure eval mode (deactivates Dropout layers)
        self.model.net.eval()
        criterion = nn.BCEWithLogitsLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                logits = self.model.net(batch_X).squeeze(-1)
                loss = criterion(logits, batch_y)

                total_loss += loss.item() * len(batch_y)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                total_correct += (preds == batch_y).sum().item()
                total_samples += len(batch_y)

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)

        return float(avg_loss), self.num_val_samples, {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
        }
