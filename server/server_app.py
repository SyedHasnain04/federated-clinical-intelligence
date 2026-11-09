"""Server application for Flower federated learning using FedAvg.

Defines LocalClientProxy and orchestration logic running non-private FedAvg
over five simulated client partitions without Ray dependencies.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

import flwr as fl
from flwr.common import (
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    ReconnectIns,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import Server
from flwr.server.strategy import FedAvg

from client.client_app import FLClient
from data.mock_schema import generate_mock_data
from data.partition import partition_clients
from data.split import patient_level_split
from models.baseline import (
    BaseModel,
    LogisticRegressionModel,
    MLPModel,
    TabularPreprocessor,
)

logger = logging.getLogger(__name__)


class LocalClientProxy(ClientProxy):
    """In-process proxy routing Flower messages directly to a local client.

    Implements all five abstract methods of flwr.server.client_proxy.ClientProxy
    for safe, transparent dispatch without network sockets or Ray.
    """

    def __init__(self, cid: str, client: fl.client.Client):
        super().__init__(cid)
        self.client = client

    def get_properties(
        self,
        ins: GetPropertiesIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> GetPropertiesRes:
        return self.client.get_properties(ins)

    def get_parameters(
        self,
        ins: GetParametersIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> GetParametersRes:
        return self.client.get_parameters(ins)

    def fit(
        self,
        ins: FitIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> FitRes:
        return self.client.fit(ins)

    def evaluate(
        self,
        ins: EvaluateIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> EvaluateRes:
        return self.client.evaluate(ins)

    def reconnect(
        self,
        ins: ReconnectIns,
        timeout: Optional[float],
        group_id: Optional[int],
    ) -> DisconnectRes:
        return DisconnectRes(reason="session_closed")


def aggregate_weighted_metrics(
    metrics: List[Tuple[int, Dict[str, Scalar]]],
) -> Dict[str, Scalar]:
    """Computes sample-weighted average of numeric client metrics."""
    if not metrics:
        return {}

    total_samples = sum(num_samples for num_samples, _ in metrics)
    if total_samples == 0:
        return {}

    aggregated: Dict[str, float] = {}
    for num_samples, m_dict in metrics:
        for key, val in m_dict.items():
            if isinstance(val, (int, float)):
                aggregated[key] = aggregated.get(key, 0.0) + float(val) * num_samples

    return {k: v / total_samples for k, v in aggregated.items()}


def get_fedavg_strategy(
    initial_parameters: Parameters,
    min_clients: int = 5,
    on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
    on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
) -> FedAvg:
    """Configures FedAvg strategy with full participation across all 5 clients.

    Parameters
    ----------
    initial_parameters : Parameters
        Initial global model parameters serialized as Flower Parameters.
    min_clients : int
        Minimum and total clients expected per round (default: 5).
    on_fit_config_fn : Optional[Callable[[int], Dict[str, Scalar]]]
        Function generating client fit configuration per round.
    on_evaluate_config_fn : Optional[Callable[[int], Dict[str, Scalar]]]
        Function generating client evaluate configuration per round.

    Returns
    -------
    FedAvg
        Configured FedAvg strategy instance.
    """
    return FedAvg(
        fraction_fit=1.0,
        min_fit_clients=min_clients,
        min_available_clients=min_clients,
        fraction_evaluate=1.0,
        min_evaluate_clients=min_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        fit_metrics_aggregation_fn=aggregate_weighted_metrics,
        evaluate_metrics_aggregation_fn=aggregate_weighted_metrics,
    )


def run_federated_simulation(
    num_rounds: int = 2,
    num_clients: int = 5,
    model_type: str = "logistic_regression",
    seed: int = 42,
    checkpoint_path: Optional[str] = None,
    max_workers: Optional[int] = None,
    local_epochs: int = 3,
    lr: float = 1e-3,
    batch_size: int = 32,
    n_stays: int = 3000,
) -> Dict[str, Any]:
    """Executes an in-process federated learning simulation using FedAvg.

    Parameters
    ----------
    num_rounds : int
        Number of federated aggregation rounds.
    num_clients : int
        Number of simulated clients (strictly 5 for careunits).
    model_type : str
        "logistic_regression" or "mlp".
    seed : int
        Random seed for cohort generation and client partitioning.
    checkpoint_path : Optional[str]
        File path to save the final verified global model state_dict.
    max_workers : Optional[int]
        Maximum worker threads in ThreadPoolExecutor (1 for sequential, None for concurrent).
    local_epochs : int
        Number of local training epochs per client round.
    lr : float
        Client learning rate.
    batch_size : int
        Client training batch size.
    n_stays : int
        Number of synthetic cohort stays to generate.

    Returns
    -------
    Dict[str, Any]
        Dictionary with history, elapsed time, final parameters, and trained global model.
    """
    logger.info("Initializing synthetic cohort and client partitions (seed=%s)...", seed)

    # 1. Build synthetic cohort and patient-level split (zero disk side-effects)
    df = generate_mock_data(n_stays=n_stays, seed=seed)
    train_df, val_df, test_df = patient_level_split(
        df,
        seed=seed,
        frac=(0.7, 0.15, 0.15),
        splits_dir="",
    )

    # 2. Fit shared preprocessor ONCE on the full global training split ONLY
    #    (strictly zero leakage from val_df or test_df)
    shared_preprocessor = TabularPreprocessor()
    shared_preprocessor.fit(train_df)

    # 3. Partition both train and val into disjoint subsets matching careunits
    train_partitions = partition_clients(
        train_df,
        num_clients=num_clients,
        strategy="natural",
        seed=seed,
        output_dir="",
        report_path="",
    )
    val_partitions = partition_clients(
        val_df,
        num_clients=num_clients,
        strategy="natural",
        seed=seed,
        output_dir="",
        report_path="",
    )

    # 4. Guard against empty partitions
    for cid in range(num_clients):
        if cid not in train_partitions or len(train_partitions[cid]) == 0:
            raise ValueError(f"Client {cid} training partition is empty. Cannot execute FedAvg.")
        if cid not in val_partitions or len(val_partitions[cid]) == 0:
            raise ValueError(f"Client {cid} validation partition is empty. Cannot execute FedAvg.")

    # 5. Instantiate 5 independent FLClient objects sharing the exact same fitted preprocessor
    client_manager = SimpleClientManager()
    clients: List[FLClient] = []

    for cid in range(num_clients):
        fl_client = FLClient(
            client_id=cid,
            train_df=train_partitions[cid],
            val_df=val_partitions[cid],
            model_type=model_type,
            batch_size=batch_size,
            base_seed=seed,
            preprocessor=shared_preprocessor,
        )
        clients.append(fl_client)
        wrapped_client = fl_client.to_client()
        proxy = LocalClientProxy(cid=str(cid), client=wrapped_client)
        client_manager.register(proxy)

    # 6. Extract initial parameters from an independent reference model with fixed seed
    torch.manual_seed(seed)
    input_dim = shared_preprocessor.transform(train_df)[0].shape[1]
    if model_type == "logistic_regression":
        ref_model: BaseModel = LogisticRegressionModel(input_dim=input_dim)
    elif model_type == "mlp":
        ref_model = MLPModel(input_dim=input_dim, hidden_dims=(64, 32), dropout=0.2)
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")

    initial_parameters_list = ref_model.get_parameters()
    initial_parameters = ndarrays_to_parameters(initial_parameters_list)

    # 6. Fit configuration hook
    def fit_config_fn(server_round: int) -> Dict[str, Scalar]:
        return {
            "local_epochs": local_epochs,
            "lr": lr,
            "server_round": server_round,
        }

    # 7. Configure FedAvg strategy and Server instance
    strategy = get_fedavg_strategy(
        initial_parameters=initial_parameters,
        min_clients=num_clients,
        on_fit_config_fn=fit_config_fn,
    )

    server = Server(client_manager=client_manager, strategy=strategy)
    if max_workers is not None:
        server.max_workers = max_workers

    # 8. Run simulation
    history, elapsed = server.fit(num_rounds=num_rounds, timeout=None)

    # 9. Retrieve and verify final aggregated global parameters
    if server.parameters is None:
        raise RuntimeError("Federated simulation ended without producing global server parameters.")

    final_ndarrays = parameters_to_ndarrays(server.parameters)

    # 10. Load final weights into clean global model
    if model_type == "logistic_regression":
        global_model: BaseModel = LogisticRegressionModel(input_dim=input_dim)
    else:
        global_model = MLPModel(input_dim=input_dim, hidden_dims=(64, 32), dropout=0.2)
    global_model.set_parameters(final_ndarrays)

    # 11. Checkpoint persistence and verification
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        if global_model.net is None:
            raise RuntimeError("Global model network is None; cannot save checkpoint.")

        torch.save(global_model.net.state_dict(), checkpoint_path)

        # Verification: reload into fresh model instance
        if model_type == "logistic_regression":
            verify_model: BaseModel = LogisticRegressionModel(input_dim=input_dim)
        else:
            verify_model = MLPModel(input_dim=input_dim, hidden_dims=(64, 32), dropout=0.2)

        if verify_model.net is None:
            raise RuntimeError("Verification model network is None.")

        verify_model.net.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        reloaded_ndarrays = verify_model.get_parameters()

        for p_exp, p_act in zip(final_ndarrays, reloaded_ndarrays):
            np.testing.assert_allclose(
                p_act,
                p_exp,
                rtol=1e-5,
                atol=1e-5,
                err_msg="Reloaded checkpoint does not match final server parameters.",
            )

    return {
        "history": history,
        "elapsed": elapsed,
        "final_parameters": final_ndarrays,
        "initial_parameters": initial_parameters_list,
        "global_model": global_model,
        "clients": clients,
        "preprocessor": shared_preprocessor,
        "test_df": test_df,
    }
