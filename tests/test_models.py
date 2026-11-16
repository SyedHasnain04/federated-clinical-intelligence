"""Unit tests for tabular models, preprocessor, and evaluation metrics."""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from data.mock_schema import generate_mock_data
from data.split import patient_level_split
from data.partition import partition_clients
from models.baseline import (
    BaseModel,
    TabularPreprocessor,
    LogisticRegressionModel,
    MLPModel,
    EXCLUDED_COLS,
    validate_batch_size,
    validate_binary_labels,
)
from eval.clinical_metrics import compute_metrics, find_best_operating_point


@pytest.fixture
def mock_df():
    return generate_mock_data(n_stays=500, seed=42)


def test_preprocessor_fit_on_train_only(mock_df):
    train, val, test = patient_level_split(mock_df, seed=42)

    preprocessor = TabularPreprocessor()
    preprocessor.fit(train)

    # 3. Leakage: anchor_year_group, ids, landmark_time not in model feature set
    for col in EXCLUDED_COLS:
        assert col not in preprocessor.feature_cols, f"Excluded col {col} found in features!"

    X_train, y_train = preprocessor.transform(train)
    X_val, y_val = preprocessor.transform(val)
    X_test, y_test = preprocessor.transform(test)

    assert X_train.shape[1] == len(preprocessor.feature_cols)
    assert X_val.shape[1] == len(preprocessor.feature_cols)
    assert X_test.shape[1] == len(preprocessor.feature_cols)
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_val).any()
    assert not np.isnan(X_test).any()


def test_client_preprocessor_statistics_differ(mock_df):
    train, val, test = patient_level_split(mock_df, seed=42)
    clients = partition_clients(train, num_clients=3, strategy="natural", seed=42, output_dir="", report_path="")

    # 5. Imputation/scaler statistics differ across clients and are never computed from val/test
    p_client0 = TabularPreprocessor().fit(clients[0])
    p_client1 = TabularPreprocessor().fit(clients[1])

    # Statistics should not be identical
    assert not np.allclose(p_client0.means_.values, p_client1.means_.values)
    assert not np.allclose(p_client0.stds_.values, p_client1.stds_.values)


def test_logistic_regression_forward_and_params(mock_df):
    preprocessor = TabularPreprocessor().fit(mock_df)
    X, y = preprocessor.transform(mock_df)

    # 6. Models: forward shapes, no NaN loss, get/set_parameters round-trip
    model = LogisticRegressionModel(input_dim=X.shape[1])
    probs = model.predict_proba(X)
    assert probs.shape == (len(X),)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Train short run
    model.fit(X, y, epochs=5, lr=0.01, batch_size=32)
    trained_probs = model.predict_proba(X)
    assert not np.isnan(trained_probs).any()

    # get_parameters / set_parameters round trip
    params = model.get_parameters()
    new_model = LogisticRegressionModel(input_dim=X.shape[1])
    new_model.set_parameters(params)
    roundtrip_probs = new_model.predict_proba(X)

    np.testing.assert_allclose(trained_probs, roundtrip_probs, rtol=1e-5, atol=1e-5)


def test_mlp_forward_and_params(mock_df):
    preprocessor = TabularPreprocessor().fit(mock_df)
    X, y = preprocessor.transform(mock_df)

    model = MLPModel(input_dim=X.shape[1], hidden_dims=(32, 16))
    probs = model.predict_proba(X)
    assert probs.shape == (len(X),)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Train short run
    model.fit(X, y, epochs=5, lr=0.01, batch_size=32)
    trained_probs = model.predict_proba(X)
    assert not np.isnan(trained_probs).any()

    # get_parameters / set_parameters round trip
    params = model.get_parameters()
    new_model = MLPModel(input_dim=X.shape[1], hidden_dims=(32, 16))
    new_model.set_parameters(params)
    roundtrip_probs = new_model.predict_proba(X)

    np.testing.assert_allclose(trained_probs, roundtrip_probs, rtol=1e-5, atol=1e-5)


def test_compute_metrics_toy_confusion_matrix():
    # 7. compute_metrics on a toy confusion matrix matches hand-computed values
    # Toy matrix:
    # 4 TP (true=1, pred=1), 1 FP (true=0, pred=1)
    # 1 FN (true=1, pred=0), 4 TN (true=0, pred=0)
    # Total = 10 samples
    # accuracy = (4 + 4) / 10 = 0.8
    # precision = 4 / (4 + 1) = 0.8
    # recall = 4 / (4 + 1) = 0.8
    # f1 = 0.8
    # specificity = 4 / (4 + 1) = 0.8
    y_true = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
    y_probs = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.2, 0.1, 0.05])

    metrics = compute_metrics(y_true, y_probs, threshold=0.5)

    assert metrics["accuracy"] == 0.8
    assert metrics["precision"] == 0.8
    assert metrics["recall"] == 0.8
    assert metrics["sensitivity"] == 0.8
    assert metrics["specificity"] == 0.8
    assert metrics["f1"] == 0.8
    assert metrics["n_samples"] == 10
    assert metrics["positive_rate"] == 0.5


def test_operating_point_tuning():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_probs = np.array([0.9, 0.85, 0.8, 0.75, 0.6, 0.4, 0.3, 0.1])
    thresh = find_best_operating_point(y_true, y_probs, metric="f1")
    assert 0.6 < thresh <= 0.75


# ============================================================================
# Task 2 Fix 2: Strict Model Parameter Ingestion Tests
# ============================================================================


def test_set_parameters_valid_input_and_flower_roundtrip():
    """Verifies set_parameters updates model weights accurately from valid arrays and Flower payloads."""
    for model in [LogisticRegressionModel(input_dim=8), MLPModel(input_dim=8)]:
        init_params = model.get_parameters()
        assert len(init_params) > 0

        # Create valid perturbed parameters
        new_params = [p + 1.25 for p in init_params]
        model.set_parameters(new_params)

        updated_params = model.get_parameters()
        assert len(updated_params) == len(new_params)
        for p_exp, p_act in zip(new_params, updated_params):
            np.testing.assert_allclose(p_act, p_exp, rtol=1e-6, atol=1e-6)

        # Flower serialization round-trip
        flower_parameters = ndarrays_to_parameters(new_params)
        deserialized_ndarrays = parameters_to_ndarrays(flower_parameters)
        model.set_parameters(deserialized_ndarrays)

        flower_updated = model.get_parameters()
        for p_exp, p_act in zip(deserialized_ndarrays, flower_updated):
            np.testing.assert_allclose(p_act, p_exp, rtol=1e-6, atol=1e-6)

        # Accepts tuple as well as list
        model.set_parameters(tuple(new_params))


def test_set_parameters_input_validation_none_and_non_collection():
    """Verifies set_parameters rejects None, non-collections, and uninitialized network."""
    model = LogisticRegressionModel(input_dim=4)

    # 1. Reject None
    with pytest.raises(TypeError, match="parameters cannot be None"):
        model.set_parameters(None)

    # 2. Reject non-collections
    for bad_input in [42, "parameters", {"w": np.zeros((1, 4))}, object()]:
        with pytest.raises(TypeError, match="parameters must be a list or tuple"):
            model.set_parameters(bad_input)

    # 3. Reject uninitialized network
    uninit_model = BaseModel()
    with pytest.raises(RuntimeError, match="Network not initialized"):
        uninit_model.set_parameters([np.zeros((1, 4))])


def test_set_parameters_incorrect_count():
    """Verifies set_parameters rejects too few, too many, or empty array collections fail-closed."""
    lr_model = LogisticRegressionModel(input_dim=4)  # Expects 2 arrays (weight, bias)
    valid_params = lr_model.get_parameters()
    assert len(valid_params) == 2

    # Empty collection
    with pytest.raises(ValueError, match="parameters cannot be empty; model expects 2 parameter arrays"):
        lr_model.set_parameters([])

    # Too few arrays
    with pytest.raises(ValueError, match="Parameter count mismatch: model expects 2 arrays, got 1"):
        lr_model.set_parameters(valid_params[:1])

    # Too many arrays
    extra_params = valid_params + [np.zeros((1,), dtype=np.float32)]
    with pytest.raises(ValueError, match="Parameter count mismatch: model expects 2 arrays, got 3"):
        lr_model.set_parameters(extra_params)

    # MLPModel (expects 6 arrays)
    mlp_model = MLPModel(input_dim=4)
    mlp_params = mlp_model.get_parameters()
    assert len(mlp_params) == 6

    with pytest.raises(ValueError, match="parameters cannot be empty; model expects 6 parameter arrays"):
        mlp_model.set_parameters([])

    with pytest.raises(ValueError, match="Parameter count mismatch: model expects 6 arrays, got 5"):
        mlp_model.set_parameters(mlp_params[:5])

    with pytest.raises(ValueError, match="Parameter count mismatch: model expects 6 arrays, got 7"):
        mlp_model.set_parameters(mlp_params + [np.zeros((1,), dtype=np.float32)])


def test_set_parameters_invalid_array_types_and_dtypes():
    """Verifies set_parameters rejects non-ndarrays, object, string, structured, and incompatible dtypes."""
    model = LogisticRegressionModel(input_dim=4)
    valid_params = model.get_parameters()

    # 1. Non-ndarray element (list, tensor, scalar)
    for bad_elem in [[1.0, 2.0], torch.zeros((1, 4)), 1.0, "weights"]:
        bad_params = [bad_elem, valid_params[1]]
        with pytest.raises(TypeError, match="must be a numpy.ndarray"):
            model.set_parameters(bad_params)

    # 2. Object dtype
    obj_array = np.array(["a", "b", "c", "d"], dtype=object).reshape(1, 4)
    with pytest.raises(TypeError, match="has non-numeric dtype 'object'"):
        model.set_parameters([obj_array, valid_params[1]])

    # 3. String dtype
    str_array = np.array([["1", "2", "3", "4"]])
    with pytest.raises(TypeError, match="has non-numeric dtype"):
        model.set_parameters([str_array, valid_params[1]])

    # 4. Structured dtype
    structured_arr = np.zeros((1, 4), dtype=[("val", np.float32)])
    with pytest.raises(TypeError, match="has non-numeric dtype"):
        model.set_parameters([structured_arr, valid_params[1]])

    # 5. Incompatible dtype: passing integer array to floating-point model parameter
    int_array = np.zeros((1, 4), dtype=np.int64)
    with pytest.raises(TypeError, match="has incompatible dtype 'int64'; expected floating-point array"):
        model.set_parameters([int_array, valid_params[1]])


def test_set_parameters_invalid_values_nan_and_inf():
    """Verifies set_parameters rejects NaN, +Inf, and -Inf in floating-point parameter arrays."""
    model = LogisticRegressionModel(input_dim=4)
    valid_params = model.get_parameters()

    for bad_val in [np.nan, np.inf, -np.inf]:
        bad_weight = valid_params[0].copy()
        bad_weight[0, 0] = bad_val
        with pytest.raises(ValueError, match="contains non-finite values \\(NaN or Inf\\)"):
            model.set_parameters([bad_weight, valid_params[1]])

        bad_bias = valid_params[1].copy()
        bad_bias[0] = bad_val
        with pytest.raises(ValueError, match="contains non-finite values \\(NaN or Inf\\)"):
            model.set_parameters([valid_params[0], bad_bias])


def test_set_parameters_shape_mismatch():
    """Verifies set_parameters rejects parameter arrays with mismatched dimensions or shapes."""
    model = LogisticRegressionModel(input_dim=4)  # weight: (1, 4), bias: (1,)
    valid_params = model.get_parameters()

    # Transposed weight shape: (4, 1) instead of (1, 4)
    with pytest.raises(ValueError, match="shape mismatch: expected \\(1, 4\\), got \\(4, 1\\)"):
        model.set_parameters([valid_params[0].T, valid_params[1]])

    # Extra dimension: (1, 4, 1) instead of (1, 4)
    with pytest.raises(ValueError, match="shape mismatch: expected \\(1, 4\\), got \\(1, 4, 1\\)"):
        model.set_parameters([valid_params[0][:, :, None], valid_params[1]])

    # Bias shape mismatch: (2,) instead of (1,)
    with pytest.raises(ValueError, match="shape mismatch: expected \\(1,\\), got \\(2,\\)"):
        model.set_parameters([valid_params[0], np.array([0.0, 0.0], dtype=np.float32)])


def test_set_parameters_atomicity_guarantee():
    """Verifies all-or-nothing atomicity: if any array is invalid, no model parameter is modified."""
    model = MLPModel(input_dim=8)
    orig_params = [p.copy() for p in model.get_parameters()]
    assert len(orig_params) == 6

    # Test 1: Final array (index 5) is invalid with NaN
    cand_params_nan = [p + 5.0 for p in orig_params]
    cand_params_nan[-1] = np.array([np.nan], dtype=np.float32)

    with pytest.raises(ValueError, match="contains non-finite values"):
        model.set_parameters(cand_params_nan)

    # Verify every parameter (including the first) remains completely unmutated
    current_params = model.get_parameters()
    for idx, (p_orig, p_cur) in enumerate(zip(orig_params, current_params)):
        np.testing.assert_allclose(p_cur, p_orig, atol=0.0, rtol=0.0, err_msg=f"Parameter {idx} mutated on failed set!")

    # Test 2: Final array has shape mismatch
    cand_params_shape = [p + 5.0 for p in orig_params]
    cand_params_shape[-1] = np.zeros((10,), dtype=np.float32)

    with pytest.raises(ValueError, match="shape mismatch"):
        model.set_parameters(cand_params_shape)

    current_params = model.get_parameters()
    for idx, (p_orig, p_cur) in enumerate(zip(orig_params, current_params)):
        np.testing.assert_allclose(p_cur, p_orig, atol=0.0, rtol=0.0)

    # Test 3: Intermediate array (index 2) has non-numeric dtype
    cand_params_dtype = [p + 5.0 for p in orig_params]
    cand_params_dtype[2] = np.array(["invalid"] * cand_params_dtype[2].size).reshape(cand_params_dtype[2].shape)

    with pytest.raises(TypeError, match="has non-numeric dtype"):
        model.set_parameters(cand_params_dtype)

    current_params = model.get_parameters()
    for idx, (p_orig, p_cur) in enumerate(zip(orig_params, current_params)):
        np.testing.assert_allclose(p_cur, p_orig, atol=0.0, rtol=0.0)


def test_set_parameters_model_buffers_and_dtype_preservation():
    """Verifies set_parameters cleanly handles modules with persistent buffers, preserving dtypes and devices."""
    class ModelWithBuffers(BaseModel):
        def __init__(self, dim: int = 4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 4),
                nn.BatchNorm1d(4),
            )

    model = ModelWithBuffers(dim=4)
    state_keys = list(model.net.state_dict().keys())
    assert "1.running_mean" in state_keys
    assert "1.num_batches_tracked" in state_keys

    orig_params = model.get_parameters()
    # Check that num_batches_tracked is integer dtype
    num_batches_idx = state_keys.index("1.num_batches_tracked")
    assert np.issubdtype(orig_params[num_batches_idx].dtype, np.integer)

    # Valid modification of both parameters and buffers
    new_params = [p.copy() for p in orig_params]
    new_params[0] = new_params[0] + 0.5  # linear.weight
    new_params[num_batches_idx] = np.array(15, dtype=np.int64)  # buffer count

    model.set_parameters(new_params)
    retrieved = model.get_parameters()
    np.testing.assert_allclose(retrieved[0], new_params[0])
    assert int(retrieved[num_batches_idx]) == 15
    assert model.net.state_dict()["1.num_batches_tracked"].dtype == torch.int64

    # Incompatible dtype: passing float to integer buffer
    bad_params = [p.copy() for p in new_params]
    bad_params[num_batches_idx] = np.array(15.5, dtype=np.float32)
    with pytest.raises(TypeError, match="expected integer array for torch.int64 parameter"):
        model.set_parameters(bad_params)


# ============================================================================
# Task 2 Fix 4 Stage 1: Batch-Size and Binary-Label Validation Tests
# ============================================================================


# --- 1. Batch-Size Helper Tests ---

@pytest.mark.parametrize("valid_bs", [1, 16, 32, 64, 128, 1024])
def test_validate_batch_size_accepts_positive_python_integers(valid_bs):
    """Verifies validate_batch_size accepts positive Python ints and returns Python int."""
    result = validate_batch_size(valid_bs)
    assert result == valid_bs
    assert type(result) is int
    assert not isinstance(result, bool)


@pytest.mark.parametrize(
    "valid_np_bs",
    [np.int64(32), np.int32(16), np.int16(8), np.int8(1), np.uint32(64)],
)
def test_validate_batch_size_accepts_positive_numpy_integers(valid_np_bs):
    """Verifies validate_batch_size accepts positive NumPy integer scalars and converts to Python int."""
    result = validate_batch_size(valid_np_bs)
    assert result == int(valid_np_bs)
    assert type(result) is int
    assert not isinstance(result, bool)


@pytest.mark.parametrize("invalid_val", [0, -1, -32, np.int64(0), np.int32(-5)])
def test_validate_batch_size_rejects_zero_and_negative(invalid_val):
    """Verifies validate_batch_size rejects zero and negative integers with ValueError."""
    with pytest.raises(ValueError, match="positive integer"):
        validate_batch_size(invalid_val)


@pytest.mark.parametrize("bool_val", [True, False, np.bool_(True), np.bool_(False)])
def test_validate_batch_size_rejects_booleans(bool_val):
    """Verifies validate_batch_size strictly rejects bool and np.bool_ with TypeError."""
    with pytest.raises(TypeError, match="must be a positive integer"):
        validate_batch_size(bool_val)


@pytest.mark.parametrize("float_val", [32.0, 1.0, 0.0, -1.0, 16.5, np.float32(32.0), np.float64(16.0)])
def test_validate_batch_size_rejects_floats(float_val):
    """Verifies validate_batch_size rejects floats (including integer floats like 32.0) with TypeError."""
    with pytest.raises(TypeError, match="must be a positive integer"):
        validate_batch_size(float_val)


@pytest.mark.parametrize(
    "bad_type",
    ["32", "0", "", "abc", None, np.nan, float("nan"), float("inf"), -float("inf"), [32], {32}],
)
def test_validate_batch_size_rejects_strings_none_nan_and_unsupported(bad_type):
    """Verifies validate_batch_size rejects strings, None, NaNs, infinities, and collections with TypeError."""
    with pytest.raises(TypeError, match="must be a positive integer"):
        validate_batch_size(bad_type)


# --- 2. Binary-Label Helper Tests ---

@pytest.mark.parametrize(
    "labels_input",
    [
        np.array([0, 1, 0, 1]),
        np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        pd.Series([0, 1, 1, 0]),
        pd.Series([0.0, 1.0, 0.0]),
        [0, 1, 0, 1],
        (0, 1, 1),
    ],
)
def test_validate_binary_labels_accepts_valid_numeric_binary_labels(labels_input):
    """Verifies validate_binary_labels accepts 1D integer/float binary labels and returns float32."""
    res = validate_binary_labels(labels_input)
    assert isinstance(res, np.ndarray)
    assert res.dtype == np.float32
    assert res.ndim == 1
    assert set(np.unique(res)).issubset({0.0, 1.0})


@pytest.mark.parametrize(
    "bool_input",
    [
        np.array([True, False, True]),
        pd.Series([False, True, False]),
        [True, False, False],
        (True, True, False),
    ],
)
def test_validate_binary_labels_accepts_booleans_and_converts(bool_input):
    """Verifies validate_binary_labels accepts boolean inputs and converts True->1.0, False->0.0."""
    res = validate_binary_labels(bool_input)
    assert isinstance(res, np.ndarray)
    assert res.dtype == np.float32
    assert res.ndim == 1
    np.testing.assert_allclose(res, [1.0 if x else 0.0 for x in bool_input])


def test_validate_binary_labels_accepts_single_class_arrays():
    """Verifies validate_binary_labels accepts all-zero and all-one arrays."""
    all_zeros = validate_binary_labels(np.zeros(10, dtype=int))
    assert np.all(all_zeros == 0.0)
    assert all_zeros.dtype == np.float32

    all_ones = validate_binary_labels(np.ones(10, dtype=float))
    assert np.all(all_ones == 1.0)
    assert all_ones.dtype == np.float32


@pytest.mark.parametrize("invalid_val", [2, -1, 0.5, 99, 1.0001, -0.01])
def test_validate_binary_labels_rejects_non_binary_values(invalid_val):
    """Verifies validate_binary_labels rejects values outside {0, 1} with ValueError."""
    bad_labels = np.array([0.0, 1.0, invalid_val])
    with pytest.raises(ValueError, match=r"only binary values in \{0, 1\}"):
        validate_binary_labels(bad_labels)


@pytest.mark.parametrize("bad_numeric", [np.nan, float("inf"), -float("inf")])
def test_validate_binary_labels_rejects_nan_and_infinities(bad_numeric):
    """Verifies validate_binary_labels rejects NaN and infinities with ValueError."""
    bad_labels = np.array([0.0, 1.0, bad_numeric])
    with pytest.raises(ValueError, match="non-finite values"):
        validate_binary_labels(bad_labels)


@pytest.mark.parametrize(
    "bad_type_input",
    [
        pd.Series(["0", "1"]),
        np.array(["0", "1"]),
        np.array([0, 1], dtype=object),
        ["0", "1"],
        ["a", "b"],
        None,
    ],
)
def test_validate_binary_labels_rejects_strings_and_object_arrays(bad_type_input):
    """Verifies validate_binary_labels rejects strings and object arrays with TypeError."""
    with pytest.raises(TypeError):
        validate_binary_labels(bad_type_input)


@pytest.mark.parametrize(
    "multidim_input",
    [
        np.zeros((10, 1)),
        np.ones((5, 2)),
        np.zeros((2, 2, 2)),
        pd.DataFrame({"label": [0, 1, 0]}),
        np.array(1),  # 0-D scalar
    ],
)
def test_validate_binary_labels_rejects_multidimensional_arrays(multidim_input):
    """Verifies validate_binary_labels strictly rejects non-1D arrays without silent flattening."""
    with pytest.raises(ValueError, match="1-dimensional"):
        validate_binary_labels(multidim_input)


def test_validate_binary_labels_context_in_error_message():
    """Verifies supplied context is formatted into the validation error message."""
    with pytest.raises(ValueError, match=r"Client 42 val: labels must contain only binary values in \{0, 1\}"):
        validate_binary_labels(np.array([0, 2]), context="Client 42 val")


# --- 3. Preprocessor Tests ---

def test_preprocessor_target_validation_valid_and_boolean(mock_df):
    """Verifies TabularPreprocessor.transform preserves valid binary and boolean targets."""
    prep = TabularPreprocessor().fit(mock_df)

    # 1. Valid binary targets
    X, y = prep.transform(mock_df)
    assert y is not None
    assert y.dtype == np.float32
    assert set(np.unique(y)).issubset({0.0, 1.0})

    # 2. Boolean targets cleanly converted
    bool_df = mock_df.copy()
    bool_df["in_hospital_mortality"] = bool_df["in_hospital_mortality"].astype(bool)
    X_bool, y_bool = prep.transform(bool_df)
    assert y_bool is not None
    assert y_bool.dtype == np.float32
    np.testing.assert_allclose(y, y_bool)


def test_preprocessor_target_validation_rejects_non_binary_and_non_finite(mock_df):
    """Verifies TabularPreprocessor.transform rejects invalid label values and non-finites."""
    prep = TabularPreprocessor().fit(mock_df)

    # Non-binary value 2
    bad_df = mock_df.copy()
    bad_df.loc[bad_df.index[0], "in_hospital_mortality"] = 2
    with pytest.raises(ValueError, match=r"only binary values in \{0, 1\}"):
        prep.transform(bad_df)

    # Fractional value 0.5
    bad_frac_df = mock_df.copy()
    bad_frac_df["in_hospital_mortality"] = bad_frac_df["in_hospital_mortality"].astype(float)
    bad_frac_df.loc[bad_frac_df.index[0], "in_hospital_mortality"] = 0.5
    with pytest.raises(ValueError, match=r"only binary values in \{0, 1\}"):
        prep.transform(bad_frac_df)

    # Non-finite NaN
    bad_nan_df = mock_df.copy()
    bad_nan_df["in_hospital_mortality"] = bad_nan_df["in_hospital_mortality"].astype(float)
    bad_nan_df.loc[bad_nan_df.index[0], "in_hospital_mortality"] = np.nan
    with pytest.raises(ValueError, match="non-finite values"):
        prep.transform(bad_nan_df)


def test_preprocessor_target_validation_missing_target_preserves_none(mock_df):
    """Verifies TabularPreprocessor.transform returns (X, None) when target column is omitted."""
    prep = TabularPreprocessor().fit(mock_df)
    features_only_df = mock_df.drop(columns=["in_hospital_mortality"])
    X, y = prep.transform(features_only_df)
    assert y is None
    assert X.shape[1] == 51


# --- 4. BaseModel.fit Tests ---

def test_base_model_fit_valid_training(mock_df):
    """Verifies BaseModel.fit succeeds and updates parameters with valid inputs."""
    prep = TabularPreprocessor().fit(mock_df)
    X, y = prep.transform(mock_df)

    model = LogisticRegressionModel(input_dim=51)
    init_params = [p.copy() for p in model.get_parameters()]

    model.fit(X, y, epochs=2, lr=1e-2, batch_size=32)
    trained_params = model.get_parameters()

    # Parameters should have changed
    assert any(not np.allclose(p_i, p_t) for p_i, p_t in zip(init_params, trained_params))


@pytest.mark.parametrize("bad_bs", [0, -1, True, False, "32", 32.0, None])
def test_base_model_fit_rejects_invalid_batch_size(mock_df, bad_bs):
    """Verifies BaseModel.fit rejects invalid batch sizes before training or parameter mutation."""
    prep = TabularPreprocessor().fit(mock_df)
    X, y = prep.transform(mock_df)

    model = LogisticRegressionModel(input_dim=51)
    init_params = [p.copy() for p in model.get_parameters()]

    expected_exc = ValueError if isinstance(bad_bs, int) and not isinstance(bad_bs, bool) and bad_bs <= 0 else TypeError
    with pytest.raises(expected_exc):
        model.fit(X, y, batch_size=bad_bs)

    # Verify parameters were not mutated
    for p_init, p_curr in zip(init_params, model.get_parameters()):
        np.testing.assert_allclose(p_init, p_curr, atol=0.0, rtol=0.0)


def test_base_model_fit_rejects_non_binary_and_non_finite_labels(mock_df):
    """Verifies BaseModel.fit rejects non-binary and non-finite labels without state mutation."""
    prep = TabularPreprocessor().fit(mock_df)
    X, y = prep.transform(mock_df)

    model = LogisticRegressionModel(input_dim=51)
    init_params = [p.copy() for p in model.get_parameters()]

    # Non-binary label value 2
    bad_y_val = y.copy()
    bad_y_val[0] = 2.0
    with pytest.raises(ValueError, match=r"only binary values in \{0, 1\}"):
        model.fit(X, bad_y_val)

    # Non-finite NaN
    bad_y_nan = y.copy()
    bad_y_nan[0] = np.nan
    with pytest.raises(ValueError, match="non-finite values"):
        model.fit(X, bad_y_nan)

    # Ensure no parameter mutation
    for p_init, p_curr in zip(init_params, model.get_parameters()):
        np.testing.assert_allclose(p_init, p_curr, atol=0.0, rtol=0.0)


def test_base_model_fit_rejects_length_and_dimension_mismatches(mock_df):
    """Verifies BaseModel.fit rejects sample count and label dimension mismatches."""
    prep = TabularPreprocessor().fit(mock_df)
    X, y = prep.transform(mock_df)

    model = LogisticRegressionModel(input_dim=51)

    # Mismatched sample count (X has 500, y has 499)
    with pytest.raises(ValueError, match="sample count mismatch"):
        model.fit(X, y[:499])

    # 2D label shape (N, 1) - must be rejected, not silently flattened
    y_2d = y.reshape(-1, 1)
    with pytest.raises(ValueError, match="1-dimensional"):
        model.fit(X, y_2d)

    # Empty training data
    with pytest.raises(ValueError, match="empty"):
        model.fit(X[:0], y[:0])
