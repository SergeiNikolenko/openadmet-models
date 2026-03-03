from pathlib import Path
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import openadmet.models.inference.inference as inference_module
from openadmet.models.tests.unit.datafiles import (
    pred_test_data_csv,
    anvil_lgbm_trained_model_dir,
    anvil_chemprop_trained_model_dir,
)


@pytest.fixture
def anvil_lgbm():
    return anvil_lgbm_trained_model_dir


@pytest.fixture
def anvil_chemprop():
    return anvil_chemprop_trained_model_dir


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="MacOS runner not enough memory"
)
@pytest.mark.parametrize("model_dir", ["anvil_lgbm", "anvil_chemprop"])
def test_predict(model_dir, request):
    # Use the fixture to get the model directory
    model_dir = request.getfixturevalue(model_dir)
    # Test the predict function with a sample input
    input_path = pred_test_data_csv
    input_col = "MY_SMILES"
    model_dir = [model_dir]
    write_csv = False
    output_path = None
    debug = False

    result = inference_module.predict(
        input_path,
        input_col,
        model_dir,
        write_csv,
        output_path,
        debug=False,
        accelerator="cpu",
    )

    # Check if the result is a DataFrame
    assert isinstance(result, pd.DataFrame)


def test_generate_pairwise_df_uses_task_idx_column():
    data = pd.DataFrame({"SMILES": ["CCO", "CCN"]})
    predictions = np.array([[1.0, 11.0], [2.0, 12.0], [3.0, 13.0]])
    feat = SimpleNamespace(how_to_pair="ut")

    pairwise_df = inference_module._generate_pairwise_df(
        data=data,
        input_col="SMILES",
        feat=feat,
        predictions=predictions,
        predictions_tag="pred",
        std_tag="std",
        task_idx=1,
    )

    assert pairwise_df["pred"].tolist() == [11.0, 12.0, 13.0]
    assert pairwise_df["std"].tolist() == [11.0, 12.0, 13.0]


def test_predict_single_task_pairwise_uses_column_zero(monkeypatch):
    class DummyPairwiseFeaturizer:
        def __init__(self, how_to_pair="ut"):
            self.how_to_pair = how_to_pair

        def featurize(self, smiles):
            return np.zeros((len(smiles), 1)), np.arange(len(smiles))

    class DummyModel:
        estimator = "dummy"

        def predict(self, X_feat, accelerator="cpu"):
            return np.array([[7.0], [8.0], [9.0]])

    def fake_loader(_):
        return (
            DummyModel(),
            DummyPairwiseFeaturizer(),
            SimpleNamespace(tag="PAIR"),
            SimpleNamespace(target_cols=["task0"]),
        )

    monkeypatch.setattr(inference_module, "PairwiseFeaturizer", DummyPairwiseFeaturizer)
    monkeypatch.setattr(inference_module, "load_anvil_model_and_metadata", fake_loader)

    input_df = pd.DataFrame({"SMILES": ["CCO", "CCN"]})
    result = inference_module.predict(
        input_path=input_df,
        input_col="SMILES",
        model_dir="dummy_model",
        write_csv=False,
        output_csv=None,
        debug=False,
        accelerator="cpu",
        log=False,
    )

    pred_col = "OADMET_PRED_PAIR_task0"
    assert pred_col in result.columns
    assert result[pred_col].tolist() == [7.0, 8.0, 9.0]


def test_predict_pairwise_multitask_passes_task_idx(monkeypatch):
    class DummyPairwiseFeaturizer:
        def __init__(self, how_to_pair="ut"):
            self.how_to_pair = how_to_pair

        def featurize(self, smiles):
            return np.zeros((len(smiles), 1)), np.arange(len(smiles))

    class DummyModel:
        estimator = "dummy"

        def predict(self, X_feat, accelerator="cpu"):
            return np.array([[1.0, 2.0, 3.0]])

    def fake_loader(_):
        return (
            DummyModel(),
            DummyPairwiseFeaturizer(),
            SimpleNamespace(tag="PAIR"),
            SimpleNamespace(target_cols=["task0", "task1", "task2"]),
        )

    task_idxs = []

    def fake_generate_pairwise_df(
        data, input_col, feat, predictions, predictions_tag, std_tag, task_idx=0
    ):
        task_idxs.append(task_idx)
        data[predictions_tag] = task_idx
        data[std_tag] = task_idx
        return data

    monkeypatch.setattr(inference_module, "PairwiseFeaturizer", DummyPairwiseFeaturizer)
    monkeypatch.setattr(inference_module, "load_anvil_model_and_metadata", fake_loader)
    monkeypatch.setattr(inference_module, "_generate_pairwise_df", fake_generate_pairwise_df)

    inference_module.predict(
        input_path=pd.DataFrame({"SMILES": ["CCO"]}),
        input_col="SMILES",
        model_dir="dummy_model",
        write_csv=False,
        output_csv=None,
        debug=False,
        accelerator="cpu",
        log=False,
    )

    assert task_idxs == [0, 1, 2]
