import joblib
import numpy as np
import pytest

from src.calibration import CalibratedModel
from src.evaluate import EvaluationError, load_model
from src.models import build_pipeline
from tests.test_models import TEST_CFG, make_xy


@pytest.fixture(scope="module")
def good_file(tmp_path_factory):
    X, y = make_xy(300, 0)
    model = CalibratedModel(build_pipeline("logreg", "baseline", TEST_CFG), "none").fit(X, y)
    path = tmp_path_factory.mktemp("models") / "final_model.joblib"
    joblib.dump(model, path)
    return path


def test_a_good_model_loads_and_scores(good_file):
    model = load_model(good_file)
    assert model.predict_proba(make_xy(10, 3)[0]).shape == (10, 2)


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(EvaluationError, match="not found"):
        load_model(tmp_path / "nope.joblib")


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"this is not a model",
        b"cno_such_module_xyz\nThing\n.",
        bytes(np.random.default_rng(0).integers(0, 256, 300, dtype=np.uint8)),
        bytes(np.random.default_rng(1).integers(0, 256, 5000, dtype=np.uint8)),
        b"\x80\x04\x95",
        b"(((((",
    ],
    ids=["empty", "text", "unknown-module", "random-300", "random-5000", "truncated-header", "unbalanced"],
)
def test_damaged_files_give_a_clear_error(tmp_path, payload):
    path = tmp_path / "final_model.joblib"
    path.write_bytes(payload)
    with pytest.raises(EvaluationError, match="Could not load"):
        load_model(path)


def test_a_truncated_real_model_file_gives_a_clear_error(tmp_path, good_file):
    data = good_file.read_bytes()
    path = tmp_path / "final_model.joblib"
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(EvaluationError, match="Could not load"):
        load_model(path)


def test_a_folder_in_place_of_the_file_gives_a_clear_error(tmp_path):
    folder = tmp_path / "final_model.joblib"
    folder.mkdir()
    with pytest.raises(EvaluationError, match="Could not load"):
        load_model(folder)