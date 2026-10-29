from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

from mlforge_core import profile_dataframe, train_dataframe


def test_profile_detects_missing_and_columns():
    df = pd.DataFrame({"age": [1, 2, None], "segment": ["a", "a", "b"]})
    result = profile_dataframe(df)
    assert result["rows"] == 3
    assert result["columns_detail"][0]["missing"] == 1


def test_train_exports_model_and_report(tmp_path: Path):
    data = load_breast_cancer(as_frame=True).frame.sample(160, random_state=42)
    report = train_dataframe(data, "target", tmp_path)
    assert report["winner"]
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "predictions.csv").exists()
