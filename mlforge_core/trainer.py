from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .profiling import profile_dataframe

SEED = 42


def _task(y: pd.Series, override: str | None = None) -> str:
    if override in {"classification", "regression"}: return override
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15: return "regression"
    return "classification"


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ], remainder="drop")


def _candidates(task: str) -> dict[str, Any]:
    if task == "classification":
        return {"logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED), "random_forest": RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=-1), "extra_trees": ExtraTreesClassifier(n_estimators=150, random_state=SEED, n_jobs=-1)}
    return {"ridge": Ridge(), "random_forest": RandomForestRegressor(n_estimators=150, random_state=SEED, n_jobs=-1), "extra_trees": ExtraTreesRegressor(n_estimators=150, random_state=SEED, n_jobs=-1)}


def train_dataframe(df: pd.DataFrame, target: str, out_dir: str | Path, task: str | None = None) -> dict[str, Any]:
    if target not in df.columns: raise ValueError(f"Target column not found: {target}")
    if len(df) < 20: raise ValueError("At least 20 rows are required")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    y = df[target]; X = df.drop(columns=[target]); detected = _task(y, task)
    profile = profile_dataframe(df)
    profile_by_name = {item["name"]: item for item in profile["columns_detail"]}
    dropped = [c for c in X.columns if profile_by_name[c]["inferred_type"] in {"id_like", "text"}]
    X_model = X.drop(columns=dropped, errors="ignore")
    if X_model.shape[1] == 0: raise ValueError("No usable feature columns remain")
    stratify = y if detected == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=0.2, random_state=SEED, stratify=stratify)
    if detected == "classification":
        min_class = int(y_train.value_counts().min()); folds = max(2, min(5, min_class)); cv = StratifiedKFold(folds, shuffle=True, random_state=SEED); scoring = {"accuracy":"accuracy", "f1":"f1_weighted"}
        if y.nunique() == 2: scoring["roc_auc"] = "roc_auc"
    else:
        folds = min(5, max(2, len(y_train) // 10)); cv = KFold(folds, shuffle=True, random_state=SEED); scoring = {"r2":"r2", "mae":"neg_mean_absolute_error", "rmse":"neg_root_mean_squared_error"}
    leaderboard=[]; best_name=None; best_score=-np.inf
    for name, model in _candidates(detected).items():
        pipe=Pipeline([("preprocess", _preprocessor(X_model)), ("model", model)])
        started=time.perf_counter(); scores=cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1, error_score="raise"); elapsed=round(time.perf_counter()-started, 3)
        primary = float(np.mean(scores["test_accuracy"] if detected=="classification" else scores["test_r2"]))
        row={"model":name,"cv_score":round(primary,6),"fit_seconds":elapsed,"folds":folds}
        for key in scoring: row[f"cv_{key}"]=round(float(np.mean(scores[f"test_{key}"])),6)
        leaderboard.append(row)
        if primary > best_score: best_score=primary; best_name=name
    winner=Pipeline([("preprocess", _preprocessor(X_model)), ("model", _candidates(detected)[best_name])]); winner.fit(X_train,y_train)
    pred=winner.predict(X_test); metrics={"task":detected,"target":target,"test_size":len(y_test)}
    if detected=="classification":
        metrics.update({"accuracy":float(accuracy_score(y_test,pred)),"f1_weighted":float(f1_score(y_test,pred,average="weighted"))})
        if y.nunique()==2: metrics["roc_auc"]=float(roc_auc_score(y_test,winner.predict_proba(X_test)[:,1]))
    else:
        metrics.update({"r2":float(r2_score(y_test,pred)),"mae":float(mean_absolute_error(y_test,pred)),"rmse":float(mean_squared_error(y_test,pred)**0.5)})
    joblib.dump(winner, out/"model.joblib")
    pd.DataFrame({"actual":y_test.to_numpy(),"prediction":pred}).to_csv(out/"predictions.csv", index=False)
    report={"seed":SEED,"target":target,"task":detected,"dropped_columns":dropped,"profile":profile,"leaderboard":leaderboard,"winner":best_name,"metrics":metrics,"versions":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,"scikit_learn":sklearn.__version__}}
    (out/"report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def predict(model_path: str | Path, data_path: str | Path, out_path: str | Path) -> pd.DataFrame:
    model=joblib.load(model_path); df=pd.read_csv(data_path); pred=model.predict(df); result=df.copy(); result["prediction"]=pred; result.to_csv(out_path,index=False); return result
