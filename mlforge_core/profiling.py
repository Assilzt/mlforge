from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    inferred_type: str
    missing: int
    missing_pct: float
    unique: int
    sample_values: list[Any]
    warnings: list[str]


def infer_column_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series): return "boolean"
    if pd.api.types.is_numeric_dtype(series): return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series): return "datetime"
    nunique = series.nunique(dropna=True)
    ratio = nunique / max(len(series), 1)
    if ratio > 0.9: return "id_like"
    if pd.api.types.is_object_dtype(series):
        return "categorical" if ratio <= 0.2 else "text"
    return "categorical"


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    columns: list[ColumnProfile] = []
    warnings: list[str] = []
    if len(df) < 100: warnings.append("small_sample_size")
    if df.duplicated().any(): warnings.append("duplicate_rows")
    for name in df.columns:
        s = df[name]
        inferred = infer_column_type(s)
        col_warnings: list[str] = []
        if s.nunique(dropna=False) <= 1: col_warnings.append("constant_column")
        if inferred == "id_like": col_warnings.append("id_like_column")
        if inferred == "text" and s.nunique(dropna=True) > min(100, len(df) * 0.5): col_warnings.append("high_cardinality_text")
        if col_warnings: warnings.extend(f"{name}:{w}" for w in col_warnings)
        sample = [x.item() if hasattr(x, "item") else x for x in s.dropna().head(3).tolist()]
        columns.append(ColumnProfile(name, str(s.dtype), inferred, int(s.isna().sum()), round(float(s.isna().mean()), 4), int(s.nunique(dropna=True)), sample, col_warnings))
    return {"rows": len(df), "columns": len(df.columns), "duplicates": int(df.duplicated().sum()), "warnings": warnings, "columns_detail": [asdict(c) for c in columns]}
