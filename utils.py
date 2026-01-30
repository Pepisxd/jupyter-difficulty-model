from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


TEXT_PARTS = [
    ("ENUNCIADO", "enunciado"),
    ("INPUT", "input_desc"),
    ("OUTPUT", "output_desc"),
    ("RESTRICCIONES", "restricciones"),
    ("EJEMPLO ENTRADA", "ejemplo_entrada"),
    ("EJEMPLO SALIDA", "ejemplo_salida"),
]


def build_text_series(df: pd.DataFrame) -> pd.Series:
    def row_text(row: pd.Series) -> str:
        parts: list[str] = []
        for label, col in TEXT_PARTS:
            val = str(row.get(col, "") or "").strip()
            if val:
                parts.append(f"{label}: {val}")
        return "\n\n".join(parts)

    return df.apply(row_text, axis=1)


def build_text_from_inputs(values: dict[str, str]) -> str:
    parts: list[str] = []
    for label, col in TEXT_PARTS:
        val = (values.get(col) or "").strip()
        if val:
            parts.append(f"{label}: {val}")
    return "\n\n".join(parts)


def normalize_weights(text_weight: float, code_weight: float) -> tuple[float, float]:
    total = text_weight + code_weight
    if total <= 0:
        return 0.5, 0.5
    return text_weight / total, code_weight / total


def combine_probs(
    text_probs: np.ndarray | None, code_probs: np.ndarray | None, text_weight: float, code_weight: float
) -> np.ndarray:
    if text_probs is None:
        return code_probs
    if code_probs is None:
        return text_probs
    tw, cw = normalize_weights(text_weight, code_weight)
    return (text_probs * tw) + (code_probs * cw)


def coerce_str_series(series: Iterable[str]) -> pd.Series:
    return pd.Series([str(x or "") for x in series])
