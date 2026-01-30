#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

from utils import build_text_series, combine_probs


LABELS = ["facil", "intermedio", "dificil"]


def build_text_pipeline() -> Pipeline:
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=60000,
    )
    clf = LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced")
    return Pipeline([("vec", vec), ("clf", clf)])


def build_code_pipeline() -> Pipeline:
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 6),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=90000,
        lowercase=False,
    )
    clf = LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced")
    return Pipeline([("vec", vec), ("clf", clf)])


def evaluate(name: str, y_true: np.ndarray, probs: np.ndarray) -> None:
    pred = np.array(LABELS)[np.argmax(probs, axis=1)]
    acc = accuracy_score(y_true, pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_true, pred, labels=LABELS, digits=4, zero_division=0))
    cm = confusion_matrix(y_true, pred, labels=LABELS)
    print(f"{name} Confusion Matrix (labels={LABELS}):")
    print(cm)


def get_model_classes(model: Pipeline | None) -> np.ndarray | None:
    if model is None:
        return None
    clf = model.named_steps.get("clf")
    if clf is None or not hasattr(clf, "classes_"):
        return None
    return clf.classes_


def align_probs(probs: np.ndarray | None, classes: np.ndarray | None) -> np.ndarray | None:
    if probs is None or classes is None:
        return None
    class_list = list(classes)
    order = [class_list.index(lbl) for lbl in LABELS]
    return probs[:, order]


def pick_weight(text_probs: np.ndarray | None, code_probs: np.ndarray | None, y_true: np.ndarray) -> tuple[float, float]:
    if text_probs is None and code_probs is None:
        return 0.5, 0.5
    if text_probs is None:
        return 0.0, 1.0
    if code_probs is None:
        return 1.0, 0.0

    best = (0.5, 0.5)
    best_f1 = -1.0
    for tw in np.linspace(0.5, 0.9, 5):
        cw = 1.0 - tw
        probs = combine_probs(text_probs, code_probs, tw, cw)
        pred = np.array(LABELS)[np.argmax(probs, axis=1)]
        score = f1_score(y_true, pred, average="macro")
        if score > best_f1:
            best_f1 = score
            best = (float(tw), float(cw))
    return best


def grid_search(pipe: Pipeline, X: pd.Series, y: np.ndarray, param_grid: dict) -> Pipeline:
    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)
    return search.best_estimator_


def main() -> None:
    base = Path(__file__).parent
    data_path = base / "miDataSet_250.csv"
    if not data_path.exists():
        data_path = base / "miDataSet_200.csv"
        if not data_path.exists():
            data_path = base / "miDataSet_clean.csv"
    out_path = base / "difficulty_model_bundle.joblib"

    df = pd.read_csv(data_path)
    df = shuffle(df, random_state=42).reset_index(drop=True)

    df["dificultad"] = df["dificultad"].astype(str).str.strip().str.lower()
    df.loc[df["dificultad"] == "easy", "dificultad"] = "facil"
    df = df[df["dificultad"].isin(LABELS)].copy()

    required = {"dificultad", "enunciado"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    df["enunciado"] = df["enunciado"].fillna("")
    df["codigo"] = df.get("codigo", "").fillna("")

    text_series = build_text_series(df)
    code_series = df["codigo"].fillna("")
    y = df["dificultad"].to_numpy()

    idx_train, idx_test, y_train, y_test = train_test_split(
        df.index,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(splitter.split(idx_train, y_train))
    idx_train_sub = idx_train[train_idx]
    idx_val = idx_train[val_idx]
    y_train_sub = y[idx_train_sub]
    y_val = y[idx_val]

    text_model = None
    code_model = None

    text_probs_val = None
    code_probs_val = None

    if text_series.loc[idx_train_sub].str.len().gt(0).any():
        text_model = grid_search(
            build_text_pipeline(),
            text_series.loc[idx_train_sub],
            y_train_sub,
            param_grid={"clf__C": [0.5, 1.0, 2.0, 4.0]},
        )
        text_probs_val = text_model.predict_proba(text_series.loc[idx_val])
    else:
        print("\nTexto: no hay suficiente texto para entrenar.")

    if code_series.loc[idx_train_sub].str.len().gt(0).any():
        code_model = grid_search(
            build_code_pipeline(),
            code_series.loc[idx_train_sub],
            y_train_sub,
            param_grid={"clf__C": [0.5, 1.0, 2.0, 4.0]},
        )
        code_probs_val = code_model.predict_proba(code_series.loc[idx_val])
    else:
        print("\nCodigo: no hay suficiente codigo para entrenar.")

    class_order = get_model_classes(text_model)
    if class_order is None:
        class_order = get_model_classes(code_model)
    text_probs_val = align_probs(text_probs_val, class_order)
    code_probs_val = align_probs(code_probs_val, class_order)
    text_weight, code_weight = pick_weight(text_probs_val, code_probs_val, y_val)

    if text_model is not None:
        text_model.fit(text_series.loc[idx_train], y_train)
    if code_model is not None:
        code_model.fit(code_series.loc[idx_train], y_train)

    text_probs_test = text_model.predict_proba(text_series.loc[idx_test]) if text_model is not None else None
    code_probs_test = code_model.predict_proba(code_series.loc[idx_test]) if code_model is not None else None

    text_probs_test = align_probs(text_probs_test, class_order)
    code_probs_test = align_probs(code_probs_test, class_order)

    if text_probs_test is not None:
        evaluate("Texto", y_test, text_probs_test)
    if code_probs_test is not None:
        evaluate("Codigo", y_test, code_probs_test)

    combined_probs = combine_probs(text_probs_test, code_probs_test, text_weight, code_weight)
    if combined_probs is not None:
        evaluate("Combinado", y_test, combined_probs)

    bundle = {
        "labels": LABELS,
        "text_model": text_model,
        "code_model": code_model,
        "text_weight": text_weight,
        "code_weight": code_weight,
        "train_file": str(data_path),
    }

    joblib.dump(bundle, out_path)
    print("\nModelo guardado en:", out_path)
    print(f"Pesos combinados: texto={text_weight:.2f}, codigo={code_weight:.2f}")


if __name__ == "__main__":
    main()
