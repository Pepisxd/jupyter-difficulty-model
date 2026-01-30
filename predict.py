 #!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from utils import build_text_from_inputs, combine_probs


def read_block(prompt: str) -> str:
    print(prompt)
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main() -> None:
    base = Path(__file__).parent
    bundle = joblib.load(base / "difficulty_model_bundle.joblib")

    enunciado = read_block("Escribe el enunciado (finaliza con END):")
    input_desc = read_block("Describe el input (finaliza con END):")
    output_desc = read_block("Describe el output (finaliza con END):")
    restricciones = read_block("Pega restricciones (finaliza con END):")
    ejemplo_entrada = read_block("Pega ejemplo de entrada (finaliza con END):")
    ejemplo_salida = read_block("Pega ejemplo de salida (finaliza con END):")
    codigo = read_block("Pega el codigo (finaliza con END). Si no hay, solo escribe END:")

    values = {
        "enunciado": enunciado,
        "input_desc": input_desc,
        "output_desc": output_desc,
        "restricciones": restricciones,
        "ejemplo_entrada": ejemplo_entrada,
        "ejemplo_salida": ejemplo_salida,
    }

    text_input = build_text_from_inputs(values)
    code_input = codigo or ""

    text_model = bundle.get("text_model")
    code_model = bundle.get("code_model")
    labels = bundle.get("labels", ["facil", "intermedio", "dificil"])
    tw = bundle.get("text_weight", 0.6)
    cw = bundle.get("code_weight", 0.4)

    text_probs = None
    code_probs = None

    if text_model is not None and text_input:
        text_probs = text_model.predict_proba([text_input])
    if code_model is not None and code_input:
        code_probs = code_model.predict_proba([code_input])

    probs = combine_probs(text_probs, code_probs, tw, cw)
    if probs is None:
        print("No hay texto ni codigo para predecir.")
        return

    probs = probs[0]
    idx = int(np.argmax(probs))
    pred = labels[idx]
    conf = float(probs[idx])

    out = {
        "dificultad": pred,
        "confianza": conf,
        "probabilidades": {labels[i]: float(probs[i]) for i in range(len(labels))},
    }

    print(out)


if __name__ == "__main__":
    main()
