# Jupyter Difficulty Model

Modelo de clasificación de dificultad para ejercicios de programación, usando texto del enunciado y código de solución.

## Estructura
- `train.py`: entrena el modelo y guarda `difficulty_model_bundle.joblib`.
- `predict.py`: prueba interactiva en consola con un ejercicio.
- `utils.py`: utilidades de formateo y combinación de probabilidades.
- `generate_dataset.py`: genera datasets sintéticos balanceados.
- `miDataSet_clean.csv`: dataset limpio base.
- `miDataSet_250.csv`: dataset generado (250 ejercicios).

## Requisitos
- Python 3.10+
- pandas
- scikit-learn
- joblib

## Entrenar
```bash
python3 train.py
```

## Probar
```bash
python3 predict.py
```

## Generar dataset
```bash
python3 generate_dataset.py
```

> Nota: Los datasets aquí son generados/curados para uso educativo.
