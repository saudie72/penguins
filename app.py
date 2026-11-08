"""Predict a Palmer penguin's body mass (g) with the Keras network from the notebook (Steps 26-34).

prep.json holds the numbers learned by the notebook's prep step (prep.joblib): fill values,
scaling and category lists, so the app can encode a row exactly as during training.
"""
import json
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import spaces  # Hugging Face ZeroGPU helper; import it before the ML libraries
import gradio as gr
import keras
import numpy as np

model = keras.models.load_model("model.keras")
with open("prep.json") as f:
    P = json.load(f)

NUM, CAT = P["numbers"], P["categories"]
columns, typical = P["columns"], P["typical"]


def encode(row):
    """Same as prep.transform: median-fill + standard-scale numbers, then one-hot categories."""
    parts = []
    for col, fill, mean, scale in zip(NUM["columns"], NUM["fill"], NUM["mean"], NUM["scale"]):
        v = row.get(col)
        v = fill if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
        parts.append((v - mean) / scale)
    for col, fill, values in zip(CAT["columns"], CAT["fill"], CAT["values"]):
        v = row.get(col)
        v = fill if v in (None, "") else str(v)
        parts.extend(1.0 if v == x else 0.0 for x in values)   # unknown value → all zeros
    return np.array([parts], dtype="float32")


@spaces.GPU
def zerogpu_check():
    """ZeroGPU Spaces refuse to start without at least one @spaces.GPU function.
    This small network runs fine on the CPU, so predict() below never needs the GPU
    and this function is never called."""
    return None


def predict(*values):
    out = model.predict(encode(dict(zip(columns, values))), verbose=0)
    return f"{float(out.ravel()[0]) * P['y_std'] + P['y_mean']:,.0f} g"


num_cols = set(NUM["columns"])
choices = dict(zip(CAT["columns"], CAT["values"]))
fields = [
    gr.Number(value=round(float(typical[c]), 2), label=c) if c in num_cols
    else gr.Dropdown(choices=choices[c], value=str(typical[c]), label=c)
    for c in columns
]

demo = gr.Interface(
    fn=predict,
    inputs=fields,
    outputs=gr.Textbox(label="Predicted body_mass_g"),
    title="Predict a penguin's body mass",
    description="Palmer Penguins · Keras network (64 → dropout 20% → 32 → 1). "
                "Every box starts at a typical penguin; change the values and click Predict.",
    submit_btn="Predict",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
