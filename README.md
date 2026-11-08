---
title: predict-my-data
emoji: 🐧
colorFrom: blue
colorTo: gray
sdk: gradio
app_file: app.py
python_version: "3.11"
pinned: false
---

# Predict a penguin's body mass

Step 15 of the Palmer Penguins homework: the Keras regression network from the notebook
(Steps 26–34), served as a permanent Gradio app on Hugging Face Spaces.

- `model.keras` — the network, saved with Keras 3.13; the empty `quantization_config` entries were
  removed from its config so it also loads on Keras 3.12 (the newest version for Python 3.10)
- `prep.json` — the numbers from the notebook's `prep.joblib` (fill values, scaling, category lists,
  `y_mean`, `y_std`, `columns`, `typical`) as plain text, so the app needs no scikit-learn.
  `app.py` encodes rows exactly like `prep.transform` (checked on all 342 penguins: identical).
- `prep.joblib` — kept in this GitHub repo for reference; the Space does not need it
- `app.py` — one input per column, prefilled with the typical penguin, plus a Predict button;
  the prediction is un-scaled back to grams (`pred * y_std + y_mean`)

- `penguins.csv` — the cleaned Palmer Penguins table (342 rows), used for the scatter plot

The app ("Penguin Scale") has species buttons with a drawn picture of each penguin, sliders for the
three measurements, a result card with a weight bar, and a scatter plot of all 342 penguins with a
star for yours. Picking a species moves the sliders to that species' averages.

Inputs: species, island, bill_length_mm, bill_depth_mm, flipper_length_mm, sex.
