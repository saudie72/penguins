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

- `model.keras` — the network (Keras 3.13)
- `prep.json` — the numbers from the notebook's `prep.joblib` (fill values, scaling, category lists,
  `y_mean`, `y_std`, `columns`, `typical`) as plain text, so the app needs no scikit-learn.
  `app.py` encodes rows exactly like `prep.transform` (checked on all 342 penguins: identical).
- `prep.joblib` — kept in this GitHub repo for reference; the Space does not need it
- `app.py` — one input per column, prefilled with the typical penguin, plus a Predict button;
  the prediction is un-scaled back to grams (`pred * y_std + y_mean`)

Inputs: species, island, bill_length_mm, bill_depth_mm, flipper_length_mm, sex.
