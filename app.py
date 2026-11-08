"""Penguin Scale: predict a Palmer penguin's body mass (g) with the Keras network from the notebook
(Steps 26-34).

prep.json holds the numbers learned by the notebook's prep step (prep.joblib): fill values,
scaling and category lists, so the app can encode a row exactly as during training.
penguins.csv is the cleaned Palmer Penguins table (342 rows), used only for the plot.
"""
import json
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import spaces  # Hugging Face ZeroGPU helper; import it before the ML libraries
import gradio as gr
import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

model = keras.models.load_model("model.keras")
with open("prep.json") as f:
    P = json.load(f)
NUM, CAT = P["numbers"], P["categories"]
columns = P["columns"]
data = pd.read_csv("penguins.csv")

# ---------- look and feel ----------
COLORS = {"Adelie": "#F28C28", "Chinstrap": "#9B51E0", "Gentoo": "#1FA187"}
ISLANDS = {"Adelie": "Torgersen", "Chinstrap": "Dream", "Gentoo": "Biscoe"}
FACTS = {
    "Adelie": "The classic tuxedo penguin, with a white ring around each eye. "
              "Lives on all three islands.",
    "Chinstrap": "Named for the thin black stripe under its chin. "
                 "In this data it lives only on Dream island.",
    "Gentoo": "The big one: long flippers, an orange bill and a white patch over each eye. "
              "Lives only on Biscoe island.",
}
SPECIES_AVG = data.groupby("species")[["bill_length_mm", "bill_depth_mm",
                                       "flipper_length_mm", "body_mass_g"]].mean()
MASS_MIN, MASS_MAX = 2700, 6300


def penguin_svg(species, size=150):
    """A small drawn penguin; each species gets its own markings."""
    c = COLORS[species]
    bill = "#F25C05" if species == "Gentoo" else "#2B2B2B"
    marks = {
        "Adelie": '<circle cx="86" cy="52" r="9" fill="none" stroke="#fff" stroke-width="4"/>'
                  '<circle cx="114" cy="52" r="9" fill="none" stroke="#fff" stroke-width="4"/>',
        "Chinstrap": '<path d="M70 72 Q100 96 130 72" fill="none" stroke="#222" stroke-width="3"/>',
        "Gentoo": '<path d="M72 40 Q100 24 128 40" fill="none" stroke="#fff" stroke-width="7" '
                  'stroke-linecap="round"/>',
    }[species]
    face = "#fff" if species == "Chinstrap" else "#23262B"
    return f"""
<svg viewBox="0 0 200 200" width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg"
     role="img" aria-label="{species} penguin">
  <circle cx="100" cy="100" r="96" fill="{c}" opacity="0.18"/>
  <ellipse cx="100" cy="185" rx="52" ry="8" fill="#000" opacity="0.12"/>
  <ellipse cx="54" cy="118" rx="14" ry="40" fill="#23262B" transform="rotate(18 54 118)"/>
  <ellipse cx="146" cy="118" rx="14" ry="40" fill="#23262B" transform="rotate(-18 146 118)"/>
  <ellipse cx="100" cy="112" rx="50" ry="70" fill="#23262B"/>
  <ellipse cx="100" cy="126" rx="35" ry="54" fill="#FFFFFF"/>
  <ellipse cx="100" cy="58" rx="34" ry="30" fill="{face}"/>
  {"<path d='M66 58 Q100 22 134 58 Q100 40 66 58Z' fill='#23262B'/>" if species == "Chinstrap" else ""}
  {marks}
  <circle cx="86" cy="52" r="4" fill="{'#23262B' if species == 'Chinstrap' else '#fff'}"/>
  <circle cx="114" cy="52" r="4" fill="{'#23262B' if species == 'Chinstrap' else '#fff'}"/>
  <path d="M92 62 L108 62 L100 76 Z" fill="{bill}"/>
  <ellipse cx="84" cy="182" rx="13" ry="5" fill="#F2A33A"/>
  <ellipse cx="116" cy="182" rx="13" ry="5" fill="#F2A33A"/>
</svg>"""


def species_card(species):
    c = COLORS[species]
    avg = SPECIES_AVG.loc[species]
    return f"""
<div class="card" style="border-top:6px solid {c}">
  <div class="card-row">
    {penguin_svg(species, 120)}
    <div>
      <div class="card-title" style="color:{c}">{species}</div>
      <p class="fact">{FACTS[species]}</p>
      <p class="small">Average: <b>{avg.body_mass_g:,.0f} g</b> ·
         flipper {avg.flipper_length_mm:.0f} mm · bill {avg.bill_length_mm:.1f} mm</p>
    </div>
  </div>
</div>"""


def result_card(species, sex, grams):
    c = COLORS[species]
    pct = 100 * (np.clip(grams, MASS_MIN, MASS_MAX) - MASS_MIN) / (MASS_MAX - MASS_MIN)
    same = data[(data.species == species) & (data.sex == sex)].body_mass_g
    diff = grams - same.mean()
    word = "heavier" if diff >= 0 else "lighter"
    return f"""
<div class="card result" style="border-top:6px solid {c}">
  <div class="label">Predicted body mass</div>
  <div class="big" style="color:{c}">{grams:,.0f} g</div>
  <div class="small">about {grams / 1000:.1f} kg · roughly {grams / 355:.0f} cans of soda</div>
  <div class="bar"><div class="fill" style="width:{pct:.0f}%;background:{c}"></div>
    <div class="tick" style="left:{pct:.0f}%"></div></div>
  <div class="bar-labels"><span>{MASS_MIN:,} g</span><span>range in the data</span>
    <span>{MASS_MAX:,} g</span></div>
  <p class="small">That's <b>{abs(diff):,.0f} g {word}</b> than the average {sex} {species}
     ({same.mean():,.0f} g).</p>
</div>"""


def scatter(species, flipper, grams):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=110)
    for sp, part in data.groupby("species"):
        ax.scatter(part.flipper_length_mm, part.body_mass_g, s=26, alpha=0.55,
                   color=COLORS[sp], edgecolor="white", linewidth=0.5, label=sp)
    ax.scatter([flipper], [grams], marker="*", s=520, color=COLORS[species],
               edgecolor="#1B2A3A", linewidth=1.6, zorder=5, label="Your penguin")
    ax.set_xlabel("Flipper length (mm)")
    ax.set_ylabel("Body mass (g)")
    ax.set_title("Your penguin among all 342 penguins", fontsize=12, loc="left", color="#1B2A3A")
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------- model ----------
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


def predict_grams(row):
    out = model.predict(encode(row), verbose=0)
    return float(out.ravel()[0]) * P["y_std"] + P["y_mean"]


def run(species, island, sex, bill_length, bill_depth, flipper):
    row = {"species": species, "island": island, "sex": sex, "bill_length_mm": bill_length,
           "bill_depth_mm": bill_depth, "flipper_length_mm": flipper}
    grams = predict_grams(row)
    return result_card(species, sex, grams), scatter(species, flipper, grams)


def pick_species(species):
    """Jump the sliders to that species' averages and its home island."""
    avg = SPECIES_AVG.loc[species]
    return (species_card(species), ISLANDS[species], round(float(avg.bill_length_mm), 1),
            round(float(avg.bill_depth_mm), 1), int(round(float(avg.flipper_length_mm))))


# ---------- page ----------
HEADER = f"""
<div class="hero">
  <div class="hero-pics">{penguin_svg("Adelie", 70)}{penguin_svg("Chinstrap", 70)}{penguin_svg("Gentoo", 70)}</div>
  <h1>🐧 Penguin Scale</h1>
  <p>Pick a penguin, move the sliders, and a neural network guesses how much it weighs.<br>
  Trained on 342 penguins from the Palmer Archipelago, Antarctica.</p>
</div>"""

CSS = """
.gradio-container {background: linear-gradient(180deg,#E8F6FF 0%,#F7FBFF 45%,#FFFFFF 100%) !important;}
.hero {text-align:center;padding:26px 16px 18px;border-radius:20px;color:#fff;
       background:linear-gradient(135deg,#0B3C5D 0%,#1D6FA3 55%,#57B8E6 100%);
       box-shadow:0 10px 30px rgba(11,60,93,.25);}
.hero h1 {color:#fff;font-size:2.4rem;margin:6px 0 4px;}
.hero p {color:#E4F3FF;margin:0;font-size:1.02rem;line-height:1.5;}
.hero-pics {display:flex;justify-content:center;gap:10px;}
.hero-pics svg {filter:drop-shadow(0 4px 6px rgba(0,0,0,.25));}
.card {background:#fff;border-radius:16px;padding:16px 18px;box-shadow:0 6px 20px rgba(16,42,67,.08);}
.card-row {display:flex;gap:14px;align-items:center;}
.card-row svg {flex:none;}
.card-title {font-size:1.5rem;font-weight:800;}
.fact {margin:4px 0;color:#33475B;}
.small {color:#52667A;font-size:.92rem;margin:6px 0 0;}
.result .label {text-transform:uppercase;letter-spacing:.08em;font-size:.78rem;color:#52667A;}
.result .big {font-size:3rem;font-weight:800;line-height:1.1;}
.bar {position:relative;height:14px;border-radius:7px;background:#E6EEF5;margin-top:14px;overflow:visible;}
.bar .fill {height:100%;border-radius:7px;opacity:.85;}
.bar .tick {position:absolute;top:-5px;width:4px;height:24px;margin-left:-2px;background:#1B2A3A;border-radius:2px;}
.bar-labels {display:flex;justify-content:space-between;font-size:.75rem;color:#7A8B9C;margin-top:4px;}
footer {display:none !important;}
"""

THEME = gr.themes.Soft(primary_hue="sky", secondary_hue="orange", neutral_hue="slate",
                       radius_size="lg", font=[gr.themes.GoogleFont("Nunito"), "sans-serif"])

# The colours above are designed for a light page, so keep it light even when the
# visitor's browser is in dark mode (otherwise Gradio's dark theme mixes with them).
FORCE_LIGHT = """
() => {
  const light = () => document.body.classList.remove('dark');
  light();
  new MutationObserver(light).observe(document.body, {attributes: true, attributeFilter: ['class']});
}
"""

start = pick_species("Adelie")

with gr.Blocks(title="Penguin Scale") as demo:
    gr.HTML(HEADER)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            species = gr.Radio(list(COLORS), value="Adelie", label="Species")
            card = gr.HTML(start[0])
            with gr.Row():
                island = gr.Radio(CAT["values"][CAT["columns"].index("island")],
                                  value=start[1], label="Island")
                sex = gr.Radio(["female", "male"], value="male", label="Sex")
            bill_length = gr.Slider(32, 60, value=start[2], step=0.1, label="Bill length (mm)")
            bill_depth = gr.Slider(13, 22, value=start[3], step=0.1, label="Bill depth (mm)")
            flipper = gr.Slider(170, 232, value=start[4], step=1, label="Flipper length (mm)")
            btn = gr.Button("⚖️  Weigh my penguin", variant="primary", size="lg")
        with gr.Column(scale=6):
            result = gr.HTML()
            plot = gr.Plot(show_label=False)

    inputs = [species, island, sex, bill_length, bill_depth, flipper]
    gr.Examples(
        examples=[["Gentoo", "Biscoe", "male", 50.0, 16.0, 225],
                  ["Adelie", "Torgersen", "female", 37.0, 17.5, 185],
                  ["Chinstrap", "Dream", "male", 51.0, 19.5, 200]],
        inputs=inputs, outputs=[result, plot], fn=run, run_on_click=True,
        label="Try these penguins")

    # picking a species yourself jumps the sliders to that species' averages;
    # an example click only swaps the picture (its own slider values stay)
    species.input(pick_species, species, [card, island, bill_length, bill_depth, flipper]) \
        .then(run, inputs, [result, plot])
    species.change(species_card, species, card)
    btn.click(run, inputs, [result, plot])
    for comp in (island, sex):
        comp.change(run, inputs, [result, plot])
    for comp in (bill_length, bill_depth, flipper):
        comp.release(run, inputs, [result, plot])
    demo.load(run, inputs, [result, plot])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, js=FORCE_LIGHT)
