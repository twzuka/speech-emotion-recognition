import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

metadata = pd.read_csv(ROOT / "data/processed/metadata.csv")
settings = json.loads(
    (ROOT / "config/settings.json").read_text(encoding="utf-8")
)

train = metadata[metadata["split"] == "train"]

print("Длительность записей train, секунды:")
print(train["duration"].describe().round(2))

print("\nКоличество записей по числу каналов во всём датасете:")
print(metadata["channels"].value_counts().sort_index())

emotion_counts = (
    train["emotion"]
    .value_counts()
    .reindex(settings["emotions"], fill_value=0)
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

emotion_counts.plot.bar(ax=axes[0], color="steelblue")
axes[0].set_title("Эмоции в обучающей выборке")
axes[0].set_xlabel("Эмоция")
axes[0].set_ylabel("Количество записей")
axes[0].tick_params(axis="x", labelrotation=45)

axes[1].hist(
    train["duration"], bins=20,
    color="steelblue", edgecolor="white"
)
axes[1].set_title("Длительность записей train")
axes[1].set_xlabel("Длительность, секунды")
axes[1].set_ylabel("Количество записей")

fig.tight_layout()

output = ROOT / "reports/figures/train_data_overview.png"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=150)
plt.close(fig)

print(f"\nГрафик сохранён: {output.relative_to(ROOT)}")