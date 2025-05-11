import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

FOLDERS = ["results", "results_dl"]

MODEL_NAMES = {
    "lr": "LR",
    "rf": "RF",
    "dt": "DT",
    "xgb": "XGB",
    "svm_lin": "SVM",
    "mlp_dl": "MLP",
    "cnn_1d": "CNN",
    "bilstm": "BiLSTM"
}

all_data = []

for folder_path in FOLDERS:
    for file in os.listdir(folder_path):
        if file.endswith("_report.json"):
            parts = file.split("_")
            model_name = parts[0].lower()
            fold = parts[-2]

            with open(os.path.join(folder_path, file), "r") as f:
                metrics = json.load(f)
                all_data.append({
                    "model": MODEL_NAMES.get(model_name, model_name.upper()),
                    "fold": fold,
                    "f1_macro": round(metrics["f1_macro"] * 100, 1),  # в процентах
                    "accuracy": round(metrics["accuracy"] * 100, 1)
                })

df = pd.DataFrame(all_data)

summary = df.groupby("model").agg(
    mean_f1=("f1_macro", "mean"),
    std_f1=("f1_macro", "std")
).reset_index()

summary = summary.sort_values(by="mean_f1", ascending=True).reset_index(drop=True)

models = summary['model'].tolist()
palette = sns.color_palette("viridis", len(models))
model_to_color = {model: palette[i] for i, model in enumerate(models)}


fig, ax = plt.subplots(figsize=(12, 6))

sns.scatterplot(data=summary, x='model', y='mean_f1', s=100, palette=model_to_color,
                hue='model', legend=False, ax=ax)
ax.plot(summary['model'], summary['mean_f1'], color='gray', linestyle='--', linewidth=1.5)

for i in range(1, len(summary)):
    prev = summary.iloc[i - 1]['mean_f1']
    curr = summary.iloc[i]['mean_f1']
    diff = curr - prev
    mid_x = (i + i - 1) / 2
    mid_y = (prev + curr) / 2
    ax.text(mid_x, mid_y + 0.2, f"+{diff:.1f}%", ha='center', fontsize=9, color='black')

ax.set_title("Сравнение моделей по F1-score (в процентах)", fontsize=14)
ax.set_xlabel("Модель", fontsize=12)
ax.set_ylabel("F1-score (%)", fontsize=12)

y_min = summary['mean_f1'].min() - 1
y_max = summary['mean_f1'].max() + 1
ax.set_ylim(y_min, y_max)

ax.grid(True, linestyle='--', alpha=0.5)

legend_elements = [Patch(facecolor=model_to_color[model], label=model) for model in models]
ax.legend(handles=legend_elements, title="Модели", loc="upper left", bbox_to_anchor=(1, 1))

for idx, row in summary.iterrows():
    ax.text(idx, row['mean_f1'] + 0.2, f"{row['mean_f1']:.1f}%", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("f1_by_model_percentage.png", dpi=300, bbox_inches='tight')
plt.show()
