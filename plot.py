import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch



folder_path = "results" 
MODEL_NAMES = {
    "lr": "LR",
    "rf": "RF",
    "dt": "DT",
    "xgb": "XGB",
    "svm_lin": "SVM"
}

all_data = []

for file in os.listdir(folder_path):
    if file.endswith("_report.json"):
        model_name = file.split("_")[0].lower()
        fold = file.split("_")[-2]
        with open(os.path.join(folder_path, file), "r") as f:
            metrics = json.load(f)
            all_data.append({
                "model": MODEL_NAMES.get(model_name, model_name.upper()),
                "fold": fold,
                "f1_macro": metrics["f1_macro"]
            })

df = pd.DataFrame(all_data)

folds = sorted(df['fold'].unique(), key=lambda x: int(x.replace('fold', '')))
models = sorted(df['model'].unique())

summary = df.groupby("model").agg(
    mean_f1=("f1_macro", "mean")
).reset_index()

folds.append('mean')

palette = sns.color_palette("viridis", len(models))
model_to_color = {model: palette[i] for i, model in enumerate(models)}

fig, axes = plt.subplots(
    nrows=(len(folds) + 1) // 2,
    ncols=2,
    figsize=(14, 5 * ((len(folds) + 1) // 2)),
    sharey=True
)
axes = axes.flatten() 

for i, fold in enumerate(folds):
    ax = axes[i]

    if fold == 'mean':
        data_plot = summary.copy()
        data_plot['model'] = data_plot['model']
        sns.barplot(data=data_plot, x='model', y='mean_f1', ax=ax, palette=model_to_color, order=models)
        ax.set_title("Среднее F1 по всем фолдам", fontsize=14)
    else:
        data_fold = df[df['fold'] == fold]
        sns.barplot(data=data_fold, x='model', y='f1_macro', ax=ax, palette=model_to_color, order=models)
        ax.set_title(f"Fold {fold.replace('fold', '')}", fontsize=14)

    ax.set_xlabel("")
    ax.set_ylabel("F1-score", fontsize=12)
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='x', rotation=45)

for j in range(len(folds), len(axes)):
    fig.delaxes(axes[j])

legend_elements = [Patch(facecolor=model_to_color[model], label=model) for model in models]
fig.legend(handles=legend_elements, title="Модели", loc="upper right", bbox_to_anchor=(0.9, 0.85))

plt.suptitle("F1-score по фолдам и моделям", fontsize=18, y=1.02)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("f1_by_fold_and_model_fixed_colors.png", dpi=300, bbox_inches='tight')
plt.show()