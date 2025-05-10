import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


folder_path = "results" 

all_data = []

for file in os.listdir(folder_path):
    if file.endswith("_report.json"):
        model_name = file.split("_")[0]
        fold = file.split("_")[-1].replace(".json", "")
        with open(os.path.join(folder_path, file), "r") as f:
            metrics = json.load(f)
            all_data.append({
                "model": model_name,
                "fold": fold,
                "f1_macro": metrics["f1_macro"]
            })

df = pd.DataFrame(all_data)

summary = df.groupby("model").agg(
    mean_f1=("f1_macro", "mean")
).reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=summary, x="model", y="mean_f1", palette="viridis")

plt.title("Сравнение моделей по среднему F1-score (macro)", fontsize=14)
plt.xlabel("Модель", fontsize=12)
plt.ylabel("Средний F1-score", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

plt.ylim(0.5, 1.0)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("models_f1_comparison_zoomed.png", dpi=300)
plt.show()

summary.to_csv("models_average_f1.csv", index=False)
print("✅ Итоговые F1-оценки сохранены в 'models_average_f1.csv'")