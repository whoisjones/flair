import glob
import re
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from prettytable import PrettyTable

def plot_overview():
    sns.set_theme()
    sns.set_style(rc={
        'xtick.bottom': False,
        'ytick.left': True,
    })
    sns.set(font_scale=1.2)
    plt.figure(figsize=(6, 6))
    data = {
        "corpus": ["", "zelda + wikidata (ours)", "conll03", "wnut17", "fewnerd", "mit-movie", "mit-restaurant",
                   "ontonotes", "i2b2",
                   "germeval"],
        "distinct_labels": [817256, 817256, 4, 6, 66, 8, 12, 18, 23, 12],
        "label_length": [99.8, 99.8, 4.78, 9.62, 16.67, 8.65, 8.76, 7.32, 8.45, 7.72],
        "entity_mentions": [2593230, 2593230, 23256, 1949, 340180, 11123, 16474, 104151, 29233, 28278]}
    df = pd.DataFrame(data)
    markers = {"conll03": "o", "wnut17": "o", "fewnerd": "o", "mit-movie": "o", "mit-restaurant": "o", "ontonotes": "o",
               "i2b2": "o", "bc2gm": "o", "germeval": "o", "": ".", "zelda + wikidata (ours)": "D"}
    ax = sns.scatterplot(data=df, x="distinct_labels", y="label_length", hue="corpus", alpha=0.95, style="corpus",
                         markers=markers,
                         s=400)
    plt.legend(loc="lower center", bbox_to_anchor=(0.44, -0.4), bbox_transform=ax.transAxes, ncols=3, facecolor="white",
               frameon=False, markerscale=0.75, fontsize=11)
    plt.ylabel("Average label length")
    plt.xlabel("Number of distinct labels")
    plt.yscale("log")
    plt.xscale("log")
    plt.tight_layout()
    plt.show()

paths = glob.glob("/glusterfs/dfs-gfs-dist/goldejon/ner4all/tag_set_extension/fewshot_evaluation/*/results*")
def run():
    scores = {}

    for path in paths:
        if "jnlpba" in path:
            print()

        if "fewnerd" in path or "ontonotes" in path:
            fewshot_config, pretraining_config = path.split("/")[-2].split("__")
            fewshot_config = fewshot_config.split("-")
            pretraining_config = pretraining_config.split("-")
        else:
            fewshot_config = path.split("/")[-2].split("-")[:3]
            pretraining_config = path.split("/")[-2].split("-")[3:]

        with open(path, "r") as f:
            run = json.load(f)

        if "english" in fewshot_config[-1] or "arabic" in fewshot_config[-1] or "chinese" in fewshot_config[-1]:
            labels = "ner_tags"
            language = fewshot_config[2].split("_")[0]
        else:
            labels = fewshot_config[-1]
            language = None
        kshot = fewshot_config[1]

        if "zelda" in pretraining_config:
            size = pretraining_config[1]
            dataset = pretraining_config[2]
            model = pretraining_config[3]
            semantics = pretraining_config[4]
            negatives = pretraining_config[5]

            key = f"{dataset} {size} ({labels}, {model}, {semantics}, {negatives} {'' if language is None else language})"
        else:
            dataset = pretraining_config[2]
            model = pretraining_config[-1]

            key = f"{dataset} ({labels}, {model}{'' if language is None else ', ' + language})"

        if key not in scores:
            scores[key] = {kshot: [round(run["test_micro avg"]["f1-score"] * 100, 2)]}
        elif kshot not in scores[key]:
            scores[key][kshot] = [round(run["test_micro avg"]["f1-score"] * 100, 2)]
        else:
            scores[key][kshot].append(round(run["test_micro avg"]["f1-score"] * 100, 2))

    table = PrettyTable()

    for c in ["Model"] + [f"{kshot}-shot" for kshot in [0, 1, 5, 10]]:
        table.add_column(c, [])

    for model, results in scores.items():
        row = [model + f"(avg over {str(len(results))})"]
        for k in ["0shot", "1shot", "5shot", "10shot"]:
            if k not in results:
                row.append("-")
                continue
            mean = np.round(np.mean(results[k]), 2)
            std = np.round(np.std(results[k]), 2)
            row.append(f"{mean} +/- {std}")
        table.add_row(row)
    print(table)

if __name__ == "__main__":
    run()