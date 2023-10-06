import glob
import re
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from prettytable import PrettyTable

paths = glob.glob("/glusterfs/dfs-gfs-dist/goldejon/ner4all/loss_function_experiments/fewnerd_fixed_targets/ce/finetuning/*/result*")

def print_table():
    for s in ["simple", "short", "long"]:
        scores = {}
        for path in paths:
            k_shot = re.search(r'(\d+)shot', path).group(1)
            num_labels = re.search(r'with-(\d+)-', path).group(1)
            label_type = re.search(r'long|short|simple', path).group()

            if label_type != s:
                continue

            with open(path, "r") as f:
                run = json.load(f)

            key = f"{k_shot}-{num_labels}"

            if key not in scores:
                scores[key] = [round(run["test_micro avg"]["f1-score"] * 100, 2)]
            else:
                scores[key].append(round(run["test_micro avg"]["f1-score"] * 100, 2))

        table = PrettyTable()

        for c in ["Number of pretraining labels"] + [f"{kshot}-shot" for kshot in [1, 5, 10]]:
            table.add_column(c, [])

        for l in [3, 5, 10, 30, 50]:
            row = [l]
            for k in [1, 5, 10]:
                mean = np.round(np.mean(scores["{k}-{l}".format(k=k, l=l)]), 2)
                std = np.round(np.std(scores["{k}-{l}".format(k=k, l=l)]), 2)
                row.append(f"{mean} +/- {std}")
            table.add_row(row)
        print(table)

def to_dataframe():
    paths = glob.glob(
        "/glusterfs/dfs-gfs-dist/goldejon/ner4all/loss_function_experiments/fewnerd_fixed_targets/ce/finetuning/*/result*")
    scores = {}
    for s in ["simple", "short", "long"]:
        for path in paths:
            k_shot = re.search(r'(\d+)shot', path).group(1)
            num_labels = re.search(r'with-(\d+)-', path).group(1)
            label_type = re.search(r'long|short|simple', path).group()

            if label_type != s:
                continue

            with open(path, "r") as f:
                run = json.load(f)

            key = f"{s}-{k_shot}-{num_labels}"

            if key not in scores:
                scores[key] = [round(run["test_micro avg"]["f1-score"] * 100, 2)]
            else:
                scores[key].append(round(run["test_micro avg"]["f1-score"] * 100, 2))

    df_scores = {"setting": [], "kshot": [], "pt_labels": [], "mean": [], "std": []}
    for k, v in scores.items():
        setting, kshot, pt_labels = k.split("-")
        df_scores["Label semantic"].append(setting)
        df_scores["k-shot"].append(kshot)
        df_scores["L"].append(pt_labels)
        df_scores["avg. F1"].append(float(np.round(np.mean(v), 2)))
        df_scores["std. F1"].append(float(np.round(np.std(v), 2)))

    return df_scores

def plot_motivation_graph():
    def to_dataframe():
        paths = glob.glob(
            "/glusterfs/dfs-gfs-dist/goldejon/ner4all/loss_function_experiments/fewnerd_fixed_targets/ce/finetuning/*/result*")
        scores = {}
        for s in ["simple", "short", "long"]:
            for path in paths:
                k_shot = re.search(r'(\d+)shot', path).group(1)
                num_labels = re.search(r'with-(\d+)-', path).group(1)
                label_type = re.search(r'long|short|simple', path).group()
                if label_type != s:
                    continue
                with open(path, "r") as f:
                    run = json.load(f)
                key = f"{s}-{k_shot}-{num_labels}"
                if key not in scores:
                    scores[key] = [round(run["test_micro avg"]["f1-score"] * 100, 2)]
                else:
                    scores[key].append(round(run["test_micro avg"]["f1-score"] * 100, 2))
        df_scores = {"Label semantic": [], "k-shot": [], "L": [], "avg. F1": [], "std. F1": []}
        for k, v in scores.items():
            setting, kshot, pt_labels = k.split("-")
            df_scores["Label semantic"].append(setting)
            df_scores["k-shot"].append(kshot)
            df_scores["L"].append(pt_labels)
            df_scores["avg. F1"].append(float(np.round(np.mean(v), 2)))
            df_scores["std. F1"].append(float(np.round(np.std(v), 2)))
        return df_scores

    sns.set(font_scale=1)
    df = pd.DataFrame(to_dataframe())
    g = sns.FacetGrid(df, col="L", row="Label semantic", hue="k-shot", margin_titles=True, palette="viridis")
    g.map(sns.barplot, "k-shot", "avg. F1", width=0.5)
    for x, s in enumerate(["simple", "short", "long"]):
        for y, k in enumerate(["3", "5", "10", "30", "50"]):
            mean = df[(df["Label semantic"] == s) & (df["L"] == k)]["avg. F1"].mean()
            g.axes[x, y].axhline(mean, c='r', ls='--')
            g.axes[x, y].annotate(f"Avg: {mean:.1f}", xy=(0.1, mean + 1), ha="center")
    g.fig.subplots_adjust(top=0.92)
    g.fig.suptitle("L distinct labels observed during pre-training")
    plt.show()

if __name__ == "__main__":
    to_dataframe()