"""
This file helps to analyze the results produced by various scripts for few-shot domain transfer.
"""
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from pathlib import Path
import matplotlib.colors as mcolors

def recompute_results(path: str):
    """
    Recompute overall results file and returns it which in turn can be saved again to the target folder as results.json

    path: str = Path to the results folder that consists of typical folder structure 1shot_0, etc.
    """
    files = glob.glob(f"{path}/*")
    results = {}
    for file in files:
        if not file.endswith(".json"):
            k = file.split("/")[-1].split("_")[0].replace("shot", "")
            with open(file + "/result.txt", "r") as f:
                for line in f.readlines():
                    if "micro avg" in line:
                        f1_score = round(float(line.split()[-2]) * 100, 2)
            if k not in results:
                results[k] = {}
                results[k]["results"] = [f1_score]
            else:
                results[k]["results"].append(f1_score)
    for key, result_dict in results.items():
        results[key]["average"] = np.mean(result_dict["results"])
        results[key]["std"] = np.std(result_dict["results"])

    return results


def scores_per_class(path: str, target_keys: list):
    """
    Computes the scores per label for all experiments in target folder. Returns a dictionary containing scores per
    class per k-shot.

    path: str = Target directory with standard result structure.
    target_keys: list = List of labels applicable for analysis. All labels can be found in local_corpora.py
    """
    files = glob.glob(f"{path}/*")
    results = {}
    for file in files:
        if file.endswith(".json") or file.endswith(".txt") or file.endswith(".pt"):
            continue
        shot = file.split("/")[-1]
        number_of_shot = int(re.search(r"\d+", shot).group())
        if number_of_shot not in results.keys():
            results[number_of_shot] = {key: [] for key in target_keys}
        with open(file + "/result.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                for key in target_keys:
                    if key in line:
                        results[number_of_shot][key].append(line.split()[3])

    score_per_class = {}
    for key in sorted(results):
        res = results[key]
        for label, scores in res.items():
            if label not in score_per_class.keys():
                score_per_class[label] = {}

            if key not in score_per_class[label].keys():
                score_per_class[label][key] = round(np.mean([float(x) for x in scores]) * 100, 2)
    return score_per_class


def print_results():
    """
    Prints a line graph with multiple subplots with scores per class.
    Things to do before running this function:
    - Adjust figure title
    - Use correct target keys. (Below are labels for CONLL03 and WNUT17)
    - Provide description + path to target directory for plotting
    - Use applicable baselines (comment in/out in figure section)
    """

    figure_title = "Domain Transfer - CONLL03 - Learn initialization - baselines"
    target_keys = ["person", "location", "organization", "miscellaneous"]
    # target_keys = ["person", "location", "corporation", "creative work", "group", "product"]

    paths = {
        "LPFT (1e-2, exact)": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-1e-2_early-stopping_exact-matching",
        "LPFT (1e-2, compose)": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-1e-2_early-stopping_compose-matching",
        "Linear (exact)": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/baseline_conll_03_exact-matching",
        "Linear (compose)": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/baseline_conll_03_compose-matching",
        "Dual Encoder (GloVe)": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/fewshot-dual-encoder/bert-base-uncased_conll_03_1e-05_123_pretrained-on-ontonotes",
    }

    def sort_dict(d):
        sorted_keys = sorted(map(int, d.keys()))
        return {k: d[str(k)] for k in sorted_keys}

    results = {}
    results_per_class = {}
    for experiment, path in paths.items():
        with open(
            f"{path}/results.json",
            "r",
        ) as f:
            result = json.load(f)
            results[experiment] = sort_dict(result)

        results_per_class[experiment] = scores_per_class(path, target_keys)

    fig, axes = plt.subplot_mosaic(
        [["top row", "top row"]] + [target_keys[i : i + 2] for i in range(0, len(target_keys), 2)], figsize=(12, 12)
    )

    fig.suptitle(figure_title)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(results)))

    # BASELINES
    axes["top row"].plot(
        [1, 5, 20, 50], [44.8, 66.9, 77.5, 82.0], label="Linear (from paper)", color="tab:orange", linestyle="dotted"
    )
    axes["top row"].plot(
        [1, 5, 20, 50],
        [68.4, 76.7, 79.7, 83.1],
        label="Dual Encoder (from paper)",
        color="tab:blue",
        linestyle="dotted",
    )
    # axes["top row"].plot([1, 5, 20, 50], [27.6, 35.2, 40.9, 42.5], label="Linear (from paper)", color="tab:orange", linestyle="dotted")
    # axes["top row"].plot([1, 5, 20, 50], [38.3, 40.8, 42.7, 43.3], label="Dual Encoder (from paper)", color="tab:blue", linestyle="dotted")
    for c, (experiment, result) in zip(colors, results.items()):
        axes["top row"].plot(result.keys(), [x["average"] for x in result.values()], label=experiment, color=c)
        for label in target_keys:
            axes[label].plot(
                list(map(int, results_per_class[experiment][label].keys())),
                results_per_class[experiment][label].values(),
                label=experiment,
                color=c,
            )
    axes["top row"].set_title("Average F1 on Domain Transfer")
    axes["top row"].set_xlabel("k-shots")
    axes["top row"].set_ylabel("F1 score")
    axes["top row"].legend(loc="lower right")

    for label in target_keys:
        axes[label].set_title(f"Average F1 on label {label}")
        axes[label].set_xlabel("k-shots")
        axes[label].set_ylabel("F1 score")
        axes[label].legend(loc="lower right")

    fig.tight_layout(pad=0.9)
    fig.show()


def to_csv(save_path: str, save_path_per_class: str):
    """
    Save results as csv.

    save_path: str = The path for overall results.
    save_path_per_class = The path where to store results per class.
    """
    # target_keys = ["person", "location", "corporation", "creative work", "group", "product"]
    target_keys = ["person", "location", "organization", "miscellaneous"]

    paths = {
        "5e-1": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-5e-1_early-stopping_compose-matching",
        "1e-1": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-1e-1_early-stopping_compose-matching",
        "5e-2": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-5e-2_early-stopping_compose-matching",
        "1e-2": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-1e-2_early-stopping_compose-matching",
        "1e-3": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-1e-3_early-stopping_compose-matching",
        "1e-4": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-1e-4_early-stopping_compose-matching",
        "1e-5": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/LPFT/bert-base-uncased_conll_03_1e-05_123_onto-LPFT-1e-5_early-stopping_compose-matching",
        "Linear Baseline (compose)": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/reuse-weights-flert/baseline_conll_03_compose-matching",
        "Dual Encoder": "/glusterfs/dfs-gfs-dist/goldejon/flair-models/fewshot-dual-encoder/bert-base-uncased_conll_03_1e-05_123_pretrained-on-ontonotes",
    }

    def sort_dict(d):
        sorted_keys = sorted(map(int, d.keys()))
        return {k: d[str(k)] for k in sorted_keys}

    results = {}
    results_per_class = {}

    for experiment, path in paths.items():
        with open(
            f"{path}/results.json",
            "r",
        ) as f:
            result = json.load(f)
            results[experiment] = sort_dict(result)
        results_per_class[experiment] = scores_per_class(path, target_keys)

    df_data = {}
    for experiment, result_dict in results.items():
        for k, scores in result_dict.items():
            if k not in df_data:
                df_data[k] = [round(scores["average"], 2)]
            else:
                df_data[k].append(round(scores["average"], 2))

    df = pd.DataFrame(data=df_data, index=results.keys())
    df.to_csv(save_path)

    df_data = {}
    index_tuples = []
    for experiment, result_dict in results_per_class.items():
        for label, scores in result_dict.items():
            index_tuples.append((experiment, label))
            for k, score in scores.items():
                if k not in df_data:
                    df_data[k] = [round(score, 2)]
                else:
                    df_data[k].append(round(score, 2))

    index = pd.MultiIndex.from_tuples(index_tuples, names=["Experiment", "Label"])
    df = pd.DataFrame(data=df_data, index=index)
    df.to_csv(save_path_per_class)


def extract_single_run(path, k="1"):
    if k == "0":
        to_check = "result.txt"
    else:
        to_check = "training.log"
    with open(path / to_check, "r") as f:
        for line in f.readlines():
            if "micro avg" in line and line.split()[0] == "micro":
                return round(float(line.split()[-2]) * 100, 2)


def extract_multiple_runs(path, nested=False):

    files = (
        glob.glob(f"{path}/*10*")
        + glob.glob(f"{path}/*20*")
        + glob.glob(f"{path}/*30*")
        + glob.glob(f"{path}/*40*")
        + glob.glob(f"{path}/*50*")
    )
    results = {}
    for file in files:
        if nested:
            pattern = r"fewnerd-(.*?)-masked"
            fewshot_granularity = file.split("/")[-1].split("_")[1]
            match = re.search(pattern, fewshot_granularity)
            if match:
                fewshot_granularity = match.group(1)
            pretrain_granularity = file.split("/")[-1].split("_")[2]
            match = re.search(pattern, pretrain_granularity)
            if match:
                pretrain_granularity = match.group(1)
            exp_key = f"{pretrain_granularity}-to-{fewshot_granularity}"
            if exp_key not in results:
                results[exp_key] = {}
        else:
            k = file.split("/")[-1].split("_")[0].replace("shot", "")

        if nested:
            experiment_files = glob.glob(f"{file}/*")
            for exp_file in experiment_files:
                if ".json" not in exp_file:
                    k = exp_file.split("/")[-1].split("_")[0].replace("shot", "")
                    f1_score = extract_single_run(Path(file) / exp_file, k)
                    if k not in results[exp_key]:
                        results[exp_key][k] = {}
                        results[exp_key][k]["results"] = [f1_score]
                    else:
                        results[exp_key][k]["results"].append(f1_score)
        else:
            f1_score = extract_single_run(path / file, k)
            if k not in results:
                results[k] = {}
                results[k]["results"] = [f1_score]
            else:
                results[k]["results"].append(f1_score)

    if nested:
        for exp, result_dicts in results.items():
            for exp_k, result_dict in result_dicts.items():
                results[exp][exp_k]["average"] = np.mean(result_dict["results"])
                results[exp][exp_k]["std"] = np.std(result_dict["results"])
    else:
        for key, result_dict in results.items():
            results[key]["average"] = np.mean(result_dict["results"])
            results[key]["std"] = np.std(result_dict["results"])
    return results


def extract_x_y(result_dict):
    result_dict = {int(k): v for k, v in result_dict.items()}

    # Sort keys
    sorted_keys = sorted(result_dict)
    result_dict = {k: result_dict[k] for k in sorted_keys}

    y = np.array([v["average"] for k, v in result_dict.items()])
    sigma = np.array([v["std"] for k, v in result_dict.items()])
    x = np.array([k for k, v in result_dict.items()])
    lower_bound = y - sigma
    upper_bound = y + sigma
    lower_bound = lower_bound.clip(min=0)
    return x, y, lower_bound, upper_bound


def get_font_color(rgba, threshold=0.5):
    # Convert the RGBA color to an RGB color
    rgb = mcolors.to_rgb(rgba)

    # Calculate the luminance of the color
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    # Return a light font color if the luminance is below the threshold, otherwise a dark font color
    if luminance < threshold:
        return "white"
    else:
        return "black"


def extended_experiments():

    granularities = ["coarse", "fine", "coarse-fine", "coarse-without-misc"]
    pretraining_seeds = [10, 20, 30, 40, 50]

    pretrained_dual_encoder_path = Path(
        "/glusterfs/dfs-gfs-dist/goldejon/flair-models/pretrained-dual-encoder/masked-models"
    )
    low_resource_dual_encoder_path = Path(
        "/glusterfs/dfs-gfs-dist/goldejon/flair-models/lowresource-dual-encoder/masked-models"
    )
    low_resource_flert_path = Path("/glusterfs/dfs-gfs-dist/goldejon/flair-models/lowresource-flert/masked-models")
    fewshot_dual_encoder_path = Path("/glusterfs/dfs-gfs-dist/goldejon/flair-models/fewshot-dual-encoder/masked-models")

    full_finetuning_scores = {}
    for granularity in granularities:
        scores = []
        for pretraining_seed in pretraining_seeds:
            scores.append(
                extract_single_run(
                    pretrained_dual_encoder_path
                    / f"bert-base-uncased_fewnerd-{granularity}-inverse-masked_1e-05-{pretraining_seed}"
                )
            )
        full_finetuning_scores[granularity] = {
            "results": np.array(scores),
            "average": np.mean(scores),
            "std": np.std(scores),
        }

    low_resource_dual_encoder_results = {
        "coarse": extract_multiple_runs(
            low_resource_dual_encoder_path / "bert-base-uncased_fewnerd-coarse-masked_1e-05_early-stopping"
        ),
        "coarse-without-misc": extract_multiple_runs(
            low_resource_dual_encoder_path / "bert-base-uncased_fewnerd-coarse-without-misc-masked_1e-05_early-stopping"
        ),
        "fine": extract_multiple_runs(
            low_resource_dual_encoder_path / "bert-base-uncased_fewnerd-fine-masked_1e-05_early-stopping"
        ),
        "coarse-fine": extract_multiple_runs(
            low_resource_dual_encoder_path / "bert-base-uncased_fewnerd-coarse-fine-masked_1e-05_early-stopping"
        ),
    }

    low_resource_flert_results = {
        "coarse": extract_multiple_runs(
            low_resource_flert_path / "bert-base-uncased_fewnerd-coarse-masked_1e-05_early-stopping"
        ),
        "coarse-without-misc": extract_multiple_runs(
            low_resource_flert_path / "bert-base-uncased_fewnerd-coarse-without-misc-masked_1e-05_early-stopping"
        ),
        "fine": extract_multiple_runs(
            low_resource_flert_path / "bert-base-uncased_fewnerd-fine-masked_1e-05_early-stopping"
        ),
    }

    fewshot_results = extract_multiple_runs(fewshot_dual_encoder_path, nested=True)

    colors = {"coarse": "tab:blue", "coarse-without-misc": "tab:orange", "fine": "tab:green", "coarse-fine": "tab:red"}
    axes = {"coarse": (0, 0), "coarse-without-misc": (0, 1), "fine": (1, 0), "coarse-fine": (1, 1)}

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    fig.suptitle("Domain transfer within FewNERD on unseen labels")
    for granularity in granularities:

        axs[axes[granularity]].set_title(f"Few-shot on: {granularity}")
        axs[axes[granularity]].set_xlabel("k-shots per class")
        axs[axes[granularity]].set_ylabel("F1-score (span-level)")
        axs[axes[granularity]].set_xscale("log")

        x = np.array([0, 1, 2, 4, 8, 16, 32, 64])
        y = np.array([full_finetuning_scores[granularity]["average"]] * 8)
        sigma = [full_finetuning_scores[granularity]["std"]] * 8
        lower_bound = y - sigma
        upper_bound = y + sigma
        lower_bound = lower_bound.clip(min=0)
        axs[axes[granularity]].plot(x, y, color="black", linestyle="--", linewidth=1, label="full-finetuning")
        axs[axes[granularity]].fill_between(x, lower_bound, upper_bound, alpha=0.15, color="black")

        if granularity == "coarse-fine":
            x, y, lower_bound, upper_bound = extract_x_y(low_resource_flert_results["fine"])
        else:
            x, y, lower_bound, upper_bound = extract_x_y(low_resource_flert_results[granularity])
        axs[axes[granularity]].plot(x, y, linewidth=1, color="tab:brown", label="FLERT")
        axs[axes[granularity]].fill_between(x, lower_bound, upper_bound, alpha=0.15, color="tab:brown")

        x, y, lower_bound, upper_bound = extract_x_y(low_resource_dual_encoder_results[granularity])
        axs[axes[granularity]].plot(x, y, linewidth=1, color="tab:gray", label="no-pretraining")
        axs[axes[granularity]].fill_between(x, lower_bound, upper_bound, alpha=0.15, color="tab:gray")

        for exp, results in fewshot_results.items():
            if exp.endswith(f"to-{granularity}"):
                pretraining = exp.split("-to-")[0]
                x, y, lower_bound, upper_bound = extract_x_y(results)
                axs[axes[granularity]].plot(x, y, linewidth=1, color=colors[pretraining], label=pretraining)
                axs[axes[granularity]].fill_between(x, lower_bound, upper_bound, alpha=0.15, color=colors[pretraining])

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    plt.show()

    for k in ["0", "1", "2", "4", "8", "16", "32", "64"]:
        df = pd.DataFrame(index=granularities, columns=granularities)
        for exp, exp_results in fewshot_results.items():
            pretraining, fewshot = exp.split("-to-")
            df[fewshot][pretraining] = exp_results[k]["average"]
        df = pd.DataFrame(data=df.values.astype("float"), index=granularities, columns=granularities)

        # Plot the DataFrame as a matrix
        fig, ax = plt.subplots(figsize=(14, 14))
        im = ax.imshow(df, cmap="viridis")
        # Set axis labels
        ax.set_xticks(range(len(df.columns)))
        ax.set_yticks(range(len(df.index)))
        ax.set_xticklabels(df.columns)
        ax.set_yticklabels(df.index)
        ax.set_xlabel("Few-Shot on:")
        ax.set_ylabel("Pretrained on:")
        ax.grid(False)
        # Set axis labels to be displayed at 45 degrees
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # Loop over data dimensions and create text annotations
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                text = ax.text(
                    j,
                    i,
                    round(float(df.iloc[i, j]), 2),
                    ha="center",
                    va="center",
                )
                text.set_color(get_font_color(im.cmap(im.norm(df.iloc[i, j]))))
                text.set_fontsize(12)

        # Set title
        ax.set_title(f"Details on {k}-shots")
        # Add colorbar
        fig.colorbar(im)
        plt.show()

    df = pd.DataFrame(
        columns=granularities,
        index=pd.MultiIndex.from_product([["0", "1", "2", "4", "8", "16", "32", "64"], granularities]),
    )
    for exp, exp_results in fewshot_results.items():
        pretraining, fewshot = exp.split("-to-")
        for k in exp_results.keys():
            df.loc[k, fewshot][pretraining] = exp_results[k]["average"]
    print(df)


def loner_hyperparameters():
    for mask_size in ["128", "256"]:
        for lr in ["1e-05", "5e-05", "1e-06", "5e-06"]:

            prefix_path = "/glusterfs/dfs-gfs-dist/goldejon/flair-models/fewshot-dual-encoder/masked-models"
            base_path = f"{prefix_path}/bert-base-uncased_fewnerd-coarse_pretrained-on-LONER_lr-{lr}_seed-123_mask-{mask_size}_size-100k_1e-05_early-stopping/*"
            results_dirs = glob.glob(base_path)
            results = {}
            for result_dir in results_dirs:
                if result_dir.endswith("json"):
                    continue
                k = result_dir.split("/")[-1].split("_")[0].replace("shot", "")
                result = extract_single_run(Path(result_dir), k=k)

                if k not in results:
                    results[k] = {}
                    results[k]["scores"] = [result]
                else:
                    results[k]["scores"].append(result)

            for k, v in results.items():
                results[k]["average"] = np.mean(v["scores"])
                results[k]["std"] = np.std(v["scores"])

            total_avg = []
            for k, v in results.items():
                print(f"mask: {mask_size} - lr: {lr} - shot: {k} - result: ${round(v['average'], 1)} \pm {round(v['std'], 1)}$")
                total_avg.append(v['average'])

            print(f"mask: {mask_size} - lr: {lr} - total avg: {round(np.mean(total_avg), 1)}")


def main_experiment_per_class(base_path):
    from prettytable import PrettyTable
    paths = glob.glob(f"{base_path}/*")
    order = ["0", "1", "2", "4", "8", "16", "full"]
    all_results = {}

    def extract_single_run(path):
        scores_per_class = {}
        extract = False
        i = 0
        with open(path, "r") as f:
            for line in f.readlines():
                if "precision" in line:
                    extract = True
                    continue

                if extract and line.split() and i < 66:
                    scores_per_class[" ".join(line.split()[:-4])] = round(float(line.split()[-2]) * 100, 2)
                    i += 1

        return scores_per_class

    for path in paths:
        files = [x for x in glob.glob(f"{path}/*/*") if "training.log" in x or "result.txt" in x]
        results = {}
        for file in files:
            k = file.split("/")[-2].split("_")[0].replace("shot", "")
            k = k if k != "-1" else "full"
            if k not in results:
                results[k] = {"scores": [extract_single_run(file)]}
            else:
                results[k]["scores"].append(extract_single_run(file))

        for k, v in results.items():
            _tmp_avg = {}
            _tmp_std = {}
            for label in v["scores"][0].keys():
                _tmp_avg[label] = np.round(np.mean([s[label] for s in v["scores"]]), 2)
                _tmp_std[label] = np.round(np.std([s[label] for s in v["scores"]]), 2)
            _tmp_avg = dict(sorted(_tmp_avg.items(), key=lambda x: x[1], reverse=True))
            _tmp_std = {k: _tmp_std[k] for k in _tmp_avg.keys()}
            results[k]["avg"] = _tmp_avg
            results[k]["std"] = _tmp_std

        results = {k: results[k] for k in order}
        model = path.split('/')[-1].split("_", 1)[-1]
        all_results[model] = results

    for model, shots in all_results.items():
        print(f"Model: {model}")
        labels = list(shots["0"]["avg"].keys())
        table = PrettyTable()
        table.field_names = ["label"] + order
        for label in labels:
            row = [label]
            for k in order:
                row = row + [f"{shots[k]['avg'][label]} +/- {shots[k]['std'][label]}"]
            table.add_row(row)
        print(table)

def main_experiments(base_path, add_graph: bool = False):
    from prettytable import PrettyTable
    paths = glob.glob(f"{base_path}/*")
    paths = [path for path in paths if not ("fewnerdfine-1e-05_pretrained-on-bert-base-uncased_" in path or "fewnerdfine-5e-05_" in path)]
    order = ["0", "1", "2", "4", "8", "16", "full"]
    all_results = {}

    def extract_single_run(path):
        with open(path, "r") as f:
            for line in f.readlines():
                if "micro avg" in line and line.split()[0] == "micro":
                    return round(float(line.split()[-2]) * 100, 2)

    for path in paths:
        files = [x for x in glob.glob(f"{path}/*/*") if "training.log" in x or "result.txt" in x]
        results = {}
        for file in files:
            k = file.split("/")[-2].split("_")[0].replace("shot", "")
            k = k if k != "-1" else "full"
            if k not in results:
                results[k] = {"scores": [extract_single_run(file)]}
            else:
                results[k]["scores"].append(extract_single_run(file))

        for k, v in results.items():
            results[k]["avg"] = np.round(np.mean(v["scores"]), 2)
            results[k]["std"] = np.round(np.std(v["scores"]), 2)

        results = {k: results[k] for k in order}
        model = path.split('/')[-1].split("_", 1)[-1]
        all_results[model] = results

    table = PrettyTable()
    table.field_names = ["model"] + order
    for k, v in all_results.items():
        table.add_row([k] + [f"{y['avg']} +/- {y['std']}" for x, y in v.items()])
    print(table)

    if add_graph:
        plt.figure(figsize=(12, 8))
        for model, scores in all_results.items():
            x_values = list(scores.keys())
            y_values = [entry['avg'] for entry in scores.values()]
            std_values = [entry['std'] for entry in scores.values()]

            plt.plot(x_values, y_values, marker='o', label=model.replace("pretrained_on", ""))
            plt.fill_between(x_values, np.array(y_values) - np.array(std_values), np.array(y_values) + np.array(std_values),
                             alpha=0.2)

        plt.xlabel('k-shots')
        plt.ylabel('F1 score')
        plt.title('Few-Shot on increasing amount of seen labels')
        plt.legend(fontsize="small")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    loner_hyperparameters()