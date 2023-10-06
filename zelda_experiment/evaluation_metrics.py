import glob
import re
import json
import numpy as np

from prettytable import PrettyTable

paths = glob.glob("/glusterfs/dfs-gfs-dist/goldejon/ner4all/tag_set_extension/fewshot_evaluation/*/results*")

scores = {}
for path in paths:
    fewshot_config, pretraining_config = path.split("/")[-2].split("__")
    fewshot_config = fewshot_config.split("-")
    pretraining_config = pretraining_config.split("-")

    with open(path, "r") as f:
        run = json.load(f)

    dataset = pretraining_config[2]
    labels = fewshot_config[-1]
    kshot = fewshot_config[1]
    model = pretraining_config[-1]

    key = f"{dataset} ({labels}, {model})"

    if key not in scores:
        scores[key] = {kshot: [round(run["test_micro avg"]["f1-score"] * 100, 2)]}
    elif kshot not in scores[key]:
        scores[key][kshot] = [round(run["test_micro avg"]["f1-score"] * 100, 2)]
    else:
        scores[key][kshot].append(round(run["test_micro avg"]["f1-score"] * 100, 2))

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