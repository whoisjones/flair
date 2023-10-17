import json
import copy
import random
from collections import Counter

import datasets
from datasets import load_dataset

from label_name_map import semantic_label_name_map
from masking import mask_dataset, mask_full_dataset




def compute_fewnerd_fixed_targets_samples():

    finetuning_labels = ['other-medical', 'product-game', 'location-park', 'product-ship', 'building-sportsfacility',
                         'other-educationaldegree', 'building-airport', 'building-hospital', 'product-train',
                         'building-library', 'building-hotel', 'building-restaurant', 'event-disaster',
                         'event-election',
                         'event-protest', 'art-painting']

    def count_entity_mentions(tags):
        return [tags[i] for i in range(len(tags)) if
                (i == 0 or tags[i] != tags[i - 1]) and tags[i] != 0]

    finetuning_corpus = "DFKI-SLT/few-nerd"
    label_column = "fine_ner_tags"
    full_dataset = load_dataset(finetuning_corpus, "supervised")

    possible_pretraining_labels = set(semantic_label_name_map["few-nerd"][label_column].keys()) - set(finetuning_labels) - set("O")
    possible_pretraining_labels = sorted(list(possible_pretraining_labels))

    pretraining_indices = {}
    for num_labels in [3, 5, 10, 20, 30, 40, 50]:
        for seed in range(3):
            print(f"Computing fixed pretraining indices for FewNERD with fixed targets for {num_labels} labels and seed {seed}.")
            random.seed(seed)
            sampled_pretraining_labels = random.sample(possible_pretraining_labels, k=num_labels)
            sampled_pretraining_labels = sorted(sampled_pretraining_labels)
            max_entity_mentions = 10000
            pretraining_dataset = mask_dataset(full_dataset, label_column, "short", sampled_pretraining_labels)
            pretraining_dataset = pretraining_dataset["train"].shuffle(seed)
            label_counter = Counter()
            selected_idx = []
            for idx, example in enumerate(pretraining_dataset):
                mention_counter = count_entity_mentions(example[label_column])
                if mention_counter:
                    label_counter = label_counter + Counter(mention_counter)
                    selected_idx.append(idx)
                if sum(label_counter.values()) >= max_entity_mentions:
                    break
            print("Done.")
            pretraining_indices[f"{num_labels}-{seed}"] = {
                "indices": selected_idx,
                "shuffle_seed": seed,
                "labels": sampled_pretraining_labels,
            }

    print("Saving pre-training file.")
    with open("/glusterfs/dfs-gfs-dist/goldejon/ner4all/loss_function_experiments/fewnerd_fixed_targets/pretraining_fewnerd_indices.json", "w") as f:
        json.dump(pretraining_indices, f)
    print("Done.")

    fewshot_indices = {}
    for k in [1, 5, 10]:
        for seed in range(3):
            print(f"Computing fixed fewshot indices for FewNERD with fixed targets for {k} shots and seed {seed}.")
            fine_tuning_dataset = mask_dataset(full_dataset, label_column, finetuning_labels)
            fine_tuning_dataset = fine_tuning_dataset["validation"].shuffle(seed)
            label_counter = Counter()
            selected_fewshots = []
            all_labels = [idx for idx, label in enumerate(fine_tuning_dataset.features[label_column].feature.names)
                          if
                          label != "outside"]
            for idx, example in enumerate(fine_tuning_dataset):
                mention_counter = count_entity_mentions(example[label_column])
                counter_if_added = Counter(mention_counter) + label_counter
                if any([tag > k for tag in counter_if_added.values()]) or not mention_counter:
                    continue
                if all([tag <= k for tag in counter_if_added.values()]):
                    label_counter = label_counter + Counter(mention_counter)
                    selected_fewshots.append(idx)
                if all([tag == k for tag in label_counter.values()]) and set(label_counter.keys()) == set(
                        all_labels):
                    break
            print("Done.")
            fewshot_indices[f"{k}-{seed}"] = {
                "indices": selected_fewshots,
                "shuffle_seed": seed,
                "labels": finetuning_labels,
            }

    print("Saving fewshot file.")
    with open("/glusterfs/dfs-gfs-dist/goldejon/ner4all/loss_function_experiments/fewnerd_fixed_targets/fewshot_fewnerd_indices.json", "w") as f:
        json.dump(fewshot_indices, f)
    print("Done.")

def compute_indices_tag_set_extension():

    def count_entity_mentions(tags):
        return [tags[i] for i in range(len(tags)) if
                (i == 0 or tags[i] != tags[i - 1]) and tags[i] != 0]

    finetuning_corpus = "DFKI-SLT/few-nerd"
    full_dataset = load_dataset(finetuning_corpus, "supervised")

    labels = sorted(list(set(full_dataset["train"].features["ner_tags"].feature.names) - set("O")))
    pretraining_indices = {}

    for kshot in [1, 5, 10]:
        for seed in range(3):
            dataset = copy.deepcopy(full_dataset)
            random.seed(seed)
            pretraining_labels = random.sample(labels, k=4)
            finetuning_labels = [x for x in labels if x not in pretraining_labels]
            dataset = mask_full_dataset(dataset, "ner_tags", "long", pretraining_labels)
            dataset["train"] = dataset["train"].shuffle(seed)
            dataset["validation"] = dataset["validation"].shuffle(seed)
            label_counter = Counter()
            selected_idx = []

            num_pretraining_mentions = 0
            for idx, example in enumerate(dataset["train"]):
                mention_counter = count_entity_mentions(example["ner_tags"])
                if mention_counter:
                    num_pretraining_mentions += len(mention_counter)

            print("Computing pretraining mentions and evaluation indices.")
            for idx, example in enumerate(dataset["validation"]):
                mention_counter = count_entity_mentions(example["ner_tags"])
                counter_if_added = Counter(mention_counter) + label_counter
                if any([tag > kshot for tag in counter_if_added.values()]) or not mention_counter:
                    continue
                if all([tag <= kshot for tag in counter_if_added.values()]):
                    label_counter = label_counter + Counter(mention_counter)
                    selected_idx.append(idx)
                if all([tag == kshot for tag in label_counter.values()]) and set(label_counter.keys()) == set(
                        finetuning_labels):
                    break

            print("Done.")
            pretraining_indices[f"{kshot}-{seed}"] = {
                "indices": selected_idx,
                "shuffle_seed": seed,
                "pretraining_labels": pretraining_labels,
                "finetuning_labels": finetuning_labels,
                "num_pretraining_mentions": num_pretraining_mentions,
            }

    print("Saving pre-training file.")
    with open("/glusterfs/dfs-gfs-dist/goldejon/ner4all/tag_set_extension/fewnerd_indices.json", "w") as f:
        json.dump(pretraining_indices, f)

    print("Done.")

def compute_halves_tag_set_extension():

    def count_entity_mentions(tags):
        return [tags[i] for i in range(len(tags)) if
                (i == 0 or tags[i] != tags[i - 1]) and tags[i] != 0]

    finetuning_corpus = "DFKI-SLT/few-nerd"
    full_dataset = load_dataset(finetuning_corpus, "supervised")
    pretraining_indices = {}

    for label_column in ["ner_tags", "fine_ner_tags"]:

        labels = sorted(list(set(full_dataset["train"].features[label_column].feature.names) - set("O")))

        for kshot in [1, 5, 10]:
            for seed in range(3):
                dataset = copy.deepcopy(full_dataset)
                random.seed(seed)
                pretraining_labels = random.sample(labels, k=4 if label_column == "ner_tags" else 33)
                finetuning_labels = [x for x in labels if x not in pretraining_labels]
                dataset = mask_full_dataset(dataset, label_column, "long", pretraining_labels)
                dataset["train"] = dataset["train"].shuffle(seed)

                num_pretraining_mentions = 0
                for idx, example in enumerate(dataset["train"]):
                    mention_counter = count_entity_mentions(example[label_column])
                    if mention_counter:
                        num_pretraining_mentions += len(mention_counter)

                print("Computing pretraining mentions and evaluation indices.")
                fewshots = {}
                for fewshot_seed in range(3):
                    fewshot_indices = []
                    label_counter = Counter()
                    eval_split = copy.deepcopy(dataset["validation"]).shuffle(fewshot_seed)
                    for idx, example in enumerate(eval_split):
                        mention_counter = count_entity_mentions(example[label_column])
                        counter_if_added = Counter(mention_counter) + label_counter
                        if any([tag > kshot for tag in counter_if_added.values()]) or not mention_counter:
                            continue
                        if all([tag <= kshot for tag in counter_if_added.values()]):
                            label_counter = label_counter + Counter(mention_counter)
                            fewshot_indices.append(idx)
                        if all([tag == kshot for tag in label_counter.values()]) and set(label_counter.keys()) == set(
                                finetuning_labels):
                            break

                    fewshots[fewshot_seed] = fewshot_indices

                print("Done.")
                pretraining_indices[f"{label_column}-{kshot}-{seed}"] = {
                    "fewshot_indices": fewshots,
                    "shuffle_seed": seed,
                    "pretraining_labels": pretraining_labels,
                    "finetuning_labels": finetuning_labels,
                    "num_pretraining_mentions": num_pretraining_mentions,
                }

    print("Saving pre-training file.")
    with open("/glusterfs/dfs-gfs-dist/goldejon/ner4all/tag_set_extension/new_fewnerd_indices_5050.json", "w") as f:
        json.dump(pretraining_indices, f)

    print("Done.")

def process_ontonotes(dataset):

    tag_info = dataset["train"].features["sentences"][0]["named_entities"]

    def flatten(examples):
        sentences = [sentence for doc in examples["sentences"] for sentence in doc]
        examples["tokens"] = [sentence["words"] for sentence in sentences]
        examples["ner_tags"] = [sentence["named_entities"] for sentence in sentences]
        return examples

    dataset = dataset.map(flatten, batched=True, remove_columns=dataset["train"].column_names)
    dataset = dataset.map(lambda example, idx: {"id": idx}, with_indices=True)

    id2biolabel = {idx: label for idx, label in enumerate(tag_info.feature.names)}
    id2iolabel = {}
    label_mapping = {}
    for idx, label in id2biolabel.items():
        if label == "O":
            id2iolabel[len(id2iolabel)] = label
            label_mapping[idx] = len(id2iolabel) - 1
        elif label.startswith("B-") or label.startswith("I-"):
            io_label = label[2:]
            if io_label not in id2iolabel.values():
                io_label_id = len(id2iolabel)
                id2iolabel[io_label_id] = io_label
                label_mapping[idx] = io_label_id
            else:
                label_mapping[idx] = [k for k, v in id2iolabel.items() if v == io_label][0]

    def io_format(examples):
        examples["ner_tags"] = [[label_mapping.get(old_id) for old_id in sample] for sample in examples["ner_tags"]]
        return examples

    dataset = dataset.map(io_format, batched=True)

    tag_info.feature.names = list(id2iolabel.values())

    features = datasets.Features({
        "id": dataset["train"].features["id"],
        "tokens": dataset["train"].features["tokens"],
        "ner_tags": tag_info
    })

    dataset = dataset.cast(features)

    return dataset

def compute_ontonotes_indices():

    def count_entity_mentions(tags):
        return [tags[i] for i in range(len(tags)) if
                (i == 0 or tags[i] != tags[i - 1]) and tags[i] != 0]

    for language in ["english_v4", "arabic_v4", "chinese_v4"]:
        full_dataset = load_dataset("conll2012_ontonotesv5", language)
        full_dataset = process_ontonotes(full_dataset)
        pretraining_indices = {}

        label_column = "ner_tags"

        labels = sorted(list(set(full_dataset["train"].features[label_column].feature.names) - set("O")))

        for kshot in [1, 5, 10]:
            for seed in range(3):
                dataset = copy.deepcopy(full_dataset)
                random.seed(seed)
                pretraining_labels = random.sample(labels, k=9)
                finetuning_labels = [x for x in labels if x not in pretraining_labels]
                dataset = mask_full_dataset(dataset, label_column, "long", pretraining_labels)
                dataset["train"] = dataset["train"].shuffle(seed)

                num_pretraining_mentions = 0
                for idx, example in enumerate(dataset["train"]):
                    mention_counter = count_entity_mentions(example[label_column])
                    if mention_counter:
                        num_pretraining_mentions += len(mention_counter)

                print("Computing pretraining mentions and evaluation indices.")
                fewshots = {}
                for fewshot_seed in range(3):
                    fewshot_indices = []
                    label_counter = Counter()
                    eval_split = copy.deepcopy(dataset["validation"]).shuffle(fewshot_seed)
                    for idx, example in enumerate(eval_split):
                        mention_counter = count_entity_mentions(example[label_column])
                        counter_if_added = Counter(mention_counter) + label_counter
                        if any([tag > kshot for tag in counter_if_added.values()]) or not mention_counter:
                            continue
                        if all([tag <= kshot for tag in counter_if_added.values()]):
                            label_counter = label_counter + Counter(mention_counter)
                            fewshot_indices.append(idx)
                        if all([tag == kshot for tag in label_counter.values()]) and set(label_counter.keys()) == set(
                                finetuning_labels):
                            break

                    fewshots[fewshot_seed] = fewshot_indices

                print("Done.")
                pretraining_indices[f"{kshot}-{seed}"] = {
                    "fewshot_indices": fewshots,
                    "shuffle_seed": seed,
                    "pretraining_labels": pretraining_labels,
                    "finetuning_labels": finetuning_labels,
                    "num_pretraining_mentions": num_pretraining_mentions,
                }

        print("Saving pre-training file.")
        with open(f"/glusterfs/dfs-gfs-dist/goldejon/ner4all/tag_set_extension/ontonotes_{language}_indices_5050.json", "w") as f:
            json.dump(pretraining_indices, f)

    print("Done.")

def to_io_format(dataset):
    id_info = dataset["train"].features["id"]
    token_info = dataset["train"].features["tokens"]
    tag_info = dataset["train"].features["ner_tags"]

    id2biolabel = {idx: label for idx, label in enumerate(tag_info.feature.names)}
    id2iolabel = {}
    label_mapping = {}
    for idx, label in id2biolabel.items():
        if label == "O":
            id2iolabel[len(id2iolabel)] = label
            label_mapping[idx] = len(id2iolabel) - 1
        elif label.startswith("B-") or label.startswith("I-"):
            io_label = label[2:]
            if io_label not in id2iolabel.values():
                io_label_id = len(id2iolabel)
                id2iolabel[io_label_id] = io_label
                label_mapping[idx] = io_label_id
            else:
                label_mapping[idx] = [k for k, v in id2iolabel.items() if v == io_label][0]

    def io_format(examples):
        examples["id"] = examples["id"]
        examples["tokens"] = examples["tokens"]
        examples["ner_tags"] = [[label_mapping.get(old_id) for old_id in sample] for sample in examples["ner_tags"]]
        return examples

    dataset = dataset.map(io_format, batched=True)
    dataset = dataset.select_columns(["id", "tokens", "ner_tags"])

    tag_info.feature.names = list(id2iolabel.values())

    features = datasets.Features({
        "id": id_info,
        "tokens": token_info,
        "ner_tags": tag_info
    })

    dataset = dataset.cast(features)

    return dataset

def compute_full_mask_for_dataset():

    def count_entity_mentions(tags):
        return [tags[i] for i in range(len(tags)) if
                (i == 0 or tags[i] != tags[i - 1]) and tags[i] != 0]

    for dataset_name in ["chembenchmark"]:
        if dataset_name == "chembenchmark":
            features = datasets.Features({'id': datasets.Value(dtype='int64', id=None),
                                          'tokens': datasets.Sequence(feature=datasets.Value(dtype='string', id=None), length=-1, id=None),
                                          'ner_tags': datasets.Sequence(feature=datasets.Value(dtype='string', id=None), length=-1, id=None)})
            dataset = load_dataset("json", data_files={"train": "../chemdata/*train*", "test": "../chemdata/*dev*"}, features=features)

            def get_label_list(labels):
                # copied from https://github.com/huggingface/transformers/blob/66fd3a8d626a32989f4569260db32785c6cbf42a/examples/pytorch/token-classification/run_ner.py#L320
                unique_labels = set()
                for label in labels:
                    if label == "O":
                        continue
                    unique_labels = unique_labels | set(label)
                label_list = list(unique_labels)
                label_list = sorted(label_list, key=lambda x: (x[2:], x[0]))
                return label_list

            all_labels = get_label_list(dataset["train"]["ner_tags"])
            full_dataset = dataset.cast_column("ner_tags", datasets.Sequence(datasets.ClassLabel(names=all_labels)))
        else:
            full_dataset = load_dataset(dataset_name)

        full_dataset = to_io_format(full_dataset)

        pretraining_indices = {}

        label_column = "ner_tags"

        labels = sorted(list(set(full_dataset["train"].features[label_column].feature.names) - set("O")))

        for kshot in [1, 5, 10]:
            finetuning_labels = labels

            print("Computing pretraining mentions and evaluation indices.")
            fewshots = {}
            for fewshot_seed in range(3):
                dataset = copy.deepcopy(full_dataset)
                fewshot_indices = []
                label_counter = Counter()
                if dataset_name in ["jnlpba", "chembenchmark"]:
                    eval_split = copy.deepcopy(dataset["train"]).shuffle(fewshot_seed)
                else:
                    eval_split = copy.deepcopy(dataset["validation"]).shuffle(fewshot_seed)
                for idx, example in enumerate(eval_split):
                    mention_counter = count_entity_mentions(example[label_column])
                    counter_if_added = Counter(mention_counter) + label_counter
                    if any([tag > kshot for tag in counter_if_added.values()]) or not mention_counter:
                        continue
                    if all([tag <= kshot for tag in counter_if_added.values()]):
                        label_counter = label_counter + Counter(mention_counter)
                        fewshot_indices.append(idx)
                    if all([tag == kshot for tag in label_counter.values()]) and set(label_counter.keys()) == set(
                            finetuning_labels):
                        break

                fewshots[fewshot_seed] = fewshot_indices

            print("Done.")
            pretraining_indices[f"{kshot}"] = {
                "fewshot_indices": fewshots,
                "shuffle_seed": None,
                "pretraining_labels": [],
                "finetuning_labels": labels,
                "num_pretraining_mentions": 0,
            }

        print("Saving pre-training file.")
        with open(f"/glusterfs/dfs-gfs-dist/goldejon/ner4all/tag_set_extension/{dataset_name}_fewshots.json", "w") as f:
            json.dump(pretraining_indices, f)

    print("Done.")

if __name__ == "__main__":
    compute_full_mask_for_dataset()