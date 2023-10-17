import os
import json
import copy
import logging
import random
import glob
from datetime import datetime
from pathlib import Path
from prettytable import PrettyTable
from seqeval.metrics import classification_report

import datasets
import lightning as L
import numpy as np
from transformers import AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset

from masking import mask_full_dataset
from models import BiEncoder
from label_name_map import semantic_label_name_map

task = "finetuning"

logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO (you can change it as needed)
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

pretained_models_folder = [
    "/glusterfs/dfs-gfs-dist/goldejon/ner4all/tag_set_extension/pretrained-models/fewnerd-bert-fine_ner_tags-seed-0",
    "/glusterfs/dfs-gfs-dist/goldejon/ner4all/tag_set_extension/pretrained-models/fewnerd-bert-fine_ner_tags-seed-1",
    "/glusterfs/dfs-gfs-dist/goldejon/ner4all/tag_set_extension/pretrained-models/fewnerd-bert-fine_ner_tags-seed-2",
    "/glusterfs/dfs-gfs-dist/goldejon/ner4all/tag_set_extension/pretrained-models/zelda-bert-small-dataset-with-full_desc-and-0-negatives-excluded-fine_ner_tags-seed-0",
    "/glusterfs/dfs-gfs-dist/goldejon/ner4all/tag_set_extension/pretrained-models/zelda-bert-small-dataset-with-full_desc-and-0-negatives-excluded-fine_ner_tags-seed-1",
    "/glusterfs/dfs-gfs-dist/goldejon/ner4all/tag_set_extension/pretrained-models/zelda-bert-small-dataset-with-full_desc-and-0-negatives-excluded-fine_ner_tags-seed-2",
    "/glusterfs/dfs-gfs-dist/goldejon/ner4all/tag_set_extension/pretrained-models/zelda-bert-small-dataset-with-sampled_desc-and-0-negatives-excluded-fine_ner_tags-seed-0",
    "/glusterfs/dfs-gfs-dist/goldejon/ner4all/tag_set_extension/pretrained-models/zelda-bert-small-dataset-with-sampled_desc-and-0-negatives-excluded-fine_ner_tags-seed-1",
    "/glusterfs/dfs-gfs-dist/goldejon/ner4all/tag_set_extension/pretrained-models/zelda-bert-small-dataset-with-sampled_desc-and-0-negatives-excluded-fine_ner_tags-seed-2",
]
finetuning_path = "/glusterfs/dfs-gfs-dist/goldejon/ner4all/tag_set_extension/fewshot_evaluation/"

def count_entity_mentions(tags):
    return [tags[i] for i in range(len(tags)) if
            (i == 0 or tags[i] != tags[i - 1]) and tags[i] != 0]

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

def get_chembenchmark():
    features = datasets.Features({'id': datasets.Value(dtype='int64', id=None),
                                  'tokens': datasets.Sequence(feature=datasets.Value(dtype='string', id=None),
                                                              length=-1, id=None),
                                  'ner_tags': datasets.Sequence(feature=datasets.Value(dtype='string', id=None),
                                                                length=-1, id=None)})
    dataset = load_dataset("json", data_files={"train": "chemdata/*train*", "validation": "chemdata/*dev*"},
                           features=features)

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
    return full_dataset

def zeroshot_evaluation():
    # --- Set seed ---
    L.seed_everything(123)

    # --- Load dataset ---
    jnlpba = to_io_format(load_dataset("jnlpba"))
    club = to_io_format(get_chembenchmark())
    datasets = [("jnlpba", jnlpba), ("chembenchmark", club)]

    for pretrained_model_path in pretained_models_folder:
        # --- Iterate over pretraining configs  ---
        for dataset_name, full_dataset in datasets:
            run_idx = int(pretrained_model_path.split("-")[-1]) + 1

            # --- Setup logging ---
            logger = logging.getLogger(f"{task}-0shot-seed-{run_idx}")

            # --- Copy sorted dataset ---
            dataset = copy.deepcopy(full_dataset)

            if "fewnerd" in pretrained_model_path:
                finetuning_extension = "run{run_idx}-{k}shot-{granularity}__pretrained-on-{pretraining_dataset}-{transformer}".format(
                    run_idx=run_idx,
                    k=0,
                    granularity=dataset_name,
                    pretraining_dataset="fewnerd",
                    transformer="bert",
                )
            elif "zelda" in pretrained_model_path:
                size = "small"
                zelda_label_granularity = "sampled_desc" if "sampled_desc" in pretrained_model_path else "full_desc"
                zelda_num_negatives = "0"
                finetuning_extension = "run{run_idx}-{k}shot-{granularity}_pretrained-on-{size}-{pretraining_dataset}-{transformer}-{zelda_label_granularity}-{zelda_num_negatives}negatives".format(
                    run_idx=run_idx,
                    k=0,
                    granularity=dataset_name,
                    size=size,
                    pretraining_dataset="zelda",
                    transformer="bert",
                    zelda_label_granularity=zelda_label_granularity,
                    zelda_num_negatives=zelda_num_negatives,
                )

            # --- Save path ---
            experiment_path = Path(finetuning_path + finetuning_extension)

            if not os.path.exists(experiment_path):
                os.makedirs(experiment_path)

            current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            file_handler_pretraining = logging.FileHandler(
                experiment_path / f"training_{current_time}.log"
            )
            logger.addHandler(file_handler_pretraining)

            # --- Mask dataset for pretraining ---
            id2label = {idx: semantic_label_name_map[dataset_name][f"ner_tags_short"].get(label) for idx, label
                        in enumerate(dataset["train"].features["ner_tags"].feature.names)}

            # --- Log config ---
            logger.info(30 * '-')
            logger.info(f"STARTING ZEROSHOT RUN")
            logger.info("Dataset: {}".format(dataset_name))
            logger.info("Label granularity: {}".format("ner_tags"))
            logger.info("Label semantic level: {}".format("short" if dataset_name == "fewnerd" else "long"))
            logger.info("Save path: {}".format(experiment_path))
            logger.info("Target labels: {}".format(id2label))
            logger.info("# Run: {}".format(run_idx))

            # --- Log pretraining config ---
            logger.info(10 * "-")
            logger.info("Pretrained model:")
            logger.info("Transformer: {}".format("bert"))
            if "zelda" in pretrained_model_path:
                logger.info("Zelda num examples: {}".format(size))
                logger.info("Zelda label granularity: {}".format(zelda_label_granularity))
                logger.info("Zelda number of negatives: {}".format(zelda_num_negatives))

            encoder_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path + "/encoder")
            decoder_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path + "/decoder")

            def align_labels_with_tokens(labels, word_ids):
                new_labels = []
                current_word = None
                for word_id in word_ids:
                    if word_id != current_word:
                        # Start of a new word!
                        current_word = word_id
                        label = -100 if word_id is None else labels[word_id]
                        new_labels.append(label)
                    elif word_id is None:
                        # Special token
                        new_labels.append(-100)
                    else:
                        # Same word as previous token
                        label = labels[word_id]
                        new_labels.append(label)

                return new_labels

            def tokenize_and_align_labels(examples):
                tokenized_inputs = encoder_tokenizer(
                    examples["tokens"], truncation=True, is_split_into_words=True
                )
                all_labels = examples["ner_tags"]
                new_labels = []
                for i, labels in enumerate(all_labels):
                    word_ids = tokenized_inputs.word_ids(i)
                    new_labels.append(align_labels_with_tokens(labels, word_ids))

                tokenized_inputs["labels"] = new_labels
                return tokenized_inputs

            def compute_metrics(eval_preds):
                logits, golds = eval_preds
                predictions = np.argmax(logits, axis=-1)

                def to_io_format(label):
                    if label == "outside" or label == "XO":
                        return "O"
                    else:
                        return "B-" + label

                # Remove ignored index (special tokens) and convert to labels
                y_true = [[to_io_format(id2label.get(l)) for l in gold if l != -100] for gold in golds]
                y_pred = [
                    [to_io_format(id2label.get(p)) for (p, l) in zip(prediction, gold) if l != -100]
                    for prediction, gold in zip(predictions, golds)
                ]

                logger.info("Scores for run:")
                all_metrics = classification_report(y_true, y_pred, output_dict=True)

                table = PrettyTable()
                for c in ["class", "precision", "recall", "f1-score", "support"]:
                    table.add_column(c, [])
                for _label, _scores in all_metrics.items():
                    if "micro" in _label:
                        table.add_row(5 * ["-"])
                    table.add_row([_label] + [round(_scores[c], 2) for c in
                                              ["precision", "recall", "f1-score", "support"]])
                logger.info(table)

                return all_metrics

            data_collator = DataCollatorForTokenClassification(encoder_tokenizer)

            training_args = TrainingArguments(
                output_dir=str(experiment_path),
                overwrite_output_dir=True,
                do_train=True,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=32,
                learning_rate=1e-5,
                warmup_ratio=0.1,
                save_strategy="no",
                save_total_limit=0,
                seed=123,
                num_train_epochs=1,
                logging_dir=str(experiment_path),
                logging_steps=5,
            )

            model = BiEncoder(
                encoder_model=pretrained_model_path + "/encoder",
                decoder_model=pretrained_model_path + "/decoder",
                tokenizer=decoder_tokenizer,
                labels=id2label,
                zelda_label_sampling=None,
                zelda_mask_size=0
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=None,
                eval_dataset=None,
                tokenizer=encoder_tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )

            test_dataset = dataset["validation"].map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=dataset["validation"].column_names,
            )

            logger.info("Start evaluation...")
            preds = trainer.predict(test_dataset)
            logger.info("Evaluation completed.")

            with open(experiment_path / "results.json", "w") as f:
                json.dump(preds.metrics, f)

            logger.info(f"ENDED ZEROSHOT RUN")
            logger.info(30 * '-')


if __name__ == "__main__":
    zeroshot_evaluation()