import os
import json
import copy
import logging
import random
from datetime import datetime
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch
from transformers import AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer
import datasets
from datasets import load_dataset

from masking import mask_full_dataset
from models import BiEncoder, LEAR
from label_name_map import semantic_label_name_map

task = "pretraining"

logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO (you can change it as needed)
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

gluster_path = "/glusterfs/dfs-gfs-dist/goldejon/ner4all/acl_submission/pretrained-models/{dataset}"


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

def count_entity_mentions(tags):
    return [tags[i] for i in range(len(tags)) if
            (i == 0 or tags[i] != tags[i - 1]) and tags[i] != 0]

def pretrain(args):
    # --- Set seed ---
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    # --- Load dataset ---
    if args.dataset == "fewnerd":
        full_dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")
    elif args.dataset == "zelda":
        full_dataset = load_dataset("json", data_files="/glusterfs/dfs-gfs-dist/goldejon/ner4all/loner/jsonl/*")
    elif args.dataset == "ontonotes":
        full_dataset = process_ontonotes(load_dataset("conll2012_ontonotesv5", "english_v4"))
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    # --- Load pretraining indices ---
    with open(f"/glusterfs/dfs-gfs-dist/goldejon/ner4all/tag_set_extension/{args.masking_dataset}_indices_5050.json", "r") as f:
        pretraining_indices = json.load(f)

    if args.masking_dataset == "ontonotes":
        label_columns = ["ner_tags"]
    else:
        label_columns = ["ner_tags", "fine_ner_tags"]

    # --- Iterate over pretraining configs  ---
    for label_column in label_columns:
        for seed in [0, 1, 2]:

            # --- Setup logging ---
            logger = logging.getLogger(f"{task}-{seed}")
            dataset = copy.deepcopy(full_dataset)

            if args.dataset in ["fewnerd", "ontonotes"]:
                pretraining_extension = "-{model}-{transformer}-{label_column}-seed-{seed}".format(
                    model=args.model,
                    transformer=args.transformer_model,
                    label_column=label_column,
                    seed=seed
                )
            elif args.dataset == "zelda":
                pretraining_extension = "-{model}-{transformer}-{entity_mentions}-dataset-with-{zelda_sampling}-and-{num_negatives}-negatives-excluded-{masking_dataset}-{label_column}-seed-{seed}".format(
                    model=args.model,
                    transformer=args.transformer_model,
                    entity_mentions=args.zelda_entity_mentions,
                    zelda_sampling=args.zelda_sampling,
                    num_negatives=args.zelda_num_negatives,
                    masking_dataset=args.masking_dataset,
                    label_column=label_column,
                    seed=seed,
                )
            else:
                raise ValueError(f"Unknown dataset {args.dataset}")

            # --- Save path ---
            experiment_path = Path(gluster_path.format(dataset=args.dataset) + pretraining_extension)

            if not os.path.exists(experiment_path):
                os.makedirs(experiment_path)

            current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            file_handler_pretraining = logging.FileHandler(
                experiment_path / f"training_{current_time}.log"
            )
            logger.addHandler(file_handler_pretraining)

            # --- Mask dataset for pretraining ---
            indices = pretraining_indices[f"{label_column}-1-{seed}"]
            pretraining_labels = indices["pretraining_labels"]

            if args.dataset in ["fewnerd", "ontonotes"]:
                random.seed(seed)
                dataset = mask_full_dataset(dataset, label_column, args.label_semantic_level, pretraining_labels)
                pretraining_dataset = dataset["train"].shuffle(seed)
                id2label = {
                    idx: label for idx, label in enumerate(pretraining_dataset.features[label_column].feature.names)
                }

                # --- QA Check that we sample the correct pre-computed entities---
                assert sum([len(count_entity_mentions(x)) for x in pretraining_dataset[label_column]]) <= indices["num_pretraining_mentions"] + 25
                pretraining_labels_qa = [
                    semantic_label_name_map["few-nerd" if args.dataset == "fewnerd" else "conll2012_ontonotesv5"][f"{label_column}_{args.label_semantic_level}"].get(x) for x in
                    pretraining_labels]
                assert set([x for x in pretraining_dataset.features[label_column].feature.names if x != "outside"]) == set(pretraining_labels_qa)
                logger.info("QA Check passed. Number of pretraining mentions is consistent.")

            elif args.dataset == "zelda":
                with open("/glusterfs/dfs-gfs-dist/goldejon/ner4all/loner/labelID2label.json", "r") as f:
                    id2label = json.load(f)

                if label_column == "fine_ner_tags":
                    mask_labels = list(set([x.split("-")[1] for x in pretraining_labels]))
                else:
                    mask_labels = pretraining_labels

                # --- Exclude labels that are not in the pretraining set ---
                logger.info("Exclude ZELDA labels that are not in the pretraining set...")
                new_label2idx = {}
                exclusions = []
                for k, vals in id2label.items():
                    if vals == "O":
                        new_label2idx[k] = vals
                        continue
                    new_label = {}
                    if vals.get("description"):
                        if all([label not in vals["description"] for label in mask_labels]):
                            new_label["description"] = vals["description"]
                    if vals.get("labels"):
                        new_label["labels"] = [label for label in vals["labels"] if not (
                                    label in mask_labels or any([x in label for x in mask_labels]))]
                    if not new_label:
                        exclusions.append(int(k))
                        new_label["description"] = "miscellaneous"
                    new_label2idx[k] = new_label

                id2label = new_label2idx
                logger.info("Done.")

                if args.zelda_entity_mentions == "small":
                    # --- Filter ZELDA until number of pretraining mentions is reached ---
                    logger.info("Filter ZELDA until number of pretraining mentions is reached...")
                    pretraining_dataset = dataset["train"].shuffle(seed)
                    num_pretraining_mentions = indices["num_pretraining_mentions"]
                    zelda_mentions = 0
                    zelda_distinct_labels = set()
                    selected_idx = []
                    for idx, example in enumerate(pretraining_dataset):
                        mention_counter = count_entity_mentions(example["ner_tags"])
                        zelda_distinct_labels = zelda_distinct_labels.union(set(mention_counter))
                        if any([x in exclusions for x in mention_counter]):
                            continue
                        if mention_counter:
                            zelda_mentions += len(mention_counter)
                            selected_idx.append(idx)
                        if zelda_mentions >= num_pretraining_mentions:
                             break
                    pretraining_dataset = pretraining_dataset.select(selected_idx)
                    logger.info("Done.")
                else:
                    logger.info("Use full ZELDA dataset...")
                    pretraining_dataset = dataset["train"].shuffle(seed)
                    logger.info("Length of full ZELDA dataset: {}".format(len(pretraining_dataset)))
                    pretraining_dataset = pretraining_dataset.filter(lambda example: list(set(example["ner_tags"])) != [0])
                    logger.info("Length of full ZELDA dataset after filtering: {}".format(len(pretraining_dataset)))
                    logger.info("Done.")
            else:
                raise ValueError(f"Unknown dataset {args.dataset}")

            # --- Log config ---
            logger.info(30 * '-')
            logger.info(f"STARTING PRETRAINING RUN")
            logger.info("Config:")
            logger.info("Dataset: {}".format(args.dataset))
            if args.dataset in ["fewnerd", "ontonotes"]:
                logger.info("Number of pretraining labels: {}".format(len(pretraining_dataset.features[label_column].feature.names)))
                logger.info("Pretraining labels: {}".format(pretraining_dataset.features[label_column].feature.names))
                logger.info("Label semantic level: {}".format(args.label_semantic_level))
            elif args.dataset == "zelda":
                if args.zelda_entity_mentions == "small":
                    logger.info("Number of entity mentions: {}".format(zelda_mentions))
                    logger.info("ZELDA number of distinct labels: {}".format(len(zelda_distinct_labels)))
                logger.info("ZELDA sampling: {}".format(args.zelda_sampling))
                logger.info("ZELDA corpus size: {}".format(args.zelda_entity_mentions))
                logger.info("ZELDA number of negatives: {}".format(args.zelda_num_negatives))
            logger.info("Pretraining seed: {}".format(seed))
            logger.info("Save path: {}".format(experiment_path))
            logger.info("Transformer model: {}".format(args.transformer_model))
            logger.info("Learning rate: {}".format(args.lr))
            logger.info("Number of epochs: {}".format(args.num_epochs))

            encoder_tokenizer = AutoTokenizer.from_pretrained(args.transformer_model)
            decoder_tokenizer = AutoTokenizer.from_pretrained(args.transformer_model)

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
                    examples["tokens"], max_length=128, truncation=True, is_split_into_words=True
                )
                if args.dataset == "zelda":
                    all_labels = examples["ner_tags"]
                else:
                    all_labels = examples[label_column]
                new_labels = []
                for i, labels in enumerate(all_labels):
                    word_ids = tokenized_inputs.word_ids(i)
                    new_labels.append(align_labels_with_tokens(labels, word_ids))

                tokenized_inputs["labels"] = new_labels
                return tokenized_inputs

            train_dataset = pretraining_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=pretraining_dataset.column_names,
            )

            data_collator = DataCollatorForTokenClassification(encoder_tokenizer)

            training_args = TrainingArguments(
                output_dir=str(experiment_path),
                overwrite_output_dir=True,
                do_train=True,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                learning_rate=args.lr,
                warmup_ratio=0.1,
                save_strategy="epoch",
                save_total_limit=1,
                seed=123,
                num_train_epochs=args.num_epochs,
                logging_dir=str(experiment_path),
                logging_steps=100,
            )

            if args.model == "biencoder":
                model = BiEncoder(
                    encoder_model=args.transformer_model,
                    decoder_model=args.transformer_model,
                    tokenizer=decoder_tokenizer,
                    labels=id2label,
                    zelda_label_sampling=args.zelda_sampling,
                    zelda_mask_size=args.zelda_num_negatives,
                )
            elif args.model == "lear":
                model = LEAR(
                    encoder_model=args.transformer_model,
                    decoder_model=args.transformer_model,
                    tokenizer=decoder_tokenizer,
                    labels=id2label,
                    zelda_label_sampling=args.zelda_sampling,
                    zelda_mask_size=args.zelda_num_negatives,
                )
            else:
                raise ValueError(f"Unknown model {args.model}")

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=None,
                tokenizer=encoder_tokenizer,
                data_collator=data_collator,
            )
            logger.info("Start training...")
            trainer.train()
            logger.info("Pretraining completed.")

            logger.info("Log history:")
            for log_step in trainer.state.log_history:
                logger.info(log_step)

            model.encoder.save_pretrained(experiment_path / "encoder")
            encoder_tokenizer.save_pretrained(experiment_path / "encoder")
            model.decoder.save_pretrained(experiment_path / "decoder")
            decoder_tokenizer.save_pretrained(experiment_path / "decoder")
            logger.info(f"Saved pretrained model and tokenizer to {experiment_path}.")

            with open(f"{experiment_path}/training_args.json", "w") as f:
                json.dump(training_args.to_json_string(), f)

            logger.info(f"ENDED PRETRAINING RUN")
            logger.info(30 * '-')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, choices=["biencoder", "lear"])
    parser.add_argument("--dataset", type=str, choices=["fewnerd", "zelda", "ontonotes"])
    parser.add_argument("--masking_dataset", type=str, choices=["fewnerd", "ontonotes"], default=None)
    parser.add_argument("--label_semantic_level", type=str, default="short")
    parser.add_argument("--zelda_entity_mentions", type=str, choices=["small", "full"], required=False)
    parser.add_argument("--zelda_sampling", type=str, choices=["full_desc", "sampled_desc", "only_labels", "only_desc"], required=False)
    parser.add_argument("--zelda_num_negatives", type=int, default=0)
    parser.add_argument("--transformer_model", type=str, default="bert-base-uncased")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    config = parser.parse_args()
    pretrain(config)
