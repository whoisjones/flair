import os
import json
import copy
import logging
import random
from datetime import datetime
from pathlib import Path
from argparse import ArgumentParser
from prettytable import PrettyTable
from seqeval.metrics import classification_report

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

pretained_models_folder = "/glusterfs/dfs-gfs-dist/goldejon/ner4all/tag_set_extension/pretrained-models/"
finetuning_path = "/glusterfs/dfs-gfs-dist/goldejon/ner4all/tag_set_extension/fewshot_evaluation/"

def count_entity_mentions(tags):
    return [tags[i] for i in range(len(tags)) if
            (i == 0 or tags[i] != tags[i - 1]) and tags[i] != 0]

def fewshot_evaluation(args):
    # --- Set seed ---
    L.seed_everything(123)

    # --- Load dataset ---
    full_dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")

    # --- Load pretraining indices ---
    with open("/glusterfs/dfs-gfs-dist/goldejon/ner4all/tag_set_extension/fewnerd_indices_5050.json", "r") as f:
        pretraining_indices = json.load(f)

    # --- Iterate over pretraining configs  ---
    for config, indices in pretraining_indices.items():

        # --- Parse config ---
        label_column, kshot, pretraining_seed = config.split("-")

        if kshot == "1":
            num_epochs = 100
        elif kshot == "5":
            num_epochs = 75
        elif kshot == "10":
            num_epochs = 50
        else:
            raise ValueError(f"Unknown kshot {kshot}")

        # --- Parse pretraining information from path ---
        pretrained_model_config = args.pretrained_model_path.split("/")[-1].split("-")
        dataset_name = pretrained_model_config[0]
        transformer = pretrained_model_config[1]
        if dataset_name == "fewnerd":
            pretraining_label_column = pretrained_model_config[2]
        elif dataset_name == "zelda":
            pretraining_label_column = pretrained_model_config[-3]
        if label_column != pretraining_label_column:
            continue
        pretrained_model_path = pretained_models_folder + args.pretrained_model_path + pretraining_seed
        pretraining_seed = int(pretraining_seed)

        for fewshot_seed, fewshot_indices in indices["fewshot_indices"].items():
            fewshot_seed = int(fewshot_seed)
            run_idx = (fewshot_seed + 1) + (pretraining_seed * 3)

            # --- Setup logging ---
            logger = logging.getLogger(f"{task}-kshot-{kshot}-seed-{run_idx}")

            # --- Copy sorted dataset ---
            dataset = copy.deepcopy(full_dataset)

            if dataset_name == "fewnerd":
                finetuning_extension = "run{run_idx}-{k}shot-{granularity}__pretrained-on-{pretraining_dataset}-{transformer}".format(
                    run_idx=run_idx,
                    k=kshot,
                    granularity=label_column,
                    pretraining_dataset=dataset_name,
                    transformer=transformer,
                )
            elif dataset_name == "zelda":
                size = pretrained_model_config[2]
                zelda_label_granularity = pretrained_model_config[5]
                zelda_num_negatives = pretrained_model_config[7]
                finetuning_extension = "run{run_idx}-{k}shot-{granularity}_pretrained-on-{size}-{pretraining_dataset}-{transformer}-{zelda_label_granularity}-{zelda_num_negatives}negatives".format(
                    run_idx=run_idx,
                    k=kshot,
                    granularity=label_column,
                    size=size,
                    pretraining_dataset=dataset_name,
                    transformer=transformer,
                    zelda_label_granularity=zelda_label_granularity,
                    zelda_num_negatives=zelda_num_negatives,
                )
            else:
                raise ValueError(f"Unknown dataset {dataset_name}")

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
            pretraining_labels = indices["pretraining_labels"]
            finetuning_labels = indices["finetuning_labels"]
            random.seed(pretraining_seed)
            dataset = mask_full_dataset(dataset, label_column, "short" if dataset_name == "fewnerd" else "long", pretraining_labels)
            fewshot_dataset = dataset["validation"].shuffle(fewshot_seed).select(fewshot_indices)
            id2label = {idx: label for idx, label in enumerate(fewshot_dataset.features[label_column].feature.names)}

            # --- QA Check that we sample the correct pre-computed entities---
            assert sum([len(count_entity_mentions(x)) for x in fewshot_dataset[label_column]]) <= int(kshot) * len(finetuning_labels)
            finetuning_labels_qa = [
                semantic_label_name_map["few-nerd"][f"{label_column}_{'short' if dataset_name == 'fewnerd' else 'long'}"].get(x) for x in
                finetuning_labels]
            assert set([x for x in fewshot_dataset.features[label_column].feature.names if x != "outside"]) == set(finetuning_labels_qa)
            logger.info("QA Check passed. Number of fewshot examples is correct.")

            # --- Log config ---
            logger.info(30 * '-')
            logger.info(f"STARTING FINETUNING RUN")
            logger.info("Fewshot on:")
            logger.info("Dataset: {}".format(dataset_name))
            logger.info("K-shot: {}".format(kshot))
            logger.info("Number of fewshot sentences: {}".format(len(fewshot_dataset)))
            logger.info("Label granularity: {}".format(label_column))
            logger.info("Label semantic level: {}".format("short" if dataset_name == "fewnerd" else "long"))
            logger.info("Save path: {}".format(experiment_path))
            logger.info("Target labels: {}".format(finetuning_labels))
            logger.info("# Run: {}".format(run_idx))

            # --- Log pretraining config ---
            logger.info(10 * "-")
            logger.info("Pretrained model:")
            logger.info("Transformer: {}".format(transformer))
            if dataset_name == "zelda":
                logger.info("Zelda num examples: {}".format(size))
                logger.info("Zelda label granularity: {}".format(zelda_label_granularity))
                logger.info("Zelda number of negatives: {}".format(zelda_num_negatives))
            logger.info("Learning rate: {}".format(args.lr))
            logger.info("Number of epochs: {}".format(num_epochs))

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
                all_labels = examples[label_column]
                new_labels = []
                for i, labels in enumerate(all_labels):
                    word_ids = tokenized_inputs.word_ids(i)
                    new_labels.append(align_labels_with_tokens(labels, word_ids))

                tokenized_inputs["labels"] = new_labels
                return tokenized_inputs

            fewshot_dataset = fewshot_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=fewshot_dataset.column_names,
            )

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
                per_device_eval_batch_size=16,
                learning_rate=args.lr,
                warmup_ratio=0.1,
                save_strategy="no",
                save_total_limit=0,
                seed=123,
                num_train_epochs=num_epochs,
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
                train_dataset=fewshot_dataset,
                eval_dataset=None,
                tokenizer=encoder_tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
            logger.info("Start training...")
            trainer.train()
            logger.info("Pretraining completed.")

            logger.info("Log history:")
            for log_step in trainer.state.log_history:
                logger.info(log_step)

            test_dataset = dataset["test"].map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=dataset["test"].column_names,
            )

            logger.info("Start evaluation...")
            preds = trainer.predict(test_dataset)
            logger.info("Evaluation completed.")

            with open(experiment_path / "results.json", "w") as f:
                json.dump(preds.metrics, f)

            logger.info(f"ENDED FINETUNING RUN")
            logger.info(30 * '-')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--lr", type=float, default=3e-5)
    config = parser.parse_args()
    fewshot_evaluation(config)