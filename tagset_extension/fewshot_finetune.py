import os
import json
import logging
import random
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from seqeval.metrics import classification_report
from transformers import AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset
from prettytable import PrettyTable

from masking import mask_dataset
from models import BiEncoder
from pretrain import gluster_path, pretraining_extension

task_name = "finetuning"

logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO (you can change it as needed)
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(task_name)


def finetune_fixed_targets(args):
    experiment_path = gluster_path + "finetuning/run-{idx}-{kshot}shot-with-{pt_num_labels}-{semantic_level}-labels"

    if not os.path.exists(gluster_path + "finetuning"):
        os.makedirs(gluster_path + "finetuning")

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")

    with open("/vol/tmp/goldejon/ner4all/loss_function_experiments/fewnerd_fixed_targets/fewshot_fewnerd_indices.json", "r") as f:
        fewshot_indices = json.load(f)

    overall_scores = {}

    for kshot in args.kshots:
        for pt_num_labels in args.num_pretraining_labels:
            for semantic_level in args.label_semantic_level:
                for pt_seed in range(args.pretraining_seeds):
                    for ft_seed in range(args.fewshot_seeds):

                        # We can skip this since we sampled all remaining labels for the pretraining set.
                        if pt_num_labels == 50 and pt_seed > 0:
                            continue

                        run_idx = (ft_seed + 1) + (pt_seed * args.pretraining_seeds)
                        save_base_path = Path(experiment_path.format(idx=run_idx, kshot=kshot, semantic_level=semantic_level, pt_num_labels=pt_num_labels))

                        if not os.path.exists(save_base_path):
                            os.makedirs(save_base_path)

                        file_handler_pretraining = logging.FileHandler(f'{save_base_path}/{task_name}.log')
                        logger.addHandler(file_handler_pretraining)

                        logger.info(30 * '-')
                        logger.info(f"STARTING FINETUNING RUN")
                        logger.info("Pretraining Model Config:")
                        logger.info("Number of pretraining labels: {}".format(pt_num_labels))
                        logger.info("Seed: {}".format(pt_seed))
                        logger.info("Label semantic level: {}".format(semantic_level))

                        logger.info("Fine-tuning Model Config:")
                        logger.info("Learning rate: {}".format(args.lr))
                        logger.info("Number of epochs: {}".format(args.num_epochs))
                        logger.info("Save path: {}".format(save_base_path))
                        logger.info("Run: {}".format(run_idx))

                        fewshot_labels = fewshot_indices[f"{kshot}-{ft_seed}"]["labels"]
                        fine_tuning_dataset = mask_dataset(dataset, "fine_ner_tags", semantic_level, fewshot_labels)
                        train_dataset = fine_tuning_dataset["validation"].shuffle(ft_seed).select(fewshot_indices[f"{kshot}-{ft_seed}"]["indices"])
                        id2label = {idx: label for idx, label in enumerate(train_dataset.features["fine_ner_tags"].feature.names)}

                        model_path = (gluster_path + pretraining_extension.format(
                                semantic_level=semantic_level, num_labels=pt_num_labels, seed=pt_seed
                            ))

                        encoder_tokenizer = AutoTokenizer.from_pretrained(
                            model_path + "/encoder"
                        )
                        decoder_tokenizer = AutoTokenizer.from_pretrained(
                            model_path + "/decoder"
                        )

                        logger.info(f"Using pre-trained model for fine-tuning: {pt_num_labels}-labels-seed-{pt_seed}")

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
                            all_labels = examples["fine_ner_tags"]
                            new_labels = []
                            for i, labels in enumerate(all_labels):
                                word_ids = tokenized_inputs.word_ids(i)
                                new_labels.append(align_labels_with_tokens(labels, word_ids))

                            tokenized_inputs["labels"] = new_labels
                            return tokenized_inputs

                        train_dataset = train_dataset.map(
                            tokenize_and_align_labels,
                            batched=True,
                            remove_columns=train_dataset.column_names,
                        )

                        data_collator = DataCollatorForTokenClassification(encoder_tokenizer)

                        training_args = TrainingArguments(
                            output_dir=str(save_base_path),
                            overwrite_output_dir=True,
                            do_train=True,
                            per_device_train_batch_size=8,
                            per_device_eval_batch_size=32,
                            learning_rate=args.lr,
                            warmup_ratio=0.1,
                            save_strategy="no",
                            save_total_limit=0,
                            seed=123,
                            num_train_epochs=args.num_epochs,
                            logging_dir=str(save_base_path),
                            logging_steps=5,
                        )

                        model = BiEncoder(
                            encoder_model=model_path + "/encoder",
                            decoder_model=model_path + "/decoder",
                            tokenizer=decoder_tokenizer,
                            labels=id2label,
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

                        trainer = Trainer(
                            model=model,
                            args=training_args,
                            train_dataset=train_dataset,
                            eval_dataset=None,
                            tokenizer=encoder_tokenizer,
                            data_collator=data_collator,
                            compute_metrics=compute_metrics,
                        )

                        logger.info("Start training...")
                        trainer.train()
                        logger.info("Fine Tuning completed.")

                        test_dataset = fine_tuning_dataset["test"].map(
                            tokenize_and_align_labels,
                            batched=True,
                            remove_columns=fine_tuning_dataset["test"].column_names,
                        )

                        logger.info("Start evaluation...")
                        preds = trainer.predict(test_dataset)
                        logger.info("Evaluation completed.")

                        if f"{kshot}shot-on-{semantic_level}-{pt_num_labels}" not in overall_scores:
                            overall_scores[f"{kshot}shot-on-{semantic_level}-{pt_num_labels}"] = [round(preds.metrics["test_micro avg"]["f1-score"], 2)]
                        else:
                            overall_scores[f"{kshot}shot-on-{semantic_level}-{pt_num_labels}"].append(round(preds.metrics["test_micro avg"]["f1-score"], 2))

                        with open(save_base_path / "results.json", "w") as f:
                            json.dump(preds.metrics, f)

                        logger.info("Log history:")
                        for log_step in trainer.state.log_history:
                            logger.info(log_step)

                        with open(f"{save_base_path}/training_args.json", "w") as f:
                            json.dump(training_args.to_json_string(), f)

                        logger.info(f"Saved results of fine-tuned model to {save_base_path}.")
                        logger.info(f"ENDED FINE-TUNING RUN")
                        logger.info(30 * '-')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--kshots", nargs="+", type=int)
    parser.add_argument("--num_pretraining_labels", nargs="+", type=int)
    parser.add_argument("--pretraining_seeds", type=int, default=3)
    parser.add_argument("--fewshot_seeds", type=int, default=3)
    parser.add_argument("--label_semantic_level", nargs="+", type=str, choices=["simple", "short", "long"])
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=30)
    args = parser.parse_args()
    finetune_fixed_targets(args)
