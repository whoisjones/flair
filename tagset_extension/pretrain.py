import os
import json
import logging
import random
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch
from transformers import AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset

from masking import mask_dataset
from models import BiEncoder

task = "pretraining"

logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO (you can change it as needed)
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(task)
gluster_path = "/vol/tmp/goldejon/ner4all/acl_submission/validation-experiment/"
pretraining_extension = "pretrained-models/{semantic_level}-{num_labels}-labels-seed-{seed}"


def pretrain_fixed_targets(args):
    experiment_path = gluster_path + pretraining_extension

    if not os.path.exists(gluster_path):
        os.makedirs(gluster_path)

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")

    with open("/vol/tmp/goldejon/ner4all/loss_function_experiments/fewnerd_fixed_targets/pretraining_fewnerd_indices.json", "r") as f:
        pretraining_indices = json.load(f)

    for key, training_info in pretraining_indices.items():
        num_labels, seed = key.split("-")
        for semantic_level in args.label_semantic_level:

            # We can skip this since we sampled all remaining labels for the pretraining set.
            if num_labels == "50" and int(seed) > 0:
                continue

            save_base_path = Path(experiment_path.format(semantic_level=semantic_level, num_labels=num_labels, seed=seed))
            if not os.path.exists(save_base_path):
                os.makedirs(save_base_path)

            file_handler_pretraining = logging.FileHandler(f'{save_base_path}/{task}.log')
            logger.addHandler(file_handler_pretraining)

            logger.info(30 * '-')
            logger.info(f"STARTING PRETRAINING RUN")
            logger.info("Config:")
            logger.info("Number of pretraining labels: {}".format(num_labels))
            logger.info("Label semantic level: {}".format(semantic_level))
            logger.info("Seed: {}".format(seed))
            logger.info("Save path: {}".format(save_base_path))
            logger.info("Transformer model: {}".format(args.transformer_model))
            logger.info("Learning rate: {}".format(args.lr))
            logger.info("Number of epochs: {}".format(args.num_epochs))

            labels_to_keep = pretraining_indices[f"{num_labels}-{seed}"]["labels"]
            logger.info(f"Original labels: {labels_to_keep}")

            pretraining_dataset = mask_dataset(
                dataset=dataset,
                label_column="fine_ner_tags",
                label_semantic_level=semantic_level,
                labels_to_keep=labels_to_keep
            )

            shuffle_seed = pretraining_indices[f"{num_labels}-{seed}"]["shuffle_seed"]
            indices = pretraining_indices[f"{num_labels}-{seed}"]["indices"]
            pretraining_dataset = pretraining_dataset["train"].shuffle(seed=shuffle_seed).select(indices)

            id2label = {
                idx: label for idx, label in enumerate(pretraining_dataset.features["fine_ner_tags"].feature.names)
            }
            logger.info(f"Labels after masking: {list(id2label.values())}")

            encoder_tokenizer = AutoTokenizer.from_pretrained(args.transformer_model if "sparselatenttyping" not in args.transformer_model else "bert-base-uncased")
            decoder_tokenizer = AutoTokenizer.from_pretrained(args.transformer_model if "sparselatenttyping" not in args.transformer_model else "bert-base-uncased")

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

            train_dataset = pretraining_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=pretraining_dataset.column_names,
            )

            data_collator = DataCollatorForTokenClassification(encoder_tokenizer)

            training_args = TrainingArguments(
                output_dir=str(save_base_path),
                overwrite_output_dir=True,
                do_train=True,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                learning_rate=args.lr,
                warmup_ratio=0.1,
                save_strategy="epoch",
                save_total_limit=1,
                seed=123,
                num_train_epochs=args.num_epochs,
                logging_dir=str(save_base_path),
                logging_steps=100,
            )

            model = BiEncoder(
                encoder_model=args.transformer_model,
                decoder_model=args.transformer_model,
                tokenizer=decoder_tokenizer,
                labels=id2label,
            )

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

            model.encoder.save_pretrained(save_base_path / "encoder")
            encoder_tokenizer.save_pretrained(save_base_path / "encoder")
            model.decoder.save_pretrained(save_base_path / "decoder")
            decoder_tokenizer.save_pretrained(save_base_path / "decoder")
            logger.info(f"Saved pretrained model and tokenizer to {save_base_path}.")

            with open(f"{save_base_path}/training_args.json", "w") as f:
                json.dump(training_args.to_json_string(), f)

            logger.info(f"ENDED PRETRAINING RUN")
            logger.info(30 * '-')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--label_semantic_level", nargs="+", type=str, choices=["simple", "short", "long"])
    parser.add_argument("--transformer_model", type=str, default="bert-base-uncased")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=5)
    config = parser.parse_args()
    pretrain_fixed_targets(config)