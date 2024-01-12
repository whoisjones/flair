import json
import random
import argparse
from pathlib import Path

import pytorch_lightning as pl
from transformers import AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer

from zelda_experiment.models import BiEncoder, LEAR
from local_datasets import get_masked_fewnerd_corpus


def pretrain_hf(args):
    pl.seed_everything(123)
    model_name = "bert-base-uncased"

    save_base_path = Path(f"{args.gluster_path}/pretrained_lear_")

    for seed in [10, 20, 30, 50]:
        dataset = get_masked_fewnerd_corpus(seed)
        if args.num_samples == "small":
            num_samples = 100000
        elif args.num_samples == "full":
            num_samples = len(dataset["train"])
        random_numbers = random.sample(range(0, len(dataset["train"]) + 1), num_samples)
        small_dataset = dataset["train"].select(random_numbers)

        encoder_tokenizer = AutoTokenizer.from_pretrained(model_name)
        decoder_tokenizer = AutoTokenizer.from_pretrained(model_name)

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
                    # If the label is B-XXX we change it to I-XXX
                    if label % 2 == 1:
                        label += 1
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

        train_dataset = small_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=small_dataset.column_names,
        )

        data_collator = DataCollatorForTokenClassification(encoder_tokenizer)

        training_args = TrainingArguments(
            output_dir=str(save_base_path),
            overwrite_output_dir=True,
            do_train=True,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=1e-6,
            warmup_ratio=0.1,
            save_strategy="epoch",
            save_total_limit=3,
            seed=123,
        )


        with open(args.label_file, 'r') as f:
            labels = json.load(f)

        if args.model == "biencoder":
            model = BiEncoder(
                labels=labels,
                encoder_model=model_name,
                decoder_model=model_name,
                tokenizer=decoder_tokenizer,
                zelda_mask_size=0,
                uniform_p=[0.5, 0.5],
                geometric_p=0.5,
            )
        elif args.model == "LEAR":
            model = LEAR(
                labels=labels,
                encoder_model=model_name,
                decoder_model=model_name,
                tokenizer=decoder_tokenizer,
                zelda_mask_size=0,
            )


        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            tokenizer=encoder_tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        model.encoder.save_pretrained(save_base_path / "encoder")
        encoder_tokenizer.save_pretrained(save_base_path / "encoder")
        model.decoder.save_pretrained(save_base_path / "decoder")
        decoder_tokenizer.save_pretrained(save_base_path / "decoder")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--gluster_path", type=str, default="/glusterfs/dfs-gfs-dist/goldejon/ner4all/acl_submission")
    # NER4ALL needs to be loaded from disk
    parser.add_argument("--dataset_path", type=str, default="/glusterfs/dfs-gfs-dist/goldejon/ner4all/loner/jsonl")
    parser.add_argument("--label_path", type=str, default='/glusterfs/dfs-gfs-dist/goldejon/ner4all/loner/zelda_labelID2label.json')
    parser.add_argument("--num_samples", type=str, default="small")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    if not any([args.dataset, args.dataset_path]):
        raise ValueError("no dataset provided.")

    pretrain_hf(args)
