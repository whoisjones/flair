
import json
import glob
from typing import Dict
import argparse
from pathlib import Path
import logging
import random
from collections import Counter

import numpy as np
import numpy.random
import torch
import torch.cuda
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer, PreTrainedTokenizer
import datasets
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO (you can change it as needed)
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


pretraining_logger = logging.getLogger("pretraining")
file_handler_pretraining = logging.FileHandler('pretraining.log')
pretraining_logger.addHandler(file_handler_pretraining)

finetuning_logger = logging.getLogger("finetuning")
file_handler_finetuning = logging.FileHandler('finetuning.log')
finetuning_logger.addHandler(file_handler_finetuning)

semantic_label_name_map = {
    "conll2012_ontonotesv5": {
        "ner_tags": {
            "O": "outside",
            "CARDINAL": "cardinal",
            "DATE": "date",
            "EVENT": "event",
            "FAC": "facility",
            "GPE": "geographical social political entity",
            "LANGUAGE": "language",
            "LAW": "law",
            "LOC": "location",
            "MONEY": "money",
            "NORP": "nationality religion political",
            "ORDINAL": "ordinal",
            "ORG": "organization",
            "PERCENT": "percent",
            "PERSON": "person",
            "PRODUCT": "product",
            "QUANTITY": "quantity",
            "TIME": "time",
            "WORK_OF_ART": "work of art",
        }
    },
    "few-nerd": {
        "ner_tags": {
            "O": "outside",
            "location": "location",
            "person": "person",
            "organization": "organization",
            "building": "building",
            "other": "other",
            "product": "product",
            "event": "event",
            "art": "art",
        },
        "fine_ner_tags": {
            "O": "outside",
            "location-GPE": "geographical social political entity",
            "person-other": "other person",
            "organization-other": "other organization",
            "organization-company": "company",
            "person-artist/author": "author artist",
            "person-athlete": "athlete",
            "person-politician": "politician",
            "building-other": "other building",
            "organization-sportsteam": "sportsteam",
            "organization-education": "eduction",
            "location-other": "other location",
            "other-biologything": "biology",
            "location-road/railway/highway/transit": "road railway highway transit",
            "person-actor": "actor",
            "product-other": "other product",
            "event-sportsevent": "sportsevent",
            "organization-government/governmentagency": "government agency",
            "location-bodiesofwater": "bodies of water",
            "organization-media/newspaper": "media newspaper",
            "art-music": "music",
            "other-chemicalthing": "chemical",
            "event-attack/battle/war/militaryconflict": "attack war battle military conflict",
            "organization-politicalparty": "political party",
            "art-writtenart": "written art",
            "other-award": "award",
            "other-livingthing": "living thing",
            "event-other": "other event",
            "art-film": "film",
            "product-software": "software",
            "organization-sportsleague": "sportsleague",
            "other-language": "language",
            "other-disease": "disease",
            "organization-showorganization": "show organization",
            "product-airplane": "airplane",
            "other-astronomything": "astronomy",
            "organization-religion": "religion",
            "product-car": "car",
            "person-scholar": "scholar",
            "other-currency": "currency",
            "person-soldier": "soldier",
            "location-mountain": "mountain",
            "art-broadcastprogram": "broadcastprogram",
            "location-island": "island",
            "art-other": "other art",
            "person-director": "director",
            "product-weapon": "weapon",
            "other-god": "god",
            "building-theater": "theater",
            "other-law": "law",
            "product-food": "food",
            "other-medical": "medical",
            "product-game": "game",
            "location-park": "park",
            "product-ship": "ship",
            "building-sportsfacility": "sportsfacility",
            "other-educationaldegree": "educational degree",
            "building-airport": "airport",
            "building-hospital": "hospital",
            "product-train": "train",
            "building-library": "library",
            "building-hotel": "hotel",
            "building-restaurant": "restaurant",
            "event-disaster": "disaster",
            "event-election": "election",
            "event-protest": "protest",
            "art-painting": "painting",
        }
    }
}

def compute_fewshot_samples():
    def count_integer_changes(input_list):
        return [input_list[i] for i in range(len(input_list)) if
                (i == 0 or input_list[i] != input_list[i - 1]) and input_list[i] != 0]

    fewshot_indexes = {}
    for finetuning_corpus, label_column in [("conll2012_ontonotesv5", "ner_tags"),
                                            ("DFKI-SLT/few-nerd", "ner_tags"),
                                            ("DFKI-SLT/few-nerd", "fine_ner_tags")]:
        dataset_config = get_finetuning_dataset_config(finetuning_corpus)
        full_dataset = load_dataset(finetuning_corpus, dataset_config)
        if "conll2012" in finetuning_corpus:
            full_dataset = process_ontonotes(full_dataset)
        for seed in [10, 30, 50]:
            for k in [1, 5, 10]:
                for run_seed in range(3):
                    finetuning_labels = get_finetuning_labels(finetuning_corpus, seed)
                    dataset = mask_dataset(full_dataset, label_column, finetuning_labels)
                    dataset = dataset.shuffle(run_seed)
                    print(
                        f"compute fewshots for: {finetuning_corpus} with seed {seed} for run {run_seed} and kshot {k}")
                    label_counter = Counter()
                    selected_fewshots = []
                    all_labels = [idx for idx, label in enumerate(dataset["train"].features[label_column].feature.names)
                                  if
                                  label != "outside"]
                    for idx, example in enumerate(dataset["train"]):
                        mention_counter = count_integer_changes(example[label_column])
                        counter_if_added = Counter(mention_counter) + label_counter
                        if any([tag > k for tag in counter_if_added.values()]) or not mention_counter:
                            continue
                        if all([tag <= k for tag in counter_if_added.values()]):
                            label_counter = label_counter + Counter(mention_counter)
                            selected_fewshots.append(idx)
                        if all([tag == k for tag in label_counter.values()]) and set(label_counter.values()) == set(
                                all_labels):
                            break

                    if finetuning_corpus not in fewshot_indexes:
                        fewshot_indexes[finetuning_corpus] = {f"{k}-{seed}-{run_seed}": selected_fewshots}
                    else:
                        fewshot_indexes[finetuning_corpus][f"{k}-{seed}-{run_seed}"] = selected_fewshots

    with open("fewshot_indices_two_halves.json", "w") as f:
        json.dump(fewshot_indexes, f)

class BiEncoder(torch.nn.Module):

    def __init__(
            self,
            encoder_model: str,
            decoder_model: str,
            labels: dict,
            tokenizer: PreTrainedTokenizer,
    ):
        super(BiEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.decoder = AutoModel.from_pretrained(decoder_model)
        labels = {int(k): v for k, v in labels.items()}
        self.labels = labels
        self.num_labels = len(labels)
        self.tokenizer = tokenizer
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels):
        token_hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoded_labels = self.tokenizer(list(self.labels.values()), padding=True, truncation=True, max_length=64, return_tensors="pt").to(labels.device)
        label_hidden_states = self.decoder(**encoded_labels)
        label_embeddings = label_hidden_states.last_hidden_state[:, 0, :]
        logits = torch.matmul(token_hidden_states.last_hidden_state, label_embeddings.T)
        return (self.loss(logits.transpose(1, 2), labels),)

class NER4ALLBiEncoder(torch.nn.Module):

    def __init__(
            self,
            encoder_model: str,
            decoder_model: str,
            labels: dict,
            tokenizer: PreTrainedTokenizer,
            uniform_p: list,
            mask_size: int = 0,
            geometric_p: float = 0.33,
            ner4all_pretraining: bool = False
    ):
        super(NER4ALLBiEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.decoder = AutoModel.from_pretrained(decoder_model)
        labels = {int(k): v for k, v in labels.items()}
        self.labels = labels
        self.num_labels = len(labels)
        self.tokenizer = tokenizer
        self.mask_size = mask_size
        self.uniform_p = uniform_p
        self.geometric_p = geometric_p
        self.loss = torch.nn.CrossEntropyLoss()
        self.ner4all_pretraining = ner4all_pretraining

    def _verbalize_labels(self, selected_labels):
        label_descriptions = []
        label_granularities = ["description", "labels"]
        for i in selected_labels:
            if i == 0:
                label_description = "outside"
            else:
                label_granularity = np.random.choice(label_granularities, p=self.uniform_p)
                fallback_option = [x for x in label_granularities if x != label_granularity][0]
                if self.labels.get(i).get(label_granularity) is None and self.labels.get(i).get(fallback_option) is not None:
                    label_granularity = fallback_option
                elif self.labels.get(i).get(label_granularity) is None and self.labels.get(i).get(fallback_option) is None:
                    label_description = "miscellaneous"
                    label_descriptions.append(label_description)
                    continue

                if label_granularity == "description":
                    label_description = f"{self.labels.get(i)[label_granularity]}"
                elif label_granularity == "labels":
                    num_labels = np.random.geometric(self.geometric_p, 1)
                    num_labels = num_labels if num_labels <= len(self.labels.get(i).get("labels")) else len(self.labels.get(i).get("labels"))
                    sampled_labels = np.random.choice(self.labels.get(i).get("labels"), num_labels, replace=False).tolist()
                    label_description = f"{', '.join(sampled_labels)}"
                else:
                    raise ValueError(f"Unknown label granularity {label_granularity}")
            label_descriptions.append(label_description)
        return label_descriptions

    def _prepare_labels(self, labels):
        positive_labels = torch.unique(labels)
        positive_labels = positive_labels[(positive_labels != -100)]
        number_negatives_needed = self.mask_size - positive_labels.size(0)
        if number_negatives_needed > 0:
            negative_labels = numpy.random.choice(np.arange(0, len(self.labels)), size=number_negatives_needed, replace=False)
            labels_for_batch = np.unique(np.concatenate([positive_labels.detach().cpu().numpy(), negative_labels]))
        else:
            labels_for_batch = positive_labels.detach().cpu().numpy()
        labels = self.adjust_batched_labels(labels_for_batch, labels)
        label_descriptions = self._verbalize_labels(labels_for_batch)
        encoded_labels = self.tokenizer(label_descriptions, padding=True, truncation=True, max_length=64, return_tensors="pt").to(labels.device)
        return encoded_labels, labels

    def adjust_batched_labels(self, labels_for_batch, original_label_tensor):
        batch_size = original_label_tensor.size(0)
        adjusted_label_tensor = torch.zeros_like(original_label_tensor)

        labels_for_batch = labels_for_batch[(labels_for_batch != 0)]

        label_mapping = {label.item(): idx + 1 for idx, label in enumerate(labels_for_batch)}
        label_mapping[-100] = -100
        label_mapping[0] = 0

        for i in range(batch_size):
            adjusted_label_tensor[i] = torch.tensor([label_mapping.get(label.item(), -1) for label in original_label_tensor[i]])

        return adjusted_label_tensor

    def forward(self, input_ids, attention_mask, labels):
        token_hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoded_labels, labels = self._prepare_labels(labels)
        label_hidden_states = self.decoder(**encoded_labels)
        label_embeddings = label_hidden_states.last_hidden_state[:, 0, :]
        logits = torch.matmul(token_hidden_states.last_hidden_state, label_embeddings.T)
        return (self.loss(logits.transpose(1, 2), labels),)

def get_pretraining_corpus_size(corpus_size, dataset, save_base_path):
    if corpus_size == "small":
        dataset["train"] = dataset["train"].shuffle(42)
        num_tags = [len(set(example)) for example in dataset["train"]["ner_tags"]]
        i = 0
        curr_entity_mentions = 0
        cut_off = 169000 # max number of entity mentions in FewNERD
        selected_idx = []
        while curr_entity_mentions < cut_off:
            curr_entity_mentions += num_tags[i]
            selected_idx.append(i)
            i += 1
        small_dataset = dataset["train"].select(selected_idx)
        pretraining_logger.info(f"Selected {len(small_dataset)} examples for pretraining based on seed 42.")
        with open(save_base_path / "pretraining_selected_idx_for_zelda.json", "w") as f:
            json.dump(selected_idx, f)

    elif corpus_size == "full":
        small_dataset = dataset["train"]
        pretraining_logger.info("Using full dataset for pretraining.")

    elif corpus_size == "debug":
        small_dataset = dataset["train"].select(range(200))
        pretraining_logger.info("Using debug dataset pretraining.")

    else:
        raise ValueError(f"Unknown corpus size {args.corpus_size}")

    return small_dataset


def pretrain_fixed_targets(args):
    pl.seed_everything(123)
    pretraining_logger.info(f"Pretraining Seed: 123")

    dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")
    pretraining_logger.info(f"Loaded pretraining dataset: FewNERD")

    finetuning_labels = ['other-medical', 'product-game', 'location-park', 'product-ship', 'building-sportsfacility', 'other-educationaldegree', 'building-airport', 'building-hospital', 'product-train', 'building-library', 'building-hotel', 'building-restaurant', 'event-disaster', 'event-election', 'event-protest', 'art-painting']
    possible_pretraining_labels = set(semantic_label_name_map["few-nerd"]["fine_ner_tags"].keys()) - set(finetuning_labels) - set("O")

    with open("/vol/tmp/goldejon/ner4all/loss_function_experiments/pretraining_indices_fewnerd.json", "r") as f:
        pretraining_indices = json.load(f)

    for num_labels in [50]:
        for seed in range(3):
            save_base_path = Path(f"{args.gluster_path}/{args.save_dir}/pretraining-{num_labels}-labels-seed-{seed}")
            pretraining_logger.info(f"Saving pretraining models to {save_base_path}")

            random.seed(seed)
            sampled_pretraining_labels = random.sample(possible_pretraining_labels, k=num_labels)
            pretraining_dataset = mask_dataset(dataset, "fine_ner_tags", sampled_pretraining_labels)
            pretraining_dataset = pretraining_dataset["train"].select(pretraining_indices[f"{num_labels}-{seed}"])
            labels = {idx: label for idx, label in enumerate(pretraining_dataset.features["fine_ner_tags"].feature.names)}

            encoder_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            decoder_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

            pretraining_logger.info(f"Using model for pretraining: bert-base-uncased")

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
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                learning_rate=5e-5,
                warmup_ratio=0.1,
                save_strategy="epoch",
                save_total_limit=1,
                seed=123,
                num_train_epochs=5,
                logging_dir=str(save_base_path),
                logging_steps=100,
            )
            pretraining_logger.info(training_args.to_json_string())

            model = BiEncoder(
                encoder_model="bert-base-uncased",
                decoder_model="bert-base-uncased",
                labels=labels,
                tokenizer=decoder_tokenizer,
            )
            pretraining_logger.info(training_args.to_json_string())

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=None,
                tokenizer=encoder_tokenizer,
                data_collator=data_collator,
            )

            trainer.train()
            pretraining_logger.info("Pretraining completed.")
            for log_step in trainer.state.log_history:
                pretraining_logger.info(log_step)

            model.encoder.save_pretrained(save_base_path / "encoder")
            encoder_tokenizer.save_pretrained(save_base_path / "encoder")
            model.decoder.save_pretrained(save_base_path / "decoder")
            decoder_tokenizer.save_pretrained(save_base_path / "decoder")

            pretraining_logger.info(f"Saved pretrained model and tokenizer to {save_base_path}.")

def pretrain(args) -> Dict[str, str]:
    pl.seed_everything(123)
    pretraining_logger.info(f"Pretraining Seed: 123")

    save_base_path = Path(f"{args.gluster_path}/{args.save_dir}/pretraining")
    pretraining_logger.info(f"Saving pretraining models to {save_base_path}")

    if args.pretraining_corpus == "ner4all":
        dataset = load_dataset("json", data_files=glob.glob('/vol/tmp/goldejon/datasets/loner/jsonl/*'))
    elif args.pretraining_corpus == "few-nerd":
        dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")
    else:
        raise ValueError(f"Unknown pretraining corpus {args.pretraining_corpus}")
    pretraining_logger.info(f"Loaded pretraining dataset: {args.pretraining_corpus}")

    dataset = get_pretraining_corpus_size(args.corpus_size, dataset, save_base_path)

    encoder_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    decoder_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    pretraining_logger.info(f"Using model for pretraining: bert-base-uncased")

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

    train_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset.column_names,
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
        save_total_limit=1,
        seed=123,
        num_train_epochs=3,
        logging_dir=str(save_base_path),
        logging_steps=100,
    )
    pretraining_logger.info(training_args.to_json_string())

    if args.pretraining_corpus == "ner4all":
        with open('/vol/tmp/goldejon/datasets/loner/labelID2label.json', 'r') as f:
            labels = json.load(f)
    else:
        labels = dataset["train"].features["ner_tags"].feature.names

    model = BiEncoder(
        encoder_model="bert-base-uncased",
        decoder_model="bert-base-uncased",
        labels=labels,
        tokenizer=decoder_tokenizer,
        mask_size=0,
        uniform_p=[0.5, 0.5],
        geometric_p=0.33,
        ner4all_pretraining=True if args.pretraining_corpus == "ner4all" else False
    )
    pretraining_logger.info(training_args.to_json_string())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=encoder_tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    pretraining_logger.info("Pretraining completed.")
    for log_step in trainer.state.log_history:
        pretraining_logger.info(log_step)

    model.encoder.save_pretrained(save_base_path / "encoder")
    encoder_tokenizer.save_pretrained(save_base_path / "encoder")
    model.decoder.save_pretrained(save_base_path / "decoder")
    decoder_tokenizer.save_pretrained(save_base_path / "decoder")

    pretraining_logger.info(f"Saved pretrained model and tokenizer to {save_base_path}.")

    return {
        "encoder_path": str(save_base_path / "encoder"),
        "decoder_path": str(save_base_path / "decoder")
    }

def get_finetuning_dataset_config(finetuning_corpus):
    if "few-nerd" in finetuning_corpus:
        return "supervised"
    elif "conll2012" in finetuning_corpus:
        return "english_v4"
    else:
        raise ValueError(f"Unknown finetuning corpus {finetuning_corpus}")

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

def get_label_columns(finetuning_corpus):
    if "few-nerd" in finetuning_corpus:
        return ["ner_tags", "fine_ner_tags"]
    elif "conll2012" in finetuning_corpus:
        return ["ner_tags"]
    else:
        raise ValueError(f"Unknown finetuning corpus {finetuning_corpus}")

def get_finetuning_labels(finetuning_corpus, seed):
    if "few-nerd" in finetuning_corpus:
        labels = ["person", "location", "organization", "product", "art", "event", "building", "other"]
    elif "conll2012" in finetuning_corpus:
        labels = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']
    else:
        raise ValueError(f"Unknown fine-tuning corpus name {finetuning_corpus}")

    random.seed(seed)
    pretraining_labels = random.sample(labels, int(0.5 * len(labels)))
    finetuning_labels = list(set(labels) - set(pretraining_labels))
    #finetuning_logger.info(f"Fine-tuning on selected labels: {finetuning_labels}")

    return finetuning_labels

def mask_dataset(dataset, label_column, finetuning_labels):
    tag_info = dataset["train"].features[label_column]
    dataset_name = dataset["train"].info.dataset_name
    labels_for_model = {}
    labels_for_masking = {}
    curr_idx = 0
    for old_label_idx, old_label in enumerate(dataset["train"].features[label_column].feature.names):
        if any([old_label.startswith(l) for l in finetuning_labels]) or old_label == "O":
            labels_for_masking[old_label_idx] = curr_idx
            labels_for_model[curr_idx] = semantic_label_name_map[dataset_name][label_column][
                dataset["train"].features[label_column].feature.names[old_label_idx]
            ]
            curr_idx += 1
        else:
            labels_for_masking[old_label_idx] = 0

    def mask(examples):
        examples[label_column] = [[labels_for_masking.get(old_id) for old_id in sample] for sample in
                                     examples[label_column]]
        return examples

    dataset = dataset.map(mask, batched=True)

    tag_info.feature.names = list(labels_for_model.values())
    features = dataset["train"].features
    features[label_column] = tag_info

    dataset = dataset.cast(features)

    return dataset

def finetune(pretrained_models: Dict):
    pl.seed_everything(123)
    finetuning_logger.info(f"Fine-tuning Seed: 123")

    save_base_path = Path(f"{args.gluster_path}/{args.save_dir}/finetuning")
    finetuning_logger.info(f"Saving fine-tuned models to {save_base_path}")

    dataset_config = get_finetuning_dataset_config(args.finetuning_corpus)
    dataset = load_dataset(args.finetuning_corpus, dataset_config)
    finetuning_logger.info(f"Loaded pretraining dataset: {args.finetuning_corpus}")

    if "conll2012" in args.finetuning_corpus:
        finetuning_logger.info("Flatten OntoNotes dataset.")
        dataset = process_ontonotes(dataset)

    label_columns = get_label_columns(args.finetuning_corpus)

    for label_column in label_columns:
        for kshot in [0, 1, 5, 10]:
            for seed in [10, 30, 50]:
                for run_seed in range(3):

                    finetuning_labels = get_finetuning_labels(args.finetuning_corpus, seed)
                    dataset = mask_dataset(dataset, label_column, finetuning_labels) # ALWAYS DO LIKE THIS
                    dataset = dataset.shuffle(run_seed) # ALWAYS DO LIKE THIS

                    with open(f"data/fewshot_masked-fewnerd-{'coarse' if label_column == 'ner_tags' else 'fine'}.json", "r") as f:
                        masked_indices = json.load(f)

                    if kshot > 0:
                        dataset["train"] = dataset["train"].select(masked_indices[f"{kshot}-{seed}-{run_seed}"])

                    encoder_tokenizer = AutoTokenizer.from_pretrained(pretrained_models["encoder_path"])
                    decoder_tokenizer = AutoTokenizer.from_pretrained(pretrained_models["decoder_path"])

                    finetuning_logger.info(f"Using model for fine-tuning: {pretrained_models['encoder_path']} and {pretrained_models['decoder_path']}")

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
                        all_labels = examples[fewnerd_setting]
                        new_labels = []
                        for i, labels in enumerate(all_labels):
                            word_ids = tokenized_inputs.word_ids(i)
                            new_labels.append(align_labels_with_tokens(labels, word_ids))

                        tokenized_inputs["labels"] = new_labels
                        return tokenized_inputs

                    train_dataset = dataset.map(
                        tokenize_and_align_labels,
                        batched=True,
                        remove_columns=dataset["train"].column_names,
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
                        save_total_limit=1,
                        seed=123,
                        num_train_epochs=100,
                        logging_dir=str(save_base_path),
                        logging_steps=2,
                    )
                    pretraining_logger.info("Fine-Tuning Model Config:")
                    pretraining_logger.info("Learning Rate: 1e-6")
                    pretraining_logger.info("Warmup Ratio: 0.1")
                    pretraining_logger.info("Save Strategy: epoch")

                    model = BiEncoder(
                        encoder_model=pretrained_models["encoder_path"],
                        decoder_model=pretrained_models["decoder_path"],
                        labels=labels_for_model,
                        tokenizer=decoder_tokenizer,
                        mask_size=0,
                        uniform_p=[0.5, 0.5],
                        geometric_p=0.33,
                        ner4all_pretraining=False
                    )
                    pretraining_logger.info("Mask Size: 0")
                    pretraining_logger.info("Uniform P: [0.5, 0.5]")
                    pretraining_logger.info("Geometric P: 0.33")

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=None,
                        tokenizer=encoder_tokenizer,
                        data_collator=data_collator,
                    )

                    if kshot != 0:
                        trainer.train()
                        pretraining_logger.info("Fine-tuning completed.")
                        for log_step in trainer.state.log_history:
                            pretraining_logger.info(log_step)

                        model.encoder.save_pretrained(save_base_path / "encoder")
                        encoder_tokenizer.save_pretrained(save_base_path / "encoder")
                        model.decoder.save_pretrained(save_base_path / "decoder")
                        decoder_tokenizer.save_pretrained(save_base_path / "decoder")

                    outputs = trainer.predict(dataset["test"])

                    pretraining_logger.info(f"Saved pretrained model and tokenizer to {save_base_path}.")


if __name__ == "__main__":
    compute_fewnerd_fixed_targets_samples()
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--pretraining_fixed_targets", action="store_true")
    parser.add_argument("--save_dir", type=str, default="cross_entropy_biencoder")
    parser.add_argument("--gluster_path", type=str, default="/vol/tmp/goldejon/ner4all/loss_function_experiments")
    # NER4ALL needs to be loaded from disk
    parser.add_argument("--pretraining_corpus", type=str, default="ner4all")
    parser.add_argument("--finetuning_corpus", type=str, default="fewnerd")
    parser.add_argument("--corpus_size", type=str, default="debug")
    parser.add_argument("--encoder_transformer", type=str, default="bert-base-uncased")
    parser.add_argument("--decoder_transformer", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    if args.pretraining_fixed_targets:
        pretrain_fixed_targets(args)
"""