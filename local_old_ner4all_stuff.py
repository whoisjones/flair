import os
import json
import copy
import glob
import random
import argparse
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import numpy.random
import torch
import torch.cuda
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer, PreTrainedTokenizer
from datasets import load_dataset

import flair
from flair.data import Sentence, Dictionary
from flair.embeddings import (
    TransformerWordEmbeddings,
    TransformerDocumentEmbeddings, Embeddings, SentenceTransformerDocumentEmbeddings
)
from flair.models import TokenClassifier
from flair.trainers import ModelTrainer
from flair.training_utils import store_embeddings
from torch.utils.data.dataset import Subset

from local_loner import LONER
from local_corpora import get_masked_fewnerd_corpus, get_corpus

class BatchedLabelVerbalizerDecoder(torch.nn.Module):
    def __init__(self, label_embedding: Embeddings, label_dictionary: Dictionary, requires_masking: bool, mask_size: int = 128):
        super().__init__()
        self.label_embedding = label_embedding
        self.verbalized_labels: List[Sentence] = self.verbalize_labels(label_dictionary)
        self.requires_masking = requires_masking
        self.mask_size = mask_size
        self.to(flair.device)

    @staticmethod
    def verbalize_labels(label_dictionary) -> List[Sentence]:
        verbalized_labels = []
        for byte_label, idx in label_dictionary.item2idx.items():
            str_label = byte_label.decode("utf-8")
            if label_dictionary.span_labels:
                if str_label == "O":
                    verbalized_labels.append("outside")
                elif str_label.startswith("B-"):
                    verbalized_labels.append("begin " + str_label.split("-")[1])
                elif str_label.startswith("I-"):
                    verbalized_labels.append("inside " + str_label.split("-")[1])
                elif str_label.startswith("E-"):
                    verbalized_labels.append("ending " + str_label.split("-")[1])
                elif str_label.startswith("S-"):
                    verbalized_labels.append("single " + str_label.split("-")[1])
            else:
                verbalized_labels.append(str_label)
        return list(map(Sentence, verbalized_labels))

    def embedding_sublist(self, labels) -> List[Sentence]:
        unique_entries = set(labels)

        # Randomly sample entries from the larger list
        while len(unique_entries) < self.mask_size:
            entry = random.choice(range(len(self.verbalized_labels)))
            unique_entries.add(entry)

        return [self.verbalized_labels[idx] for idx in unique_entries], unique_entries

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:

        if self.training and self.requires_masking:
            labels_to_include = labels.cpu().numpy().tolist()
            labels, indices = self.embedding_sublist(labels_to_include)
            self.label_embedding.embed(labels)
        elif not self.training or not self.requires_masking:
            labels = self.label_embedding.embed(self.verbalized_labels)

        label_tensor = torch.stack([label.get_embedding() for label in labels])

        if self.training:
            store_embeddings(labels, "none")

        scores = torch.mm(inputs, label_tensor.T)

        if self.training and self.requires_masking:
            all_scores = torch.zeros(scores.shape[0], len(self.verbalized_labels), device=flair.device)
            all_scores[:, torch.LongTensor(list(indices))] = scores
        elif not self.training or not self.requires_masking:
            all_scores = scores

        return all_scores

class HfDualEncoder(torch.nn.Module):
    def __init__(self, labels: dict, encoder_model: str, decoder_model: str, tokenizer: PreTrainedTokenizer, uniform_p: list, mask_size: int = 128, geometric_p: float = 0.5):
        super(HfDualEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model)
        if decoder_model == "all-mpnet-base-v2":
            self.decoder = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        else:
            self.decoder = AutoModel.from_pretrained(decoder_model)
        labels = {int(k): v for k, v in labels.items()}
        self.labels = labels
        self.num_labels = len(labels)
        self.tokenizer = tokenizer
        self.mask_size = mask_size
        self.uniform_p = uniform_p
        self.geometric_p = geometric_p
        self.loss = torch.nn.CrossEntropyLoss()

    def _verbalize_labels(self, selected_labels):
        #label_descriptions = [self.labels[i] for i in selected_labels]
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
                    label_description = f"{'begin' if self.labels.get(i)['bio_tag'] == 'B-' else 'inside'} {self.labels.get(i)[label_granularity]}"
                elif label_granularity == "labels":
                    num_labels = np.random.geometric(self.geometric_p, 1)
                    num_labels = num_labels if num_labels <= len(self.labels.get(i).get("labels")) else len(self.labels.get(i).get("labels"))
                    sampled_labels = np.random.choice(self.labels.get(i).get("labels"), num_labels, replace=False).tolist()
                    label_description = f"{'begin' if self.labels.get(i)['bio_tag'] == 'B-' else 'inside'} {', '.join(sampled_labels)}"
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

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask, labels):
        token_hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoded_labels, labels = self._prepare_labels(labels)
        label_hidden_states = self.decoder(**encoded_labels)
        if "sentence-transformers" in self.decoder.name_or_path:
            label_embeddings = self.mean_pooling(label_hidden_states, encoded_labels['attention_mask'])
            label_embeddings = torch.nn.functional.normalize(label_embeddings, p=2, dim=1)
        else: # take cls token
            label_embeddings = label_hidden_states.last_hidden_state[:, 0, :]
        logits = torch.matmul(token_hidden_states.last_hidden_state, label_embeddings.T)
        return (self.loss(logits.transpose(1, 2), labels),)

def get_save_base_path(args, task_name):
    is_pretraining = True if "pretrain" in task_name else False
    if args.dataset_path:
        is_zelda = True if "zelda" in args.dataset_path.lower() else False
        dataset = f"LONER{'-ZELDA' if is_zelda else ''}"
    else:
        dataset = f"{args.dataset}{args.fewnerd_granularity if args.dataset == 'fewnerd' else ''}"

    if is_pretraining:
        sampling = "-".join([str(x) for x in args.uniform_p])
        training_arguments = f"_{args.lr}_seed-{args.seed}_mask-{args.mask_size}_size-{args.corpus_size}{f'_sampling-{sampling}' if sampling != '0.5-0.5' else ''}"

        if args.encoder_transformer == args.decoder_transformer:
            model_arguments = f"{args.encoder_transformer}"
        else:
            model_arguments = f"{args.encoder_transformer}_{args.decoder_transformer}"
    else:
        if args.model_to_use == "flair":
            pretraining_model = args.pretrained_flair_model.split('/')[-2].split('_', 1)[-1]
        elif args.model_to_use == "hf":
            if "checkpoint" in args.pretrained_hf_encoder:
                pretraining_model = f"{args.pretrained_hf_encoder.split('/')[-2]}_checkpoint{args.pretrained_hf_encoder.split('/')[-1].replace('checkpoint', '')}"
            else:
                pretraining_model = args.pretrained_hf_encoder.split('/')[-2]
        else:
            pretraining_model = "plain-bert"
        training_arguments =  f"-{args.lr}_pretrained-on-{pretraining_model}"

    return Path(
        f"{args.cache_path}/{task_name}/"
        f"{model_arguments + '_' if is_pretraining else ''}"
        f"{dataset}"
        f"{training_arguments}"
    )

def get_corpus_size(args):
    if args.corpus_size == "100k":
        num_samples = 100000
    elif args.corpus_size == "500k":
        num_samples = 500000
    elif args.corpus_size == "1M":
        num_samples = 1000000
    else:
        raise ValueError("Invalid corpus size")
    return num_samples

def pretrain_flair(args):
    flair.set_seed(args.seed)

    if torch.cuda.is_available():
        flair.device = f"cuda:{args.cuda_device}"

    save_base_path = get_save_base_path(args, task_name="pretrained-dual-encoder")

    label_type = "ner"
    if args.dataset_path:
        corpus = LONER(base_path=args.dataset_path)
        num_samples = get_corpus_size(args)
        indices = random.sample(range(len(corpus.train)), k=num_samples)
        corpus._train = Subset(corpus._train, indices)
    else:
        corpus = get_corpus(args.dataset, args.fewnerd_granularity)

    # Create label dictionary and decoder. The decoder needs the label dictionary using BIO encoding from TokenClassifier.
    label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
    decoder_dict = TokenClassifier._create_internal_label_dictionary(label_dict, span_encoding="BIO")
    decoder_dict.span_labels = True

    # Create model
    embeddings = TransformerWordEmbeddings(args.encoder_transformer)
    decoder = BatchedLabelVerbalizerDecoder(label_embedding=TransformerDocumentEmbeddings(args.decoder_transformer), label_dictionary=decoder_dict, requires_masking=True, mask_size=args.mask_size)
    model = TokenClassifier(embeddings=embeddings, decoder=decoder, label_dictionary=label_dict, label_type=label_type, span_encoding="BIO")

    trainer = ModelTrainer(model, corpus)

    trainer.fine_tune(
        save_base_path,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        mini_batch_size=args.bs,
        mini_batch_chunk_size=args.mbs,
        save_model_each_k_epochs=1,
    )


def pretrain_hf(args):
    pl.seed_everything(args.seed)

    save_base_path = get_save_base_path(args, task_name="pretrained-dual-encoder-hf")

    dataset = load_dataset("json", data_files=glob.glob(f'{args.dataset_path}/*'))
    num_samples = get_corpus_size(args)
    random_numbers = random.sample(range(0, len(dataset["train"]) + 1), num_samples)
    small_dataset = dataset["train"].select(random_numbers)

    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_transformer)
    if args.decoder_transformer == "all-mpnet-base-v2":
        decoder_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    else:
        decoder_tokenizer = AutoTokenizer.from_pretrained(args.decoder_transformer)

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
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        save_strategy="epoch",
        save_total_limit=3,
        seed=args.seed,
    )


    with open('/vol/tmp/goldejon/datasets/loner/zelda_labelID2label.json', 'r') as f:
        labels = json.load(f)

    model = HfDualEncoder(labels=labels, encoder_model=args.encoder_transformer, decoder_model=args.decoder_transformer, tokenizer=decoder_tokenizer, mask_size=args.mask_size, uniform_p=args.uniform_p, geometric_p=args.geometric_p)

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

def train_fewshot(args):
    flair.set_seed(args.seed)

    if torch.cuda.is_available():
        flair.device = f"cuda:{args.cuda_device}"

    save_base_path = get_save_base_path(args, task_name=args.task_name)

    with open(f"data/fewshot_fewnerdfine.json", "r") as f:
        fewshot_indices = json.load(f)

    results = {}

    # every pretraining seed masks out different examples in the dataset
    base_corpus = get_corpus(args.dataset, args.fewnerd_granularity)

    # iterate over k-shots
    for k in args.k:

        # average k-shot scores over 3 seeds for pretraining seed
        results[f"{k}"] = {"results": []}

        for seed in range(0, 5):

            if seed > 0 and k == 0:
                continue

            # ensure same sampling strategy for each seed
            flair.set_seed(seed)
            corpus = copy.copy(base_corpus)
            if k != 0:
                if k == -1:
                    pass
                else:
                    corpus._train = Subset(base_corpus._train, fewshot_indices[f"{k}-{seed}"])
                    corpus._dev = Subset(base_corpus._train, [])
            else:
                pass

            # mandatory for flair to work
            tag_type = "ner"
            label_dictionary = corpus.make_label_dictionary(tag_type, add_unk=False)
            decoder_dict = TokenClassifier._create_internal_label_dictionary(label_dictionary, span_encoding="BIO")
            decoder_dict.span_labels = True

            if args.model_to_use == "flair":
                model = TokenClassifier.load(args.pretrained_flair_model)
                # Set the new label_dictionary in the decoder and disable masking
                model.decoder.verbalized_labels = model.decoder.verbalize_labels(decoder_dict)
                model.decoder.requires_masking = False
                # Also set the label_dictionary in the model itself for evaluation
                model.label_dictionary = decoder_dict
                model.span_encoding = "BIO"
            elif args.model_to_use == "hf":
                if "checkpoint" in args.pretrained_hf_encoder and "checkpoint" in args.pretrained_hf_decoder:
                    state_dict = torch.load(Path(args.pretrained_hf_encoder) / "pytorch_model.bin")
                    encoder_state = OrderedDict({k.replace("encoder.", "", 1): v for k, v in state_dict.items() if k.startswith("encoder")})
                    decoder_state = OrderedDict({k.replace("decoder.", ""): v for k, v in state_dict.items() if k.startswith("decoder")})

                    encoder = TransformerWordEmbeddings("bert-base-uncased", use_context_separator=False, use_context=False)
                    encoder.model.load_state_dict(encoder_state)
                    if "all-mpnet-base-v2" in args.pretrained_hf_decoder:
                        label_embeddings = SentenceTransformerDocumentEmbeddings("sentence-transformers/all-mpnet-base-v2")
                    else:
                        label_embeddings = TransformerDocumentEmbeddings("bert-base-uncased", use_context_separator=False, use_context=False)
                    label_embeddings.model.load_state_dict(decoder_state)
                else:
                    # Create model
                    encoder = TransformerWordEmbeddings(args.pretrained_hf_encoder, use_context_separator=False, use_context=False)
                    if "all-mpnet-base-v2" in args.pretrained_hf_decoder:
                        label_embeddings = SentenceTransformerDocumentEmbeddings(args.pretrained_hf_decoder)
                    else:
                        label_embeddings = TransformerDocumentEmbeddings(args.pretrained_hf_decoder, use_context_separator=False, use_context=False)
                decoder = BatchedLabelVerbalizerDecoder(
                    label_embedding=label_embeddings, label_dictionary=decoder_dict,
                    requires_masking=False, mask_size=args.mask_size)
                model = TokenClassifier(embeddings=encoder, decoder=decoder, label_dictionary=label_dictionary,
                                        label_type=tag_type, span_encoding="BIO")
            else:
                raise ValueError("model_to_use must be one of 'flair' or 'hf'")

            if k != 0:
                trainer = ModelTrainer(model, corpus)

                save_path = save_base_path / f"{k}shot_{seed}"

                # 7. run fine-tuning
                result = trainer.train(
                    save_path,
                    learning_rate=args.lr,
                    mini_batch_size=args.bs,
                    mini_batch_chunk_size=args.mbs,
                    max_epochs=args.epochs if k != -1 else 3,
                    optimizer=torch.optim.AdamW,
                    train_with_dev=True,
                    min_learning_rate=args.lr * 1e-2,
                    save_final_model=False,
                )

                results[f"{k}"]["results"].append(result["test_score"])

                for sentence in corpus.train:
                    for token in sentence:
                        token.remove_labels(tag_type)
            else:
                save_path = save_base_path / f"{k}shot_{seed}"
                import os

                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                result = model.evaluate(corpus.test, "ner", out_path=save_path / "predictions.txt")
                results[f"{k}"]["results"].append(result.main_score)
                with open(save_path / "result.txt", "w") as f:
                    f.write(result.detailed_results)

    def postprocess_scores(scores: dict):
        rounded_scores = [round(float(score) * 100, 2) for score in scores["results"]]
        return {"results": rounded_scores, "average": np.mean(rounded_scores), "std": np.std(rounded_scores)}

    def add_total_average(results: dict):
        kshots = set([x.split("-")[0] for x in results.keys()])
        for kshot in kshots:
            vals = [value["average"] for key, value in results.items() if key.split("_")[0].replace("shot", "")]
            results[f"total-average-{kshot}"] = np.round(np.mean(vals))
        return results

    results = {setting: postprocess_scores(result) for setting, result in results.items()}
    results = add_total_average(results)

    with open(save_base_path / "results.json", "w") as f:
        json.dump(results, f)


def train_low_resource(args):
    flair.set_seed(args.seed)

    if torch.cuda.is_available():
        flair.device = f"cuda:{args.cuda_device}"

    save_base_path = get_save_base_path(args, task_name="low-resource-dual-encoder-main-experiment")

    with open(f"data/fewshot_{args.dataset}{args.fewnerd_granularity if args.dataset == 'fewnerd' else ''}.json", "r") as f:
        fewshot_indices = json.load(f)

    results = {}

    # every pretraining seed masks out different examples in the dataset
    base_corpus = get_corpus(args.dataset, args.fewnerd_granularity)

    # iterate over k-shots
    for k in args.k:

        # average k-shot scores over 3 seeds for pretraining seed
        results[f"{k}"] = {"results": []}

        for seed in range(0, 5):

            if seed > 0 and k == 0:
                continue

            # ensure same sampling strategy for each seed
            flair.set_seed(seed)
            corpus = copy.copy(base_corpus)
            if k != 0:
                if k == -1:
                    pass
                else:
                    corpus._train = Subset(base_corpus._train, fewshot_indices[f"{k}-{seed}"])
                    corpus._dev = Subset(base_corpus._train, [])
            else:
                pass

            # mandatory for flair to work
            tag_type = "ner"
            label_dictionary = corpus.make_label_dictionary(tag_type, add_unk=False)
            decoder_dict = TokenClassifier._create_internal_label_dictionary(label_dictionary, span_encoding="BIO")
            decoder_dict.span_labels = True

            if args.model_to_use == "flair":
                model = TokenClassifier.load(args.pretrained_flair_model)
                # Set the new label_dictionary in the decoder and disable masking
                model.decoder.verbalized_labels = model.decoder.verbalize_labels(decoder_dict)
                model.decoder.requires_masking = False
                # Also set the label_dictionary in the model itself for evaluation
                model.label_dictionary = decoder_dict
                model.span_encoding = "BIO"
            elif args.model_to_use == "hf":
                if "checkpoint" in args.pretrained_hf_encoder and "checkpoint" in args.pretrained_hf_decoder:
                    state_dict = torch.load(Path(args.pretrained_hf_encoder) / "pytorch_model.bin")
                    encoder_state = OrderedDict({k.replace("encoder.", "", 1): v for k, v in state_dict.items() if k.startswith("encoder")})
                    decoder_state = OrderedDict({k.replace("decoder.", ""): v for k, v in state_dict.items() if k.startswith("decoder")})

                    encoder = TransformerWordEmbeddings("bert-base-uncased", use_context_separator=False, use_context=False)
                    encoder.model.load_state_dict(encoder_state)
                    if "all-mpnet-base-v2" in args.pretrained_hf_decoder:
                        label_embeddings = SentenceTransformerDocumentEmbeddings("sentence-transformers/all-mpnet-base-v2")
                    else:
                        label_embeddings = TransformerDocumentEmbeddings("bert-base-uncased", use_context_separator=False, use_context=False)
                    label_embeddings.model.load_state_dict(decoder_state)
                else:
                    # Create model
                    encoder = TransformerWordEmbeddings(args.pretrained_hf_encoder, use_context_separator=False, use_context=False)
                    if "all-mpnet-base-v2" in args.pretrained_hf_decoder:
                        label_embeddings = SentenceTransformerDocumentEmbeddings(args.pretrained_hf_decoder)
                    else:
                        label_embeddings = TransformerDocumentEmbeddings(args.pretrained_hf_decoder, use_context_separator=False, use_context=False)
                decoder = BatchedLabelVerbalizerDecoder(
                    label_embedding=label_embeddings, label_dictionary=decoder_dict,
                    requires_masking=False, mask_size=args.mask_size)
                model = TokenClassifier(embeddings=encoder, decoder=decoder, label_dictionary=label_dictionary,
                                        label_type=tag_type, span_encoding="BIO")
            elif args.model_to_use == "bert":
                label_dict = corpus.make_label_dictionary(label_type=tag_type, add_unk=False)
                decoder_dict = TokenClassifier._create_internal_label_dictionary(label_dict, span_encoding="BIO")
                decoder_dict.span_labels = True

                # Create model
                embeddings = TransformerWordEmbeddings("bert-base-uncased")
                decoder = BatchedLabelVerbalizerDecoder(
                    label_embedding=TransformerDocumentEmbeddings("bert-base-uncased"),
                    label_dictionary=decoder_dict, requires_masking=False, mask_size=args.mask_size)
                model = TokenClassifier(embeddings=embeddings, decoder=decoder, label_dictionary=label_dict,
                                        label_type=tag_type, span_encoding="BIO")
            else:
                raise ValueError("model_to_use must be one of 'flair' or 'hf'")

            if k != 0:
                trainer = ModelTrainer(model, corpus)

                save_path = save_base_path / f"{k}shot_{seed}"

                # 7. run fine-tuning
                result = trainer.train(
                    save_path,
                    learning_rate=args.lr,
                    mini_batch_size=args.bs,
                    mini_batch_chunk_size=args.mbs,
                    max_epochs=args.epochs if k != -1 else 3,
                    optimizer=torch.optim.AdamW,
                    train_with_dev=True,
                    min_learning_rate=args.lr * 1e-2,
                    save_final_model=False,
                )

                results[f"{k}"]["results"].append(result["test_score"])

                for sentence in corpus.train:
                    for token in sentence:
                        token.remove_labels(tag_type)
            else:
                save_path = save_base_path / f"{k}shot_{seed}"
                import os

                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                result = model.evaluate(corpus.test, "ner", out_path=save_path / "predictions.txt")
                results[f"{k}"]["results"].append(result.main_score)
                with open(save_path / "result.txt", "w") as f:
                    f.write(result.detailed_results)

    def postprocess_scores(scores: dict):
        rounded_scores = [round(float(score) * 100, 2) for score in scores["results"]]
        return {"results": rounded_scores, "average": np.mean(rounded_scores), "std": np.std(rounded_scores)}

    def add_total_average(results: dict):
        kshots = set([x.split("-")[0] for x in results.keys()])
        for kshot in kshots:
            vals = [value["average"] for key, value in results.items() if key.split("_")[0].replace("shot", "")]
            results[f"total-average-{kshot}"] = np.round(np.mean(vals))
        return results

    results = {setting: postprocess_scores(result) for setting, result in results.items()}
    results = add_total_average(results)

    with open(save_base_path / "results.json", "w") as f:
        json.dump(results, f)

def train_masked_fewshot(args):
    flair.set_seed(args.seed)

    if torch.cuda.is_available():
        flair.device = f"cuda:{args.cuda_device}"

    save_base_path = get_save_base_path(args, task_name=args.task_name)

    with open(f"data/fewshot_masked-fewnerd-{args.fewnerd_granularity}.json", "r") as f:
        fewshot_indices = json.load(f)

    results = {}
    for pretraining_seed in args.pretraining_seeds:

        # every pretraining seed masks out different examples in the dataset
        base_corpus, kept_labels = get_masked_fewnerd_corpus(
            pretraining_seed, args.fewnerd_granularity, inverse_mask=True
        )

        # iterate over k-shots
        for k in args.k:

            # average k-shot scores over 3 seeds for pretraining seed
            results[f"{k}-{pretraining_seed}"] = {"results": []}

            for seed in range(0, 3):
                # ensure same sampling strategy for each seed
                flair.set_seed(seed)
                corpus = copy.copy(base_corpus)
                if k != 0:
                    corpus._train = Subset(base_corpus._train, fewshot_indices[f"{k}-{pretraining_seed}-{seed}"])
                else:
                    pass
                corpus._dev = Subset(base_corpus._train, [])

                # mandatory for flair to work
                tag_type = "ner"
                label_dictionary = corpus.make_label_dictionary(tag_type, add_unk=False)
                decoder_dict = TokenClassifier._create_internal_label_dictionary(label_dictionary, span_encoding="BIO")
                decoder_dict.span_labels = True

                if args.model_to_use == "flair":
                    model = TokenClassifier.load(args.pretrained_flair_model)
                    # Set the new label_dictionary in the decoder and disable masking
                    model.decoder.verbalized_labels = model.decoder.verbalize_labels(decoder_dict)
                    model.decoder.requires_masking = False
                    # Also set the label_dictionary in the model itself for evaluation
                    model.label_dictionary = decoder_dict
                    model.span_encoding = "BIO"
                elif args.model_to_use == "hf":
                    # Create model
                    encoder = TransformerWordEmbeddings(args.pretrained_hf_encoder)
                    if "all-mpnet-base-v2" in args.pretrained_hf_decoder:
                        label_embeddings = SentenceTransformerDocumentEmbeddings(args.pretrained_hf_decoder)
                    else:
                        label_embeddings = TransformerDocumentEmbeddings(args.pretrained_hf_decoder)
                    decoder = BatchedLabelVerbalizerDecoder(
                        label_embedding=label_embeddings, label_dictionary=decoder_dict,
                        requires_masking=False, mask_size=args.mask_size)
                    model = TokenClassifier(embeddings=encoder, decoder=decoder, label_dictionary=label_dictionary,
                                            label_type=tag_type, span_encoding="BIO")
                else:
                    raise ValueError("model_to_use must be one of 'flair' or 'hf'")

                if k > 0:
                    trainer = ModelTrainer(model, corpus)

                    save_path = save_base_path / f"{k}shot_{pretraining_seed}_{seed}"

                    # 7. run fine-tuning
                    result = trainer.train(
                        save_path,
                        learning_rate=args.lr,
                        mini_batch_size=args.bs,
                        mini_batch_chunk_size=args.mbs,
                        max_epochs=args.epochs,
                        optimizer=torch.optim.AdamW,
                        train_with_dev=True,
                        min_learning_rate=args.lr * 1e-2,
                        save_final_model=False,
                    )

                    results[f"{k}-{pretraining_seed}"]["results"].append(result["test_score"])

                    for sentence in corpus.train:
                        for token in sentence:
                            token.remove_labels(tag_type)
                else:
                    save_path = save_base_path / f"{k}shot_{pretraining_seed}_{seed}"
                    import os

                    if not os.path.exists(save_path):
                        os.mkdir(save_path)

                    result = model.evaluate(corpus.test, "ner", out_path=save_path / "predictions.txt")
                    results[f"{k}-{pretraining_seed}"]["results"].append(result.main_score)
                    with open(save_path / "result.txt", "w") as f:
                        f.write(result.detailed_results)

    def postprocess_scores(scores: dict):
        rounded_scores = [round(float(score) * 100, 2) for score in scores["results"]]
        return {"results": rounded_scores, "average": np.mean(rounded_scores), "std": np.std(rounded_scores)}

    def add_total_average(results: dict):
        kshots = set([x.split("-")[0] for x in results.keys()])
        for kshot in kshots:
            vals = [value["average"] for key, value in results.items() if key.split("_")[0].replace("shot", "")]
            results[f"total-average-{kshot}"] = np.round(np.mean(vals))
        return results

    results = {setting: postprocess_scores(result) for setting, result in results.items()}
    results = add_total_average(results)

    with open(save_base_path / "results.json", "w") as f:
        json.dump(results, f)

def find_hyperparameters(args):
    flair.device = f"cuda:{args.cuda_device}"

    save_base_path = Path(
        f"{args.cache_path}/fewshot-loner-hyperparams/"
        f"fewshot-fewnerd-{args.fewnerd_granularity}"
        f"_{args.lr}"
        f"_pretrained-on-{args.pretrained_model_path.split('/')[-2].split('_', 1)[-1]}"
        f"-epoch-{args.pretrained_model_path.split('/')[-1].split('_')[-1].replace('.pt', '')}"
    )

    with open(f"data/fewshot_masked-fewnerd-{args.fewnerd_granularity}.json", "r") as f:
        fewshot_indices = json.load(f)

    results = {}

    for pretraining_seed in args.pretraining_seeds:

        base_corpus, kept_labels = get_masked_fewnerd_corpus(
            pretraining_seed, args.fewnerd_granularity, inverse_mask=True
        )

        model_paths = [
            "/vol/tmp/goldejon/flair-models/pretrained-dual-encoder/bert-base-uncased_LONER_lr-1e-06_seed-123_mask-128_size-100k/model_epoch_1.pt",
            "/vol/tmp/goldejon/flair-models/pretrained-dual-encoder/bert-base-uncased_LONER_lr-1e-06_seed-123_mask-128_size-100k/model_epoch_2.pt",
            "/vol/tmp/goldejon/flair-models/pretrained-dual-encoder/bert-base-uncased_LONER_lr-1e-06_seed-123_mask-128_size-100k/model_epoch_3.pt"
        ]

        for path in model_paths:
            for k in args.k:
                for lr in [5e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6, 5e-6]:
                    for seed in range(2):
                        flair.set_seed(seed)
                        corpus = copy.copy(base_corpus)
                        if k != 0:
                            corpus._train = Subset(base_corpus._train, fewshot_indices[f"{k}-{pretraining_seed}-{seed}"])
                        else:
                            pass
                        corpus._dev = Subset(base_corpus._train, [])

                        tag_type = "ner"
                        label_dictionary = corpus.make_label_dictionary(tag_type, add_unk=False)

                        model = TokenClassifier.load(path)
                        decoder_dict = TokenClassifier._create_internal_label_dictionary(label_dictionary,  "BIO")
                        decoder_dict.span_labels = True

                        # Set the new label_dictionary in the decoder and disable masking
                        model.decoder.verbalized_labels = model.decoder.verbalize_labels(decoder_dict)
                        model.decoder.requires_masking = False
                        # Also set the label_dictionary in the model itself for evaluation
                        model.label_dictionary = decoder_dict
                        model.span_encoding = "BIO"

                        trainer = ModelTrainer(model, corpus)

                        save_path = save_base_path / f"{k}shot_{pretraining_seed}_{seed}"

                        # 7. run fine-tuning
                        result = trainer.train(
                            save_path,
                            learning_rate=lr,
                            mini_batch_size=args.bs,
                            mini_batch_chunk_size=args.mbs,
                            max_epochs=args.epochs,
                            optimizer=torch.optim.AdamW,
                            train_with_dev=True,
                            min_learning_rate=args.lr * 1e-2,
                            save_final_model=False,
                        )

                        model_path_identifier = path.split("/")[-2]
                        key = f"{model_path_identifier}-{lr}"
                        if not key in results:
                            results[key] = [round(result["test_score"] * 100, 2)]
                        else:
                            results[key].append(round(result["test_score"] * 100, 2))

    with open("hyperparam_results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--pretrain_flair", action="store_true")
    parser.add_argument("--pretrain_hf", action="store_true")
    parser.add_argument("--train_fewshot", action="store_true")
    parser.add_argument("--train_low_resource", action="store_true")
    parser.add_argument("--train_masked_fewshot", action="store_true")
    parser.add_argument("--find_hyperparameters", action="store_true")
    parser.add_argument("--task_name", type=str)

    # Pretraining arguments
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--pretraining_seeds", type=int, nargs="+", default=[10])
    parser.add_argument("--cache_path", type=str, default="/vol/tmp/goldejon/flair-models")
    # All datasets from flair or hugginface
    parser.add_argument("--dataset", type=str, default="") # fewnerd
    parser.add_argument("--fewnerd_granularity", type=str, default="") # coarse
    # LONER needs to be loaded from disk
    parser.add_argument("--dataset_path", type=str, default="") # "/vol/tmp/goldejon/datasets/loner/zelda_jsonl_bio"
    parser.add_argument("--corpus_size", type=str, default="100k")
    parser.add_argument("--encoder_transformer", type=str, default="bert-base-uncased")
    parser.add_argument("--decoder_transformer", type=str, default="bert-base-uncased")

    # Training arguments
    parser.add_argument("--mask_size", type=int, default=128)
    parser.add_argument("--uniform_p", type=float, nargs="+", default=[0.5, 0.5])
    parser.add_argument("--geometric_p", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--mbs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--k", type=int, nargs="+", default=[1])
    parser.add_argument("--model_to_use", type=str, default="flair")
    parser.add_argument("--pretrained_flair_model", type=str, default="/vol/tmp/goldejon/flair-models/pretrained-dual-encoder/bert-base-uncased_LONER_lr-1e-05_seed-123_mask-128_size-100k/model_epoch_3.pt")
    parser.add_argument("--pretrained_hf_encoder", type=str, default="/vol/tmp/goldejon/flair-models/pretrained-dual-encoder-hf/bert-base-uncased_LONER_lr-1e-06_seed-123_mask-128_size-500k/encoder")
    parser.add_argument("--pretrained_hf_decoder", type=str, default="/vol/tmp/goldejon/flair-models/pretrained-dual-encoder-hf/bert-base-uncased_LONER_lr-1e-06_seed-123_mask-128_size-500k/decoder")
    args = parser.parse_args()

    if not any([args.dataset, args.dataset_path]):
        raise ValueError("no dataset provided.")

    if args.pretrain_flair:
        pretrain_flair(args)
    if args.pretrain_hf:
        pretrain_hf(args)
    if args.train_fewshot:
        train_fewshot(args)
    if args.train_low_resource:
        train_low_resource(args)
    if args.train_masked_fewshot:
        train_masked_fewshot(args)
    if args.find_hyperparameters:
        find_hyperparameters(args)
