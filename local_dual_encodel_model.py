import os
import json
import copy
import glob
import random
import argparse
from pathlib import Path
from typing import List

import numpy as np
import numpy.random
import torch
import torch.cuda
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer, \
    BatchEncoding, PreTrainedTokenizer
from datasets import load_dataset

import flair
from flair.data import Sentence, Dictionary
from flair.embeddings import (
    TransformerWordEmbeddings,
    TransformerDocumentEmbeddings, Embeddings
)
from flair.models import TokenClassifier
from flair.trainers import ModelTrainer
from flair.training_utils import store_embeddings
from torch.utils.data.dataset import Subset

from local_loner import LONER
from local_corpora import get_masked_fewnerd_corpus

class MemoryEfficientLabelVerbalizerDecoder(torch.nn.Module):
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

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor = None, inference: bool = False) -> torch.Tensor:

        if self.training and self.requires_masking:
            labels_to_include = labels.cpu().numpy().tolist()
            labels, indices = self.embedding_sublist(labels_to_include)
            self.label_embedding.embed(labels)
        elif inference or not self.requires_masking:
            labels = self.label_embedding.embed(self.verbalized_labels)

        label_tensor = torch.stack([label.get_embedding() for label in labels])

        if self.training:
            store_embeddings(labels, "none")

        scores = torch.mm(inputs, label_tensor.T)

        if self.training and self.requires_masking:
            all_scores = torch.zeros(scores.shape[0], len(self.verbalized_labels), device=flair.device)
            all_scores[:, torch.LongTensor(list(indices))] = scores
        elif inference or not self.requires_masking:
            all_scores = scores

        return all_scores

class HfDualEncoder(torch.nn.Module):
    def __init__(self, labels: List[str], tokenizer: PreTrainedTokenizer, mask_size: int = 128):
        super(HfDualEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.decoder = AutoModel.from_pretrained("bert-base-uncased")
        self.labels = labels
        self.num_labels = len(labels)
        self.tokenizer = tokenizer
        self.mask_size = mask_size
        self.loss = torch.nn.CrossEntropyLoss()

    def _filter_labels(self, labels):
        np_labels = labels.detach().cpu().numpy()
        unique_labels = np.unique(np.clip(np_labels, a_min=0, a_max=None))
        additional_ids_needed = self.mask_size - unique_labels.shape[0]
        sampled_labels = numpy.random.choice(np.arange(0, len(self.labels)), size=additional_ids_needed, replace=False)
        selected_labels = np.unique(np.concatenate([unique_labels, sampled_labels]))
        label_description = [self.labels[i] for i in selected_labels]
        encoded_labels = self.tokenizer(label_description, padding=True, truncation=True, max_length=64, return_tensors="pt").to(labels.device)
        return encoded_labels, selected_labels
    def forward(self, input_ids, attention_mask, labels):
        device = f'cuda:{str(labels.device.index)}'
        token_hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoded_labels, selected_label_ids = self._filter_labels(labels)
        label_hidden_states = self.decoder(**encoded_labels)
        label_cls_token = label_hidden_states.last_hidden_state[:, 0, :]
        scores = torch.matmul(token_hidden_states.last_hidden_state, label_cls_token.T)
        logits = torch.zeros(labels.size(0), labels.size(1) , self.num_labels, device=device)
        logits[:, :, selected_label_ids] = scores
        return (self.loss(logits.transpose(1, 2), labels),)

def train_flair(args):
    flair.set_seed(args.seed)

    if torch.cuda.is_available():
        flair.device = f"cuda:{args.cuda_device}"

    save_base_path = Path(
        f"{args.cache_path}/pretrained-dual-encoder/"
        f"{args.transformer}"
        f"_LONER_lr-{args.lr}_seed-{args.seed}_mask-{args.mask_size}_size-{args.corpus_size}"
    )

    label_type = "ner"
    corpus = LONER(base_path=args.dataset_path)

    # Downsample corpus to either 100k or 1M samples
    if not args.corpus_size == "full":
        if args.corpus_size == "100k":
            samples = 100000
        elif args.corpus_size == "1M":
            samples = 1000000
        elif args.corpus_size == "5M":
            samples = 5000000
        else:
            raise ValueError("Invalid corpus size")
        indices = random.sample(range(len(corpus.train)), k=samples)
        corpus._train = Subset(corpus._train, indices)

    # Create label dictionary and decoder. The decoder needs the label dictionary using BIO encoding from TokenClassifier.
    label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
    decoder_dict = TokenClassifier._create_internal_label_dictionary(label_dict, args.span_encoding)
    decoder_dict.span_labels = True

    # Create model
    embeddings = TransformerWordEmbeddings(args.transformer)
    decoder = MemoryEfficientLabelVerbalizerDecoder(label_embedding=TransformerDocumentEmbeddings(args.transformer), label_dictionary=decoder_dict, requires_masking=True, mask_size=args.mask_size)
    model = TokenClassifier(embeddings=embeddings, decoder=decoder, label_dictionary=label_dict, label_type=label_type, span_encoding=args.span_encoding)

    trainer = ModelTrainer(model, corpus)

    trainer.fine_tune(
        save_base_path,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        mini_batch_size=args.bs,
        mini_batch_chunk_size=args.mbs,
        save_model_each_k_epochs=1,
    )


def train_hf(args):

    is_zelda = True if "zelda" in args.dataset_path.lower() else False

    save_base_path = Path(
        f"{args.cache_path}/pretrained-dual-encoder-hf/"
        f"{args.transformer}"
        f"_LONER{'-ZELDA' if is_zelda else ''}"
        f"_lr-{args.lr}"
        f"_seed-{args.seed}"
        f"_mask-{args.mask_size}"
        f"_size-{args.corpus_size}"
    )

    pl.seed_everything(args.seed)
    dataset = load_dataset("json", data_files=glob.glob(f'{args.dataset_path}/*'))
    if not args.corpus_size == "full":
        if args.corpus_size == "100k":
            num_samples = 100000
        elif args.corpus_size == "500k":
            num_samples = 500000
        elif args.corpus_size == "1M":
            num_samples = 1000000
        else:
            raise ValueError("Invalid corpus size")
    else:
        num_samples = len(dataset["train"])
    random_numbers = random.sample(range(0, len(dataset["train"]) + 1), num_samples)
    small_dataset = dataset["train"].select(random_numbers)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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
        tokenized_inputs = tokenizer(
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

    data_collator = DataCollatorForTokenClassification(tokenizer)

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

    with open("/glusterfs/dfs-gfs-dist/goldejon/datasets/loner/labelID2label_bio.json", "r") as f:
        label_mapping = json.load(f)

    labels = []
    for label in label_mapping.values():
        if label.startswith("B-"):
            labels.append(f'begin {label[2:]}')
        elif label.startswith("I-"):
            labels.append(f'inside {label[2:]}')
        elif label == "O":
            labels.append("outside")
        else:
            print("error")

    model = HfDualEncoder(labels=labels, tokenizer=tokenizer, mask_size=args.mask_size)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    model.encoder.save_pretrained(save_base_path / "encoder")
    model.decoder.save_pretrained(save_base_path / "decoder")



def eval_fewshot_fewnerd(args):
    flair.device = f"cuda:{args.cuda_device}"

    save_base_path = Path(
        f"{args.cache_path}/fewshot-dual-encoder/masked-models/"
        f"{args.transformer}"
        f"_fewnerd-{args.fewnerd_granularity}"
        f"_pretrained-on-{args.pretrained_model_path.split('/')[-2].split('_', 1)[-1]}"
        f"_{args.lr}"
        f"{'_early-stopping' if args.early_stopping else ''}"
    )

    with open(f"data/fewshot_masked-fewnerd-{args.fewnerd_granularity}.json", "r") as f:
        fewshot_indices = json.load(f)

    for pretraining_seed in [10, 20, 30, 40, 50]:

        base_corpus, kept_labels = get_masked_fewnerd_corpus(
            pretraining_seed, args.fewnerd_granularity, inverse_mask=True
        )

        results = {}
        for k in args.k:
            results[f"{k}"] = {"results": []}
            for seed in range(5):
                flair.set_seed(seed)
                corpus = copy.copy(base_corpus)
                if k != 0:
                    corpus._train = Subset(base_corpus._train, fewshot_indices[f"{k}-{pretraining_seed}-{seed}"])
                else:
                    pass
                corpus._dev = Subset(base_corpus._train, [])

                tag_type = "ner"
                label_dictionary = corpus.make_label_dictionary(tag_type, add_unk=False)

                model = TokenClassifier.load(args.pretrained_model_path)
                decoder_dict = TokenClassifier._create_internal_label_dictionary(label_dictionary,  args.span_encoding)
                decoder_dict.span_labels = True

                # Set the new label_dictionary in the decoder and disable masking
                model.decoder.verbalized_labels = model.decoder.verbalize_labels(decoder_dict)
                model.decoder.requires_masking = False
                # Also set the label_dictionary in the model itself for evaluation
                model.label_dictionary = decoder_dict
                model.span_encoding = args.span_encoding

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
                        train_with_dev=args.early_stopping,
                        min_learning_rate=args.min_lr if args.early_stopping else 0.001,
                        save_final_model=False,
                    )

                    results[f"{k}"]["results"].append(result["test_score"])
                else:
                    save_path = save_base_path / f"{k}shot_{pretraining_seed}_{seed}"
                    import os

                    if not os.path.exists(save_path):
                        os.mkdir(save_path)

                    decoder_dict = TokenClassifier._create_internal_label_dictionary(label_dictionary, args.span_encoding)
                    decoder_dict.span_labels = True

                    # Set the new label_dictionary in the decoder and disable masking
                    model.decoder.verbalized_labels = model.decoder.verbalize_labels(decoder_dict)
                    model.decoder.requires_masking = False
                    # Also set the label_dictionary in the model itself for evaluation
                    model.label_dictionary = decoder_dict

                    result = model.evaluate(corpus.test, "ner", out_path=save_path / "predictions.txt")
                    results[f"{k}"]["results"].append(result.main_score)
                    with open(save_path / "result.txt", "w") as f:
                        f.write(result.detailed_results)

    def postprocess_scores(scores: dict):
        rounded_scores = [round(float(score) * 100, 2) for score in scores["results"]]
        return {"results": rounded_scores, "average": np.mean(rounded_scores), "std": np.std(rounded_scores)}

    results = {setting: postprocess_scores(result) for setting, result in results.items()}

    with open(save_base_path / "results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_flair", action="store_true")
    parser.add_argument("--train_hf", action="store_true")
    parser.add_argument("--do_eval_fewshot_fewnerd", action="store_true")
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cache_path", type=str, default="/glusterfs/dfs-gfs-dist/goldejon/flair-models")
    parser.add_argument("--dataset_path", type=str, default="/glusterfs/dfs-gfs-dist/goldejon/datasets/loner")
    parser.add_argument("--span_encoding", type=str, default="BIO")
    parser.add_argument("--mask_size", type=int, default=128)
    parser.add_argument("--corpus_size", type=str, default="100k")
    parser.add_argument("--transformer", type=str, default="bert-base-uncased")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--mbs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--anneal_factor", type=float, default=0.5)
    parser.add_argument("--fewnerd_granularity", type=str, default="coarse")
    parser.add_argument("--k", type=int, default=1, nargs="+")
    parser.add_argument("--pretrained_model_path", type=str, default="/glusterfs/dfs-gfs-dist/goldejon/flair-models/pretrained-dual-encoder/bert-base-uncased_LONER_lr-1e-05_seed-123_mask-128_size-100k/model_epoch_3.pt")
    args = parser.parse_args()

    if args.train_flair:
        train_flair(args)
    if args.train_hf:
        train_hf(args)
    if args.do_eval_fewshot_fewnerd:
        eval_fewshot_fewnerd(args)