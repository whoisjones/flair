import argparse
from pathlib import Path
import random
import json
import copy
import itertools
from typing import List

import numpy as np
import torch
import torch.cuda

import flair
from flair.data import Sentence, Dictionary
from flair.datasets import CONLL_03
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

def train(args):
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
    samples = 100000 if args.corpus_size =="100k" else 1000000
    indices = random.sample(range(len(corpus.train)), k=samples)
    corpus._train = Subset(corpus._train, indices)

    # Create label dictionary and decoder. The decoder needs the label dictionary using BIO encoding from TokenClassifier.
    label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
    decoder_dict = TokenClassifier._create_internal_label_dictionary(label_dict, args.span_encoding)
    decoder_dict.span_labels = True

    # Create model
    embeddings = TransformerWordEmbeddings(args.transformer)
    decoder = BatchedLabelVerbalizerDecoder(label_embedding=TransformerDocumentEmbeddings(args.transformer), label_dictionary=decoder_dict, requires_masking=True, mask_size=args.mask_size)
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

def eval(args):
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

    #train(args)
    eval(args)


