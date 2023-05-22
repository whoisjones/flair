import argparse
from pathlib import Path
import random
from typing import List

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

def eval(model_path):
    #model_path = "/glusterfs/dfs-gfs-dist/goldejon/flair-models/pretrained-dual-encoder/bert-base-uncased_LONER_1e-05-123/model_epoch_3.pt"
    corpus = CONLL_03(
        column_format={0: "text", 1: "pos", 2: "chunk", 3: "ner"},
        label_name_map={"PER": "person", "LOC": "location", "ORG": "organization", "MISC": "miscellaneous"},
    )
    label_dict = corpus.make_label_dictionary(label_type="ner", add_unk=False)

    # Load model and create new label dictionary
    model = TokenClassifier.load(model_path)
    decoder_dict = TokenClassifier._create_internal_label_dictionary(label_dict, "BIO")
    decoder_dict.span_labels = True

    # Set the new label_dictionary in the decoder and disable masking
    model.decoder.verbalized_labels = model.decoder.verbalize_labels(decoder_dict)
    model.decoder.requires_masking = False
    # Also set the label_dictionary in the model itself for evaluation
    model.label_dictionary = decoder_dict

    # Evaluate
    results = model.evaluate(corpus.test, "ner")
    print(results.detailed_results)

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
    args = parser.parse_args()

    train(args)
    #eval()

