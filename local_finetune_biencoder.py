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

def train_fewnerd_fine(args):
    flair.set_seed(123)

    if torch.cuda.is_available():
        flair.device = f"cuda:{args.cuda_device}"

    save_base_path = get_save_base_path(args, task_name=args.task_name)

    with open(f"data/fewshot_fewnerdfine.json", "r") as f:
        fewshot_indices = json.load(f)

    results = {}

    # every pretraining seed masks out different examples in the dataset
    base_corpus = get_corpus(args.dataset, args.fewnerd_granularity)

    # iterate over k-shots
    for k in [1, 2, 4, 8, 16, 0]:

        # average k-shot scores over 3 seeds for pretraining seed
        results[f"{k}"] = {"results": []}

        for seed in range(3):

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


def train_masked_fewshot(args):
    flair.set_seed(123)

    if torch.cuda.is_available():
        flair.device = f"cuda:{args.cuda_device}"

    save_base_path = get_save_base_path(args, task_name=args.task_name)

    with open(f"data/fewshot_masked-fewnerd-{args.fewnerd_granularity}.json", "r") as f:
        fewshot_indices = json.load(f)

    results = {}
    for pretraining_seed in [10, 20, 30, 40]:

        # every pretraining seed masks out different examples in the dataset
        base_corpus, kept_labels = get_masked_fewnerd_corpus(
            pretraining_seed, args.fewnerd_granularity, inverse_mask=True
        )

        # iterate over k-shots
        for k in range(3):

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
