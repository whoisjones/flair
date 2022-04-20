Multitask Learning (beta)
----------

We currently support two variants of multitask learning: standard
sequence tagging (similar to [MT-DNN](https://arxiv.org/abs/1901.11504))
and TARS tagging.

---------------------------

Prerequisites
------
In your local flair repository, you need to have two supporting
files: `local_args.py` (takes care of argument parsing) 
and `local_get_corpora.py` (takes care of correctly preparing corpora for MTL).
Latter file will be merged at some point into the flair structure.
<details>
    <summary>local_args.py</summary>

```
import os
import re
import yaml
import argparse


def get_arguments():
    args = argparser_train()

    if args.config != "":
        args = read_config(args)

    if args.path is None and args.corpus is None:
        raise ValueError("Please provided information where to store model with --path and which corpus"
                         "to use with --corpus arguments.")

    args.corpus = args.corpus.split(",")

    setattr(args, "run", _get_current_run_number(args))

    os.makedirs(f"resources/{args.path}/run{args.run}/")

    _log_config(args)

    return args


def argparser_train():
    parser = argparse.ArgumentParser(
        description="TARS Multitask Models"
    )
    parser.add_argument("--config", type=str, default="",
                        help="Configration file (YAML) for all arguments, if empty, use command lines arguments")
    parser.add_argument(
        "--path",
        type=str,
        help="save training under given path",
    )

    # DEVICE ARGUMENTS
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="use CUDA"
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="which GPU to use"
    )

    # CORPUS ARGUMENTS
    parser.add_argument(
        "--corpus",
        nargs='+',
        help="what corpora to use"
    )
    parser.add_argument(
        "--corpus_type",
        type=str,
        help="Corpus with or without label name mapping for TARS.",
        choices=["standard", "tars"],
        default="standard"
    )
    parser.add_argument(
        "--label_type",
        type=str,
        default="ner",
    )

    # TRAINER ARGUMENTS
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="learning rate to use",
        default=5e-6
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4
    )
    parser.add_argument(
        "--batch_chunk_size",
        type=int,
        help="chunk size for gradient accumulation",
        default=4
    )

    # TRANSFORMER ARGUMENTS
    parser.add_argument(
        "--transformer_model",
        type=str,
        default="xlm-roberta-large",
    )
    parser.add_argument(
        "--fine_tune",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--use_context",
        help="whether to use document context or not",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--subtoken_pooling",
        type=str,
        default="first"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="-1"
    )

    # TARS ARGUMENTS
    parser.add_argument(
        "--num_negatives",
        type=int,
        default=2,
        help="number of negative examples to be sampled",
    )
    return parser.parse_args()

def read_config(args):
    """Return namespace object like argparser of yaml file"""

    with open(args.config, "r") as f:
        conf = yaml.safe_load(f)

    if conf.get("path") is None or conf.get("corpus") is None:
        raise ValueError("path and corpus are mandatory values in a config file.")
    else:
        for k, v in conf.items():
            setattr(args, k, v)

    return args

def _get_current_run_number(args):
    run = []
    if os.path.isdir(f"resources/{args.path}"):
        for file in os.listdir(f"resources/{args.path}"):
            if file.startswith("run"):
                run += [r for r in re.findall(r'\d+', file)]
    return max([int(r) for r in run]) + 1 if run else 1

def _log_config(args):
    with open(f"resources/{args.path}/run{args.run}/configuration.yaml", 'w') as f:
        yaml.dump(args.__dict__, f)

```
</details>

<details>
  <summary>local_get_corpora.py</summary>
  
```
import os
from pathlib import Path
from flair.datasets import CONLL_03, CONLL_03_DUTCH, CONLL_03_GERMAN, CONLL_03_SPANISH, WNUT_17, ColumnCorpus, BC2GM, CDR, NCBI_DISEASE

def get_corpus(corpus: list, corpus_type: str, label_type: str = None):
    if len(corpus) == 1:
        corpus_obj = get_single_corpus(corpus.pop(), corpus_type)
        return [
            {
                "id": f"{corpus_obj.__class__.__name__}_{label_type}_0",
                "corpus": corpus_obj,
                "label_space": corpus_obj.make_label_dictionary(label_type),
                "label_type": label_type
            }
        ]
    elif len(corpus) > 1:
        return get_mutitask_corpus(corpora=corpus, corpus_type=corpus_type, label_type=label_type)
    else:
        raise ValueError("no combination of corpus and type found.")

def get_single_corpus(corpus, corpus_type):
    if corpus == "conll" and corpus_type == "standard":
        return get_conll()
    elif corpus == "conll" and corpus_type == "tars":
        return get_conll_for_tars()
    elif corpus == "wnut" and corpus_type == "standard":
        return get_wnut()
    elif corpus == "wnut" and corpus_type == "tars":
        return get_wnut_for_tars()
    elif corpus == "ontonotes" and corpus_type == "standard":
        return get_ontonotes()
    elif corpus == "ontonotes" and corpus_type == "tars":
        return get_ontonotes_for_tars()
    elif corpus == "bc2gm" and corpus_type == "standard":
        return get_bc2gm()
    elif corpus == "bc2gm" and corpus_type == "tars":
        return get_bc2gm_for_tars()
    elif corpus == "cdr" and corpus_type == "standard":
        return get_cdr()
    elif corpus == "cdr" and corpus_type == "tars":
        return get_cdr_for_tars()
    elif corpus == "ncbi" and corpus_type == "standard":
        return get_ncbi()
    elif corpus == "ncbi" and corpus_type == "tars":
        return get_ncbi_for_tars()
    elif corpus == "conll_es" and corpus_type == "standard":
        return get_conll_es()
    elif corpus == "conll_es" and corpus_type == "tars":
        return get_conll_es_for_tars()
    elif corpus == "conll_de" and corpus_type == "standard":
        return get_conll_de()
    elif corpus == "conll_de" and corpus_type == "tars":
        return get_conll_de_for_tars()
    elif corpus == "conll_nl" and corpus_type == "standard":
        return get_conll_nl()
    elif corpus == "conll_nl" and corpus_type == "tars":
        return get_conll_nl_for_tars()

def get_mutitask_corpus(corpora: list, corpus_type: str, label_type: str):
    tagger_configurations = []
    for idx, corpus in enumerate(corpora):
        corpus = get_single_corpus(corpus, corpus_type)
        tagger_configurations.append(
            {
                "id": f"{corpus.__class__.__name__}_{label_type}_{idx}",
                "corpus": corpus,
                "label_space": corpus.make_label_dictionary(label_type),
                "label_type": label_type
            }
        )
    return tagger_configurations

def get_conll():
    return CONLL_03()

def get_conll_for_tars():
    return CONLL_03(label_name_map={
        'LOC': 'location',
        'ORG': 'organization',
        'PER': 'person',
        'MISC': 'miscellaneous'
    })

def get_ontonotes():
    return ColumnCorpus(
        os.path.join(Path(Path.home(), Path(".flair/datasets/onto_ner"))),
        column_format={0: "text", 1: "pos", 2: "upos", 3: "ner"}
    )

def get_ontonotes_for_tars():
    return ColumnCorpus(
        os.path.join(Path(Path.home(), Path(".flair/datasets/onto_ner"))),
        column_format={0: "text", 1: "pos", 2: "upos", 3: "ner"},
        label_name_map={
            'CARDINAL': 'cardinal',
            'DATE': 'date',
            'EVENT': 'event',
            'FAC': 'facility',
            'GPE': 'geopolitical',
            'LANGUAGE': 'language',
            'LAW': 'law documents',
            'LOC': 'location',
            'MONEY': 'money',
            'NORP': 'nationality, religious or political',
            'ORDINAL': 'ordinal',
            'ORG': 'organization',
            'PERCENT': 'percentage',
            'PERSON': 'person',
            'PRODUCT': 'product',
            'QUANTITY': 'quantity',
            'TIME': 'time',
            'WORK_OF_ART': 'art'}
    )

def get_wnut():
    return WNUT_17()

def get_wnut_for_tars():
    return WNUT_17(label_name_map={
        "location": "location",
        "corporation": "corporation",
        "person": "person",
        "creative-work": "creative work",
        "product": "product",
        "group": "group",
    })

def get_conll_de():
    return CONLL_03_GERMAN()

def get_conll_de_for_tars():
    return CONLL_03_GERMAN(label_name_map={
        'LOC': 'location',
        'ORG': 'organization',
        'PER': 'person',
        'MISC': 'miscellaneous'
    })

def get_conll_es():
    return CONLL_03_SPANISH()

def get_conll_es_for_tars():
    return CONLL_03_SPANISH(label_name_map={
        'LOC': 'location',
        'ORG': 'organization',
        'PER': 'person',
        'MISC': 'miscellaneous'})

def get_conll_nl():
    return CONLL_03_DUTCH()

def get_conll_nl_for_tars():
    return CONLL_03_DUTCH(label_name_map={
        'LOC': 'location',
        'ORG': 'organization',
        'PER': 'person',
        'MISC': 'miscellaneous'})

def get_bc2gm():
    return BC2GM()

def get_bc2gm_for_tars():
    return BC2GM()

def get_cdr():
    return CDR()

def get_cdr_for_tars():
    return CDR()

def get_ncbi():
    return NCBI_DISEASE()

def get_ncbi_for_tars():
    return NCBI_DISEASE()
```
</details>

Add new corpora
-----
You can easily extend MTL with corpora of your choice by adding two
functions and one reference.

Append two functions to the script - one calling the standard corpus,
one calling the corpus with your label name map for TARS:
```
def get_conll():
    return CONLL_03()

def get_conll_for_tars():
    return CONLL_03(label_name_map={
        'LOC': 'location',
        'ORG': 'organization',
        'PER': 'person',
        'MISC': 'miscellaneous'
    })
```

In the function `get_single_corpus()` you need to return the
desired corpus. You will pass the information with command line
arguments or in .yaml files which you will see in the section
how to run MTL.

```
def get_single_corpus(corpus, corpus_type):
    if corpus == "conll" and corpus_type == "standard":
        return get_conll()
    elif corpus == "conll" and corpus_type == "tars":
        return get_conll_for_tars()
```

How to run a model
------
In the following are 4 different template files to run certain
MTL models and their baselines:

- Single SequenceTagger model (with document context)
- Multitask SequenceTagger model (with document context)
- Single TARS model (with document context)
- Multitask TARS model (with document context)

Simply copy the respective model file in your local repository
and configure the entire model with following YAML file:

<details>
    <summary>YAML file</summary>

```
### General parameters
path: mtl_model
cuda: True
cuda_device: 2

### Corpus parameters
corpus: wnut,conll # If using multiple corpora seperate with comma. Choose from:
  # conll, wnut, ontonotes, conll_es, conll_de, conll_nl, bc2gm, cdr, ncbi
corpus_type: tars
label_type: ner

### Trainer parameters
# learning_rate: 5e-6
batch_size: 4
batch_chunk_size: 4

# Transformers parameters
# transformer_model: xlm-roberta-large
# fine_tune: True
# use_context: True
# subtoken_pooling: first
# layers: -1

# TARS parameters
num_negatives: 1
```
</details>

<b>Notice</b>: make sure that you take the correct script for your configuration.
If you want to do MTL for standard sequence tagger and thus define in your YAML
`corpus=wnut,conll` and `corpus_type=standard`, take the script from Multitask SequenceTagger model section.

Finally run your models like:

`python [local_model.py] --config [config.yaml]`

where `local_model.py` is one of the scripts below and `config.yaml`
the respective YAML file.

Single TARS model
------
Add this file also to your local repository.

<b>Important</b>:
Be sure that you are on the branch `tars_multitask.`

<details>
<summary>local_tars_model.py</summary>

```
import torch
import flair
from flair.models import TARSTagger
from flair.embeddings import TransformerWordEmbeddings
from flair.trainers import ModelTrainer

from local_args import get_arguments
from local_get_corpora import get_corpus


def run_train(args):

    # SET CUDA DEVICE
    if torch.cuda.is_available():
        if not args.cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably run with --cuda"
            )
    flair.device = f"cuda:{args.cuda_device}" if args.cuda else "cpu"

    configuration = get_corpus(args.corpus, corpus_type=args.corpus_type).pop()
    corpus = configuration.get("corpus")
    print(corpus)

    label_dict = configuration.get("label_space")
    print(label_dict)

    embeddings = TransformerWordEmbeddings(model=args.transformer_model,
                                           layers=args.layers,
                                           subtoken_pooling=args.subtoken_pooling,
                                           fine_tune=args.fine_tune,
                                           use_context=args.use_context,
                                           )

    tagger = TARSTagger(embeddings=embeddings,
                        label_dictionary=label_dict,
                        task_name=args.label_type,
                        label_type=args.label_type,
                        num_negative_labels_to_sample=args.num_negatives
                        )

    trainer = ModelTrainer(tagger, corpus)

    trainer.fine_tune(f'resources/{args.path}/run{args.run}',
                      learning_rate=args.learning_rate,
                      mini_batch_size=args.batch_size,
                      mini_batch_chunk_size=args.batch_chunk_size
                      )


if __name__ == "__main__":
    args = get_arguments()
    run_train(args)

```
</details>

Multitask TARS model
------
Add this file also to your local repository.

<b>Important</b>:
Be sure that you are on the branch `tars_multitask.`

<details>
<summary>local_multitask_tars_model.py</summary>

```
import torch
import flair
from flair.data import MultiCorpus
from flair.models import TARSTagger, MultitaskModel
from flair.embeddings import TransformerWordEmbeddings
from flair.trainers import ModelTrainer

from local_args import get_arguments
from local_get_corpora import get_corpus

def run_train(args):

    # SET CUDA DEVICE
    if torch.cuda.is_available():
        if not args.cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably run with --cuda"
            )
    flair.device = f"cuda:{args.cuda_device}" if args.cuda else "cpu"

    model_corpus_configurations = get_corpus(args.corpus, args.corpus_type, label_type=args.label_type)

    # ----- SHARED EMBEDDING LAYERS -----
    embeddings = TransformerWordEmbeddings(model=args.transformer_model,
                                           layers=args.layers,
                                           subtoken_pooling=args.subtoken_pooling,
                                           fine_tune=args.fine_tune,
                                           use_context=args.use_context,
                                           )

    # ----- TASKS -----
    for _idx, configuration in enumerate(model_corpus_configurations):
        if _idx == 0:
            tagger: TARSTagger = TARSTagger(embeddings=embeddings,
                                            label_dictionary=configuration.get("label_space"),
                                            task_name=configuration.get("id"),
                                            label_type=configuration.get("label_type"),
                                            num_negative_labels_to_sample=args.num_negatives)
        else:
            tagger.add_and_switch_to_new_task(label_dictionary=configuration.get("label_space"),
                                              task_name=configuration.get("id"),
                                              label_type=configuration.get("label_type"),
                                              num_negative_labels_to_sample=args.num_negatives)


    # ----- MULTITASK CORPUS -----
    multicorpus = MultiCorpus(
        corpora=[config.get("corpus") for config in model_corpus_configurations],
        task_ids=[config.get("id") for config in model_corpus_configurations]
    )

    # ----- MULTITASK MODEL -----
    multitask_model: MultitaskModel = MultitaskModel(
        models=[tagger] * len(tagger.list_existing_tasks()),
        task_ids=[config.get("id") for config in model_corpus_configurations]
    )

    # ----- TRAINING ON MODEL AND CORPUS -----
    trainer: ModelTrainer = ModelTrainer(multitask_model, multicorpus)
    trainer.fine_tune(f'resources/{args.path}/run{args.run}',
                      learning_rate=args.learning_rate,
                      mini_batch_size=args.batch_size,
                      mini_batch_chunk_size=args.batch_chunk_size)


if __name__ == "__main__":
    args = get_arguments()
    run_train(args)

```
</details>


