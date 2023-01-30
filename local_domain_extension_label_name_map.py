from flair.data import Corpus
from flair.datasets import CONLL_03, ColumnCorpus, WNUT_17


def get_corpus(name: str, map: str = "short", path: str = "/") -> Corpus:
    if name == "conll03":

        if map == "short":
            return CONLL_03(
                base_path=f"{path}/datasets",
                in_memory=True,
                label_name_map={
                    "LOC": "location",
                    "ORG": "organization",
                    "PER": "person",
                    "MISC": "miscellaneous",
                },
            )
        if map == "long":
            return CONLL_03(
                base_path=f"{path}/datasets",
                in_memory=True,
                label_name_map={
                    "LOC": "location name",
                    "ORG": "organization name",
                    "PER": "person name",
                    "MISC": "other name (not person name, not organization name, not location name)",
                },
            )

    if name == "ontonotes":

        if map == "short":
            return ColumnCorpus(
                f"{path}/datasets/onto-ner",
                column_format={0: "text", 1: "pos", 2: "upos", 3: "ner"},
                label_name_map={
                    "CARDINAL": "cardinal",
                    "DATE": "date",
                    "EVENT": "event",
                    "FAC": "facility",
                    "GPE": "geo-political entity",
                    "LANGUAGE": "language",
                    "LAW": "law",
                    "LOC": "location",
                    "MONEY": "money",
                    "NORP": "affiliation",
                    "ORDINAL": "ordinal",
                    "ORG": "organization",
                    "PERCENT": "percent",
                    "PERSON": "person",
                    "PRODUCT": "product",
                    "QUANTITY": "quantity",
                    "TIME": "time",
                    "WORK_OF_ART": "work of Art",
                },
            )

        if map == "long":
            return ColumnCorpus(
                f"{path}/datasets/onto-ner",
                column_format={0: "text", 1: "pos", 2: "upos", 3: "ner"},
                label_name_map={
                    "CARDINAL": "cardinal value",
                    "DATE": "reference to a date or period",
                    "EVENT": "event name",
                    "FAC": "name of man-made structure or facility",
                    "GPE": "name of country, city, state, province or municipality",
                    "LANGUAGE": "language",
                    "LAW": "named treaty or chapter of named legal document",
                    "LOC": "name of geographical location",
                    "MONEY": "monetary value",
                    "NORP": "adjectival form of named religion, heritage, geographical or political affiliation",
                    "ORDINAL": "ordinal number or adverbial",
                    "ORG": "organization name",
                    "PERCENT": "percent value",
                    "PERSON": "person name",
                    "PRODUCT": "product name",
                    "QUANTITY": "quantity value",
                    "TIME": "time reference",
                    "WORK_OF_ART": "title of book, song, movie or award",
                },
            )

    if name == "wnut17":
        if map == "short":
            return WNUT_17(
                label_name_map={
                    "location": "Location",
                    "corporation": "Corporation",
                    "person": "Person",
                    "creative-work": "Creative Work",
                    "product": "Product",
                    "group": "Group",
                }
            )

        if map == "long":
            return WNUT_17(
                label_name_map={
                    "location": "location name",
                    "corporation": "corporation name",
                    "person": "person name",
                    "creative-work": "name of song, movie, book or other creative work",
                    "product": "name of product or consumer good",
                    "group": "name of music band, sports team or non-corporate organization",
                }
            )


def get_label_name_map(corpus: str):
    if corpus == "conll03":
        label_name_map = {
                    "LOC": "location",
                    "ORG": "organization",
                    "PER": "person",
                    "MISC": "miscellaneous",
                }
    elif corpus == "wnut17":
        label_name_map = {
                "location": "Location",
                "corporation": "Corporation",
                "person": "Person",
                "creative-work": "Creative Work",
                "product": "Product",
                "group": "Group",
            }
    else:
        raise Exception("unknown corpus")
    return label_name_map
