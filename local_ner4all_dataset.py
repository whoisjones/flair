import glob
import json
import logging

from pathlib import Path
from typing import Optional, Union

from flair.datasets.sequence_labeling import MultiFileColumnCorpus

log = logging.getLogger("flair")

class NER4ALL(MultiFileColumnCorpus):
    def __init__(
        self,
        base_path: Union[str, Path],
        in_memory: bool = False,
        **corpusargs,
    ) -> None:
        if type(base_path) is str:
            base_path = Path(base_path)

        dataset_path = Path(base_path) / "conll"

        train_files = glob.glob(str(dataset_path / "*.conll"))

        with open(base_path / "labelID2label.json", "r") as f:
            label_mapping = json.load(f)

        super().__init__(
            train_files=train_files,
            column_format={0: "text", 1: "ner"},
            in_memory=in_memory,
            sample_missing_splits=False,
            label_name_map=label_mapping,
            **corpusargs,
        )
