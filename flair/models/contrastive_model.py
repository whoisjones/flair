import itertools
import logging
import typing
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

import flair
from flair import file_utils
from flair.data import DT, Dictionary
from flair.embeddings import Embeddings
from flair.training_utils import Result

import torch.nn as nn

log = logging.getLogger("flair")


class CosineSimilarityLoss(nn.Module):
    """
    CosineSimilarityLoss expects, that the InputExamples consists of two texts and a float label.
    It computes the vectors u = model(input_text[0]) and v = model(input_text[1]) and measures the cosine-similarity between the two.
    By default, it minimizes the following loss: ||input_label - cos_score_transformation(cosine_sim(u,v))||_2.
    :param model: SentenceTranformer model
    :param loss_fct: Which pytorch loss function should be used to compare the cosine_similartiy(u,v) with the input_label? By default, MSE:  ||input_label - cosine_sim(u,v)||_2
    :param cos_score_transformation: The cos_score_transformation function is applied on top of cosine_similarity. By default, the identify function is used (i.e. no change).
    Example::
            from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
                InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
            train_dataset = SentencesDataset(train_examples, model)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
            train_loss = losses.CosineSimilarityLoss(model=model)
    """
    def __init__(self, loss_fct=nn.MSELoss(), cos_score_transformation=nn.Identity()):
        super(CosineSimilarityLoss, self).__init__()
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation


    def forward(self, embeddings, labels):
        output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
        return self.loss_fct(output, labels.view(-1))


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the
    two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.
    Further information: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    :param model: SentenceTransformer model
    :param distance_metric: Function that returns a distance between two embeddings. The class SiameseDistanceMetric contains pre-defined metrices that can be used
    :param margin: Negative samples (label == 0) should have a distance of at least the margin value.
    :param size_average: Average by the size of the mini-batch.
    Example::
        from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
        from torch.utils.data import DataLoader
        model = SentenceTransformer('all-MiniLM-L6-v2')
        train_examples = [
            InputExample(texts=['This is a positive pair', 'Where the distance will be minimized'], label=1),
            InputExample(texts=['This is a negative pair', 'Their distance will be increased'], label=0)]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
        train_loss = losses.ContrastiveLoss(model=model)
        model.fit([(train_dataloader, train_loss)], show_progress_bar=True)
    """

    def __init__(self, margin: float = 0.5, size_average: bool = True):
        super(ContrastiveLoss, self).__init__()
        self.distance_metric = lambda x, y: 1-F.cosine_similarity(x, y)
        self.margin = margin
        self.size_average = size_average

    def forward(self, embeddings, labels):
        rep_anchor, rep_other = embeddings
        distances = self.distance_metric(rep_anchor, rep_other)
        losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
        return losses.mean() if self.size_average else losses.sum()


class Model(torch.nn.Module, typing.Generic[DT]):
    """Abstract base class for all downstream task models in Flair,
    such as SequenceTagger and TextClassifier.
    Every new type of model must implement these methods."""

    def __init__(
        self,
        embeddings: flair.embeddings.TokenEmbeddings,
    ):
        self.embeddings = embeddings
        self.loss = CosineSimilarityLoss()

    @property
    def label_type(self):
        """Each model predicts labels of a certain type."""
        pass

    def forward_loss(self, data_points: List[DT]) -> Tuple[torch.Tensor, int]:
        """Performs a forward pass and returns a loss tensor for backpropagation.
        Implement this to enable training."""
        hidden_states = self.embeddings.embed(data_points)
        return self.loss(hidden_states)

    def evaluate(
        self,
        data_points: Union[List[DT], Dataset],
        gold_label_type: str,
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: Optional[int] = 8,
        main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
        exclude_labels: List[str] = [],
        gold_label_dictionary: Optional[Dictionary] = None,
        return_loss: bool = True,
        **kwargs,
    ) -> Result:
        """Evaluates the model. Returns a Result object containing evaluation
        results and a loss value. Implement this to enable evaluation.
        :param data_loader: DataLoader that iterates over dataset to be evaluated
        :param out_path: Optional output path to store predictions
        :param embedding_storage_mode: One of 'none', 'cpu' or 'gpu'. 'none' means all embeddings are deleted and freshly recomputed, 'cpu' means all embeddings are stored on CPU, or 'gpu' means all embeddings are stored on GPU  # noqa: E501
        :return: Returns a Tuple consisting of a Result object and a loss float value
        """
        raise NotImplementedError

    def _get_state_dict(self):
        """Returns the state dictionary for this model."""
        state_dict = {"state_dict": self.state_dict()}

        return state_dict

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        """Initialize the model from a state dictionary."""
        model = cls(**kwargs)

        model.load_state_dict(state["state_dict"])
        return model

    @staticmethod
    def _fetch_model(model_name) -> str:
        return model_name

    def save(self, model_file: Union[str, Path], checkpoint: bool = False):
        """
        Saves the current model to the provided file.
        :param model_file: the model file
        """
        model_state = self._get_state_dict()

        # in Flair <0.9.1, optimizer and scheduler used to train model are not saved
        optimizer = scheduler = None

        # write out a "model card" if one is set
        if self.model_card is not None:

            # special handling for optimizer:
            # remember optimizer class and state dictionary
            if "training_parameters" in self.model_card:
                training_parameters = self.model_card["training_parameters"]

                if "optimizer" in training_parameters:
                    optimizer = training_parameters["optimizer"]
                    if checkpoint:
                        training_parameters["optimizer_state_dict"] = optimizer.state_dict()
                    training_parameters["optimizer"] = optimizer.__class__

                if "scheduler" in training_parameters:
                    scheduler = training_parameters["scheduler"]
                    if checkpoint:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            training_parameters["scheduler_state_dict"] = scheduler.state_dict()
                    training_parameters["scheduler"] = scheduler.__class__

            model_state["model_card"] = self.model_card

        # save model
        torch.save(model_state, str(model_file), pickle_protocol=4)

        # restore optimizer and scheduler to model card if set
        if self.model_card is not None:
            if optimizer:
                self.model_card["training_parameters"]["optimizer"] = optimizer
            if scheduler:
                self.model_card["training_parameters"]["scheduler"] = scheduler

    @classmethod
    def load(cls, model_path: Union[str, Path]):
        """
        Loads the model from the given file.
        :param model_path: the model file
        :return: the loaded text classifier model
        """
        model_file = cls._fetch_model(str(model_path))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # load_big_file is a workaround byhttps://github.com/highway11git
            # to load models on some Mac/Windows setups
            # see https://github.com/zalandoresearch/flair/issues/351
            f = file_utils.load_big_file(str(model_file))
            state = torch.load(f, map_location="cpu")

        model = cls._init_model_with_state_dict(state)

        if "model_card" in state:
            model.model_card = state["model_card"]

        model.eval()
        model.to(flair.device)

        return model

    def print_model_card(self):
        if hasattr(self, "model_card"):
            param_out = "\n------------------------------------\n"
            param_out += "--------- Flair Model Card ---------\n"
            param_out += "------------------------------------\n"
            param_out += "- this Flair model was trained with:\n"
            param_out += f"-- Flair version {self.model_card['flair_version']}\n"
            param_out += f"-- PyTorch version {self.model_card['pytorch_version']}\n"
            if "transformers_version" in self.model_card:
                param_out += "-- Transformers version " f"{self.model_card['transformers_version']}\n"
            param_out += "------------------------------------\n"

            param_out += "------- Training Parameters: -------\n"
            param_out += "------------------------------------\n"
            training_params = "\n".join(
                f'-- {param} = {self.model_card["training_parameters"][param]}'
                for param in self.model_card["training_parameters"]
            )
            param_out += training_params + "\n"
            param_out += "------------------------------------\n"

            log.info(param_out)
        else:
            log.info(
                "This model has no model card (likely because it is not yet "
                "trained or was trained with Flair version < 0.9.1)"
            )
