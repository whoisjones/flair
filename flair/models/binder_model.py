from typing import List, Tuple
import logging

import numpy as np
import torch.nn.functional as F

import flair
from flair.data import Sentence, Dictionary, DT, Span
from flair.embeddings import TokenEmbeddings, DocumentEmbeddings

log = logging.getLogger("flair")


class BINDER(flair.nn.Classifier[Sentence]):
    """This model implements the BINDER architecture for token classification using contrastive learning and a bi-encoder.
    Paper: https://openreview.net/forum?id=9EAQVEINuum
    """

    def __init__(
        self,
        token_encoder: TokenEmbeddings,
        label_encoder: DocumentEmbeddings,
        label_dictionary: Dictionary,
        label_type: str,
        dropout: float = 0.1,
        linear_size: int = 128,
        use_span_width_embeddings: bool = True,
        max_span_width: int = 8,
        init_temperature: float = 0.07,
        start_loss_weight: float = 0.2,
        end_loss_weight: float = 0.2,
        span_loss_weight: float = 0.6,
        threshold_loss_weight: float = 0.5,
        ner_loss_weight: float = 0.5,
    ):
        super().__init__()
        self.token_encoder = token_encoder
        self.label_decoder = label_encoder
        self._make_span_label_dictionary(label_dictionary)
        self._label_type: str = label_type

        self.dropout = torch.nn.Dropout(dropout)
        self.label_start_linear = torch.nn.Linear(self.label_encoder.embedding_length, linear_size)
        self.label_end_linear = torch.nn.Linear(self.label_encoder.embedding_length, linear_size)
        self.label_span_linear = torch.nn.Linear(self.label_decoder.embedding_length, linear_size)
        self.token_start_linear = torch.nn.Linear(self.token_encoder.embedding_length, linear_size)
        self.token_end_linear = torch.nn.Linear(self.token_encoder.embedding_length, linear_size)

        if use_span_width_embeddings:
            self.span_width_embeddings = torch.nn.Embedding(
                self.token_encoder.embedding_length * 2 + linear_size, linear_size
            )
            self.width_embeddings = torch.nn.Embedding(max_span_width, linear_size, padding_idx=0)
        else:
            self.span_linear = torch.nn.Linear(self.token_encoder.embedding_length * 2, linear_size)
            self.width_embeddings = None

        self.start_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / init_temperature))
        self.end_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / init_temperature))
        self.span_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / init_temperature))

        self.start_loss_weight = start_loss_weight
        self.end_loss_weight = end_loss_weight
        self.span_loss_weight = span_loss_weight
        self.threshold_loss_weight = threshold_loss_weight
        self.ner_loss_weight = ner_loss_weight

        self.to(flair.device)

    def forward_loss(self, data_points: List[DT]) -> Tuple[torch.Tensor, int]:
        """Forwards the BINDER model and returns the combined loss."""
        # Quality checks + get gold labels
        if len(data_points) == 0:
            return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=True), 0
        sentences = [data_points] if not isinstance(data_points, list) else data_points
        gold_labels = self._get_gold_labels(sentences)

        # Get hidden states for tokens and labels
        self.token_encoder.embed(sentences)
        lengths, token_hidden = self._get_token_hidden_states(sentences)

        labels = [Sentence(label) for label in self.label_dictionary.get_items()]
        self.label_encoder.embed(labels)
        label_hidden = torch.cat([label.get_embedding() for label in labels])

        # get shapes
        batch_size, seq_length, _ = token_hidden.size()
        num_types, _ = label_hidden.size()

        # num_types x hidden_size
        label_start_output = F.normalize(self.dropout(self.type_start_linear(label_hidden)), dim=-1)
        label_end_output = F.normalize(self.dropout(self.type_end_linear(label_hidden)), dim=-1)
        # batch_size x seq_length x hidden_size
        token_start_output = F.normalize(self.dropout(self.start_linear(token_hidden)), dim=-1)
        token_end_output = F.normalize(self.dropout(self.end_linear(token_hidden)), dim=-1)

        # batch_size x num_types x seq_length
        start_scores = self.start_logit_scale.exp() * label_start_output.unsqueeze(0) @ token_start_output.transpose(1, 2)
        end_scores = self.end_logit_scale.exp() * label_end_output.unsqueeze(0) @ token_end_output.transpose(1, 2)

        # batch_size x seq_length x seq_length x hidden_size*2
        span_output = torch.cat(
            [
                token_hidden.unsqueeze(2).expand(-1, -1, seq_length, -1),
                token_hidden.unsqueeze(1).expand(-1, seq_length, -1, -1),
            ],
            dim=3
        )

        # span_width_embeddings
        if self.width_embeddings is not None:
            range_vector = torch.cuda.LongTensor(seq_length, device=token_hidden.device).fill_(1).cumsum(0) - 1
            span_width = range_vector.unsqueeze(0) - range_vector.unsqueeze(1) + 1
            # seq_length x seq_length x hidden_size
            span_width_embeddings = self.width_embeddings(span_width * (span_width > 0))
            span_output = torch.cat([
                span_output, span_width_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)], dim=3)

        # batch_size x seq_length x seq_length x hidden_size
        span_linear_output = F.normalize(
            self.dropout(self.span_linear(span_output)).view(batch_size, seq_length * seq_length, -1), dim=-1
        )
        # num_types x hidden_size
        label_linear_out = F.normalize(self.dropout(self.type_span_linear(label_hidden)), dim=-1)

        # batch_size x num_types x seq_length x seq_length
        span_scores = self.span_logit_scale.exp() * label_linear_out.unsqueeze(0) @ span_linear_output.transpose(1, 2)
        span_scores = span_scores.view(batch_size, num_types, seq_length, seq_length)

        flat_start_scores = start_scores.view(batch_size * num_types, seq_length)
        flat_end_scores = end_scores.view(batch_size * num_types, seq_length)
        flat_span_scores = span_scores.view(batch_size * num_types, seq_length, seq_length)



        #TODO: convert predictions into spans at right point



        return

    def _make_span_label_dictionary(self, label_dictionary: Dictionary, allow_unk_predictions: bool = False):
        # span-labels need special encoding (BIO or BIOES)
        if label_dictionary.span_labels:
            # the big question is whether the label dictionary should contain an UNK or not
            # without UNK, we cannot evaluate on data that contains labels not seen in test
            # with UNK, the model learns less well if there are no UNK examples
            self.span_dictionary = Dictionary(add_unk=allow_unk_predictions)
            assert self.tag_format in ["BIOES", "BIO"]
            for label in label_dictionary.get_items():
                if label == "<unk>":
                    continue
                self.span_dictionary.add_item("O")
                if self.tag_format == "BIOES":
                    self.span_dictionary.add_item("S-" + label)
                    self.span_dictionary.add_item("B-" + label)
                    self.span_dictionary.add_item("E-" + label)
                    self.span_dictionary.add_item("I-" + label)
                if self.tag_format == "BIO":
                    self.span_dictionary.add_item("B-" + label)
                    self.span_dictionary.add_item("I-" + label)
        else:
            self.span_dictionary = label_dictionary

        self.label_dictionary = label_dictionary

        # is this a span prediction problem?
        if any(item.startswith(("B-", "S-", "I-")) for item in self.span_dictionary.get_items()):
            self.predict_spans = True
        else:
            self.predict_spans = False

        self.tagset_size = len(self.span_dictionary)
        log.info(f"BINDER model predicts: {self.span_dictionary}")

    def _get_token_hidden_states(self, sentences: List[Sentence]) -> Tuple[torch.LongTensor, torch.Tensor]:
        """Returns the token hidden states for the given sentences."""
        names = self.token_encoder.get_names()
        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)
        pre_allocated_zero_tensor = torch.zeros(
            self.embeddings.embedding_length * longest_token_sequence_in_batch,
            dtype=self.linear.weight.dtype,
            device=flair.device,
        )
        all_embs = []
        for sentence in sentences:
            all_embs += [emb for token in sentence for emb in token.get_each_embedding(names)]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[: self.embeddings.embedding_length * nb_padding_tokens]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embeddings.embedding_length,
            ]
        )
        return torch.LongTensor(lengths), sentence_tensor

    def _get_gold_labels(self, sentences: List[Sentence]) -> torch.Tensor:
        if self.label_dictionary.span_labels:
            all_sentence_labels = []
            for sentence in sentences:
                sentence_labels = ["O"] * len(sentence)
                for label in sentence.get_labels(self.label_type):
                    span: Span = label.data_point
                    if self.tag_format == "BIOES":
                        if len(span) == 1:
                            sentence_labels[span[0].idx - 1] = "S-" + label.value
                        else:
                            sentence_labels[span[0].idx - 1] = "B-" + label.value
                            sentence_labels[span[-1].idx - 1] = "E-" + label.value
                            for i in range(span[0].idx, span[-1].idx - 1):
                                sentence_labels[i] = "I-" + label.value
                    else:
                        sentence_labels[span[0].idx - 1] = "B-" + label.value
                        for i in range(span[0].idx, span[-1].idx):
                            sentence_labels[i] = "I-" + label.value
                all_sentence_labels.extend(sentence_labels)
            gold_labels = all_sentence_labels

        # all others are regular labels for each token
        else:
            gold_labels = [token.get_label(self.label_type, "O").value for sentence in sentences for token in sentence]

        labels = torch.tensor(
            [self.label_dictionary.get_idx_for_item(label) for label in gold_labels],
            dtype=torch.long,
            device=flair.device,
        )

        return labels
