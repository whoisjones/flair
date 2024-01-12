import numpy as np
import numpy.random
import torch
import torch.cuda
from transformers import AutoModel, PreTrainedTokenizer, BertConfig, AutoModelForTokenClassification


class BiEncoder(torch.nn.Module):
    def __init__(
            self,
            encoder_model: str,
            decoder_model: str,
            tokenizer: PreTrainedTokenizer,
            labels: dict,
            zelda_label_sampling: str = None,
            uniform_p: list = [0.5, 0.5],
            geometric_p: float = 0.5,
            zelda_mask_size: int = 0,
    ):
        super(BiEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.decoder = AutoModel.from_pretrained(decoder_model)
        labels = {int(k): v for k, v in labels.items()}
        self.labels = labels
        self.num_labels = len(labels)
        self.tokenizer = tokenizer
        self.zelda_label_sampling = zelda_label_sampling
        self.zelda_mask_size = zelda_mask_size
        self.uniform_p = uniform_p
        self.geometric_p = geometric_p
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels):
        token_hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if self.training and self.zelda_label_sampling:
            encoded_labels, labels = self._prepare_labels(labels)
            label_hidden_states = self.decoder(**encoded_labels)
        else:
            encoded_labels = self.tokenizer(
                list(self.labels.values()),
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt").to(labels.device)
            label_hidden_states = self.decoder(**encoded_labels)

        if "sentence-transformers" in self.decoder.name_or_path:
            label_embeddings = self.mean_pooling(label_hidden_states, encoded_labels['attention_mask'])
            label_embeddings = torch.nn.functional.normalize(label_embeddings, p=2, dim=1)
        else:
            label_embeddings = label_hidden_states.last_hidden_state[:, 0, :]

        logits = torch.matmul(token_hidden_states.last_hidden_state, label_embeddings.T)

        return (self.loss(logits.transpose(1, 2), labels), logits)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _prepare_labels(self, labels):
        positive_labels = torch.unique(labels)
        positive_labels = positive_labels[(positive_labels != -100)]
        number_negatives_needed = self.zelda_mask_size - positive_labels.size(0)

        if number_negatives_needed > 0:
            negative_labels = numpy.random.choice(np.arange(0, len(self.labels)), size=number_negatives_needed, replace=False)
            labels_for_batch = np.unique(np.concatenate([positive_labels.detach().cpu().numpy(), negative_labels]))
        else:
            labels_for_batch = positive_labels.detach().cpu().numpy()
        labels = self.adjust_batched_labels(labels_for_batch, labels)

        if self.zelda_label_sampling == "full_desc":
            label_descriptions = self._full_labels(labels_for_batch)
        elif self.zelda_label_sampling == "only_desc":
            label_descriptions = self._only_desc(labels_for_batch)
        elif self.zelda_label_sampling == "only_labels":
            label_descriptions = self._only_labels(labels_for_batch)
        elif self.zelda_label_sampling == "sampled_desc":
            label_descriptions = self._sample_labels(labels_for_batch)
        else:
            raise ValueError(f"Unknown label sampling {self.label_sampling}")

        encoded_labels = self.tokenizer(
            label_descriptions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(labels.device)
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

    def _full_labels(self, selected_labels):
        label_descriptions = []
        for i in selected_labels:
            if i == 0:
                label_descriptions.append("outside")
            else:
                label_description = []
                if self.labels.get(i).get("description") is not None:
                    label_description.append(self.labels.get(i).get("description"))
                if self.labels.get(i).get("labels") is not None:
                    for _l in self.labels.get(i).get("labels"):
                        label_description.append(_l)
                label_descriptions.append(", ".join(label_description))
        return label_descriptions

    def _only_labels(self, selected_labels):
        label_descriptions = []
        for i in selected_labels:
            fallback = []
            if i == 0:
                label_descriptions.append("outside")
            else:
                if self.labels.get(i).get("description") is not None:
                    fallback.append("description")
                label_description = []
                if self.labels.get(i).get("labels") is not None:
                    num_labels = np.random.geometric(self.geometric_p, 1)
                    num_labels = num_labels if num_labels <= len(self.labels.get(i).get("labels")) else len(self.labels.get(i).get("labels"))
                    sampled_labels = np.random.choice(self.labels.get(i).get("labels"), num_labels, replace=False).tolist()
                    label_descriptions.append(', '.join(sampled_labels))
                elif fallback:
                    label_descriptions.append(self.labels.get(i).get("description"))
                else:
                    label_description.append("miscellaneous")
                label_descriptions.append(", ".join(label_description))
        return label_descriptions

    def _only_desc(self, selected_labels):
        label_descriptions = []
        for i in selected_labels:
            fallback = []
            if i == 0:
                label_descriptions.append("outside")
            else:
                if self.labels.get(i).get("labels") is not None:
                    fallback.append("labels")
                if self.labels.get(i).get("description") is not None:
                    label_descriptions.append(self.labels.get(i).get("description"))
                elif fallback:
                    num_labels = np.random.geometric(self.geometric_p, 1)
                    num_labels = num_labels if num_labels <= len(self.labels.get(i).get("labels")) else len(self.labels.get(i).get("labels"))
                    sampled_labels = np.random.choice(self.labels.get(i).get("labels"), num_labels, replace=False).tolist()
                    label_descriptions.append(', '.join(sampled_labels))
                else:
                    label_descriptions.append("miscellaneous")
        return label_descriptions

    def _sample_labels(self, selected_labels):
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
                    label_description = f"{self.labels.get(i)[label_granularity]}"
                elif label_granularity == "labels":
                    num_labels = np.random.geometric(self.geometric_p, 1)
                    num_labels = num_labels if num_labels <= len(self.labels.get(i).get("labels")) else len(self.labels.get(i).get("labels"))
                    sampled_labels = np.random.choice(self.labels.get(i).get("labels"), num_labels, replace=False).tolist()
                    label_description = f"{', '.join(sampled_labels)}"
                else:
                    raise ValueError(f"Unknown label granularity {label_granularity}")
            label_descriptions.append(label_description)
        return label_descriptions


class LEAR(BiEncoder):

    def __init__(
            self,
            encoder_model: str,
            decoder_model: str,
            tokenizer: PreTrainedTokenizer,
            labels: dict,
            zelda_label_sampling: str = None,
            zelda_mask_size: int = 0,
    ):
        super(LEAR, self).__init__(
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            tokenizer=tokenizer,
            labels=labels,
            zelda_label_sampling=zelda_label_sampling,
            zelda_mask_size=zelda_mask_size,
        )
        self.token_fc = torch.nn.Linear(768, 768, bias=False)
        self.label_fc = torch.nn.Linear(768, 768, bias=False)
        self.fusion_fc = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, labels):
        token_hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = token_hidden_states.last_hidden_state
        token_embeddings = self.dropout(token_embeddings)

        if self.training and self.zelda_label_sampling:
            encoded_labels, labels = self._prepare_labels(labels)
            label_hidden_states = self.decoder(**encoded_labels)
        else:
            encoded_labels = self.tokenizer(
                list(self.labels.values()),
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt").to(labels.device)
            label_hidden_states = self.encoder(**encoded_labels)

        label_embeddings = label_hidden_states.last_hidden_state

        batch_size, token_seq_len, token_hidden_size = token_embeddings.shape
        num_labels, label_seq_len, label_hidden_size = label_embeddings.shape

        token_features_fc = self.token_fc(token_embeddings)
        label_features_fc = self.label_fc(label_embeddings)

        label_features_t = label_features_fc.permute(2, 0, 1).view(label_hidden_size, -1)

        # [bs, context_seq_len, class_num, label_seq_len]
        scores = torch.matmul(token_features_fc, label_features_t).view(batch_size, token_seq_len, num_labels, -1)

        extended_mask = encoded_labels["attention_mask"][None, None, :, :]
        extended_mask = (1.0 - extended_mask) * -10000.0
        scores = scores + extended_mask

        # [bs, context_seq_len, class_num, label_seq_len]
        scores = torch.softmax(scores, dim=-1)

        weighted_label_feature = label_features_fc.unsqueeze(0).unsqueeze(0).repeat(batch_size, token_seq_len, 1, 1, 1) * scores.unsqueeze(-1)
        token_feature_fc = token_features_fc.unsqueeze(2).repeat(1, 1, num_labels, 1)
        weighted_label_feature_sum = torch.sum(weighted_label_feature, dim=-2)

        fused_feature = token_feature_fc + weighted_label_feature_sum
        fused_feature = torch.tanh(self.fusion_fc(fused_feature))
        logits = torch.sum(fused_feature, -1)

        return self.loss(logits.transpose(1, 2), labels), logits
