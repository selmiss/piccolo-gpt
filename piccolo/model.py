from enum import Enum
from pathlib import Path
from typing import ClassVar, Type
import numpy as np
import torch
import os

from transformers import AutoConfig, AutoModel, PreTrainedModel, AutoModelForCausalLM

from piccolo.criteria import (
    CoSentLoss,
    ClsContrastLoss,
    PairInBatchNegSoftmaxContrastLoss,
    PairInBatchHardNegSoftmaxContrastLoss,
)


class PoolingStrategy(str, Enum):
    cls = "cls"
    last_mean = "last_mean"
    last_mean_dropout = "last_mean_dropout"


class InBatchNegLossType(str, Enum):
    cosent = "cosent"
    softmax = "softmax"
    hardneg_softmax = "hardneg_softmax"
    cls_contrast = "cls_contrast"


def build_loss(loss_type, temperature, **kwargs):
    loss_type = InBatchNegLossType(loss_type)
    match loss_type:
        case InBatchNegLossType.cosent:
            return CoSentLoss(temperature)
        case InBatchNegLossType.softmax:
            return PairInBatchNegSoftmaxContrastLoss(temperature)
        case InBatchNegLossType.hardneg_softmax:
            return PairInBatchHardNegSoftmaxContrastLoss(temperature, **kwargs)
        case InBatchNegLossType.cls_contrast:
            return ClsContrastLoss(temperature)


def creat_attention_mask_from_input_ids(
    input_ids: torch.Tensor, pad_token_id: int
) -> torch.Tensor:
    return input_ids != pad_token_id


def mean_pooling(
    hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    if attention_mask is None:
        return torch.mean(hidden_state, dim=1)
    attention_mask = attention_mask.float()
    return torch.sum(hidden_state * attention_mask.unsqueeze(-1), dim=1) / torch.sum(
        attention_mask, dim=-1, keepdim=True
    )


def last_pooling(
    hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    last_hidden = hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        emb = last_hidden[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden.shape[0]
        emb = last_hidden[torch.arange(batch_size), sequence_lengths]

    return emb


def load_hf_pretrained_model(model_name_or_path: str) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if config.model_type == "t5":
        from transformers import T5EncoderModel

        pretrained_model = T5EncoderModel.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
    else:
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    return pretrained_model


StrategyEmbedderClsMap: dict[PoolingStrategy, Type["Embedder"]] = {}


class Embedder(torch.nn.Module):
    pooling_strategy: ClassVar[PoolingStrategy]

    def __init__(self, encoder: PreTrainedModel, pad_token_id: int | None = None):
        super().__init__()
        self.encoder = encoder
        self.encoder.config.piccolo_pooling_strategy = str(self.pooling_strategy.value)

        if pad_token_id is None:
            if encoder.config.pad_token_id is not None:
                self.pad_token_id = encoder.config.pad_token_id
            else:
                self.pad_token_id = 0
        else:
            self.pad_token_id = pad_token_id

    def __init_subclass__(cls) -> None:
        StrategyEmbedderClsMap[cls.pooling_strategy] = cls

    def forward(
        self, input_ids: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def save_pretrained(self, path: str | Path):
        self.encoder.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        encoder = load_hf_pretrained_model(model_name_or_path)
        return cls(encoder)

    @property
    def max_length(self):
        return self.encoder.config.max_position_embeddings


class LastMeanEmbedder(Embedder):
    pooling_strategy: ClassVar[PoolingStrategy] = PoolingStrategy.last_mean

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = creat_attention_mask_from_input_ids(
                input_ids, self.pad_token_id
            )
        embeddings = self.encoder(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        last_hidden_state = embeddings.hidden_states[-1]
        embeddings = last_pooling(last_hidden_state, attention_mask)
        return embeddings


class EmbedderForTrain(torch.nn.Module):
    embedder: Embedder

    def __init__(self, embedder: Embedder):
        super().__init__()
        self.embedder = embedder

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        self.embedder.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs
        )


class GPTEmbedder(EmbedderForTrain):
    def __init__(
        self,
        model_name_or_path: str,
        loss_kwargs: dict,
        embedding_strategy: PoolingStrategy | str = PoolingStrategy.last_mean,
        freeze_pos_emb: bool = False,
        add_scaling_layer: bool = False,
        use_mrl: bool = False,
        add_cls_head: bool = False,
    ):
        pretrained_model = load_hf_pretrained_model(model_name_or_path)
        embedder = StrategyEmbedderClsMap[PoolingStrategy(embedding_strategy)](
            pretrained_model
        )
        super().__init__(embedder)
        self.criterion = build_loss(**loss_kwargs)
        self.cosent_loss = build_loss("cosent", temperature=0.05)
        self.cls_contrast_loss = build_loss("cls_contrast", temperature=0.05)
        self.use_mrl = use_mrl
        self.add_scaling_layer = add_scaling_layer

        if add_scaling_layer:
            scaling_layer_state_dict = torch.load(
                os.path.join(model_name_or_path, "2_Dense/pytorch_model.bin")
            )
            self.scaling_layer = ScalingLayer(
                origin_dim=1024, scaling_dim=1792
            )  # hard code here
            self.scaling_layer.load_state_dict(scaling_layer_state_dict, strict=True)

        if use_mrl:
            self.mrl_nesting_list = [
                256,
                512,
                768,
                1024,
                1280,
                1536,
                1792,
            ]  # hard code here

        if freeze_pos_emb:
            for name, param in self.embedder.encoder.embeddings.named_parameters():
                if "position_embeddings" in name:
                    param.requires_grad = False

        if add_cls_head:
            self.cls_head = torch.nn.Linear(1024, 2)  # hard code here

    def get_embedding(self, text_ids):
        if text_ids is None:
            return None
        text_embeddings = self.embedder(text_ids)
        if self.add_scaling_layer:
            text_embeddings = self.scaling_layer(text_embeddings.half()).float()
        return text_embeddings

    def compute_cls_loss(self, text_ids: torch.Tensor, text_labels: torch.tensor):
        text_embeddings = self.get_embedding(text_ids)
        pred_cls = self.cls_head(text_embeddings.half())
        loss = torch.nn.functional.cross_entropy(pred_cls, text_labels)
        return {"loss": loss}

    def compute_cls_contrast_loss(
        self,
        text_ids: torch.Tensor,
        text_pos_ids: torch.Tensor,
        text_neg_ids: torch.Tensor = None,
        type: str = "cls_contrast",
    ) -> dict[str, torch.Tensor]:
        text_embeddings = self.get_embedding(text_ids)
        text_pos_embeddings = self.get_embedding(text_pos_ids)
        text_neg_embeddings = self.get_embedding(text_neg_ids)

        if self.use_mrl:
            loss = torch.tensor(0.0, device=text_embeddings.device)
            for num_feat in self.mrl_nesting_list:
                emb, pos_emb, neg_emb = (
                    text_embeddings[..., :num_feat],
                    text_pos_embeddings[..., :num_feat],
                    text_neg_embeddings[..., :num_feat],
                )
                loss += self.cls_contrast_loss(emb, pos_emb, neg_emb) / len(
                    self.mrl_nesting_list
                )
        else:
            loss = self.cls_contrast_loss(
                text_embeddings, text_pos_embeddings, text_neg_embeddings
            )
        print("cls contrast loss: ", loss)
        return {"loss": loss}

    def compute_triplet_loss(
        self,
        text_ids: torch.Tensor,
        text_pos_ids: torch.Tensor,
        text_neg_ids: torch.Tensor = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        text_embeddings = self.get_embedding(text_ids)
        text_pos_embeddings = self.get_embedding(text_pos_ids)
        text_neg_embeddings = self.get_embedding(text_neg_ids)

        if self.use_mrl:
            loss = torch.tensor(0.0, device=text_embeddings.device)
            for num_feat in self.mrl_nesting_list:
                emb, pos_emb, neg_emb = (
                    text_embeddings[..., :num_feat],
                    text_pos_embeddings[..., :num_feat],
                    text_neg_embeddings[..., :num_feat],
                )
                loss += self.criterion(emb, pos_emb, neg_emb, **kwargs) / len(
                    self.mrl_nesting_list
                )
        else:
            loss = self.criterion(
                text_embeddings, text_pos_embeddings, text_neg_embeddings, **kwargs
            )
        print("triplet loss: ", loss)
        return {"loss": loss}

    def compute_scored_pair_loss(
        self, text_ids: torch.Tensor, text_pair_ids: torch.Tensor, labels: torch.Tensor
    ):
        text_embeddings = self.get_embedding(text_ids)
        text_pair_embeddings = self.get_embedding(text_pair_ids)
        if self.use_mrl:
            loss = torch.tensor(0.0, device=text_embeddings.device)
            for num_feat in self.mrl_nesting_list:
                emb, emb_pair = (
                    text_embeddings[..., :num_feat],
                    text_pair_embeddings[..., :num_feat],
                )
                predict_labels = torch.cosine_similarity(emb, emb_pair, dim=-1)
                loss += self.cosent_loss(predict_labels, labels) / len(
                    self.mrl_nesting_list
                )
        else:
            predict_labels = torch.cosine_similarity(
                text_embeddings, text_pair_embeddings, dim=-1
            )
            loss = self.cosent_loss(predict_labels, labels)
        print("cosent loss: ", loss)
        return {"loss": loss, "predict_labels": predict_labels}

    def forward(self, **kwargs):
        if "type" in kwargs and "cls_contrast" == kwargs["type"]:
            return self.compute_cls_contrast_loss(**kwargs)
        elif "text_ids" in kwargs and "text_pos_ids" in kwargs:
            return self.compute_triplet_loss(**kwargs)
        elif "text_ids" in kwargs and "text_pair_ids" in kwargs and "labels" in kwargs:
            return self.compute_scored_pair_loss(**kwargs)
        elif "text_ids" in kwargs and "text_labels" in kwargs:
            return self.compute_cls_loss(**kwargs)
        else:
            raise NotImplementedError("not suuport current input kwargs")
