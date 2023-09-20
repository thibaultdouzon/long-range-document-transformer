# coding=utf-8
# Copyright 2018 The Microsoft Research Asia Locallm Team Authors and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Locallm model. """


import math

import numpy as np

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from transformers.activations import ACT2FN
from transformers.file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers.models.layoutlm.configuration_layoutlm import LayoutLMConfig
from transformers.models.layoutlm import modeling_layoutlm
from transformers.models.layoutlm.modeling_layoutlm import *

from src.modeling.cosformer.configuration_cosformer import CosformerConfig


class CosformerEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(CosformerEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.x_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.y_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.h_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.w_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.page_embeddings = nn.Embedding(
            config.max_pages_embeddings, config.hidden_size
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )

    def forward(
        self,
        input_ids=None,
        bbox=None,
        pages=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        words_embeddings = inputs_embeds
        position_embeddings = self.position_embeddings(position_ids)
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError(
                "The :obj:`bbox`coordinate values should be within 0-1000 range."
            ) from e

        try:
            h_position_embeddings = self.h_position_embeddings(
                bbox[:, :, 3] - bbox[:, :, 1]
            )
            w_position_embeddings = self.w_position_embeddings(
                bbox[:, :, 2] - bbox[:, :, 0]
            )
        except IndexError as e:
            raise IndexError("The :obj:`bbox` should have positive or null area") from e

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        page_embeddings = self.page_embeddings(pages)

        embeddings = (
            words_embeddings
            + position_embeddings
            + left_position_embeddings
            + upper_position_embeddings
            + right_position_embeddings
            + lower_position_embeddings
            + h_position_embeddings
            + w_position_embeddings
            + token_type_embeddings
            + page_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

        # (N, L, E) –> (L, N, E)
        hidden_states.transpose_(0, 1)

        # get q, k, v
        # (L, N, E)
        q = self.query(hidden_states)
        # (L, N, E)
        k = self.key(hidden_states)
        # (L, N, E)
        v = self.value(hidden_states)


class CosformerSelfAttention(nn.Module):
    """O(n * d^2) Multi-Head Attention.
    If 'kernel' == 'relu' and 'use_cos' == True, the model from
    'COSFORMER : RETHINKING SOFTMAX IN ATTENTION' (https://openreview.net/pdf?id=Bl8CQrx2Up4)
    is constructed. If 'kernel' == 'elu' and 'use_cos' == False, this class implements the
    model from: 'Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention'
    (https://arxiv.org/pdf/2006.16236.pdf).
    Attributes:
      d_model: (int) The full dimension of the hidden activations.
      n_heads: (int) Number of attention heads calculated in parallel.
      d_head: (int) Dimension of each head. d_model = n_heads * d_head.
      denom_eps: (float) Small positive constant that is added to the denominator of the
        linear self attention for stabilization. See self.linear_attention().
      kernel: (func) Kernel function used to approximate softmax. Either F.relu() or
        F.elu() + 1.
      attention_func: (func) Function used to compute linear attention. Either
        self.linear_attention (without cos reweigting) or self.cos_linear_attention
        (with cos reweighting).
      w_qkv: (nn.Linear) Used to calculate Q, K, V all at the same time.
      w_o: (nn.Linear) Applied to the output after self attention.
      dropout: (nn.Dropout) Applied after w_o.
    """

    def __init__(self, config):
        """Initialized a MHA Module.
        Args:
          d_model: (int) The full dimension of the hidden activations.
          n_heads: (int) Number of attention heads calculated in parallel.
          use_cos: (bool) If True, the cos reweighting mechanism from
            https://openreview.net/pdf?id=Bl8CQrx2Up4 is implemented and positional
            embeddings are not used. If false, sinusoidal positional embeddings are used.
          kernel: (str) If 'relu' is given, softmax is approximated with F.relu(). If 'elu'
            is given, F.elu() + 1 is used.
          dropout: (float) Dropout rate.
          denom_eps: (float) Small positive constant that is added to the denominator of the
            linear self attention for stabilization. See self.linear_attention().
          bias: (bool) Whether to add bias to all linear layers.
        """
        super(CosformerSelfAttention, self).__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.d_model = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.d_head = self.d_model // self.n_heads

        # Smaller values lead to NaN in half precision
        self.denom_eps = 1e-4
        self.seq_size = config.max_position_embeddings

        # Probably should experiment with more different kernels.
        if config.attention_activation_function == "relu":
            self.kernel = self.relu_kernel
        elif config.attention_activation_function == "elu":
            self.kernel = self.elu_kernel
        else:
            raise NotImplementedError(
                "The only options for 'kernel' are 'relu and 'elu'."
            )

        if self.config.relative_attention_mode == "circ":
            self.attention_func = self.cos_linear_attention_circ
        elif self.config.relative_attention_mode == "cross":
            self.attention_func = self.cos_linear_attention_cross
        elif self.config.relative_attention_mode == "cos":
            self.attention_func = self.cos_linear_attention
        else:
            self.attention_func = self.linear_attention

        self.query = nn.Linear(self.d_model, self.d_model)
        self.key = nn.Linear(self.d_model, self.d_model)
        self.value = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def apply_mask(self, x, mask):
        """Zeroes out elements specified by the attention mask.
        Args:
          x: (torch.Tensor of float32)[batch_size, seq_len, d_model] Tensor to be masked.
          mask: (torch.Tensor of bool)[batch_size, seq_len, 1] or None: True for elements that
            will be replaced with zero, False for the ones that will remain unchanged. If None,
            the function will return x unchanged.
        Returns:
          x: (torch.Tensor of float32)[batch_size, seq_len, d_model] Tensor after masking.
        """
        if not mask is None:
            x = x.masked_fill(~mask, 0)

        return x

    def split_heads(self, x):
        """Splits the last dimension of a tensor d_model into [n_heads, d_head].
        Args:
          x: (torch.Tensor of float32)[batch_size, seq_len, d_model]
        Returns:
          x: (torch.Tensor of float32)[batch_size, seq_len, n_heads, d_head]
        """
        batch_size, seq_len = x.shape[:2]
        # x -> [batch_size, seq_len, d_model]
        x = x.view(batch_size, seq_len, self.n_heads, self.d_head)
        # x -> [batch_size, seq_len, n_heads, d_head]

        return x

    def merge_heads(self, x):
        """Merges the 2 last dimensions of a tensor [n_heads, d_head] into d_model.
        Args:
          x: (torch.Tensor of float32)[batch_size, seq_len, n_heads, d_head]
        Returns:
          x: (torch.Tensor of float32)[batch_size, seq_len, d_model]
        """
        batch_size, seq_len = x.shape[:2]
        # x -> [batch_size, seq_len, n_heads, d_head]
        x = x.view(batch_size, seq_len, self.d_model).contiguous()
        # x -> [batch_size, seq_len, d_model]

        return x

    def elu_kernel(self, x):
        """Kernel proposed in https://arxiv.org/pdf/2006.16236.pdf"""
        return F.elu(x) + 1

    def relu_kernel(self, x):
        """Kernel proposed in https://openreview.net/pdf?id=Bl8CQrx2Up4"""
        return F.relu(x)

    def linear_attention(self, q, k, v):
        """Implements linear attention as proposed in https://arxiv.org/pdf/2006.16236.pdf
        Translated from tensorflow to pytorch based on:
        https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/kernel_attention.py
        Args:
          q, k, v: (torch.Tensor of float32)[batch_size, seq_len, n_heads, d_head] The query, key
            and value tensors. The kernel function must be already applied to q and k. The
            attention mask must be already applied to k.
          weights: None. Unused.
        Retruns:
          output: (torch.Tensor of float32)[batch_size, seq_len, n_heads, d_head] The result of
            linear self attention.
        """
        # q, k, v -> [batch_size, seq_len, n_heads, d_head]
        kv = torch.einsum("bsnx,bsnz->bnxz", k, v)
        # kv -> [batch_size, n_heads, d_head, d_head]
        # add dropout here
        denominator = 1.0 / (
            torch.einsum("bsnd,bnd->bsn", q, k.sum(axis=1)) + self.denom_eps
        )
        # denominator -> [batch_size, seq_len, n_heads]

        output = torch.einsum("bsnx,bnxz,bsn->bsnz", q, kv, denominator).contiguous()
        # output -> [batch_size, seq_len, n_heads, d_head]

        return output

    def cos_linear_attention_cross(self, q, k, v, cos_weights):
        """Implements linear attention with inverse exponential reweighting based on 2D layout.
        Words closer in 2D space get greater attention.
        Args:
          q, k, v: (torch.Tensor of float32)[batch_size, seq_len, n_heads, d_head] The query, key
            and value tensors. The kernel function must be already applied to q and k. The
            attention mask must be already applied to k.
          weights: (tuple of (torch.Tensor of float32)[batch_size, seq_len])
            weights = (cos, sin), where cos[i, j] = cos(pi * i / 2 / M[i]) whre M[i] is the length
            of the i-th sample in the batch. Similarly for sin.
        Retruns:
          output: (torch.Tensor of float32)[batch_size, seq_len, n_heads, d_head] The result of
            linear self attention.
        """
        attn_x = self.cos_linear_attention(q, k, v, cos_weights[:2])
        attn_y = self.cos_linear_attention(q, k, v, cos_weights[2:])

        max_attn = torch.where(torch.abs(attn_x) > torch.abs(attn_y), attn_x, attn_y)

        return max_attn.contiguous()

        # denominator -> [batch_size, seq_len, n_heads]

    def cos_linear_attention_circ(self, q, k, v, cos_weights):
        """Implements linear attention with cos reweighting based on 2D layout.
        Words closer in 2D space get greater attention.
        Args:
          q, k, v: (torch.Tensor of float32)[batch_size, seq_len, n_heads, d_head] The query, key
            and value tensors. The kernel function must be already applied to q and k. The
            attention mask must be already applied to k.
          weights: (tuple of (torch.Tensor of float32)[batch_size, seq_len])
            weights = (cos, sin), where cos[i, j] = cos(pi * i / 2 / M[i]) whre M[i] is the length
            of the i-th sample in the batch. Similarly for sin.
        Retruns:
          output: (torch.Tensor of float32)[batch_size, seq_len, n_heads, d_head] The result of
            linear self attention.
        """
        # q, k, v -> [batch_size, seq_len, n_heads, d_head]

        cos_x, sin_x, cos_y, sin_y = cos_weights
        # cos, sin -> [batch_size, seq_len]

        q_cx_cy = torch.einsum("bsnd,bs,bs->bsnd", q, cos_x, cos_y)
        q_cx_sy = torch.einsum("bsnd,bs,bs->bsnd", q, cos_x, sin_y)
        q_sx_cy = torch.einsum("bsnd,bs,bs->bsnd", q, sin_x, cos_y)
        q_sx_sy = torch.einsum("bsnd,bs,bs->bsnd", q, sin_x, sin_y)

        k_cx_cy = torch.einsum("bsnd,bs,bs->bsnd", k, cos_x, cos_y)
        k_cx_sy = torch.einsum("bsnd,bs,bs->bsnd", k, cos_x, sin_y)
        k_sx_cy = torch.einsum("bsnd,bs,bs->bsnd", k, sin_x, cos_y)
        k_sx_sy = torch.einsum("bsnd,bs,bs->bsnd", k, sin_x, sin_y)
        # q_cos, q_sin, k_cos, k_sin -> [batch_size, seq_len, n_heads, d_head]

        kv_cx_cy = torch.einsum("bsnx,bsnz->bnxz", k_cx_cy, v)
        # kv_cos -> [batch_size, n_heads, d_head, d_head]
        qkv_cx_cy = torch.einsum("bsnx,bnxz->bsnz", q_cx_cy, kv_cx_cy)
        # qkv_cos -> [batch_size, seq_len, n_heads, d_head]

        kv_cx_sy = torch.einsum("bsnx,bsnz->bnxz", k_cx_sy, v)
        qkv_cx_sy = torch.einsum("bsnx,bnxz->bsnz", q_cx_sy, kv_cx_sy)

        kv_sx_cy = torch.einsum("bsnx,bsnz->bnxz", k_sx_cy, v)
        qkv_sx_cy = torch.einsum("bsnx,bnxz->bsnz", q_sx_cy, kv_sx_cy)

        kv_sx_sy = torch.einsum("bsnx,bsnz->bnxz", k_sx_sy, v)
        qkv_sx_sy = torch.einsum("bsnx,bnxz->bsnz", q_sx_sy, kv_sx_sy)

        # denominator
        denominator = 1.0 / torch.clamp_min(
            torch.einsum("bsnd,bnd->bsn", q_cx_cy, k_cx_cy.sum(axis=1))
            + torch.einsum("bsnd,bnd->bsn", q_cx_sy, k_cx_sy.sum(axis=1))
            + torch.einsum("bsnd,bnd->bsn", q_sx_cy, k_sx_cy.sum(axis=1))
            + torch.einsum("bsnd,bnd->bsn", q_sx_sy, k_sx_sy.sum(axis=1)),
            self.denom_eps,
        )
        # denominator -> [batch_size, seq_len, n_heads]

        output = torch.einsum(
            "bsnz,bsn->bsnz", qkv_cx_cy + qkv_cx_sy + qkv_sx_cy + qkv_sx_sy, denominator
        ).contiguous()
        # output -> [batch_size, seq_len, n_heads, d_head]

        return output

    def cos_linear_attention(self, q, k, v, cos_weights):
        """Implements linear attention with cos reweighting as in https://openreview.net/pdf?id=Bl8CQrx2Up4.
        Args:
          q, k, v: (torch.Tensor of float32)[batch_size, seq_len, n_heads, d_head] The query, key
            and value tensors. The kernel function must be already applied to q and k. The
            attention mask must be already applied to k.
          weights: (tuple of (torch.Tensor of float32)[batch_size, seq_len])
            weights = (cos, sin), where cos[i, j] = cos(pi * i / 2 / M[i]) whre M[i] is the length
            of the i-th sample in the batch. Similarly for sin.
        Retruns:
          output: (torch.Tensor of float32)[batch_size, seq_len, n_heads, d_head] The result of
            linear self attention.
        """
        # q, k, v -> [batch_size, seq_len, n_heads, d_head]
        cos, sin = cos_weights
        # cos, sin -> [batch_size, seq_len]
        q_cos = torch.einsum("bsnd,bs->bsnd", q, cos)
        q_sin = torch.einsum("bsnd,bs->bsnd", q, sin)
        k_cos = torch.einsum("bsnd,bs->bsnd", k, cos)
        k_sin = torch.einsum("bsnd,bs->bsnd", k, sin)
        # q_cos, q_sin, k_cos, k_sin -> [batch_size, seq_len, n_heads, d_head]

        kv_cos = torch.einsum("bsnx,bsnz->bnxz", k_cos, v)
        # kv_cos -> [batch_size, n_heads, d_head, d_head]
        qkv_cos = torch.einsum("bsnx,bnxz->bsnz", q_cos, kv_cos)
        # qkv_cos -> [batch_size, seq_len, n_heads, d_head]

        kv_sin = torch.einsum("bsnx,bsnz->bnxz", k_sin, v)
        # kv_sin -> [batch_size, n_heads, d_head, d_head]
        qkv_sin = torch.einsum("bsnx,bnxz->bsnz", q_sin, kv_sin)
        # qkv_sin -> [batch_size, seq_len, n_heads, d_head]

        # denominator
        denominator = 1.0 / torch.clamp_min(
            torch.einsum("bsnd,bnd->bsn", q_cos, k_cos.sum(axis=1))
            + torch.einsum("bsnd,bnd->bsn", q_sin, k_sin.sum(axis=1)),
            self.denom_eps,
        )
        # denominator -> [batch_size, seq_len, n_heads]

        output = torch.einsum(
            "bsnz,bsn->bsnz", qkv_cos + qkv_sin, denominator
        ).contiguous()
        # output -> [batch_size, seq_len, n_heads, d_head]

        return output

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        pad_mask=None,
        cos_weights=None,
    ):
        """Implements forward pass.
        Args:
          x: (torch.Tensor of float32)[batch_size, seq_len, d_model] Input batch.
          mask: (torch.Tensor of bool)[batch_size, seq_len, 1] Attention mask.
            True for elements that must be masked. If mask is None, masking is not
            applied.
          weights: (tuple of (torch.Tensor of float32)[batch_size, seq_len, n_heads, d_head])
            weights = (cos, sin), where cos[i, j] = cos(pi * i / 2 / M[i]) whre M[i]
            is the length of the i-th sample in the batch. Similarly for sin. If cos
            reweighting is not applied, weights = None.
        Retruns:
          x: (torch.Tensor of float32)[batch_size, seq_len, d_head]
        """

        # x -> [batch_size, seq_len, d_model]
        # mask -> [batch_size, seq_len, 1] or None
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        # q, k, v -> [batch_size, seq_len, d_model]

        # A note about padding & masking in linear kernel attention:
        # In the f(Q) * (f(K^T) * V) attention, f(Q) is mutiplied by a dxd matrix.
        # Therefore padded elements must be removed from (f(K^T) * V).
        # This can be done by replacing padded elements (in the seq_len) dimension
        # of either f(K^T) or V). However, as seen in linear_attention, K is
        # summed in the seq_len dimension in the denominator, which means that
        # K must be masked.

        q = self.split_heads(self.kernel(q))
        k = self.split_heads(self.apply_mask(self.kernel(k), pad_mask))
        v = self.split_heads(v)
        # q, k, v -> [batch_size, seq_len, n_heads, d_head]
        x = self.attention_func(q, k, v, cos_weights)
        # x -> [batch_size, seq_len, n_heads, d_head]
        x = self.merge_heads(x)
        x = self.dropout(self.w_o(x))
        # x -> [batch_size, seq_len, d_model]

        return (x,)


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->Cosformer
class CosformerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Cosformer
class CosformerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = CosformerSelfAttention(config)
        self.output = CosformerSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        pad_mask=None,
        cos_weights=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            pad_mask,
            cos_weights,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class CosformerIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->Cosformer
class CosformerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Cosformer
class CosformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = CosformerAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert (
                self.is_decoder
            ), f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = CosformerAttention(config)
        self.intermediate = CosformerIntermediate(config)
        self.output = CosformerOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        pad_mask=None,
        cos_weights=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            pad_mask=pad_mask,
            cos_weights=cos_weights,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[
                1:
            ]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:-1]
            )  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Cosformer
class CosformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [CosformerLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        bbox=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        pad_mask=None,
        cos_weights=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    pad_mask,
                    cos_weights,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    pad_mask,
                    cos_weights,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->Cosformer
class CosformerPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->Cosformer
class CosformerLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = CosformerPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->Cosformer
class CosformerOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = CosformerLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class CosformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CosformerConfig
    pretrained_model_archive_map = LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "layoutlm"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayoutLMLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LayoutLMEncoder):
            module.gradient_checkpointing = value


LAYOUTLM_START_DOCSTRING = r"""
    The LayoutLM model was proposed in [LayoutLM: Pre-training of Text and Layout for Document Image
    Understanding](https://arxiv.org/abs/1912.13318) by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei and
    Ming Zhou.

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`LayoutLMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

LAYOUTLM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`LayoutLMTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        bbox (`torch.LongTensor` of shape `({0}, 4)`, *optional*):
            Bounding boxes of each input sequence tokens. Selected in the range `[0,
            config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
            format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
            y1) represents the position of the lower right corner. See [Overview](#Overview) for normalization.
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`: `1` for
            tokens that are NOT MASKED, `0` for MASKED tokens.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`: `0` corresponds to a *sentence A* token, `1` corresponds to a *sentence B* token

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`: `1`
            indicates the head is **not masked**, `0` indicates the head is **masked**.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            If set to `True`, the attentions tensors of all attention layers are returned. See `attentions` under
            returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            If set to `True`, the hidden states of all layers are returned. See `hidden_states` under returned tensors
            for more detail.
        return_dict (`bool`, *optional*):
            If set to `True`, the model will return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LayoutLM Model transformer outputting raw hidden-states without any specific head on top.",
    LAYOUTLM_START_DOCSTRING,
)
class CosformerModel(CosformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = CosformerEmbeddings(config)
        self.encoder = CosformerEncoder(config)
        self.pooler = modeling_layoutlm.LayoutLMPooler(config)

        if self.config.relative_attention_mode in "circ cross".split():
            self.cos_weights_function = self.get_cos_weights_layout
        else:
            self.cos_weights_function = self.get_cos_weights

        # Initialize weights and apply final processing
        self.post_init()

    def get_mask(self, input_ids, max_len=None):
        """Creates Attention Mask.
        Args:
          lengths: (torch.Tensor of long)[batch_size]. Length of each input sequence.
          max_len: (int or None) Maximum length inside the batch. If None, assigns
            max_len = lengths.max()
        Returns:
          mask: (torch.Tensor of long)[batch_size, max_len]. Attention Mask where
            padded elements are 0 and valid ones are 1.s
        """

        # 0 is the [PAD] default token
        lengths = torch.sum(input_ids != 0, dim=1)

        # lens -> [batch_size]
        if max_len is None:
            max_len = lengths.max()
        mask = torch.arange(max_len)[None, :].to(lengths) < lengths[:, None]
        # mask -> [batch_size, max_len]

        return mask.unsqueeze(-1)

    def get_cos_weights_layout(self, bbox, mask, max_len):
        """Returns cosine weights based on layout information instead of linear position.
        Args:
          lengths: (torch.Tensor of long)[batch_size]. Length of each input sequence.
          max_len: (int or None) Maximum length inside the batch. If None, assigns
            max_len = lengths.max()
        Returns:
          (cos, sin): (tuple of torch.Tensor of float32)[batch_size, seq_len]).
            cos[i, j] = cos(pi * i / 2 / M[i]) whre M[i] is the length
            of the i-th sample in the batch. Similarly for sin.
        """
        x = (bbox[..., 0] + bbox[..., 2]) / 2  # [batch_size, max_len]
        y = (bbox[..., 1] + bbox[..., 3]) / 2  # [batch_size, max_len]

        assert x.size(1) == max_len and y.size(1) == max_len

        cos_x = torch.cos(math.pi / 2 * x / 1000)
        sin_x = torch.sin(math.pi / 2 * x / 1000)
        cos_y = torch.cos(math.pi / 2 * y / 1000)
        sin_y = torch.sin(math.pi / 2 * y / 1000)

        cos_x.masked_fill_(~mask, 0)
        sin_x.masked_fill_(~mask, 0)
        cos_y.masked_fill_(~mask, 0)
        sin_y.masked_fill_(~mask, 0)

        return cos_x.detach(), sin_x.detach(), cos_y.detach(), sin_y.detach()

    def get_cos_weights(self, input_ids, mask, max_len=None):
        """Returns cosine weights.
        Used for reweighting as described in https://openreview.net/pdf?id=Bl8CQrx2Up4.
        Args:
          lengths: (torch.Tensor of long)[batch_size]. Length of each input sequence.
          max_len: (int or None) Maximum length inside the batch. If None, assigns
            max_len = lengths.max()
        Returns:
          (cos, sin): (tuple of torch.Tensor of float32)[batch_size, seq_len]).
            cos[i, j] = cos(pi * i / 2 / M[i]) whre M[i] is the length
            of the i-th sample in the batch. Similarly for sin.
        """
        # 0 is the [PAD] default token
        lengths = torch.sum(input_ids != 0, dim=1)
        # lengths = torch.ones(input_ids.size(0), dtype=int).to(input_ids) * self.config.max_position_embeddings

        # lengths -> [batch_size]
        if max_len is None:
            max_len = lengths.max()
        # For each sample x in the batch, calculate M(x) = len(x)
        M = lengths
        # M -> [batch_size]
        idxs = math.pi / 2 * torch.arange(max_len).to(lengths)
        # idxs -> [max_len]
        idxs = torch.outer(1.0 / M, idxs)  # [..., None, None]
        # idxs -> [batch_size, max_len]

        cos = torch.cos(idxs)
        sin = torch.sin(idxs)

        cos.masked_fill_(~mask, 0)
        sin.masked_fill_(~mask, 0)

        # cos, sin -> [batch_size, max_len]
        return cos.detach(), sin.detach()

    @add_start_docstrings_to_model_forward(
        LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=modeling_layoutlm._CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        bbox=None,
        pages=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import LayoutLMTokenizer, LayoutLMModel
        >>> import torch

        >>> tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        >>> model = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")

        >>> words = ["Hello", "world"]
        >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

        >>> token_boxes = []
        >>> for word, box in zip(words, normalized_word_boxes):
        ...     word_tokens = tokenizer.tokenize(word)
        ...     token_boxes.extend([box] * len(word_tokens))
        >>> # add bounding boxes of cls + sep tokens
        >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        >>> encoding = tokenizer(" ".join(words), return_tensors="pt")
        >>> input_ids = encoding["input_ids"]
        >>> attention_mask = encoding["attention_mask"]
        >>> token_type_ids = encoding["token_type_ids"]
        >>> bbox = torch.tensor([token_boxes])

        >>> outputs = model(
        ...     input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids
        ... )

        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if bbox is None:
            bbox = torch.zeros(
                tuple(list(input_shape) + [4]), dtype=torch.long, device=device
            )

        pad_mask = self.get_mask(input_ids, max_len=self.config.max_position_embeddings)
        # pad_mask = None

        cos_weights = self.cos_weights_function(
            bbox if self.config.relative_attention_mode in "circ cross".split() else input_ids,
            pad_mask.squeeze(-1),
            max_len=self.config.max_position_embeddings,
        )

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids,
            bbox=bbox,
            pages=pages,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pad_mask=pad_mask,
            cos_weights=cos_weights,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings(
    """LayoutLM Model with a `language modeling` head on top.""",
    LAYOUTLM_START_DOCSTRING,
)
class CosformerForMaskedLM(CosformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layoutlm = CosformerModel(config)
        self.cls = LayoutLMOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(
        LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=MaskedLMOutput, config_class=modeling_layoutlm._CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids=None,
        bbox=None,
        pages=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import LayoutLMTokenizer, LayoutLMForMaskedLM
        >>> import torch

        >>> tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        >>> model = LayoutLMForMaskedLM.from_pretrained("microsoft/layoutlm-base-uncased")

        >>> words = ["Hello", "[MASK]"]
        >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

        >>> token_boxes = []
        >>> for word, box in zip(words, normalized_word_boxes):
        ...     word_tokens = tokenizer.tokenize(word)
        ...     token_boxes.extend([box] * len(word_tokens))
        >>> # add bounding boxes of cls + sep tokens
        >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        >>> encoding = tokenizer(" ".join(words), return_tensors="pt")
        >>> input_ids = encoding["input_ids"]
        >>> attention_mask = encoding["attention_mask"]
        >>> token_type_ids = encoding["token_type_ids"]
        >>> bbox = torch.tensor([token_boxes])

        >>> labels = tokenizer("Hello world", return_tensors="pt")["input_ids"]

        >>> outputs = model(
        ...     input_ids=input_ids,
        ...     bbox=bbox,
        ...     attention_mask=attention_mask,
        ...     token_type_ids=token_type_ids,
        ...     labels=labels,
        ... )

        >>> loss = outputs.loss
        ```"""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.layoutlm(
            input_ids,
            bbox,
            pages=pages,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    LayoutLM Model with a sequence classification head on top (a linear layer on top of the pooled output) e.g. for
    document image classification tasks such as the [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset.
    """,
    LAYOUTLM_START_DOCSTRING,
)
@add_start_docstrings(
    """
    LayoutLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    sequence labeling (information extraction) tasks such as the [FUNSD](https://guillaumejaume.github.io/FUNSD/)
    dataset and the [SROIE](https://rrc.cvc.uab.es/?ch=13) dataset.
    """,
    LAYOUTLM_START_DOCSTRING,
)
class CosformerForTokenClassification(CosformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlm = CosformerModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(
        LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=TokenClassifierOutput,
        config_class=modeling_layoutlm._CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        bbox=None,
        pages=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
        >>> import torch

        >>> tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        >>> model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")

        >>> words = ["Hello", "world"]
        >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

        >>> token_boxes = []
        >>> for word, box in zip(words, normalized_word_boxes):
        ...     word_tokens = tokenizer.tokenize(word)
        ...     token_boxes.extend([box] * len(word_tokens))
        >>> # add bounding boxes of cls + sep tokens
        >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        >>> encoding = tokenizer(" ".join(words), return_tensors="pt")
        >>> input_ids = encoding["input_ids"]
        >>> attention_mask = encoding["attention_mask"]
        >>> token_type_ids = encoding["token_type_ids"]
        >>> bbox = torch.tensor([token_boxes])
        >>> token_labels = torch.tensor([1, 1, 0, 0]).unsqueeze(0)  # batch size of 1

        >>> outputs = model(
        ...     input_ids=input_ids,
        ...     bbox=bbox,
        ...     attention_mask=attention_mask,
        ...     token_type_ids=token_type_ids,
        ...     labels=token_labels,
        ... )

        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            pages=pages,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
