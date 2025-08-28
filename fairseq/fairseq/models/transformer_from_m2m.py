# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import re
from typing import Any, Dict

import torch
from fairseq import checkpoint_utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.roberta import XLMRModel
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    base_architecture as transformer_base_architecture,
)


# https://github.com/pytorch/fairseq/tree/main/examples/xlmr
#from fairseq.checkpoint_utils import load_checkpoint_to_cpu


@register_model("transformer_from_pretrained_m2m")
class TransformerFromPretrainedXLMModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--pretrained-m2m-checkpoint",
            type=str,
            metavar="STR",
            help="M2M model to use for initializing transformer encoder and/or decoder",
        )
        parser.add_argument(
            "--pretrained-m2m-dictionary",
            type=str,
            metavar="STR",
            help="M2M model to use for initializing transformer embedding",
        )
        parser.add_argument(
            "--init-encoder-only",
            action="store_true",
            help="if set, don't load the M2M weights and embeddings into decoder",
        )
        parser.add_argument(
            "--init-decoder-only",
            action="store_true",
            help="if set, don't load the M2M weights and embeddings into encoder",
        )

    @classmethod
    def build_model(self, args, task):
        assert hasattr(args, "pretrained_xlm_checkpoint"), (
            "You must specify a path for --pretrained-m2m-checkpoint to use "
            "--arch transformer_from_pretrained_m2m"
        )
        assert not (
                getattr(args, "init_encoder_only", False)
                and getattr(args, "init_decoder_only", False)
        ), "Only one of --init-encoder-only and --init-decoder-only can be set."
        return super().build_model(args, task)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        set_pretrain_emb(args, src_dict, embed_tokens)
        return TransformerEncoderFromPretrainedM2M(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        set_pretrain_emb(args, tgt_dict, embed_tokens)
        return TransformerDecoderFromPretrainedM2M(args, tgt_dict, embed_tokens)


def set_pretrain_emb(args, dico, embeddings):
    """
    Pretrain word embeddings.
    """
    reloaded = checkpoint_utils.load_checkpoint_to_cpu(args.pretrained_m2m_checkpoint)

    from fairseq.data import Dictionary
    if hasattr(args, "pretrained_m2m_dictionary"):
        id2word = Dictionary.load(args.pretrained_m2m_dictionary).symbols
        n_found = 0
        with torch.no_grad():
            for i in range(len(id2word)):
                idx = dico.indices.get(id2word[i], None)
                if idx is None:
                    continue
                n_found += 1
                embeddings.weight.data[idx] = reloaded["model"]["encoder.embed_tokens.weight"][i]
        logging.info("Pretrained %i/%i words (%.3f%%)."
                     % (n_found, len(dico), 100. * n_found / len(dico)))
    else:
        embeddings.weight.data = reloaded["model"]["encoder.embed_tokens.weight"]
        logging.info("re-initial embedding with m2m checkpoint ")


def upgrade_state_dict_with_m2m_weights(
        state_dict: Dict[str, Any], pretrained_m2m_checkpoint: str, prefix:str
) -> Dict[str, Any]:
    """
    Load XLM weights into a Transformer encoder or decoder model.

    Args:
        state_dict: state dict for either TransformerEncoder or
            TransformerDecoder
        pretrained_xlm_checkpoint: checkpoint to load M2M weights from

    Raises:
        AssertionError: If architecture (num layers, attention heads, etc.)
            does not match between the current Transformer encoder or
            decoder and the pretrained_xlm_checkpoint
    """

    m2m_state_dict = checkpoint_utils.load_checkpoint_to_cpu(pretrained_m2m_checkpoint)["model"]
    for key in m2m_state_dict.keys():
        if "%s.layers" % prefix not in key:
            continue
        for search_key in ["layers"]:
            if search_key in key:
                subkey = key[key.find(search_key):]
                assert subkey in state_dict, "sub key: %s\nm2m key: %s " % (subkey, key)
                state_dict[subkey] = m2m_state_dict[key]
    return state_dict


class TransformerEncoderFromPretrainedM2M(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if getattr(args, "init_decoder_only", False):
            # Don't load XLM weights for encoder if --init-decoder-only
            return

        assert hasattr(args, "pretrained_m2m_checkpoint"), (
            "--pretrained-m2m-checkpoint must be specified to load Transformer "
            "encoder from pretrained M2M"
        )
        m2m_loaded_state_dict = upgrade_state_dict_with_m2m_weights(
            state_dict=self.state_dict(),
            pretrained_m2m_checkpoint=args.pretrained_m2m_checkpoint,
            prefix="encoder",
        )
        self.load_state_dict(m2m_loaded_state_dict, strict=True)


class TransformerDecoderFromPretrainedM2M(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        if getattr(args, "init_encoder_only", False):
            # Don't load XLM weights for decoder if --init-encoder-only
            return

        assert hasattr(args, "pretrained_m2m_checkpoint"), (
            "--pretrained-m2m-checkpoint must be specified to load Transformer "
            "decoder from pretrained M2M"
        )

        m2m_loaded_state_dict = upgrade_state_dict_with_m2m_weights(
            state_dict=self.state_dict(),
            pretrained_m2m_checkpoint=args.pretrained_m2m_checkpoint,
            prefix="decoder"
        )
        self.load_state_dict(m2m_loaded_state_dict, strict=True)


@register_model_architecture(
    "transformer_from_pretrained_m2m", "transformer_from_pretrained_m2m"
)
def base_architecture(args):
    transformer_base_architecture(args)
