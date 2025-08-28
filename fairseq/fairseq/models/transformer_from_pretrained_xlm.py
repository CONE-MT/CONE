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


@register_model("transformer_from_pretrained_xlm")
class TransformerFromPretrainedXLMModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--pretrained-xlm-checkpoint",
            type=str,
            metavar="STR",
            help="XLM model to use for initializing transformer encoder and/or decoder",
        )
        parser.add_argument(
            "--init-encoder-only",
            action="store_true",
            help="if set, don't load the XLM weights and embeddings into decoder",
        )
        parser.add_argument(
            "--init-decoder-only",
            action="store_true",
            help="if set, don't load the XLM weights and embeddings into encoder",
        )

    @classmethod
    def build_model(self, args, task):
        assert hasattr(args, "pretrained_xlm_checkpoint"), (
            "You must specify a path for --pretrained-xlm-checkpoint to use "
            "--arch transformer_from_pretrained_xlm"
        )
        assert not (
                getattr(args, "init_encoder_only", False)
                and getattr(args, "init_decoder_only", False)
        ), "Only one of --init-encoder-only and --init-decoder-only can be set."
        return super().build_model(args, task)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        set_pretrain_emb(args, src_dict, embed_tokens, dict_type="src")
        return TransformerEncoderFromPretrainedXLM(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        set_pretrain_emb(args, tgt_dict, embed_tokens, dict_type="tgt")
        return TransformerDecoderFromPretrainedXLM(args, tgt_dict, embed_tokens)


def set_pretrain_emb(args, dico, embeddings, dict_type):
    """
    Pretrain word embeddings.
    """
    pretrained_dir, pretrained_fname = os.path.dirname(args.pretrained_xlm_checkpoint), os.path.basename(
        args.pretrained_xlm_checkpoint)
    reloaded = XLMRModel.from_pretrained(pretrained_dir, checkpoint_file=pretrained_fname)
    model, id2word = reloaded.model.state_dict(), reloaded.task.target_dictionary.symbols if dict_type == "tgt" else reloaded.task.source_dictionary.symbols
    n_found = 0
    with torch.no_grad():
        for i in range(len(id2word)):
            idx = dico.indices.get(id2word[i], None)
            if idx is None:
                continue
            n_found += 1
            embeddings.weight[idx] = model["encoder.sentence_encoder.embed_tokens.weight"][i]
    logging.info("Pretrained %i/%i words (%.3f%%)."
                 % (n_found, len(dico), 100. * n_found / len(dico)))


def upgrade_state_dict_with_xlm_weights(
        state_dict: Dict[str, Any], pretrained_xlm_checkpoint: str, n_layer: int
) -> Dict[str, Any]:
    """
    Load XLM weights into a Transformer encoder or decoder model.

    Args:
        state_dict: state dict for either TransformerEncoder or
            TransformerDecoder
        pretrained_xlm_checkpoint: checkpoint to load XLM weights from

    Raises:
        AssertionError: If architecture (num layers, attention heads, etc.)
            does not match between the current Transformer encoder or
            decoder and the pretrained_xlm_checkpoint
    """
    if not os.path.exists(pretrained_xlm_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_xlm_checkpoint))

    pretrained_dir, pretrained_fname = os.path.dirname(pretrained_xlm_checkpoint), os.path.basename(
        pretrained_xlm_checkpoint)
    xlm_state_dict = XLMRModel.from_pretrained(pretrained_dir, checkpoint_file=pretrained_fname).state_dict()
    for key in xlm_state_dict.keys():

        for search_key in ["layers"]:
            if search_key in key:
                subkey = key[key.find(search_key):]
                nums = [float(i) for i in re.findall(r"\d+", subkey) if i.isdigit()]
                max_value = 0 if len(nums) == 0 else max(nums)
                if max_value >= n_layer:
                    continue
                assert subkey in state_dict, (
                    "{} Transformer encoder / decoder "
                    "state_dict does not contain {}. Cannot "
                    "load {} from pretrained XLM checkpoint "
                    "{} into Transformer.".format(
                        str(state_dict.keys()), subkey, key, pretrained_xlm_checkpoint
                    )
                )

                state_dict[subkey] = xlm_state_dict[key]
    return state_dict


class TransformerEncoderFromPretrainedXLM(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if getattr(args, "init_decoder_only", False):
            # Don't load XLM weights for encoder if --init-decoder-only
            return

        assert hasattr(args, "pretrained_xlm_checkpoint"), (
            "--pretrained-xlm-checkpoint must be specified to load Transformer "
            "encoder from pretrained XLM"
        )
        xlm_loaded_state_dict = upgrade_state_dict_with_xlm_weights(
            state_dict=self.state_dict(),
            pretrained_xlm_checkpoint=args.pretrained_xlm_checkpoint,
            n_layer=args.encoder_layers,
        )
        self.load_state_dict(xlm_loaded_state_dict, strict=True)


class TransformerDecoderFromPretrainedXLM(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        if getattr(args, "init_encoder_only", False):
            # Don't load XLM weights for decoder if --init-encoder-only
            return

        assert hasattr(args, "pretrained_xlm_checkpoint"), (
            "--pretrained-xlm-checkpoint must be specified to load Transformer "
            "decoder from pretrained XLM"
        )

        xlm_loaded_state_dict = upgrade_state_dict_with_xlm_weights(
            state_dict=self.state_dict(),
            pretrained_xlm_checkpoint=args.pretrained_xlm_checkpoint,
            n_layer=args.decoder_layers
        )
        self.load_state_dict(xlm_loaded_state_dict, strict=True)


@register_model_architecture(
    "transformer_from_pretrained_xlm", "transformer_from_pretrained_xlm"
)
def base_architecture(args):
    transformer_base_architecture(args)
