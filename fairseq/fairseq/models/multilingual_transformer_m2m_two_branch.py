# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import logging
from collections import OrderedDict

import torch

logger = logging.getLogger(__name__)

from fairseq import utils, checkpoint_utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    Embedding,
)
from fairseq.utils import safe_hasattr

from fairseq.models.multilingual_transformer import base_multilingual_architecture
from fairseq.models.multilingual_transformer_from_m2m import MultilingualTransformerModel

from fairseq.models.transformer_from_m2m import TransformerDecoderFromPretrainedM2M, \
    TransformerEncoderFromPretrainedM2M

from fairseq.models.transformer_from_m2m import set_pretrain_emb

from fairseq.data.multilingual.multilingual_data_manager import (
    MultilingualDatasetManager,
)
from fairseq.data import (
    LanguagePairDataset,
    ListDataset,
)

from cm2m_utils.ceph_file_util import CEPHFileUtil

try:
    ceph_util = CEPHFileUtil()
except:
    # logger.warning("no ceph manager")
    ceph_util = None


@register_model("multilingual_m2m_with_main_branch")
class MultilingualTransformerWithMainBranchModel(MultilingualTransformerModel):
    """Train Transformer models for multiple language pairs simultaneously.

    Requires `--task multilingual_translation`.

    We inherit all arguments from TransformerModel and assume that all language
    pairs use a single Transformer architecture. In addition, we provide several
    options that are specific to the multilingual setting.

    Args:
        --share-encoder-embeddings: share encoder embeddings across all source languages
        --share-decoder-embeddings: share decoder embeddings across all target languages
        --share-encoders: share all encoder params (incl. embeddings) across all source languages
        --share-decoders: share all decoder params (incl. embeddings) across all target languages
    """

    def __init__(self, encoders, decoders, lang_pairs=None, family_type=None):
        super().__init__(encoders, decoders, lang_pairs, family_type)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        from fairseq.tasks.multilingual_translation import MultilingualTranslationTask

        assert isinstance(task, MultilingualTranslationTask)

        # make sure all arguments are present in older models
        base_multilingual_architecture(args)

        if not safe_hasattr(args, "max_source_positions"):
            args.max_source_positions = 1024
        if not safe_hasattr(args, "max_target_positions"):
            args.max_target_positions = 1024

        # one branch + main branch
        core_lg = task.args.core_langs.split(",")[0]
        family_info = utils.get_lg_family_info(core_lg, args.family_type)
        original_model_lang_pairs = ["main-%s" % family_info, "%s-m2m" % family_info] + ["main-m2m"]

        if hasattr(args, "only_two_branch") and args.only_two_branch:
            original_model_lang_pairs = ["%s-m2m" % family_info] + ["main-m2m"]

        if hasattr(args, "lg_adapter") and args.lg_adapter:
            original_model_lang_pairs = ["main-%s" % family_info]

        src_langs = [lang_pair.split("-")[0] for lang_pair in original_model_lang_pairs]
        tgt_langs = [lang_pair.split("-")[1] for lang_pair in original_model_lang_pairs]
        # assumption: all language shares embedding
        if task.args.source_lang is not None or task.args.target_lang is not None:
            dictionary = task.dicts[task.args.source_lang]
        else:
            single_lang = task.lang_pairs[0].split("-")[0]
            dictionary = task.dicts[single_lang]

        args.share_encoder_embeddings = True
        args.share_decoder_embeddings = True

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        # build shared embeddings (if applicable)
        shared_encoder_embed_tokens, shared_decoder_embed_tokens = None, None
        if args.encoder_embed_dim != args.decoder_embed_dim:
            raise ValueError(
                "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
            )
        args.share_decoder_input_output_embed = True

        # encoders/decoders for each language
        lang_encoders, lang_decoders = {}, {}

        if hasattr(args, "pretrained_m2m_checkpoint") and task.training:
            if ceph_util is None:
                with open(args.pretrained_m2m_checkpoint, "rb") as f:
                    reloaded = torch.load(f, map_location=torch.device("cpu"))
            else:
                reloaded = ceph_util.load_checkpoint(args.pretrained_m2m_checkpoint, torch.device("cpu"))
            saved_embedding_weight = reloaded["model"]["encoder.embed_tokens.weight"]
            embed_tokens = build_embedding(
                dictionary,
                args.encoder_embed_dim,
                args.encoder_embed_path,
            )
            embed_tokens = set_pretrain_emb(args, dictionary, embed_tokens, saved_embedding_weight)
            shared_encoder_embed_tokens = embed_tokens
            shared_decoder_embed_tokens = embed_tokens

        def get_encoder(lang):
            family_info = utils.get_lg_family_info(lang, args.family_type)
            logger.info("current lange is : %s, family information is: %s" % (lang, family_info))
            if lang not in lang_encoders:
                if shared_encoder_embed_tokens is not None:
                    encoder_embed_tokens = shared_encoder_embed_tokens
                else:
                    encoder_embed_tokens = build_embedding(
                        dictionary,
                        args.encoder_embed_dim,
                        args.encoder_embed_path,
                    )
                if hasattr(args, "pretrained_m2m_checkpoint") and task.training:
                    lang_encoders[family_info] = TransformerEncoderFromPretrainedM2M(args, dictionary,
                                                                                     encoder_embed_tokens)
                else:
                    lang_encoders[family_info] = cls._get_module_class(
                        True, args, dictionary, encoder_embed_tokens, src_langs
                    )
            return lang_encoders[family_info]

        def get_decoder(lang):
            family_info = utils.get_lg_family_info(lang, args.family_type)
            logger.info("current lange is : %s, family information is: %s" % (lang, family_info))
            if lang not in lang_decoders:
                if shared_decoder_embed_tokens is not None:
                    decoder_embed_tokens = shared_decoder_embed_tokens
                else:
                    decoder_embed_tokens = build_embedding(
                        dictionary,
                        args.decoder_embed_dim,
                        args.decoder_embed_path,
                    )
                if hasattr(args, "pretrained_m2m_checkpoint") and task.training:
                    lang_decoders[family_info] = TransformerDecoderFromPretrainedM2M(args, dictionary,
                                                                                     decoder_embed_tokens)
                else:
                    lang_decoders[family_info] = cls._get_module_class(
                        False, args, dictionary, decoder_embed_tokens, tgt_langs
                    )
            return lang_decoders[family_info]

        # shared encoders/decoders (if applicable)
        shared_encoder, shared_decoder = None, None
        if args.share_encoders:
            shared_encoder = get_encoder(src_langs[0])
        if args.share_decoders:
            shared_decoder = get_decoder(tgt_langs[0])

        encoders, decoders = OrderedDict(), OrderedDict()
        lg_family_dict = {}
        for lang_pair, src, tgt in zip(original_model_lang_pairs, src_langs, tgt_langs):
            src_info = utils.get_lg_family_info(src, args.family_type)
            tgt_info = utils.get_lg_family_info(tgt, args.family_type)
            lg_family_dict[lang_pair] = "%s-%s" % (src_info, tgt_info)
            if src_info not in encoders:
                encoders[src_info] = (
                        shared_encoder if shared_encoder is not None else get_encoder(src)
                    )
            if tgt_info not in decoders:
                decoders[tgt_info] = (
                        shared_decoder if shared_decoder is not None else get_decoder(tgt)
                    )
        return MultilingualTransformerWithMainBranchModel(encoders, decoders, list(lg_family_dict.values()),
                                                          family_type=args.family_type)

    def load_state_dict(self, state_dict, strict=True, model_cfg=None):
        def get_lang_pairs(state_dict):
            keys = set()
            for k, _ in state_dict.items():
                assert k.startswith("models.")
                lang_pair = k.split(".")[1]
                if lang_pair != "main-m2m":
                    keys.add(lang_pair)
                    continue
            if len(keys) > 0:
                return list(keys)[0] if list(keys)[0] not in self.keys else None
            return None

        lang_pair = get_lang_pairs(state_dict)
        if lang_pair is not None and lang_pair not in self.keys:
            core_lang = [part for part in lang_pair.split("-") if part not in ["main", "m2m"]]
            if len(core_lang) > 0:
                core_lang = core_lang[0]
                utils.reset_model_state_dict_(self, core_lang=core_lang, keep_main=True)
        super().load_state_dict(state_dict, strict=strict, model_cfg=model_cfg)

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the multilingual_translation task is not supported"
            )

        src_data = ListDataset(src_tokens, src_lengths)
        dataset = LanguagePairDataset(src_data, src_lengths, self.source_dictionary)
        src_langtok_spec, tgt_langtok_spec = self.args.langtoks["main"]
        if self.args.lang_tok_replacing_bos_eos:
            dataset = self.data_manager.alter_dataset_langtok(
                dataset,
                src_eos=self.source_dictionary.eos(),
                src_lang=self.args.source_lang,
                tgt_eos=self.target_dictionary.eos(),
                tgt_lang=self.args.target_lang,
                src_langtok_spec=src_langtok_spec,
                tgt_langtok_spec=tgt_langtok_spec,
            )
        else:
            dataset.src = self.data_manager.src_dataset_tranform_func(
                self.args.source_lang,
                self.args.target_lang,
                dataset=dataset.src,
                spec=src_langtok_spec,
            )
        return dataset

    def get_part_model(self, language_pair):
        assert language_pair in self.keys
        sub_model = copy.deepcopy(self)
        from torch import nn
        src, tgt = language_pair.split("-")
        sub_model_dict = {language_pair: self.models[language_pair]}
        tmp_models = nn.ModuleDict(sub_model_dict)
        sub_model.encoder_keys, sub_model.decoder_keys, sub_model.keys = src, tgt, list(sub_model_dict)
        sub_model.models = tmp_models
        return sub_model

    def _get_main_model(self):
        return self.get_part_model("main-m2m")

    def _specific_model(self, cfg, family_pair):
        tmp_model = copy.deepcopy(self)
        checkpoint_utils.reload_branch_model(tmp_model, cfg.task.last_ckpt_dir, family_pair,
                                             reload_one_branch=True, reload_lg_specific_branch=False,
                                             reload_embedding=True)
        return tmp_model

    def prepare_model_for_generate(self, cfg, family_type):
        if cfg.task.last_ckpt_dir is None:
            return [self]

        src, tgt = utils.get_lg_family_info(cfg.task.source_lang, family_type), \
                   utils.get_lg_family_info(cfg.task.target_lang, family_type)

        family_pair_dict = {
            1: "main-%s" % tgt,
            2: "%s-m2m" % src,
            4: "main-m2m",
            8: "%s-%s" % (src, tgt)
        }

        # 1: M-L model; 2: L-M model; 4: M-M Model
        if cfg.task.ensemble_type == 1:
            family_pair = family_pair_dict[1]
            logger.info("language pair: %s, family info: %s" % (self.keys, family_pair))
            current_model = self._specific_model(cfg, family_pair)
            return [current_model]

        elif cfg.task.ensemble_type == 2:
            family_pair = family_pair_dict[2]
            logger.info("language pair: %s, family info: %s" % (self.keys, family_pair))
            current_model = self._specific_model(cfg, family_pair)
            return [current_model]

        elif cfg.task.ensemble_type == 4:
            family_pair = family_pair_dict[4]
            logger.info("language pair: %s, family info: %s" % (self.keys, family_pair))
            current_model = self._specific_model(cfg, family_pair)
            return [current_model]

        elif cfg.task.ensemble_type == 3:
            res_models = []
            for family_pair in [family_pair_dict[1], family_pair_dict[2]]:
                logger.info("language pair: %s, family info: %s" % (self.keys, family_pair))
                current_model = self._specific_model(cfg, family_pair)
                res_models.append(current_model)
            return res_models

        elif cfg.task.ensemble_type == 5:
            res_models = []
            for family_pair in [family_pair_dict[1], family_pair_dict[4]]:
                logger.info("language pair: %s, family info: %s" % (self.keys, family_pair))
                current_model = self._specific_model(cfg, family_pair)
                res_models.append(current_model)
            return res_models

        elif cfg.task.ensemble_type == 6:
            res_models = []
            for family_pair in [family_pair_dict[2], family_pair_dict[4]]:
                logger.info("language pair: %s, family info: %s" % (self.keys, family_pair))
                current_model = self._specific_model(cfg, family_pair)
                res_models.append(current_model)
            return res_models

        elif cfg.task.ensemble_type == 7:
            res_models = []
            for family_pair in [family_pair_dict[1], family_pair_dict[2], family_pair_dict[4]]:
                logger.info("language pair: %s, family info: %s" % (self.keys, family_pair))
                current_model = self._specific_model(cfg, family_pair)
                res_models.append(current_model)
            return res_models
        elif cfg.task.ensemble_type == 8:
            res_models = []
            for family_pair in [family_pair_dict[8]]:
                logger.info("language pair: %s, family info: %s" % (self.keys, family_pair))
                current_model = self._specific_model(cfg, family_pair)
                res_models.append(current_model)
            return res_models

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        args = self.encoder.args
        return (args.max_source_positions, args.max_target_positions)

    @classmethod
    def setup_task(cls, args, **kwargs):
        langs, dicts, training = MultilingualDatasetManager.prepare(
            cls.load_dictionary, args, **kwargs
        )
        cls.langs = langs
        return cls(args, langs, dicts, training)

@register_model_architecture(
    "multilingual_m2m_with_main_branch", "multilingual_m2m_with_main_branch_418"
)
def multilingual_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.05)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.05)
    base_multilingual_architecture(args)


@register_model_architecture("multilingual_m2m_with_main_branch", "multilingual_m2m_with_main_branch_1.2B")
def multilingual__ransformer_wmt_en_de_big_1B(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 8192)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.05)
    args.encoder_layers = getattr(args, "encoder_layers", 24)

    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", 1024)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.05)
    args.decoder_output_dim = getattr(args, "decoder_output_dim", 1024)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)

    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 8192)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.1)
    base_multilingual_architecture(args)


@register_model_architecture("multilingual_m2m_with_main_branch", "multi_nllb_200_distilled_1.2B")
def multi_nllb_200_distilled_1B(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 8192)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.encoder_layers = getattr(args, "encoder_layers", 24)

    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", 1024)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_output_dim = getattr(args, "decoder_output_dim", 1024)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)

    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 8192)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.1)
    args.dropout = getattr(args, "attention_dropout", 0.1)
    base_multilingual_architecture(args)