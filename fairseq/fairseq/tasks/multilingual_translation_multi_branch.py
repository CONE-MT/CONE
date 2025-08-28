# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import logging
import os
import random
import time
from argparse import ArgumentError

import torch
import math
from fairseq import checkpoint_utils
from fairseq import utils, metrics
from fairseq.data import (
    Dictionary,
    LanguagePairDataset, FairseqDataset, iterators, ListDataset, )
from fairseq.data import (
    data_utils,
)
from fairseq.data.multilingual.multilingual_data_manager import (
    MultilingualDatasetManager,
)
from fairseq.data.multilingual.sampling_method import SamplingMethod
from fairseq.models import FairseqMultiModel
from fairseq.models.transformer import (
    TransformerModel,
)
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
from fairseq.utils import FileContentsAction

from . import register_task

logger = logging.getLogger(__name__)
import datetime

import time
global f
f = open('translation_multi_branch.txt','w+')

from fairseq.data import LanguagePairDatasetForDemo


def get_time_gap(s, e):
    return (
        datetime.datetime.fromtimestamp(e) - datetime.datetime.fromtimestamp(s)
    ).__str__()


def _lang_token(lang: str):
    return "__{}__".format(lang)


def _lang_token_index(dic: Dictionary, lang: str):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, "cannot find language token for lang {}".format(lang)
    return idx


@register_task("multilingual_translation_branch")
class MultilingualTranslationBranchTask(MultilingualTranslationTask):
    """A task for training multiple translation models simultaneously.

    We iterate round-robin over batches from multiple language pairs, ordered
    according to the `--lang-pairs` argument.

    The training loop is roughly:

        for i in range(len(epoch)):
            for lang_pair in args.lang_pairs:
                batch = next_batch_for_lang_pair(lang_pair)
                loss = criterion(model_for_lang_pair(lang_pair), batch)
                loss.backward()
            optimizer.step()

    In practice, `next_batch_for_lang_pair` is abstracted in a FairseqDataset
    (e.g., `RoundRobinZipDatasets`) and `model_for_lang_pair` is a model that
    implements the `FairseqMultiModel` interface.

    During inference it is required to specify a single `--source-lang` and
    `--target-lang`, which indicates the inference langauge direction.
    `--lang-pairs`, `--encoder-langtok`, `--decoder-langtok` have to be set to
    the same value as training.
    """

    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)
        self.langs = args.langs
        self.dicts = dicts
        self.training = training
        if training:
            self.lang_pairs = args.lang_pairs
        else:
            self.lang_pairs = ["{}-{}".format(args.source_lang, args.target_lang)]
        # eval_lang_pairs for multilingual translation is usually all of the
        # lang_pairs. However for other multitask settings or when we want to
        # optimize for certain languages we want to use a different subset. Thus
        # the eval_lang_pairs class variable is provided for classes that extend
        # this class.
        self.eval_lang_pairs = self.lang_pairs
        # model_lang_pairs will be used to build encoder-decoder model pairs in
        # models.build_model(). This allows multitask type of sub-class can
        # build models other than the input lang_pairs
        self.model_lang_pairs = self.lang_pairs
        self.source_langs = [d.split("-")[0] for d in self.lang_pairs]
        self.target_langs = [d.split("-")[1] for d in self.lang_pairs]
        self.check_dicts(self.dicts, self.source_langs, self.target_langs)

        self.sampling_method = SamplingMethod.build_sampler(args, self)
        self.data_manager = MultilingualDatasetManager.setup_data_manager(
            args, self.lang_pairs, args.langs, dicts, self.sampling_method
        )

    def check_dicts(self, dicts, source_langs, target_langs):
        if self.args.source_dict is not None or self.args.target_dict is not None:
            # no need to check whether the source side and target side are sharing dictionaries
            return
        src_dict = dicts[source_langs[0]]
        tgt_dict = dicts[target_langs[0]]
        for src_lang in source_langs:
            assert (
                    src_dict == dicts[src_lang]
            ), "Diffrent dictionary are specified for different source languages; "
            "TranslationMultiSimpleEpochTask only supports one shared dictionary across all source languages"
        for tgt_lang in target_langs:
            assert (
                    tgt_dict == dicts[tgt_lang]
            ), "Diffrent dictionary are specified for different target languages; "
            "TranslationMultiSimpleEpochTask only supports one shared dictionary across all target languages"
    @classmethod
    def setup_task(cls, args, **kwargs):
        if args.langs is not None and isinstance(args.langs, str):
            args.langs = args.langs.split(",")
        langs, dicts, training = MultilingualDatasetManager.prepare(
            cls.load_dictionary, args, **kwargs
        )
        return cls(args, dicts, training)

    @staticmethod
    def add_args(parser):
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='inference source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='inference target language')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr',
                            action=FileContentsAction)
        parser.add_argument('--keep-inference-langtok', action='store_true',
                            help='keep language tokens in inference output (e.g. for analysis or debugging)')
        parser.add_argument('--family_type', default="original", help="using family cluster result")
        parser.add_argument("--core_langs", default="zh,en,de,fr")
        parser.add_argument("--reverse_lg_pairs", default=False)
        parser.add_argument("--async_save", default=False, type=bool)
        parser.add_argument("--branch_type", default=7, type=int)
        parser.add_argument("--only_two_branch", default=False, type=bool)
        parser.add_argument("--lg_adapter", default=False, type=bool)
        parser.add_argument("--lg_adapter_ckpt_path", default=None, type=str)
        try:
            parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                                help='max number of tokens in the source sequence')
            parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                                help='max number of tokens in the target sequence')
        except ArgumentError:
            # this might have already been defined. Once we transition this to hydra it should be fine to add it here.
            pass
        TransformerModel.add_args(parser)
        SamplingMethod.add_arguments(parser)
        MultilingualDatasetManager.add_args(parser)

    def has_sharded_data(self, split):
        return self.data_manager.has_sharded_data(split)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a dataset split."""
        shard_epoch = None
        if split in self.datasets:
            dataset = self.datasets[split]
            if self.has_sharded_data(split):
                if self.args.virtual_epoch_size is not None:
                    if dataset.load_next_shard:
                        shard_epoch = dataset.shard_epoch
                    else:
                        # no need to load next shard so skip loading
                        # also this avoid always loading from beginning of the data
                        return
                else:
                    shard_epoch = epoch
        else:
            # estimate the shard epoch from virtual data size and virtual epoch size
            shard_epoch = self.data_manager.estimate_global_pass_epoch(epoch)

        def language_pair_dataset(index):
            param_list =self.data_manager.get_split_data_param_list(
            split, epoch, shard_epoch=shard_epoch)[index]
            paths = utils.split_paths(self.args.data)
            assert len(paths) > 0
            data_path = paths[(epoch - 1) % len(paths)]
            param_list['data_path'] = data_path
            logger.info("current data path: %s" % param_list["data_path"])
            langpair_dataset = self.data_manager.load_a_dataset(
                combine=True,
                dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                **param_list
            )
            return langpair_dataset

        def get_core_lg_from_family(core_lg, lg, family_info):
            family_cluster = utils.FAMILY_DEFAULT
            if family_info.lower() == "family_7":
                family_cluster = utils.FAMILY_7
            elif family_info.lower() == "family_3":
                family_cluster = utils.FAMILY_3
            elif family_info.lower() == "english_centered_family":
                from cm2m_utils.english_centered_family_info import ENGLISH_CENTRAL_FAMILY
                family_cluster = ENGLISH_CENTRAL_FAMILY
            elif family_info.lower() == "x_x_family_8":
                from cm2m_utils.english_centered_family_info import X_X_FAMILY_8
                family_cluster = X_X_FAMILY_8

            if core_lg in family_cluster.keys():
                return lg in family_cluster[core_lg]
            else:
                return core_lg == lg

        def is_core_lg_data(lg, core_lg, family_type):

            if core_lg == 'low_resource':
                return lg in utils.LOW_RESOURCES_ZH
            elif "family" in core_lg:
                return get_core_lg_from_family(core_lg, lg, family_type)
            else:
                return lg == core_lg

        if "cur_core_lg" in kwargs and self.training:
            if kwargs["is_src"]:
                core_langs = [lg_pair for lg_pair in self.args.lang_pairs if is_core_lg_data(lg_pair.split("-")[0], kwargs["cur_core_lg"], self.args.family_type)]
            else:
                if self.args.reverse_lg_pairs:
                    core_langs = ["%s-%s" % (lg_pair.split("-")[1], lg_pair.split("-")[0]) for lg_pair in self.args.lang_pairs if is_core_lg_data(lg_pair.split("-")[0], kwargs["cur_core_lg"], self.args.family_type)]
                else:
                    core_langs = [lg_pair for lg_pair in self.args.lang_pairs if is_core_lg_data(lg_pair.split("-")[1], kwargs["cur_core_lg"], self.args.family_type)]
            assert len(core_langs) > 0
            self.data_manager.lang_pairs = core_langs

        elif split != "test":
            lgs_pairs = [lg_pair for lg_pair in self.args.lang_pairs if lg_pair.split("-")[0] != "trg"]
            self.data_manager.lang_pairs = lgs_pairs
        self.datasets[split] = self.data_manager.load_dataset(
        split,
        self.training,
        epoch=epoch,
        combine=combine,
        shard_epoch=shard_epoch,
        **kwargs,
        )


    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the multilingual_translation task is not supported"
            )

        src_data = ListDataset(src_tokens, src_lengths)
        dataset = LanguagePairDatasetForDemo(src_data, src_lengths, self.source_dictionary, shuffle=False)
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


    def build_generator(
            self,
            models,
            args,
            seq_gen_cls=None,
            extra_gen_cls_kwargs=None,
    ):
        if not getattr(args, "keep_inference_langtok", False):
            _, tgt_langtok_spec = self.args.langtoks["main"]
            if tgt_langtok_spec:
                tgt_lang_tok = self.data_manager.get_decoder_langtok(
                    self.args.target_lang, tgt_langtok_spec
                )
                extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
                extra_gen_cls_kwargs["symbols_to_strip_from_output"] = {tgt_lang_tok}

        return super().build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def inference_step(
            self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            _, tgt_langtok_spec = self.args.langtoks["main"]
            t = time.time()
            f.write('task-ckpt 1: ' + str(t) + '\n')
            f.flush()
            if not self.args.lang_tok_replacing_bos_eos:
                if prefix_tokens is None and tgt_langtok_spec:
                    tgt_lang_tok = self.data_manager.get_decoder_langtok(
                        self.args.target_lang, tgt_langtok_spec
                    )
                    src_tokens = sample["net_input"]["src_tokens"]
                    bsz = src_tokens.size(0)
                    prefix_tokens = (
                        torch.LongTensor([[tgt_lang_tok]]).expand(bsz, 1).to(src_tokens)
                    )
                t = time.time()
                f.write('task-ckpt 2: ' + str(t) + '\n')
                f.flush()
                return generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    constraints=constraints,
                )
            else:
                t = time.time()
                f.write('task-ckpt 3: ' + str(t) + '\n')
                f.flush()
                return generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    bos_token=self.data_manager.get_decoder_langtok(
                        self.args.target_lang, tgt_langtok_spec
                    )
                    if tgt_langtok_spec
                    else self.target_dictionary.eos(),
                )

    def build_model(self, args, from_checkpoint=False):
        def check_args():
            messages = []
            if (
                    len(set(self.args.lang_pairs).symmetric_difference(args.lang_pairs))
                    != 0
            ):
                messages.append(
                    "--lang-pairs should include all the language pairs {}.".format(
                        args.lang_pairs
                    )
                )
            # if self.args.encoder_langtok != args.encoder_langtok:
            #     messages.append(
            #         "--encoder-langtok should be {}.".format(args.encoder_langtok)
            #     )
            # if self.args.decoder_langtok != args.decoder_langtok:
            #     messages.append(
            #         "--decoder-langtok should {} be set.".format(
            #             "" if args.decoder_langtok else "not"
            #         )
            #     )

            if len(messages) > 0:
                raise ValueError(" ".join(messages))

        # Update args -> the fact that the constructor here
        # changes the args object doesn't mean you get the same one here
        self.update_args(args)

        # Check if task args are consistant with model args
        # check_args()

        from fairseq import models
        if not self.training:
            model = models.build_model(args, self, from_checkpoint)
        else:
            model = models.build_model(args, self, from_checkpoint)
            model_lang_pairs, lang_pairs = self.model_lang_pairs, self.lang_pairs
            for core_lg in self.args.core_langs.split(","):
                family_info = utils.get_lg_family_info(core_lg, self.args.family_type)
                model_dir = os.path.join(self.args.save_dir, family_info, "checkpoint0")
                cur_langs = ["main-%s" % family_info, "%s-m2m" % family_info]
                if self.args.only_two_branch:
                    cur_langs = ["%s-m2m" % family_info]
                if self.args.lg_adapter:
                    cur_langs = ["main-%s" % family_info]

                self.model_lang_pairs = [cur_langs]
                self.lang_pairs = [cur_langs]
                tmp_model = utils.reset_model_state_dict_(copy.deepcopy(model), core_lang=family_info, keep_main=True)
                if self.args.lg_adapter:
                    checkpoint_utils.save_model_partition(tmp_model, model_dir, cur_langs, overwrite=False,
                                                          is_lg_adapter=True)
                else:
                    checkpoint_utils.save_model_partition(tmp_model, model_dir, cur_langs + ["main-m2m"], False)
                logger.info("save %s model" % core_lg)

            self.model_lang_pairs = model_lang_pairs
            self.lang_pairs = lang_pairs

        if not isinstance(model, FairseqMultiModel):
            raise ValueError(
                "MultilingualTranslationTask requires a FairseqMultiModel architecture"
            )
        return model

    def _freeze_main_branch(self, model, prefix):
        for name, parameter in model.named_parameters():
            if prefix in name or "embed" in name:
                parameter.requires_grad = False

    def _recover_grad(self, model):
        for name, parameter in model.named_parameters():
                parameter.requires_grad = True

    def _per_lang_pair_train_loss(
            self, lang_pair, model, update_num, criterion, sample, optimizer, ignore_grad
    ):
        family_pair = utils.get_lg_family_pair(lang_pair, self.args.family_type)
        if lang_pair == "main-m2m":
            single_sample = sample[-1]  # the last sample just for main branch
        elif lang_pair.startswith("main"):
            single_sample = sample if self.args.lg_adapter else sample[1]# M-L
        elif lang_pair.endswith("m2m"):
            single_sample = sample[0]  # L-M
        else:
            raise ValueError("unknown lang pair:%s" % lang_pair)
        loss, sample_size, logging_output = criterion(
            model.models[family_pair], single_sample
        )
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def _per_lang_pair_valid_loss(self, lang_pair, model, criterion, sample):
        family_pair = utils.get_lg_family_pair(lang_pair, self.args.family_type)
        return criterion(model.models[family_pair], sample)

    def valid_step(self, sample, model, criterion):
        model.eval()
        lang_pair = self.model_lang_pairs[0]
        with torch.no_grad():
            from collections import defaultdict

            agg_loss, agg_sample_size, agg_logging_output = 0.0, 0.0, defaultdict(float)
            loss, sample_size, logging_output = self._per_lang_pair_valid_loss(
                lang_pair, model, criterion, sample
            )
            agg_loss += loss.data.item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[k] += logging_output[k]
                agg_logging_output[f"{lang_pair}:{k}"] += logging_output[k]
        return agg_loss, agg_sample_size, agg_logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        with metrics.aggregate():
            # pass 'sample_size', 'nsentences', 'ntokens' stats to fairseq_task
            super().reduce_metrics(logging_outputs, criterion)

        def reduce_metrics(logging_outputs, loss_key, nll_loss_key, ntokens_key, sample_size_key) -> None:
            """Aggregate logging outputs from data parallel training."""
            loss_sum = sum(log.get(loss_key, 0) for log in logging_outputs)
            nll_loss_sum = sum(log.get(nll_loss_key, 0) for log in logging_outputs)
            ntokens = sum(log.get(ntokens_key, 0) for log in logging_outputs)
            sample_size = sum(log.get(sample_size_key, 0) for log in logging_outputs)

            metrics.log_scalar(
                loss_key, loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_scalar(
                nll_loss_key, nll_loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl_%s" % nll_loss_key.split(":")[0], lambda meters: utils.get_perplexity(meters[nll_loss_key].avg)
            )


        def get_useful_keys():
            useful_set = []
            for k in logging_outputs[0].keys():
                if ":" in k:
                    lg_pair = k.split(":")[0]
                    useful_set.append(lg_pair)
            return useful_set

        lg_pairs = get_useful_keys()
        for lg_pair in lg_pairs:
            reduce_metrics(logging_outputs, "%s:loss" % lg_pair, "%s:nll_loss" % lg_pair,
                           "%s:ntokens" % lg_pair, "%s:sample_size" % lg_pair)

    def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        from collections import defaultdict

        agg_loss, agg_sample_size, agg_logging_output = 0.0, 0.0, defaultdict(float)
        # 1: main-lg
        # 2: lg-m2m
        # 4: main-m2m
        # 3: main-lg, lg-m2m
        # 5: main-lg, main-m2m
        # 6: lg-m2m, main-m2m
        # 7: main-lg, lg-m2m, main-m2m

        curr_lang_pairs = []
        if self.args.lg_adapter:
            curr_lang_pairs = self.model_lang_pairs
        else:
            if self.args.branch_type in [4, 5, 6, 7]:
                curr_lang_pairs.append("main-m2m")
            if not self.args.only_two_branch and self.args.branch_type in [1, 3, 5, 7]:
                main_lg = [lg for lg in self.model_lang_pairs if lg != "main-m2m" and lg.startswith("main")][0]
                curr_lang_pairs.append(main_lg)
            if not self.args.lg_adapter and self.args.branch_type in [2, 3, 6, 7]:
                lg_main = [lg for lg in self.model_lang_pairs if lg != "main-m2m" and lg.endswith("m2m")][0]
                curr_lang_pairs.append(lg_main)

        for idx, lang_pair in enumerate(curr_lang_pairs):
            def maybe_no_sync():
                if (
                        self.args.distributed_world_size > 1
                        and hasattr(model, "no_sync")
                        and idx < len(curr_lang_pairs) - 1
                ):
                    return model.no_sync()
                else:
                    return contextlib.ExitStack()  # dummy contextmanager

            with maybe_no_sync():
                loss, sample_size, logging_output = self._per_lang_pair_train_loss(
                    lang_pair,
                    model,
                    update_num,
                    criterion,
                    sample,
                    optimizer,
                    ignore_grad,
                )
            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[k] += logging_output[k]
                agg_logging_output[f"{lang_pair}:{k}"] += logging_output[k]
        return agg_loss, agg_sample_size, agg_logging_output

    def create_batch_sampler_func(
            self,
            max_positions,
            ignore_invalid_inputs,
            max_tokens,
            max_sentences,
            required_batch_size_multiple=1,
            seed=1,
    ):
        def construct_batch_sampler(dataset, epoch):
            splits = [
                s for s, _ in self.datasets.items() if self.datasets[s] == dataset
            ]
            split = splits[0] if len(splits) > 0 else None
            # NEW implementation
            if epoch is not None:
                # initialize the dataset with the correct starting epoch
                dataset.set_epoch(epoch)

            # get indices ordered by example size
            start_time = time.time()
            logger.info(f"start batch sampler: mem usage: {data_utils.get_mem_usage()}")

            with data_utils.numpy_seed(seed):
                indices = dataset.ordered_indices()
            logger.info(
                f"[{split}] @batch_sampler order indices time: {get_time_gap(start_time, time.time())}"
            )
            logger.info(f"mem usage: {data_utils.get_mem_usage()}")

            # filter examples that are too large
            if max_positions is not None:
                my_time = time.time()
                indices = self.filter_indices_by_size(
                    indices, dataset, max_positions, ignore_invalid_inputs
                )
                logger.info(
                    f"[{split}] @batch_sampler filter_by_size time: {get_time_gap(my_time, time.time())}"
                )
                logger.info(f"mem usage: {data_utils.get_mem_usage()}")

            # create mini-batches with given size constraints
            my_time = time.time()
            batch_sampler = dataset.batch_by_size(
                indices,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )

            logger.info(
                f"[{split}] @batch_sampler batch_by_size time: {get_time_gap(my_time, time.time())}"
            )
            logger.info(
                f"[{split}] per epoch batch_sampler set-up time: {get_time_gap(start_time, time.time())}"
            )
            logger.info(f"mem usage: {data_utils.get_mem_usage()}")

            return batch_sampler

        return construct_batch_sampler

        # we need to override get_batch_iterator because we want to reset the epoch iterator each time

    def get_batch_iterator(
            self,
            dataset,
            max_tokens=None,
            max_sentences=None,
            max_positions=None,
            ignore_invalid_inputs=False,
            required_batch_size_multiple=1,
            seed=1,
            num_shards=1,
            shard_id=0,
            num_workers=0,
            epoch=1,
            data_buffer_size=0,
            disable_iterator_cache=False,
            skip_remainder_batch=False,
            grouped_shuffling=False,
            disable_shuffling=False,
            update_epoch_batch_itr=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 0).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
            grouped_shuffling (bool, optional): group batches with each groups
                containing num_shards batches and shuffle groups. Reduces difference
                between sequence lengths among workers for batches sorted by length.
            update_epoch_batch_itr (bool optional): if true then donot use the cached
                batch iterator for the epoch

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        # initialize the dataset with the correct starting epoch
        assert isinstance(dataset, FairseqDataset)
        if dataset in self.dataset_to_epoch_iter:
            return self.dataset_to_epoch_iter[dataset]
        if self.args.sampling_method == "RoundRobin":
            batch_iter = super().get_batch_iterator(
                dataset,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                max_positions=max_positions,
                ignore_invalid_inputs=ignore_invalid_inputs,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed,
                num_shards=num_shards,
                shard_id=shard_id,
                num_workers=num_workers,
                epoch=epoch,
                data_buffer_size=data_buffer_size,
                disable_iterator_cache=disable_iterator_cache,
                skip_remainder_batch=skip_remainder_batch,
                update_epoch_batch_itr=update_epoch_batch_itr,
            )
            self.dataset_to_epoch_iter[dataset] = batch_iter
            return batch_iter

        construct_batch_sampler = self.create_batch_sampler_func(
            max_positions,
            ignore_invalid_inputs,
            max_tokens,
            max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
            seed=seed,
        )

        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=construct_batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            disable_shuffling=disable_shuffling
        )
        return epoch_iter
