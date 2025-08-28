#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import argparse
import ast
import copy
import logging
import os
from typing import Any, Dict, Iterator, List, Optional
from argparse import Namespace

import numpy as np
import torch
from omegaconf import open_dict
from torch import nn

# from fairseq import checkpoint_utils, tasks
from fairseq import utils
from fairseq.data import encoders
from fairseq.dataclass.configs import FairseqConfig
import time
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap

from fairseq_cli.demo_utils import content_detection

logger = logging.getLogger(__name__)
#global f
#f = open('generator_log.txt','w+')

def from_pretrained(
    model_name_or_path,
    checkpoint_file="model.pt",
    data_name_or_path=".",
    archive_map=None,
    **kwargs
):
    from fairseq import checkpoint_utils, file_utils

    if archive_map is not None:
        if model_name_or_path in archive_map:
            model_name_or_path = archive_map[model_name_or_path]
        if data_name_or_path is not None and data_name_or_path in archive_map:
            data_name_or_path = archive_map[data_name_or_path]

        # allow archive_map to set default arg_overrides (e.g., tokenizer, bpe)
        # for each model
        if isinstance(model_name_or_path, dict):
            for k, v in model_name_or_path.items():
                if k == "checkpoint_file":
                    checkpoint_file = v
                elif (
                    k != "path"
                    # only set kwargs that don't already have overrides
                    and k not in kwargs
                ):
                    kwargs[k] = v
            model_name_or_path = model_name_or_path["path"]

    model_path = file_utils.load_archive_file(model_name_or_path)

    # convenience hack for loading data and BPE codes from model archive
    if data_name_or_path.startswith("."):
        kwargs["data"] = os.path.abspath(os.path.join(model_path, data_name_or_path))
    else:
        kwargs["data"] = file_utils.load_archive_file(data_name_or_path)
    for file, arg in {
        "code": "bpe_codes",
        "bpecodes": "bpe_codes",
        "sentencepiece.bpe.model": "sentencepiece_model",
        "merges.txt": "bpe_merges",
        "vocab.json": "bpe_vocab",
    }.items():
        path = os.path.join(model_path, file)
        if os.path.exists(path):
            kwargs[arg] = path

    if "user_dir" in kwargs:
        utils.import_user_module(argparse.Namespace(user_dir=kwargs["user_dir"]))

    models, args, task = checkpoint_utils.load_model_ensemble_and_task(
        [os.path.join(model_path, cpt) for cpt in checkpoint_file.split(os.pathsep)],
        arg_overrides=kwargs,
    )

    return {
        "args": args,
        "task": task,
        "models": models,
    }


class GeneratorHubInterface(nn.Module):
    """
    PyTorch Hub interface for generating sequences from a pre-trained
    translation or language model.
    """

    def __init__(self, cfg, task, models):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.models = nn.ModuleList(models)
        self.src_dict = task.source_dictionary
        self.tgt_dict = task.target_dictionary

        # optimize model for generation
        for model in self.models:
            model.prepare_for_inference_(cfg)

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(cfg.generation.replace_unk)

        self.tokenizer = encoders.build_tokenizer(cfg.tokenizer)
        self.bpe = encoders.build_bpe(cfg.bpe)

        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(), *[model.max_positions() for model in models]
        )
        gen_args = copy.deepcopy(self.cfg.generation)
        with open_dict(gen_args):
            gen_args.beam = beam
            for k, v in kwargs.items():
                setattr(gen_args, k, v)
        self.generator = self.task.build_generator(
            self.models,
            gen_args,
            prefix_allowed_tokens_fn=None,
        )

        # this is useful for determining the device
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def translate(
        self, sentences: List[str], beam: int = 5, verbose: bool = False, **kwargs
    ) -> List[str]:
        return self.sample(sentences, beam, verbose, **kwargs)

    def sample(
        self, sentences: List[str], beam: int = 1, verbose: bool = False, **kwargs
    ) -> List[str]:
        if isinstance(sentences, str):
            return self.sample([sentences], beam=beam, verbose=verbose, **kwargs)[0]
        tokenized_sentences = [self.encode(sentence) for sentence in sentences]
        batched_hypos = self.generate(tokenized_sentences, beam, verbose, **kwargs)
        return [self.decode(hypos[0]["tokens"]) for hypos in batched_hypos]

    def score(
        self, sentences: List[str], replace_newline_with_eos: bool = False, **kwargs
    ):
        if isinstance(sentences, str):
            return self.score(
                [sentences], replace_newline_with_eos=replace_newline_with_eos, **kwargs
            )[0]

        def encode(sentence):
            if replace_newline_with_eos:
                return torch.cat([self.encode(line) for line in sentence.splitlines()])
            else:
                return self.encode(sentence)

        # NOTE: this doesn't support translation tasks currently
        tokenized_sentences = [encode(sentence) for sentence in sentences]
        return [
            hypos[0]
            for hypos in self.generate(
                tokenized_sentences, score_reference=True, **kwargs
            )
        ]

    def generate(
        self,
        tokenized_sentences: List[torch.LongTensor],
        beam: int = 5,
        verbose: bool = False,
        skip_invalid_size_inputs=False,
        inference_step_args=None,
        prefix_allowed_tokens_fn=None,
        **kwargs
    ) -> List[List[Dict[str, torch.Tensor]]]:
        if torch.is_tensor(tokenized_sentences) and tokenized_sentences.dim() == 1:
            return self.generate(
                tokenized_sentences.unsqueeze(0), beam=beam, verbose=verbose, **kwargs
            )[0]

        # build generator using current args as well as any kwargs
        #gen_args = copy.deepcopy(self.cfg.generation)
        #with open_dict(gen_args):
        #    gen_args.beam = beam
        #    for k, v in kwargs.items():
        #        setattr(gen_args, k, v)
        #generator = self.task.build_generator(
        #    self.models,
        #    gen_args,
        #    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        #)

        inference_step_args = inference_step_args or {}
        results = []
        for batch in self._build_batches(tokenized_sentences, skip_invalid_size_inputs):
            batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
            translations = self.task.inference_step(
                self.generator, self.models, batch, **inference_step_args
            )
            for id, hypos in zip(batch["id"].tolist(), translations):
                results.append((id, hypos))

        # sort output to match input order
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]

        if verbose:

            def getarg(name, default):
                return getattr(gen_args, name, getattr(self.cfg, name, default))

            for source_tokens, target_hypotheses in zip(tokenized_sentences, outputs):
                src_str_with_unk = self.string(source_tokens)
                logger.info("S\t{}".format(src_str_with_unk))
                for hypo in target_hypotheses:
                    hypo_str = self.decode(hypo["tokens"])
                    logger.info("H\t{}\t{}".format(hypo["score"], hypo_str))
                    logger.info(
                        "P\t{}".format(
                            " ".join(
                                map(
                                    lambda x: "{:.4f}".format(x),
                                    hypo["positional_scores"].tolist(),
                                )
                            )
                        )
                    )
                    if hypo["alignment"] is not None and getarg(
                        "print_alignment", False
                    ):
                        logger.info(
                            "A\t{}".format(
                                " ".join(
                                    [
                                        "{}-{}".format(src_idx, tgt_idx)
                                        for src_idx, tgt_idx in hypo["alignment"]
                                    ]
                                )
                            )
                        )
        return outputs

    def encode(self, sentence: str) -> torch.LongTensor:
        sentence = self.tokenize(sentence)
        sentence = self.apply_bpe(sentence)
        return self.binarize(sentence)

    def decode(self, tokens: torch.LongTensor) -> str:
        sentence = self.string(tokens)
        sentence = self.remove_bpe(sentence)
        return self.detokenize(sentence)

    def tokenize(self, sentence: str) -> str:
        if self.tokenizer is not None:
            sentence = self.tokenizer.encode(sentence)
        return sentence

    def detokenize(self, sentence: str) -> str:
        if self.tokenizer is not None:
            sentence = self.tokenizer.decode(sentence)
        return sentence

    def apply_bpe(self, sentence: str) -> str:
        if self.bpe is not None:
            sentence = self.bpe.encode(sentence)
        return sentence

    def remove_bpe(self, sentence: str) -> str:
        if self.bpe is not None:
            sentence = self.bpe.decode(sentence)
        return sentence

    def binarize(self, sentence: str) -> torch.LongTensor:
        return self.src_dict.encode_line(sentence, add_if_not_exist=False).long()

    def string(self, tokens: torch.LongTensor) -> str:
        return self.tgt_dict.string(tokens)

    def _build_batches(
        self, tokens: List[List[int]], skip_invalid_size_inputs: bool
    ) -> Iterator[Dict[str, Any]]:
        lengths = torch.LongTensor([t.numel() for t in tokens])
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.build_dataset_for_inference(tokens, lengths),
            max_tokens=self.cfg.dataset.max_tokens,
            max_sentences=self.cfg.dataset.batch_size,
            max_positions=self.max_positions,
            ignore_invalid_inputs=skip_invalid_size_inputs,
            disable_iterator_cache=True,
        ).next_epoch_itr(shuffle=False)
        return batch_iterator


class BPEHubInterface(object):
    """PyTorch Hub interface for Byte-Pair Encoding (BPE)."""

    def __init__(self, bpe, **kwargs):
        super().__init__()
        args = argparse.Namespace(bpe=bpe, **kwargs)
        self.bpe = encoders.build_bpe(args)
        assert self.bpe is not None

    def encode(self, sentence: str) -> str:
        return self.bpe.encode(sentence)

    def decode(self, sentence: str) -> str:
        return self.bpe.decode(sentence)


class TokenizerHubInterface(object):
    """PyTorch Hub interface for tokenization."""

    def __init__(self, tokenizer, **kwargs):
        super().__init__()
        args = argparse.Namespace(tokenizer=tokenizer, **kwargs)
        self.tokenizer = encoders.build_tokenizer(args)
        assert self.tokenizer is not None

    def encode(self, sentence: str) -> str:
        return self.tokenizer.encode(sentence)

    def decode(self, sentence: str) -> str:
        return self.tokenizer.decode(sentence)



class GeneratorInterface:
    """
    PyTorch Hub interface for generating sequences from a pre-trained
    translation or language model.
    """

    def __init__(self, cfg: FairseqConfig):
        self.cfg = cfg
        self.tokenizer = encoders.build_tokenizer(cfg.tokenizer)
        if isinstance(self.cfg, Namespace):
            self.cfg = convert_namespace_to_omegaconf(self.cfg)

    def load_model(self):
        from fairseq import checkpoint_utils, tasks
        utils.import_user_module(self.cfg.common)

        # Fix seed for stochastic decoding
        if (
            self.cfg.common.seed is not None
            and not self.cfg.generation.no_seed_provided
        ):
            np.random.seed(self.cfg.common.seed)
            utils.set_torch_seed(self.cfg.common.seed)

        # Setup task, e.g., translation
        task = tasks.setup_task(self.cfg.task)

        def _build_model(cfg, task):
            model = task.build_model(cfg.model).half().cuda()
            model.make_generation_fast_()
            return fsdp_wrap(model)

        # Load the model
        overrides = ast.literal_eval(self.cfg.common_eval.model_overrides)
        logger.info("loading model(s) from {}".format(self.cfg.common_eval.path))
        with fsdp_enable_wrap(
            self.cfg.distributed_training,
            #use_sharded_state=self.cfg.distributed_training.use_sharded_state,
        ):
            #from fairseq.pdb import pdb; pdb.set_trace()
            models, _model_args, _task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(self.cfg.common_eval.path),
            arg_overrides=overrides,
            task=task,
            suffix=self.cfg.checkpoint.checkpoint_suffix,
            strict=(self.cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=self.cfg.checkpoint.checkpoint_shard_count,
            #build_model_hook=_build_model,
            )
        # Set dictionaries
        src_dict = task.source_dictionary
        tgt_dict = task.target_dictionary

        # Handle tokenization and BPE
        bpe = task.build_bpe(self.cfg.bpe)

        # Set state
        self.bpe = bpe
        self.task = task
        self.models = models
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        return models


    def build_generator(self):
        logger.info(f"Preparing generator with settings {self.cfg.generation}")
        src = 'zh'
        tar = 'en'
        self.task.args.source_lang = src
        self.task.args.target_lang = tar
        self.task.lang_pairs = ["%s-%s" % (src, tar)]
        self.task.source_langs = [src]
        self.task.target_langs = [tar]
        self.task.eval_lang_pairs = ["%s-%s" % (src, tar)]
        self.task.model_lang_pairs = ["%s-%s" % (src, tar)]
        # just used for shared dicts
        shared_dict = list(self.task.data_manager.dicts.values())[0]
        self.task.data_manager.dicts = {
            src: shared_dict,
            tar: shared_dict,
        }
        self.generator = self.task.build_generator(
                self.models,
                self.cfg.generation,
        )
    def generate(
        self,
        inputs: List[List[int]],
        min_tokens: List[int] = None,
        max_tokens: List[int] = None,
        temperature: float = 1.0,
        top_p: float = -1.0,
        logprobs: int = 0,
        n: int = 1,
        best_of: Optional[int] = None,
        echo: bool = False,
        stop: Optional[List[int]] = None,
        seed: Optional[int] = None,
        use_cuda: bool = True,
        srcs:[str]=None,
        tars:[str]=None,
        sample_ids: [int] = None,
        content_detection_flag = True
    ):
        """
        Generate from sequences.
        Parameters match those of the OpenAI API.
        https://beta.openai.com/docs/api-reference/completions/create
        inputs: a list of pre-tokenized prompts
        min_tokens: blocks EOS until at least this many tokens is provided
        max_tokens: forces EOS after this many tokens
        temperature: softmax temperature
        top_p: nucleus probability
        log_probs: return this cutoff of the probability distribution
        n: beam size
        best_of: number of beams to return. must be <= n
        echo: if true, returned text/tokens/scores includes the prompt.
            This is useful for getting PPL evaluations.
        stop: a list of terminating tokens
        seed: an integer if desired
        use_cuda: should we use GPUs.
        """

        def get_symbols_to_strip_from_output(generator):
            if hasattr(generator, "symbols_to_strip_from_output"):
                return generator.symbols_to_strip_from_output
            else:
                return {generator.eos}

        def decode_fn(x):
            if self.bpe is not None:
                x = self.bpe.decode(x)
            if self.tokenizer is not None:
                x = self.tokenizer.decode(x)
            return x

        if seed:
            utils.set_torch_seed(seed)
        total_generation_time = 0

        # Initialize generator
        if not best_of:
            best_of = n
        assert best_of >= n

        retval = []

        def encoder_line(x):
            return x
        src, tar = srcs[0], tars[0]
        if tar == "zho_cn":
            tar = "zh"
        assert src is not None, tar is not None
        self.task.args.source_lang = src
        self.task.args.target_lang = tar
        self.task.lang_pairs = ["%s-%s" % (src, tar)]
        self.task.source_langs = [src]
        self.task.target_langs = [tar]
        self.task.eval_lang_pairs = ["%s-%s" % (src, tar)]
        self.task.model_lang_pairs = ["%s-%s" % (src, tar)]
        # just used for shared dicts
        shared_dict = list(self.task.data_manager.dicts.values())[0]
        self.task.data_manager.dicts = {
            src: shared_dict,
            tar: shared_dict,
        }
        # inputs:[['sent1','sent2','sent3']]  org inputs:[['whole_sent']]
        logger.info("begin")
        emb_inputs = []
        for i, input in enumerate(inputs):
            emb_input = "__%s__ " % srcs[i] + input[0]
            emb_input = emb_input + " __%s__" % tars[i]
            emb_inputs.append(emb_input)
        tokens, lengths = self.task.get_interactive_tokens_and_lengths(emb_inputs, encoder_line)
        #
        #tokens, lengths = [], []
        #for input in inputs:
        #    tmp_t, tmp_l = self.task.get_interactive_tokens_and_lengths(input, encoder_line)
        #     tokens.extend(tmp_t)
        #    lengths.extend(tmp_l)
        # print("tokens ", tokens)
        # print("lengths ", lengths)
        # print("tokens [0]", self.task.get_interactive_tokens_and_lengths(inputs[0], encoder_line))

        max_positions = utils.resolve_max_positions(
            self.task.max_positions(), *[model.max_positions() for model in self.models]
        )
        #f.write("tokens:" + str(tokens)+'\n')
        # tokens:[tensor([ 94124, 248079,  15697, 248075,      2]), tensor([ 94124, 248079,  15697, 248075,      2]), tensor([ 94124, 248079,  15697, 248075,      2]), tensor([   117,  13121,  12620, 248203,      2])]
        #f.flush()

        # NOTE: make all sentences in the same batch
        # self.cfg.dataset.max_tokens=512
        _, tgt_langtok_spec = self.task.args.langtoks["main"]
        #self.cfg.dataset.max_tokens=2048
        batches = self.task.get_batch_iterator(
            dataset=self.task.build_dataset_for_inference(tokens, lengths),
            max_tokens=self.cfg.dataset.max_tokens,
            max_sentences=self.cfg.dataset.batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=self.cfg.dataset.skip_invalid_size_inputs_valid_test,
            disable_shuffling=True
        ).next_epoch_itr(shuffle=False)
        print("sample ids", sample_ids)
        for batch in batches:

            net_input_tokens = batch["net_input"]["src_tokens"]

            previous_tokens = net_input_tokens[:, -2:-1]
            src_tokens = torch.cat([net_input_tokens[:, :-2], net_input_tokens[:, -1:]], dim=-1)
            src_lengths = batch["net_input"]["src_lengths"] - 1

            batch["net_input"]["src_tokens"] = src_tokens
            batch["net_input"]["src_lengths"] = src_lengths

            # okay actually generate
            logger.info(f"Executing generation on input tensor size {src_tokens.shape}")
            if use_cuda:
                batch = utils.move_to_cuda(batch)
                previous_tokens = utils.move_to_cuda(previous_tokens)

            translate_start_time = time.time()
            # NOTE: The src order only supports batch = 1 condition
            src_order = batch["id"]
            invert_order = [0] * len(src_order)
            for i, s in enumerate(src_order):
                invert_order[s] = i

            translations = self.task.inference_step(self.generator, self.models, batch, prefix_tokens=previous_tokens)
            logger.info("compete inferenfe")
            translate_time = time.time() - translate_start_time
            total_generation_time += translate_time

            # possibly cut off any bsz padding we did
            hypos = translations[: len(inputs)]
            # actually turn everything into strings
            align_dict = utils.load_align_dict(self.cfg.generation.replace_unk)
            # with open('decode_str.txt', 'a+') as f:
            #     f.writelines('come to generate in  GeneratorInterface')
            #     f.writelines('hypos'+str(hypos))

            prev_len = len(self.tgt_dict)

            for i, hyps in enumerate(hypos):
                beams = []
                for hypo in hyps[:1]:
                    #print("hypo: ", hypo)
                    if not isinstance(hypo, dict):
                        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo.int().cpu(),
                        src_str=batch["net_input"]["src_tokens"][i].int().cpu(),
                        alignment=True,
                        align_dict=align_dict,
                        tgt_dict=self.tgt_dict, # copy.deepcopy(self.tgt_dict),
                        remove_bpe=self.cfg.common_eval.post_process,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.generator),
                        )
                    else:
                        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo["tokens"].int().cpu(),
                        src_str=batch["net_input"]["src_tokens"][i].int().cpu(),
                        alignment=hypo["alignment"],
                        align_dict=align_dict,
                        tgt_dict=self.tgt_dict, # copy.deepcopy(self.tgt_dict),
                        remove_bpe=self.cfg.common_eval.post_process,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.generator),
                        )
                    detok_hypo_str=hypo_str
                    # detok_hypo_str = decode_fn(hypo_str)
                    #score = hypo["score"] / math.log(2)
                    text = re.sub(r"__(\w+)(-*)(_*)(\w*)__", '', detok_hypo_str)
                    if content_detection_flag:
                        # logger.info("content_detection")
                        suggestion = content_detection(text)
                        if suggestion == "block":
                            text = "您输入的内容中存在某些词汇，xxx~"
                    text = re.sub(r"(.{2,}?)\1+", r"\1", text)
                    text = text.replace("<unk>", "")
                    result = {
                        "text":text,
                        #"tokens": hypo["tokens"].tolist(),
                        # text offset is useful for cutting off prompts or prefixes
                        # or evaluating PPL on just a subset of tokens
                        # "text_offset": token_offsets,
                        #"token_scores": score.tolist(),
                    }
                    beams.append(result)
                retval.append(beams)

            for sym in self.tgt_dict.symbols[prev_len:]:
                del self.tgt_dict.indices[sym]
            self.tgt_dict.symbols = self.tgt_dict.symbols[:prev_len]
            self.tgt_dict.count = self.tgt_dict.count[:prev_len]
            print("invert order", invert_order)
            retval_new = [retval[i] for i in invert_order]
            retval = retval_new

        print('retval:', retval)
        #logger.info("return")
        #f.write('retval:'+str(retval) + '\n')
        #f.flush()
        return retval

    @staticmethod
    def _filter_special(
        tokens: List[int],
        scores: List[float],
        distributions,
        pad_token: int = 1,
    ):
        """
        Cut off tokens after finding a special tokens.
        """

        # tokens is a 1D list of token IDs of length seqlen
        # scores is a 1D list of log-probability scores for those tokens (length seqlen)
        # distributions (optional) is a seqlen x vocab_size tensor corresponding to
        # the full distribution of predictions at each timestep

        output = []
        mask = []
        for t, s in zip(tokens, scores):
            if t == pad_token:
                # simply skip pads
                mask.append(False)
                continue
            if t <= 3:
                # and other special tokens should end things
                break
            mask.append(True)
            output.append((t, s))
        new_tokens, new_scores = zip(*output)

        # cut off at stop and drop pads
        if distributions is not None:
            distributions = distributions[: len(mask)][mask]
            distributions = distributions[: len(output)]
        return new_tokens, new_scores, distributions
