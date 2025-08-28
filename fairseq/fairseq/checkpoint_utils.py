# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import collections
import contextlib
import inspect
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import gc
import random
import re
import time
from datetime import datetime
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from fairseq.data import data_utils, EpochBatchIterator
from fairseq.dataclass.configs import CheckpointConfig
from fairseq.dataclass.utils import (
    convert_namespace_to_omegaconf,
    overwrite_args_by_name,
)
from fairseq.distributed.fully_sharded_data_parallel import FSDP, has_FSDP
from fairseq.file_io import PathManager
from fairseq.models import FairseqDecoder, FairseqEncoder
from omegaconf import DictConfig, OmegaConf, open_dict

from fairseq import utils
from cm2m_utils.ceph_file_util import CEPHFileUtil

logger = logging.getLogger(__name__)
try:
    ceph_util = CEPHFileUtil()
except:
    logger.warning("no ceph manager")
    ceph_util = None


def save_checkpoint(cfg: CheckpointConfig, trainer, epoch_itr, val_loss, end_epoch_by_user=False):
    from fairseq import meters

    # only one worker should attempt to create the required dir
    if trainer.data_parallel_rank == 0:
        if ceph_util is None:
            os.makedirs(cfg.save_dir, exist_ok=True)
        else:
            ceph_util.make_dirs(cfg.save_dir, exist_ok=True)

    prev_best = getattr(save_checkpoint, "best", val_loss)
    if val_loss is not None:
        best_function = max if cfg.maximize_best_checkpoint_metric else min
        save_checkpoint.best = best_function(val_loss, prev_best)

    if cfg.no_save:
        return

    trainer.consolidate_optimizer()  # TODO(SS): do we need this if no_save_optimizer_state

    if not trainer.should_save_checkpoint_on_current_rank:
        if trainer.always_call_state_dict_during_save_checkpoint:
            trainer.state_dict()
        return

    write_timer = meters.StopwatchMeter()
    write_timer.start()

    epoch = epoch_itr.epoch if isinstance(epoch_itr, EpochBatchIterator) else epoch_itr["epoch"]
    end_flag = epoch_itr.end_of_epoch() if isinstance(epoch_itr, EpochBatchIterator) else epoch_itr["end_of_epoch"]
    end_of_epoch = end_flag if not end_epoch_by_user else end_epoch_by_user
    updates = trainer.get_num_updates()

    logger.info(f"Preparing to save checkpoint for epoch {epoch} @ {updates} updates")

    def is_better(a, b):
        return a >= b if cfg.maximize_best_checkpoint_metric else a <= b

    suffix = trainer.checkpoint_suffix
    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds["checkpoint{}{}.pt".format(epoch, suffix)] = (
        end_of_epoch and not cfg.no_epoch_checkpoints and epoch % cfg.save_interval == 0
    )
    checkpoint_conds["checkpoint_{}_{}{}.pt".format(epoch, updates, suffix)] = (
        not end_of_epoch
        and cfg.save_interval_updates > 0
        and updates % cfg.save_interval_updates == 0
    )
    checkpoint_conds["checkpoint_best{}.pt".format(suffix)] = val_loss is not None and (
        not hasattr(save_checkpoint, "best")
        or is_better(val_loss, save_checkpoint.best)
    )
    if val_loss is not None and cfg.keep_best_checkpoints > 0:
        worst_best = getattr(save_checkpoint, "best", None)
        chkpts = checkpoint_paths(
            cfg.save_dir,
            pattern=r"checkpoint\.best_{}_(\d+\.?\d*){}\.pt".format(
                cfg.best_checkpoint_metric, suffix
            ),
        )
        if len(chkpts) > 0:
            p = chkpts[-1] if cfg.maximize_best_checkpoint_metric else chkpts[0]
            worst_best = float(p.rsplit("_")[-1].replace("{}.pt".format(suffix), ""))
        # add random digits to resolve ties
        with data_utils.numpy_seed(epoch, updates, val_loss):
            rand_sfx = np.random.randint(0, cfg.keep_best_checkpoints)

        checkpoint_conds[
            "checkpoint.best_{}_{:.3f}{}{}.pt".format(
                cfg.best_checkpoint_metric, val_loss, rand_sfx, suffix
            )
        ] = worst_best is None or is_better(val_loss, worst_best)
    checkpoint_conds[
        "checkpoint_last{}.pt".format(suffix)
    ] = not cfg.no_last_checkpoints

    epoch_state = epoch_itr.state_dict() if isinstance(epoch_itr, EpochBatchIterator) else epoch_itr["state_dict"]
    extra_state = {"train_iterator": epoch_state, "val_loss": val_loss}
    if hasattr(save_checkpoint, "best"):
        extra_state.update({"best": save_checkpoint.best})

    checkpoints = [
        os.path.join(cfg.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond
    ]
    if len(checkpoints) > 0 and trainer.should_save_checkpoint_on_current_rank:
        trainer.save_checkpoint(checkpoints[0], extra_state)
        for cp in checkpoints[1:]:
            if cfg.write_checkpoints_asynchronously:
                # TODO[ioPath]: Need to implement a delayed asynchronous
                # file copying/moving feature.
                logger.warning(
                    f"ioPath is not copying {checkpoints[0]} to {cp} "
                    "since async write mode is on."
                )
            else:
                assert PathManager.copy(
                    checkpoints[0], cp, overwrite=True
                ), f"Failed to copy {checkpoints[0]} to {cp}"

        write_timer.stop()
        logger.info(
            "Saved checkpoint {} (epoch {} @ {} updates, score {}) (writing took {} seconds)".format(
                checkpoints[0], epoch, updates, val_loss, write_timer.sum
            )
        )

    def remove_obj(old_chk):
        import shutil
        if os.path.lexists(old_chk):
            shutil.rmtree(old_chk) if os.path.isdir(old_chk) else os.remove(old_chk)
        elif PathManager.exists(old_chk):
            PathManager.rm(old_chk)

    if not end_of_epoch and cfg.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        if cfg.keep_interval_updates_pattern == -1:
            checkpoints = checkpoint_paths(
                cfg.save_dir, pattern=r"checkpoint_\d+_(\d+){}\.pt".format(suffix)
            )
        else:
            checkpoints = checkpoint_paths(
                cfg.save_dir,
                pattern=r"checkpoint_\d+_(\d+){}\.pt".format(suffix),
                keep_match=True,
            )
            checkpoints = [
                x[0]
                for x in checkpoints
                if x[1] % cfg.keep_interval_updates_pattern != 0
            ]

        for old_chk in checkpoints[cfg.keep_interval_updates :]:
            remove_obj(old_chk)
            # remove "checkpoint1" directory
            remove_obj(old_chk[:-3])  # remove ".pt"

    if cfg.keep_last_epochs > 0:
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_paths(
            cfg.save_dir, pattern=r"checkpoint(\d+){}\.pt".format(suffix)
        )
        for old_chk in checkpoints[cfg.keep_last_epochs :]:
            remove_obj(old_chk)
            # remove "checkpoint1" directory
            if cfg.keep_last_epochs != 1:
                remove_obj(old_chk[:-3])# remove ".pt"

    if cfg.keep_best_checkpoints > 0:
        # only keep the best N checkpoints according to validation metric
        checkpoints = checkpoint_paths(
            cfg.save_dir,
            pattern=r"checkpoint\.best_{}_(\d+\.?\d*){}\.pt".format(
                cfg.best_checkpoint_metric, suffix
            ),
        )
        if not cfg.maximize_best_checkpoint_metric:
            checkpoints = checkpoints[::-1]
        for old_chk in checkpoints[cfg.keep_best_checkpoints :]:
            remove_obj(old_chk)
            # remove "checkpoint1" directory
            remove_obj(old_chk[:-3]) # remove ".pt"


ENCODER_FILE_NAME = "encoder_embed.pt"
DECODER_FILE_NAME = "decoder_embed.pt"
EMBEDDING_META_FILE = "embedding_meta_file.json"
import json


def _save_obj(obj, path, async_save, executor, p_list):
    if async_save:
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            p = executor.submit(torch_persistent_save, obj, path)
            p_list.append(p)
        else:
            p = executor.submit(torch_persistent_save, obj, path)
            p_list.append(p)
    else:
        torch_persistent_save(obj, path)


def _save_embedding(model_state_dict, save_dir, overwrite, is_lg_adapter, async_save, executor, save_process_list):
    if not overwrite and PathManager.exists(os.path.join(save_dir, ENCODER_FILE_NAME)):
        return
    encoder_embed_dict, decoder_embed_dict = {}, {}
    for key, value in model_state_dict.items():
        key_name = ".".join(key.split(".")[2:])  # drop model.src_lg-tgt_lg info
        if "main-m2m" in key or is_lg_adapter:
            if "embed" in key and "encoder" in key:
                encoder_embed_dict[key_name] = value
            elif "embed" in key and "decoder" in key:
                decoder_embed_dict[key_name] = value
    _save_obj(encoder_embed_dict, os.path.join(save_dir, ENCODER_FILE_NAME), async_save, executor, save_process_list)
    _save_obj(decoder_embed_dict, os.path.join(save_dir, DECODER_FILE_NAME), async_save, executor, save_process_list)
    embedding_meta_info = {
        "time": datetime.now().timestamp(),
        "current_path": save_dir
    }
    parent_dir = os.path.dirname(save_dir)
    _save_obj(json.dumps(embedding_meta_info), os.path.join(parent_dir, EMBEDDING_META_FILE), async_save, executor, save_process_list)
    logger.info("save embedding info in %s" % save_dir)


def _save_layers(model_state_dict, lg_families, save_dir, type, overwrite, async_save, executor, save_process_list):
    assert type in ["encoder.", "decoder."], "unsupported layer type"
    for family_pair in lg_families:
        res_state_dict = {}
        lg_name = family_pair.split("-")[0] if type == "encoder." else family_pair.split("-")[1]
        save_file_name = os.path.join(save_dir, "%s_%s.pt" % (type[:-1], lg_name))
        if not overwrite and PathManager.exists(save_file_name):
            continue
        for key, value in model_state_dict.items():
            if family_pair in key and type in key and "embed" not in key:
                new_key = ".".join(key.split(".")[2:])  # drop model.src_lg-tgt_lg info
                res_state_dict[new_key] = value
        _save_obj(res_state_dict, save_file_name, async_save, executor, save_process_list)
        logger.info("save %s info in %s" % (type, save_file_name))


# just for multilingual transformer - multi branch
def save_model_partition(model, save_dir, lg_families, overwrite=True, is_lg_adapter=False, async_save=True):
    with ThreadPoolExecutor(max_workers=2) as executor:
        p_list = []

        # assert isinstance(model, FairseqMultiModel)
        if "s3://" not in save_dir:
            os.makedirs(save_dir, exist_ok=True)
        state_dict = utils.move_to_cpu(model.state_dict())
        _save_embedding(state_dict, save_dir,  overwrite, is_lg_adapter, async_save, executor, p_list)
        _save_layers(state_dict, lg_families, save_dir, "encoder.", overwrite, async_save, executor, p_list)
        _save_layers(state_dict, lg_families, save_dir, "decoder.", overwrite, async_save, executor, p_list)
        while not all([p.done() for p in p_list]):
            time.sleep(30)
        logging.info("save thread pool closed.")


def check_specific_model_exists(model_dir, lang, is_two_branch=False):
    encoder_layers_path = os.path.join(model_dir, "%s_%s.pt" % ("encoder", lang))
    decoder_layers_path = os.path.join(model_dir, "%s_%s.pt" % ("decoder", lang))
    res = []
    encoder_exits = PathManager.exists(encoder_layers_path)
    res.append(encoder_exits)
    if not is_two_branch:
        decoder_exits = PathManager.exists(decoder_layers_path)
        res.append(decoder_exits)
    return all(res)


def _get_one_branch_parameter(state_dicts,lg_pair, encoder, decoder):
    res_dict = {}
    for state_dict in state_dicts:
        for key, value in state_dict.items():
            if key.startswith("encoder") and encoder:
                new_key = "models.%s.%s" % (lg_pair, key)
            elif key.startswith("decoder") and decoder:
                new_key = "models.%s.%s" % (lg_pair, key)
            else:
                continue
            res_dict[new_key] = value
    return res_dict


def _generate_required_state(state_dicts, lg_pair, reload_one_branch, reload_lg_specific_branch):
    if reload_one_branch:
        return _get_one_branch_parameter(state_dicts,lg_pair, encoder=True, decoder=True)

    if reload_lg_specific_branch:
        res_dict = {}
        # main-L
        res_dict.update(_get_one_branch_parameter(state_dicts, "main-%s" % lg_pair, encoder=False, decoder=True))
        # L-m2m
        res_dict.update(_get_one_branch_parameter(state_dicts, "%s-m2m" % lg_pair, encoder=True, decoder=False))
        return res_dict


def _update_version(required_state, keys, value):
    for key in keys:
        required_state['models.%s.encoder.version' % key] = value
        required_state['models.%s.decoder.version' % key] = value
    return required_state


def reload_branch_model(model, model_dir, lg_pair, reload_one_branch, reload_lg_specific_branch, reload_embedding,
                        reset_main=False, is_two_branch=False, is_lg_adapter=False):

    def _get_reset_model(model, lg_pair):
        # for example: main-en, only one branch
        if reload_one_branch:
            model = utils.reset_model_state_dict_(model, lg_pair=lg_pair)


        # for example: main-en, en-m2m, keep main-m2m branch
        if reload_lg_specific_branch:
            core_lang = lg_pair
            model = utils.reset_model_state_dict_(model, core_lang=core_lang, keep_main=True)

        return model

    def _get_ckpt_list():
        ckpt_list = None
        # for example: main-en, only one branch
        if reload_one_branch:
            ckpt_list = _reload_one_branch_state(model_dir, lg_pair, reload_embedding, is_lg_adapter)

        # for example: main-en, en-m2m, keep main-m2m branch
        if reload_lg_specific_branch:
            core_lang = lg_pair
            ckpt_list = _reload_all_lg_specific_branch(model_dir, core_lang, reload_embedding, is_two_branch)

        assert ckpt_list is not None
        return ckpt_list

    version_value = [v for k, v in model.state_dict().items() if ".version" in k][0]
    model = _get_reset_model(model, lg_pair)
    ckpt_list = _get_ckpt_list()
    if reset_main:
        ckpt_main_branch = _reload_one_branch_state(model_dir, "main-m2m", reload_embedding=True, is_lg_adapter=False)
        ckpt_list.extend(ckpt_main_branch)
    required_state = _generate_required_state(ckpt_list, lg_pair, reload_one_branch, reload_lg_specific_branch)
    # update version
    _update_version(required_state, model.keys, version_value)
    logger.info("load state dict ...")
    model.load_state_dict(required_state, strict=False)

    if hasattr(model, "module"):
        model.module.module.load_state_dict(required_state, strict=False)


def _reload_one_branch_state(model_dir, lg_pair, reload_embedding, is_lg_adapter):
    old_src, old_tgt = lg_pair.split("-")
    ckpt_list = []
    encoder_layer = _load_encoder_layers(model_dir, old_src)
    ckpt_list.append(encoder_layer)
    if is_lg_adapter:
        decoder_layer = _load_decoder_layers(model_dir, "m2m")
        ckpt_list.append(decoder_layer)
    else:
        decoder_layer = _load_decoder_layers(model_dir, old_tgt)
        ckpt_list.append(decoder_layer)
    if reload_embedding:
        embeddings = _load_multi_branch_embedding(model_dir)
        ckpt_list.extend(embeddings)
    return ckpt_list


def _reload_all_lg_specific_branch(model_dir, core_lang, reload_embedding, is_two_branch):
    ckpt_list = []
    encoder_layer = _load_encoder_layers(model_dir, core_lang)
    if is_two_branch:
        ckpt_list.extend([encoder_layer])
    else:
        decoder_layer = _load_decoder_layers(model_dir, core_lang)
        ckpt_list.extend([encoder_layer, decoder_layer])
    if reload_embedding:
        embeddings = _load_multi_branch_embedding(model_dir)
        ckpt_list.extend(embeddings)
    return ckpt_list


def _load_encoder_layers(model_dir, new_src):
    # load encoder layers
    encoder_layers_path = os.path.join(model_dir, "%s_%s.pt" % ("encoder", new_src))
    encoder_layers = load_checkpoint_to_cpu(encoder_layers_path)
    logger.info("loaded encoder layers from %s" % encoder_layers_path)
    return encoder_layers


def _load_decoder_layers(model_dir, new_tgt):
    # load decoder layers
    decoder_layers_path = os.path.join(model_dir, "%s_%s.pt" % ("decoder", new_tgt))
    decoder_layers = load_checkpoint_to_cpu(decoder_layers_path)
    logger.info("loaded decoder layers from %s" % decoder_layers_path)
    return decoder_layers


def _load_multi_branch_embedding(model_dir):
    # load embedding
    encoder_embed_path = os.path.join(model_dir, ENCODER_FILE_NAME)
    decoder_embed_path = os.path.join(model_dir, DECODER_FILE_NAME)
    encoder_embedding = load_checkpoint_to_cpu(encoder_embed_path)
    decoder_embedding = load_checkpoint_to_cpu(decoder_embed_path)
    logger.info(
        "loaded encoder embedding from %s and decoder embedding %s" % (encoder_embed_path, decoder_embed_path))
    return [encoder_embedding, decoder_embedding]


def get_latest_embedding(model, save_root_dir, current_lg, current_time, thresholds):
    def get_satisfied_embedding():
        def _load_from_local(local_path, m_location = torch.device("cpu")):
            with open(local_path, "rb") as f:
                state = torch.load(f, map_location=m_location)
            return state

        def _load_from_ceph(url, m_location = torch.device("cpu")):
            return PathManager.get_ceph_manager().load_model(url, map_location=m_location)

        dirs = PathManager.ls(save_root_dir)
        satisfied_paths = []
        for sub_dir_name in dirs:
            sub_dir_path = os.path.join(save_root_dir, sub_dir_name)
            if not PathManager.isdir(sub_dir_path) or current_lg == sub_dir_name or current_lg + "/" == sub_dir_name:
                continue
            embedding_path = os.path.join(save_root_dir, sub_dir_name, EMBEDDING_META_FILE)
            logging.info("embedding path: %s" % embedding_path)
            embedding_meta_info = _load_from_ceph(embedding_path) if "s3://" in embedding_path else _load_from_local(
                embedding_path)
            embedding_meta_info = json.loads(embedding_meta_info)
            saved_time = datetime.fromtimestamp(embedding_meta_info["time"])
            interval = (current_time - saved_time).total_seconds() / 60
            if abs(interval) < thresholds:
                s_path = embedding_meta_info["current_path"]
                satisfied_paths.append(s_path)
        return satisfied_paths

    def load_satisfied_embedding(model, satisfied_path):
        lg_pair = "main-m2m"
        ckpt_list = _reload_one_branch_state(satisfied_path, lg_pair, reload_embedding=True, is_lg_adapter=False)
        version_value = [v for k, v in model.state_dict().items() if ".version" in k][0]
        required_state = _generate_required_state(ckpt_list, lg_pair, reload_one_branch=True,
                                                  reload_lg_specific_branch=False)
        # update version
        _update_version(required_state, model.keys, version_value)
        logger.info("load state dict ...")
        model.load_state_dict(required_state, strict=False)

        if hasattr(model, "module"):
            model.module.module.load_state_dict(required_state, strict=False)

        return model

    satisfied_paths = get_satisfied_embedding()
    # random select one
    if len(satisfied_paths) == 0:
        return model
    satisfied_path = random.choice(satisfied_paths)
    logger.info("loading latest main-m2m branch from %s" % satisfied_path)
    for i in range(3):
        try:
            logger.info("repeat load: %s times" % i)
            model=load_satisfied_embedding(model, satisfied_path)
            return model
        except Exception:
            # wait 5 seconds
            time.sleep(40)
            if i == 2:
                logger.error(traceback.format_exc())
                raise
    return model

def load_checkpoint(cfg: CheckpointConfig, trainer, **passthrough_args):
    """
    Load a checkpoint and restore the training iterator.

    *passthrough_args* will be passed through to
    ``trainer.get_train_iterator``.
    """

    reset_optimizer = cfg.reset_optimizer
    reset_lr_scheduler = cfg.reset_lr_scheduler
    optimizer_overrides = ast.literal_eval(cfg.optimizer_overrides)
    reset_meters = cfg.reset_meters
    reset_dataloader = cfg.reset_dataloader

    if cfg.finetune_from_model is not None and (
        reset_optimizer or reset_lr_scheduler or reset_meters or reset_dataloader
    ):
        raise ValueError(
            "--finetune-from-model can not be set together with either --reset-optimizer"
            " or reset_lr_scheduler or reset_meters or reset_dataloader"
        )

    suffix = trainer.checkpoint_suffix
    if (
        cfg.restore_file == "checkpoint_last.pt"
    ):  # default value of restore_file is 'checkpoint_last.pt'
        checkpoint_path = os.path.join(
            cfg.save_dir, "checkpoint_last{}.pt".format(suffix)
        )
        first_launch = not PathManager.exists(checkpoint_path)
        if first_launch and getattr(cfg, "continue_once", None) is not None:
            checkpoint_path = cfg.continue_once
        elif cfg.finetune_from_model is not None and first_launch:
            # if there is no last checkpoint to restore, start the finetune from pretrained model
            # else just use usual logic to load checkpoint, e.g. restart from last checkpoint and etc.
            if PathManager.exists(cfg.finetune_from_model):
                checkpoint_path = cfg.finetune_from_model
                reset_optimizer = True
                reset_lr_scheduler = True
                reset_meters = True
                reset_dataloader = True
                logger.info(
                    f"loading pretrained model from {checkpoint_path}: "
                    "optimizer, lr scheduler, meters, dataloader will be reset"
                )
            else:
                raise ValueError(
                    f"--funetune-from-model {cfg.finetune_from_model} does not exist"
                )
    elif suffix is not None:
        checkpoint_path = cfg.restore_file.replace(".pt", suffix + ".pt")
    else:
        checkpoint_path = cfg.restore_file

    if cfg.restore_file != "checkpoint_last.pt" and cfg.finetune_from_model:
        raise ValueError(
            "--finetune-from-model and --restore-file (non-default value) "
            "can not be specified together: " + str(cfg)
        )

    extra_state = trainer.load_checkpoint(
        checkpoint_path,
        reset_optimizer,
        reset_lr_scheduler,
        optimizer_overrides,
        reset_meters=reset_meters,
    )

    if (
        extra_state is not None
        and "best" in extra_state
        and not reset_optimizer
        and not reset_meters
    ):
        save_checkpoint.best = extra_state["best"]

    if extra_state is not None and not reset_dataloader:
        # restore iterator from checkpoint
        itr_state = extra_state["train_iterator"]
        epoch_itr = trainer.get_train_iterator(
            epoch=itr_state["epoch"], load_dataset=True, **passthrough_args
        )
        epoch_itr.load_state_dict(itr_state)
    else:
        epoch_itr = trainer.get_train_iterator(
            epoch=1, load_dataset=True, **passthrough_args
        )

    trainer.lr_step(epoch_itr.epoch)

    return extra_state, epoch_itr


def load_checkpoint_to_cpu(path, arg_overrides=None, load_on_all_ranks=False):
    """Loads a checkpoint to CPU (with upgrading for backward compatibility).

    If doing single-GPU training or if the checkpoint is only being loaded by at
    most one process on each node (current default behavior is for only rank 0
    to read the checkpoint from disk), load_on_all_ranks should be False to
    avoid errors from torch.distributed not having been initialized or
    torch.distributed.barrier() hanging.

    If all processes on each node may be loading the checkpoint
    simultaneously, load_on_all_ranks should be set to True to avoid I/O
    conflicts.

    There's currently no support for > 1 but < all processes loading the
    checkpoint on each node.
    """
    try:
        ceph_util = CEPHFileUtil()
    except:
        # logger.warning("no ceph manager")
        ceph_util = None

    local_path = PathManager.get_local_path(path)
    # The locally cached file returned by get_local_path() may be stale for
    # remote files that are periodically updated/overwritten (ex:
    # checkpoint_last.pt) - so we remove the local copy, sync across processes
    # (if needed), and then download a fresh copy.
    if local_path != path and PathManager.path_requires_pathmanager(path):
        try:
            if ceph_util is None:
                os.remove(local_path)
            else:
                ceph_util.remove(local_path)
        except FileNotFoundError:
            # With potentially multiple processes removing the same file, the
            # file being missing is benign (missing_ok isn't available until
            # Python 3.8).
            pass
        if load_on_all_ranks:
            torch.distributed.barrier()
        local_path = PathManager.get_local_path(path)

    if ceph_util is None:
        with open(local_path, "rb") as f:
            state = torch.load(f, map_location=torch.device("cpu"))
    else:
        state = ceph_util.load_checkpoint(local_path, torch.device("cpu"))

    if "args" in state and state["args"] is not None and arg_overrides is not None:
        args = state["args"]
        for arg_name, arg_val in arg_overrides.items():
            setattr(args, arg_name, arg_val)

    if "cfg" in state and state["cfg"] is not None:

        # hack to be able to set Namespace in dict config. this should be removed when we update to newer
        # omegaconf version that supports object flags, or when we migrate all existing models
        from omegaconf import _utils

        old_primitive = _utils.is_primitive_type
        _utils.is_primitive_type = lambda _: True

        state["cfg"] = OmegaConf.create(state["cfg"])

        _utils.is_primitive_type = old_primitive
        OmegaConf.set_struct(state["cfg"], True)

        if arg_overrides is not None:
            overwrite_args_by_name(state["cfg"], arg_overrides)

    state = _upgrade_state_dict(state)
    return state


def load_model_ensemble(
    filenames,
    arg_overrides: Optional[Dict[str, Any]] = None,
    task=None,
    strict=True,
    suffix="",
    num_shards=1,
    state=None,
):
    """Loads an ensemble of models.

    Args:
        filenames (List[str]): checkpoint files to load
        arg_overrides (Dict[str,Any], optional): override model args that
            were used during model training
        task (fairseq.tasks.FairseqTask, optional): task to use for loading
    """
    assert not (
        strict and num_shards > 1
    ), "Cannot load state dict with strict=True and checkpoint shards > 1"
    ensemble, args, _task = load_model_ensemble_and_task(
        filenames,
        arg_overrides,
        task,
        strict,
        suffix,
        num_shards,
        state,
    )
    return ensemble, args


def get_maybe_sharded_checkpoint_filename(
    filename: str, suffix: str, shard_idx: int, num_shards: int
) -> str:
    orig_filename = filename
    filename = filename.replace(".pt", suffix + ".pt")
    fsdp_filename = filename[:-3] + f"-shard{shard_idx}.pt"
    model_parallel_filename = orig_filename[:-3] + f"_part{shard_idx}.pt"
    if PathManager.exists(fsdp_filename):
        return fsdp_filename
    elif num_shards > 1:
        return model_parallel_filename
    else:
        return filename


def load_model_ensemble_and_task(
    filenames,
    arg_overrides: Optional[Dict[str, Any]] = None,
    task=None,
    strict=True,
    suffix="",
    num_shards=1,
    state=None,
):
    assert state is None or len(filenames) == 1

    from fairseq import tasks

    assert not (
        strict and num_shards > 1
    ), "Cannot load state dict with strict=True and checkpoint shards > 1"
    ensemble = []
    cfg = None
    for filename in filenames:
        orig_filename = filename
        model_shard_state = {"shard_weights": [], "shard_metadata": []}
        assert num_shards > 0
        st = time.time()
        for shard_idx in range(num_shards):
            filename = get_maybe_sharded_checkpoint_filename(
                orig_filename, suffix, shard_idx, num_shards
            )

            if not PathManager.exists(filename):
                raise IOError("Model file not found: {}".format(filename))
            if state is None:
                state = load_checkpoint_to_cpu(filename, arg_overrides)
            if "args" in state and state["args"] is not None:
                cfg = convert_namespace_to_omegaconf(state["args"])
            elif "cfg" in state and state["cfg"] is not None:
                cfg = state["cfg"]
            else:
                raise RuntimeError(
                    f"Neither args nor cfg exist in state keys = {state.keys()}"
                )

            if task is None:
                task = tasks.setup_task(cfg.task)

            if "task_state" in state:
                task.load_state_dict(state["task_state"])

            if "fsdp_metadata" in state and num_shards > 1:
                model_shard_state["shard_weights"].append(state["model"])
                model_shard_state["shard_metadata"].append(state["fsdp_metadata"])
                # check FSDP import before the code goes too far
                if not has_FSDP:
                    raise ImportError(
                        "Cannot find FullyShardedDataParallel. "
                        "Please install fairscale with: pip install fairscale"
                    )
                if shard_idx == num_shards - 1:
                    consolidated_model_state = FSDP.consolidate_shard_weights(
                        shard_weights=model_shard_state["shard_weights"],
                        shard_metadata=model_shard_state["shard_metadata"],
                    )
                    model = task.build_model(cfg.model)
                    if (
                        "optimizer_history" in state
                        and len(state["optimizer_history"]) > 0
                        and "num_updates" in state["optimizer_history"][-1]
                    ):
                        model.set_num_updates(
                            state["optimizer_history"][-1]["num_updates"]
                        )
                    model.load_state_dict(
                        consolidated_model_state, strict=strict, model_cfg=cfg.model
                    )
            else:
                # model parallel checkpoint or unsharded checkpoint
                # support old external tasks

                argspec = inspect.getfullargspec(task.build_model)
                if "from_checkpoint" in argspec.args:
                    model = task.build_model(cfg.model, from_checkpoint=True)
                else:
                    model = task.build_model(cfg.model)
                if (
                    "optimizer_history" in state
                    and len(state["optimizer_history"]) > 0
                    and "num_updates" in state["optimizer_history"][-1]
                ):
                    model.set_num_updates(state["optimizer_history"][-1]["num_updates"])
                model.load_state_dict(
                    state["model"], strict=strict, model_cfg=cfg.model
                )

            # reset state so it gets loaded for the next model in ensemble
            state = None
            if shard_idx % 10 == 0 and shard_idx > 0:
                elapsed = time.time() - st
                logger.info(
                    f"Loaded {shard_idx} shards in {elapsed:.2f}s, {elapsed / (shard_idx+1):.2f}s/shard"
                )

        # build model for ensemble
        ensemble.append(model)
    return ensemble, cfg, task


def load_model_ensemble_and_task_from_hf_hub(
    model_id,
    cache_dir: Optional[str] = None,
    arg_overrides: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "You need to install huggingface_hub to use `load_from_hf_hub`. "
            "See https://pypi.org/project/huggingface-hub/ for installation."
        )

    library_name = "fairseq"
    cache_dir = cache_dir or (Path.home() / ".cache" / library_name).as_posix()
    cache_dir = snapshot_download(
        model_id, cache_dir=cache_dir, library_name=library_name, **kwargs
    )

    _arg_overrides = arg_overrides or {}
    _arg_overrides["data"] = cache_dir
    return load_model_ensemble_and_task(
        [p.as_posix() for p in Path(cache_dir).glob("*.pt")],
        arg_overrides=_arg_overrides,
    )


def checkpoint_paths(path, pattern=r"checkpoint(\d+)\.pt", keep_match=False):
    """Retrieves all checkpoints found in `path` directory.

    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the first group in
    descending order.
    """
    pt_regexp = re.compile(pattern)
    files = PathManager.ls(path)

    entries = []
    for i, f in enumerate(files):
        m = pt_regexp.fullmatch(f)
        if m is not None:
            idx = float(m.group(1)) if len(m.groups()) > 0 else i
            entries.append((idx, m.group(0)))
    if keep_match:
        return [(os.path.join(path, x[1]), x[0]) for x in sorted(entries, reverse=True)]
    else:
        return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]


def torch_persistent_save(obj, filename, async_write: bool = False):
    is_s3_path = "s3://" in filename
    if async_write:
        with PathManager.opena(filename, "wb") as f:
            _torch_persistent_save(obj, f, is_s3_path)
    else:
        # fallback to non-atomic save
        if is_s3_path:
            _torch_persistent_save(obj, filename, is_s3_path)
        else:
            with PathManager.open(filename, "wb") as f:
                _torch_persistent_save(obj, f, is_s3_path)


def _torch_persistent_save(obj, f, is_s3_path):
    if isinstance(f, str) and not is_s3_path:
        with PathManager.open(f, "wb") as h:
            torch_persistent_save(obj, h)
        return
    for i in range(3):
        try:
            if is_s3_path:
                with io.BytesIO() as stream:
                    torch.save(obj, stream)
                    stream.seek(0)
                    return PathManager.get_ceph_manager().write(f, stream)
            else:
                return torch.save(obj, f)
        except Exception:
            if i == 2:
                logger.error(traceback.format_exc())
                raise


def _upgrade_state_dict(state):
    """Helper for upgrading old model checkpoints."""

    # add optimizer_history
    if "optimizer_history" not in state:
        if "best_loss" in state:
            state["optimizer_history"] = [
                {"criterion_name": "CrossEntropyCriterion", "best_loss": state["best_loss"]}
            ]
            del state["best_loss"]
        else:
            state["optimizer_history"] = [
                {"criterion_name": "CrossEntropyCriterion"}
            ]
        if "optimizer" in state:
            state["last_optimizer_state"] = state["optimizer"]
            del state["optimizer"]
    # move extra_state into sub-dictionary
    if "epoch" in state and "extra_state" not in state:
        state["extra_state"] = {
            "epoch": state["epoch"],
        }
        del state["epoch"]
        if "batch_offset" in state:
            state["extra_state"]["batch_offset"] = state["batch_offset"]
            del state["batch_offset"]
        if "val_loss" in state:
            state["extra_state"]["val_loss"] = state["val_loss"]
            del state["val_loss"]

    # reduce optimizer history's memory usage (only keep the last state)
    if "optimizer" in state["optimizer_history"][-1]:
        state["last_optimizer_state"] = state["optimizer_history"][-1]["optimizer"]
        for optim_hist in state["optimizer_history"]:
            del optim_hist["optimizer"]
    # record the optimizer class name
    if "optimizer_name" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["optimizer_name"] = "FairseqNAG"
    # move best_loss into lr_scheduler_state
    if "lr_scheduler_state" not in state["optimizer_history"][-1]:
        if "best_loss" in state["optimizer_history"][-1]:
            state["optimizer_history"][-1]["lr_scheduler_state"] = {
                "best": state["optimizer_history"][-1]["best_loss"]
            }
            del state["optimizer_history"][-1]["best_loss"]
    # keep track of number of updates
    if "num_updates" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["num_updates"] = 0
    # use stateful training data iterator
    if "extra_state" in state and "train_iterator" not in state["extra_state"]:
        state["extra_state"]["train_iterator"] = {
            "epoch": state["extra_state"].get("epoch", 0),
            "iterations_in_epoch": state["extra_state"].get("batch_offset", 0),
        }

    # backward compatibility, cfg updates
    if "args" in state and state["args"] is not None:
        # old model checkpoints may not have separate source/target positions
        if hasattr(state["args"], "max_positions") and not hasattr(
            state["args"], "max_source_positions"
        ):
            state["args"].max_source_positions = state["args"].max_positions
            state["args"].max_target_positions = state["args"].max_positions
        # default to translation task
        if not hasattr(state["args"], "task"):
            state["args"].task = "translation"
        # --raw-text and --lazy-load are deprecated
        if getattr(state["args"], "raw_text", False):
            state["args"].dataset_impl = "raw"
        elif getattr(state["args"], "lazy_load", False):
            state["args"].dataset_impl = "lazy"
        # epochs start at 1
        if state["extra_state"]["train_iterator"] is not None:
            state["extra_state"]["train_iterator"]["epoch"] = max(
                state["extra_state"]["train_iterator"].get("epoch", 1), 1
            )
        # --remove-bpe ==> --postprocess
        if hasattr(state["args"], "remove_bpe"):
            state["args"].post_process = state["args"].remove_bpe
        # --min-lr ==> --stop-min-lr
        if hasattr(state["args"], "min_lr"):
            state["args"].stop_min_lr = state["args"].min_lr
            del state["args"].min_lr
        # binary_cross_entropy / kd_binary_cross_entropy => wav2vec criterion
        if hasattr(state["args"], "criterion") and state["args"].criterion in [
            "binary_cross_entropy",
            "kd_binary_cross_entropy",
        ]:
            state["args"].criterion = "wav2vec"
        # remove log_keys if it's None (criteria will supply a default value of [])
        if hasattr(state["args"], "log_keys") and state["args"].log_keys is None:
            delattr(state["args"], "log_keys")
        # speech_pretraining => audio pretraining
        if (
            hasattr(state["args"], "task")
            and state["args"].task == "speech_pretraining"
        ):
            state["args"].task = "audio_pretraining"
        # audio_cpc => wav2vec
        if hasattr(state["args"], "arch") and state["args"].arch == "audio_cpc":
            state["args"].arch = "wav2vec"
        # convert legacy float learning rate to List[float]
        if hasattr(state["args"], "lr") and isinstance(state["args"].lr, float):
            state["args"].lr = [state["args"].lr]
        # convert task data arg to a string instead of List[string]
        if (
            hasattr(state["args"], "data")
            and isinstance(state["args"].data, list)
            and len(state["args"].data) > 0
        ):
            state["args"].data = state["args"].data[0]

        state["cfg"] = convert_namespace_to_omegaconf(state["args"])

    if "cfg" in state and state["cfg"] is not None:
        cfg = state["cfg"]
        with open_dict(cfg):
            # any upgrades for Hydra-based configs
            if (
                "task" in cfg
                and "eval_wer_config" in cfg.task
                and isinstance(cfg.task.eval_wer_config.print_alignment, bool)
            ):
                cfg.task.eval_wer_config.print_alignment = "hard"
            if "generation" in cfg and isinstance(cfg.generation.print_alignment, bool):
                cfg.generation.print_alignment = (
                    "hard" if cfg.generation.print_alignment else None
                )
            if (
                "model" in cfg
                and "w2v_args" in cfg.model
                and cfg.model.w2v_args is not None
                and (
                    hasattr(cfg.model.w2v_args, "task") or "task" in cfg.model.w2v_args
                )
                and hasattr(cfg.model.w2v_args.task, "eval_wer_config")
                and cfg.model.w2v_args.task.eval_wer_config is not None
                and isinstance(
                    cfg.model.w2v_args.task.eval_wer_config.print_alignment, bool
                )
            ):
                cfg.model.w2v_args.task.eval_wer_config.print_alignment = "hard"

    return state


def prune_state_dict(state_dict, model_cfg: Optional[DictConfig]):
    """Prune the given state_dict if desired for LayerDrop
    (https://arxiv.org/abs/1909.11556).

    Training with LayerDrop allows models to be robust to pruning at inference
    time. This function prunes state_dict to allow smaller models to be loaded
    from a larger model and re-maps the existing state_dict for this to occur.

    It's called by functions that load models from checkpoints and does not
    need to be called directly.
    """
    arch = None
    if model_cfg is not None:
        arch = (
            model_cfg._name
            if isinstance(model_cfg, DictConfig)
            else getattr(model_cfg, "arch", None)
        )

    if not model_cfg or arch is None or arch == "ptt_transformer":
        # args should not be none, but don't crash if it is.
        return state_dict

    encoder_layers_to_keep = getattr(model_cfg, "encoder_layers_to_keep", None)
    decoder_layers_to_keep = getattr(model_cfg, "decoder_layers_to_keep", None)

    if not encoder_layers_to_keep and not decoder_layers_to_keep:
        return state_dict

    # apply pruning
    logger.info(
        "Pruning model to specified layer configuration - this works best if the model was trained with LayerDrop"
    )

    def create_pruning_pass(layers_to_keep, layer_name):
        keep_layers = sorted(
            int(layer_string) for layer_string in layers_to_keep.split(",")
        )
        mapping_dict = {}
        for i in range(len(keep_layers)):
            mapping_dict[str(keep_layers[i])] = str(i)

        regex = re.compile(r"^{layer}.*\.layers\.(\d+)".format(layer=layer_name))
        return {"substitution_regex": regex, "mapping_dict": mapping_dict}

    pruning_passes = []
    if encoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(encoder_layers_to_keep, "encoder"))
    if decoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(decoder_layers_to_keep, "decoder"))

    new_state_dict = {}
    for layer_name in state_dict.keys():
        match = re.search(r"\.layers\.(\d+)\.", layer_name)
        # if layer has no number in it, it is a supporting layer, such as an
        # embedding
        if not match:
            new_state_dict[layer_name] = state_dict[layer_name]
            continue

        # otherwise, layer should be pruned.
        original_layer_number = match.group(1)
        # figure out which mapping dict to replace from
        for pruning_pass in pruning_passes:
            if original_layer_number in pruning_pass["mapping_dict"] and pruning_pass[
                "substitution_regex"
            ].search(layer_name):
                new_layer_number = pruning_pass["mapping_dict"][original_layer_number]
                substitution_match = pruning_pass["substitution_regex"].search(
                    layer_name
                )
                new_state_key = (
                    layer_name[: substitution_match.start(1)]
                    + new_layer_number
                    + layer_name[substitution_match.end(1) :]
                )
                new_state_dict[new_state_key] = state_dict[layer_name]

    # Since layers are now pruned, *_layers_to_keep are no longer needed.
    # This is more of "It would make it work fix" rather than a proper fix.
    if isinstance(model_cfg, DictConfig):
        context = open_dict(model_cfg)
    else:
        context = contextlib.ExitStack()
    with context:
        if hasattr(model_cfg, "encoder_layers_to_keep"):
            model_cfg.encoder_layers_to_keep = None
        if hasattr(model_cfg, "decoder_layers_to_keep"):
            model_cfg.decoder_layers_to_keep = None

    return new_state_dict


def load_pretrained_component_from_model(
    component: Union[FairseqEncoder, FairseqDecoder],
    checkpoint: str,
    strict: bool = True,
):
    """
    Load a pretrained FairseqEncoder or FairseqDecoder from checkpoint into the
    provided `component` object. If state_dict fails to load, there may be a
    mismatch in the architecture of the corresponding `component` found in the
    `checkpoint` file.
    """
    if not PathManager.exists(checkpoint):
        raise IOError("Model file not found: {}".format(checkpoint))
    state = load_checkpoint_to_cpu(checkpoint)
    if isinstance(component, FairseqEncoder):
        component_type = "encoder"
    elif isinstance(component, FairseqDecoder):
        component_type = "decoder"
    else:
        raise ValueError(
            "component to load must be either a FairseqEncoder or "
            "FairseqDecoder. Loading other component types are not supported."
        )
    component_state_dict = OrderedDict()
    for key in state["model"].keys():
        if key.startswith(component_type):
            # encoder.input_layers.0.0.weight --> input_layers.0.0.weight
            component_subkey = key[len(component_type) + 1 :]
            component_state_dict[component_subkey] = state["model"][key]
    component.load_state_dict(component_state_dict, strict=strict)
    return component


def verify_checkpoint_directory(save_dir: str) -> None:
    def local_path_check(save_dir):
        temp_file_path = os.path.join(save_dir, "dummy")
        os.makedirs(save_dir, exist_ok=True)
        try:
            with open(temp_file_path, "w"):
                pass
        except OSError as e:
            logging.warning(
                "Unable to access checkpoint save directory: {}".format(save_dir)
            )
            raise e
        else:
            os.remove(temp_file_path)

    def ceph_path_check(save_dir):
        data = torch.tensor([0, 1, 2, 3])
        tmp_path = os.path.join(save_dir, "tensor_data")
        try:
            with io.BytesIO() as f:
                torch.save(data, f)
                ceph_util.ceph_handler.write(tmp_path, f.getvalue())
        except OSError as e:
            logging.warning(
                "Unable to access checkpoint save directory: {}".format(save_dir)
            )
            raise e
        else:
            ceph_util.ceph_handler.remove(tmp_path)

    if "s3://" in save_dir:
        ceph_util.make_dirs(save_dir, exist_ok=True)
        ceph_path_check(save_dir)
    else:
        os.path.exists(save_dir)
        local_path_check(save_dir)


def save_ema_as_checkpoint(src_path, dst_path):
    state = load_ema_from_checkpoint(src_path)
    torch_persistent_save(state, dst_path)


def load_ema_from_checkpoint(fpath):
    """Loads exponential moving averaged (EMA) checkpoint from input and
    returns a model with ema weights.

    Args:
      fpath: A string path of checkpoint to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    new_state = None

    with PathManager.open(fpath, "rb") as f:
        new_state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, "cpu")
            ),
        )

        # EMA model is stored in a separate "extra state"
        model_params = new_state["extra_state"]["ema"]

        for key in list(model_params.keys()):
            p = model_params[key]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if key not in params_dict:
                params_dict[key] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                raise ValueError("Key {} is repeated in EMA model params.".format(key))

        if len(params_dict) == 0:
            raise ValueError(
                f"Input checkpoint path '{fpath}' does not contain "
                "ema model weights, is this model trained with EMA?"
            )

    new_state["model"] = params_dict
    return new_state
