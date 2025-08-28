#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import sys
from argparse import Namespace
from itertools import chain

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import pandas as pd
import ast

candidates_lgs = ["zh", "en", "es", "fr", "ru", "ar", "pt", "de", "it", "tr", "nl", "pl", "cs", "ja",
                  "ro", "sv", "bg", "el", "hu", "he", "fi", "sr", "vi", "id", "da"]


def get_lg_cos_smi(lg_embed_dict):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    columns = list(lg_embed_dict.keys())
    topk_res, output = [], []
    topk = 5
    for r_index, row in enumerate(lg_embed_dict.values()):
        row_res = []
        for c_index, col in enumerate(lg_embed_dict.values()):
            row_res.append(round(cos(row, col).item(), 4))
        tmp = (-np.array(row_res)).argsort()[1:(topk + 1)]
        output.append(row_res)
        topk_res.append([(columns[i], row_res[i]) for i in tmp])
    res_df = pd.DataFrame(output, columns=lg_embed_dict.keys(), index=lg_embed_dict.keys())
    topk_df = pd.DataFrame(topk_res, columns=["top %s" % (i + 1) for i in range(topk)], index=lg_embed_dict.keys())
    return res_df, topk_df


def cluster_lgs(res_pairs, save_file_path, png_name=None):
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.DiGraph()
    G.add_edges_from(res_pairs)
    black_edges = [edge for edge in G.edges()]

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size=200)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=False)
    if png_name is not None:
        plt.title(png_name)
    # plt.show()
    plt.savefig(save_file_path)
    plt.cla()


def get_lg_family_cos_smi_topk(topk_df, save_file):
    tmp = topk_df.T.to_dict()
    lg_dict = {}
    for key, value in tmp.items():
        lg_dict[key] = [v[0] for k, v in value.items() if v[0] in candidates_lgs ]
    res_pairs, lgs = [], set()
    for s_lg, topk_lgs in lg_dict.items():
        for t_lg in topk_lgs:
            if s_lg not in lg_dict[t_lg]:
                continue
            res_pairs.append((s_lg, t_lg))
            lgs.add(s_lg)
            lgs.add(t_lg)
    cluster_lgs(res_pairs, "png_topk_%s_%s.png" % (len(topk_df.columns), save_file))


def get_lg_family_cos_smi_threshold(res_df, save_file):
    candidates_df = res_df[candidates_lgs][res_df.index.isin(candidates_lgs)].round(2)
    get_uniques = set()
    [get_uniques.update(candidates_df[lg].unique()) for lg in candidates_lgs]
    for threshold in get_uniques:
        if threshold == 1.0:
            continue
        threshold_pairs = []
        for row in candidates_df.iterrows():
            index_lg = row[0]
            [threshold_pairs.append((index_lg, c_lg)) for c_lg in candidates_lgs if row[1][c_lg] >= threshold and c_lg != index_lg]
        cluster_lgs(threshold_pairs, "png_threshold_%s_%s.png" % (threshold, save_file), png_name="similarity greater than %s" % threshold)


def get_lg_embedding(model, tgt_dict, partition_num=0):
    import re
    lgs_embedding = {}
    is_partition = any(["partitions" in key for key in model.state_dict().keys()])
    print("is partition: %s" % is_partition)
    for token, index in tgt_dict.indices.items():
        res = re.findall(r"__([a-zA-Z]+)(-*)(_*)(\w*)__", token)
        if len(res) > 0:
            if is_partition:
                lg_emb = model.state_dict()["encoder.model.partitions.0.0.embed_tokens.%s.weight" % partition_num]
            else:
                lg_emb = model.state_dict()['encoder.embed_tokens.weight']
            lgs_embedding[token[2:-2]] = lg_emb[index]
    return lgs_embedding


def hierarchical_clustering(lg_df, save_file_name, n_clusters=6):
    import matplotlib.pyplot as plt  # for visualization
    import scipy.cluster.hierarchy as sch  # importing scipy.cluster.hierarchy for dendrogram
    from sklearn.manifold import TSNE
    dendrogram = sch.dendrogram(
        sch.linkage(lg_df.values, method="ward"))  # finding the optimal number of clusters using dendrogram
    plt.title('Language Family')  # title of the dendrogram
    plt.xlabel('Language ID')  # label of the x-axis
    plt.ylabel('Similarity')  # label of the y-axis
    # plt.show() # show the dendrogram
    plt.savefig("Dendrogram_%s.png" % save_file_name)
    plt.cla()

    from sklearn.cluster import \
        AgglomerativeClustering  # this line of code imports AgglomerativeClustering model from sk-learn
    Agg_hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    # plotting cluster 1
    # X_tsne = TSNE(n_components=2, random_state=33).fit_transform(lg_df.values)
    y_hc = Agg_hc.fit_predict(lg_df.values)
    y_hc_df = pd.DataFrame(y_hc, columns=["hc_predict"], index=lg_df.index.values)
    y_hc_df.to_csv("%s.csv" % save_file_name)


def main(cfg: DictConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
            not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
            cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)

    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    if cfg.generation.lm_path is not None:
        overrides["data"] = cfg.task.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [cfg.generation.lm_path], arg_overrides=overrides, task=None
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({cfg.task.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    model_name = os.path.basename(cfg.common_eval.path)
    is_partition = any(["partitions" in key for key in models[0].state_dict().keys()])
    if is_partition:
        partition_nums = len(
            [key for key in models[0].state_dict().keys() if "encoder.model.partitions.0.0.embed_tokens" in key])
        print("partition num: %s" % partition_nums)
        ensemble_lg_embed = None
        for partition in range(partition_nums):
            lg_embed_dict = get_lg_embedding(models[0], tgt_dict, partition_num=partition)
            if ensemble_lg_embed is None:
                ensemble_lg_embed = lg_embed_dict
            else:
                for k, v in lg_embed_dict.items():
                    ensemble_lg_embed[k] += lg_embed_dict[k]
            res_df, topk_df = get_lg_cos_smi(lg_embed_dict)
            lg_embed_list = torch.stack(list(lg_embed_dict.values()), dim=0).tolist()
            lg_embed_cols = ["col_%s" % i for i in range(len(lg_embed_list[0]))]
            lg_df = pd.DataFrame(lg_embed_list, columns=lg_embed_cols, index=lg_embed_dict.keys())
            res_df.to_csv("res_df_%s_partition_%s.csv" % (model_name, partition))
            topk_df.to_csv("topk%s_df_%s_partition_%s.csv" % (len(topk_df.columns), model_name, partition))
            lg_df.to_csv("embedding_data_%s_partition_%s.csv" % (model_name, partition))
            # hierarchical_clustering(lg_df, "png_hc_%s_%s" % (model_name, partition))
            # get_lh_family_cos_smi(topk_df, "png_hc_%s_%s" % (model_name, partition))

        ensemble_lg_embed = dict([(k, v / partition_nums) for k, v in ensemble_lg_embed.items()])
        lg_embed_list = torch.stack(list(ensemble_lg_embed.values()), dim=0).tolist()
        lg_embed_cols = ["col_%s" % i for i in range(len(lg_embed_list[0]))]
        lg_df = pd.DataFrame(lg_embed_list, columns=lg_embed_cols, index=ensemble_lg_embed.keys())
        res_df, topk_df = get_lg_cos_smi(ensemble_lg_embed)

        res_df.to_csv("res_df_%s_partition_%s.csv" % (model_name, "ensemble"))
        topk_df.to_csv("topk%s_df_%s_partition_%s.csv" % (len(topk_df.columns), model_name, "ensemble"))
        lg_df.to_csv("embedding_data_%s_partition_%s.csv" % (model_name, "ensemble"))
        # hierarchical_clustering(lg_df, "png_hc_%s_%s" % (model_name, "ensemble"))
        get_lg_family_cos_smi_topk(topk_df, "topk_png_hc_%s_%s" % (model_name, "ensemble"))
        get_lg_family_cos_smi_threshold(res_df, "%s_%s" % (model_name, "ensemble"))

    else:
        lg_embed_dict = get_lg_embedding(models[0], tgt_dict)
        res_df, topk_df = get_lg_cos_smi(lg_embed_dict)
        lg_embed_list = torch.stack(list(lg_embed_dict.values()), dim=0).tolist()
        lg_embed_cols = ["col_%s" % i for i in range(len(lg_embed_list[0]))]
        lg_df = pd.DataFrame(lg_embed_list, columns=lg_embed_cols, index=lg_embed_dict.keys())
        res_df.to_csv("res_df_%s.csv" % model_name)
        topk_df.to_csv("topk%s_df_%s.csv" % (len(topk_df.columns), model_name))
        lg_df.to_csv("embedding_data_%s.csv" % model_name)
        # hierarchical_clustering(lg_df, "png_hc_%s" % model_name)
        get_lg_family_cos_smi_topk(topk_df, "topk_png_hc_%s" % model_name)
        get_lg_family_cos_smi_threshold(res_df, model_name)


def cli_main():
    parser = options.get_generation_parser()
    # TODO: replace this workaround with refactoring of `AudioPretraining`
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="wav2vec2",
        help="Model architecture. For constructing tasks that rely on "
             "model args (e.g. `AudioPretraining`)",
    )
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
