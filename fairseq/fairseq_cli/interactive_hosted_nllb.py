#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Host the demo.

Launch with `python -m fairseq_cli.interactive_hosted` to run locally.

See docs/api.md for more information.
"""

import os
import queue
import sys

import pkg_resources
import random
import threading
import traceback

import requests
import torch
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException

import sentencepiece as spm

from fairseq import options
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import utils as dist_utils
from fairseq.service.queue import PriorityQueueRingShard
from fairseq.service.workers import WorkItem
from fairseq.service.constants_nllb import (
    MAX_SEQ_LEN,
    MAX_BATCH_TOKENS,
    DEFAULT_PORT,
    TOTAL_WORLD_SIZE,
    LAUNCH_ARGS,
)
from fairseq.service.utils import get_my_ip, encode_fn, build_logger
from fairseq.service.responses import OAIResponse
import sacrebleu
from fairseq.hub_utils import GeneratorInterface
from fairseq_cli.demo_utils import content_detection

import logging
from flask_cors import CORS
app = Flask(__name__)
CORS(app, supports_credentials=True)

# global state (mutable!)
cfg = None
port = 10056
BATCH_QUEUE = PriorityQueueRingShard()

logger = build_logger()
#global f
#f = open('log_encode_sent.txt','w+')

#def flush_write(s,f):
#    f.writelines(s)
#    f.flush()
# flush_write('down load punkt', f)
# import nltk
# nltk.download('punkt') # download this package at first time
# logging.info('down load punkt')
# from nltk.tokenize import sent_tokenize
# flush_write('down loaded punkt', f)


def cmb_generations(generations):
    cmb_list = []
    for i in range(len(generations)):
        cmb_list.append(generations[i][0]['text'])
    cmb_str = ' '.join(cmb_list)
    generations[0][0]['text'] = cmb_str
    return generations

nllb_langs="ace,ace_1,acm,acq,aeb,af,ajp,ak,am,apc,ar,ars,ary,arz,as,ast,awa,ay,az,azb,ba,ban,be,ber,bg,bi,bjn,bjn_1," \
           "bm,bn,bo,bs,bug,ca,ceb,cjk,ckb,crh,cs,cy,da,de,dik,dyu,dz,ee,el,en,eo,es,et,eu,fa,fi,fj,fo,fon,fr,fur,fuv,ga," \
           "gd,gl,gn,gu,ha,he,hi,hne,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kab,kac,kam,kbp,kea,kg,kik,kk,km,kmb,kmr,kn,ko," \
           "kr,kr_1,ks,ks_1,ky,lb,lg,li,lij,lmo,ln,lo,lt,ltg,lua,luo,lus,lv,mag,mai,mi,min,mk,ml,mn,mni,mos,mr,ms,mt,my,ne," \
           "nl,nn,no,ns,nus,ny,oc,om,or,pa,pag,pap,pl,plt,prs,ps,pt,quy,rn,ro,ru,rw,sa,sat,sc,scn,sd,sg,shn,si,sk,sl,sm,sn," \
           "so,sq,sr,ss,st,su,sv,sw,szl,ta,taq,taq_1,te,tg,th,ti,tk,tl,tn,tpi,tr,ts,tt,tum,tw,tzm,ug,uk,umb,ur,uz,vec,vi,war," \
           "wo,xh,yi,yo,zh,zhtrad".split(",")


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1))


def generate_jaccard(origin_prompts, generation_seq):
    scores = []
    for origin_prompt, input in zip(origin_prompts, generation_seq):
        score = 0
        if origin_prompt is not None:
            translation_list = [c for c in input]
            src_list = [c for c in origin_prompt]
            score = jaccard_similarity(translation_list, src_list)
        scores.append(score)
    return scores

def generate_sentence_bleu(origin_prompts, generation_seq):
    scores = []
    for origin_prompt, input in zip(origin_prompts, generation_seq):
        score = 0
        # if origin_prompt is not None:
        #     translation_list = " ".join([c for c in input])
        #     src_list = " ".join([c for c in origin_prompt])
        #     score = sacrebleu.sentence_bleu(translation_list, [src_list]).score
        scores.append(score)
    return scores


def batching_loop(timeout=100, max_tokens=MAX_BATCH_TOKENS):
    """
    batching_loop is an infinite loop responsible for executing generations.

    GPUs benefit from batching requests, but we expect workloads to come
    in non-uniformly. This loop groups requests together (via BATCH_QUEUE)
    and executes them in one batch. In order to keep latency low, unfilled
    batches are executed within a window of :timeout: milliseconds.

    batching_loop also performs dynamic batching, in order to minimize the
    amount of padding by grouping like-sized workloads together. As a result
    batching loop will provide preferential treatment to smaller workloads.  At
    the current moment, there is no TTL logic to ensure a maximum wait time.

    For a rough overview of dynamic batching, see
    https://parl.ai/docs/tutorial_worlds.html#dynamic-batching.

    :param timeout: The max queue time before a non-full batch is launched.
    :param max_tokens: the maximum number of tokens that can be processed
        concurrently. model specific and empirical.
    """
    # TODO(roller):
    # - group by generation type, topp etc, as we cannot share these
    # - modify timeout logic to be cumulative
    global BATCH_QUEUE

    batch = []
    while True:
        try:
            # for now, we only have 1 worker, so can always index to shard 0
            target_queue = BATCH_QUEUE.queue_shards[0].get_largest_queue()
            if not target_queue:
                continue
            # dynamic batching: group like-sized items to reduce the cost
            # of padding. See PR#20 for additional context.
            item = target_queue.get(timeout=timeout / 1000)
            # print("in batching_loop item=target_queue.get:", item)
            #WorkItem(cost=1, uid=0, return_queue=<queue.Queue object at 0x7f1116c9b4c0>,
            # data={'input': ['▁how ▁are ▁you'], 'src': 'en', 'tar': 'zh', 'temperature': 1.0, 'top_p': 1.0, 'n': 1})
            # accumulate the batch until it gets too big
            longest = max([item] + batch).cost
            batch_cost = longest * (len(batch) + 1)
            if batch and batch_cost > max_tokens:
                # we're over budget, put it back in the queue
                target_queue.put(item)
                raise queue.Empty
            else:
                # batch is empty or under budget
                batch.append(item)
                raise queue.Empty
        except queue.Empty:
            try:
                back_translate = False
                origin_prompt, generated_prompt = [], []
                if batch:
                    request_object = {
                        "inputs": [],
                        "min_tokens": [],
                        "max_tokens": [],
                        "srcs": [],
                        "tars": [],
                    }
                    request_object["content_detection_flag"] = True
                    for work_item in batch:
                        ro = work_item.data
                        # ro, {'input': ['▁how ▁are ▁you'], 'src': 'en', 'tar': 'zh', 'temperature': 1.0, 'top_p': 1.0,
                        #      'n': 1}
                        if ro["src"] not in nllb_langs or ro["tar"] not in nllb_langs:
                            ro["src"] = "zh"
                            ro["tar"] = "zh"
                            ro["input"] = ['▁N LL B ▁目 前 暂 不 支持 的 翻 译 语言']

                        request_object["inputs"].append(ro["input"])
                        request_object["min_tokens"].append(ro.get("min_tokens", 0))
                        request_object["max_tokens"].append(
                            ro.get("max_tokens", MAX_SEQ_LEN)
                        )
                        request_object["srcs"].append(ro.get("src"))
                        request_object["tars"].append(ro.get("tar"))

                        if "back_translate" in ro:
                            back_translate = bool(ro["back_translate"])

                        if "content_detection" in ro:
                            content_detection_flag = bool(ro["content_detection"])
                            request_object["content_detection_flag"] = content_detection_flag

                        if "origin_prompt" in ro:
                            origin_prompt.append(ro["origin_prompt"])
                        else:
                            origin_prompt.append(None)

                        if "generated_prompt" in ro:
                            generated_prompt.append(ro["generated_prompt"])
                        else:
                            generated_prompt.append(None)

                        # assumption: everyone has the same remaining args
                        for key in [
                            "temperature",
                            "top_p",
                            "n",
                            "best_of",
                            "echo",
                            "logprobs",
                            "stop"
                        ]:
                            if key in ro:
                                request_object[key] = ro[key]
                    # do the actual generations
                    request_object["seed"] = random.randint(1, 20000)
                    request_object["seed"] = random.randint(1, 20000)
                    returns = [None for _ in range(len(batch))]
                    has_generate_prompt = [False if (g is None or len(g) == 0) else True
                                           for g in generated_prompt]
                    if back_translate and not all(has_generate_prompt):
                        logging.info('request_object: ', request_object)
                        # flush_write('request_object: '+ str(request_object), f)
                        generations = generator.generate(**request_object)
                        request_object_back = {}
                        request_object_back['srcs'] = request_object['tars']
                        request_object_back['tars'] = request_object['srcs']
                        request_object_back["inputs"] = []
                        for i, generation in enumerate(generations):
                            print("before encoder", generation)
                            encoder_generation = encode_and_get_right_type(generation[0]["text"])
                            print("after encoder: ", encoder_generation)
                            request_object_back["inputs"].extend(encoder_generation)
                        generations_back = generator.generate(**request_object_back)

                        scores = generate_sentence_bleu(request_object["inputs"], [g[0]["text"] for g in generations_back])

                        for i in range(len(batch)):
                            returns[i] = []

                            generations[i][0]["score"] = scores[i]
                            returns[i].append(generations[i][0])

                            generations_back[i][0]["score"] = scores[i]
                            returns[i].append(generations_back[i][0])

                    elif back_translate and all(has_generate_prompt):
                        request_object_back = {}
                        request_object_back['srcs'] = request_object['tars']
                        request_object_back['tars'] = request_object['srcs']
                        request_object_back["inputs"] = []
                        for i, generation in enumerate(generated_prompt):
                            print("has generate prompt before encoder", generation)
                            encoder_generation = encode_and_get_right_type(generated_prompt[i])
                            print("has generate prompt after encoder: ", encoder_generation)
                            request_object_back["inputs"].extend(encoder_generation)
                        generations_back = generator.generate(**request_object_back)

                        scores = generate_sentence_bleu([g[0]["text"] for g in generations_back], request_object["inputs"])

                        for i in range(len(batch)):
                            returns[i] = []
                            generations_back[i][0]["score"] = scores[i]
                            returns[i].append(generations_back[i][0])
                    else:
                        logging.info('request_object: ', request_object)
                        # flush_write('request_object: '+ str(request_object), f)
                        generations = generator.generate(**request_object)
                        scores = generate_sentence_bleu(generation_seq=[g[0]["text"] for g in generations], origin_prompts=origin_prompt)

                        for i in range(len(batch)):
                            returns[i] = []
                            generations[i][0]["score"] = scores[i]
                            returns[i].append(generations[i][0])

                    for work_item, gen in zip(batch, returns):
                        #print(gen)
                        work_item.return_queue.put((work_item.uid, gen))

                    batch.clear()
                else:
                    # back to the loop
                    batch.clear()
                    continue
            except Exception as e:
                batch.clear()
                print(traceback.format_exc())
                print("current error: ", e)
                continue


def worker_main(cfg1: FairseqConfig, namespace_args=None):
    # disable multithreading in tokenizers and torch, as different Flask threads
    # may then fight for resources.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    global generator
    global MODE

    # make sure generations are stochastic since we have many workers
    torch.manual_seed(random.randint(1, 20000))
    torch.cuda.manual_seed(random.randint(1, 20000))
    MODE = "worker"
    # from fairseq.pdb import pdb; pdb.set_trace()
    cfg = cfg1

    generator = GeneratorInterface(cfg)
    models = generator.load_model()  # noqa: F841
    generator.build_generator()
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        [model.cuda() for model in models]

    logger.info(f"loaded model {cfg.distributed_training.distributed_rank}")
    # request_object = dist_utils.broadcast_object(
    #     None, src_rank=0, group=dist_utils.get_global_group()
    # )
    # if torch.distributed.get_rank() == 0:
    logger.info(f"Worker engaged! {get_my_ip()}:{port}")
    thread = threading.Thread(target=batching_loop, daemon=True)
    thread.start()
    app.run(host="0.0.0.0", port=port, threaded=True)
    # else:
    #     # useful in FSDP setting
    #     logger.info(f"Looping engaged! {get_my_ip()}:{port}")
    #     while True:
    #         try:
    #             request_object = dist_utils.broadcast_object(
    #                 None, src_rank=0, group=dist_utils.get_global_group()
    #             )
    #             _ = generator.generate(**request_object)
    #         except Exception:
    #             # continue looping for the next generation so we don't lock up
    #             pass


@app.errorhandler(Exception)
def handle_exception(e):
    # pass through HTTP errors
    if isinstance(e, HTTPException):
        return e
    # now you're handling non-HTTP exceptions only
    response = jsonify(
        {
            "error": {
                "message": str(e),
                "type": "oops",
                "stacktrace": traceback.format_tb(e.__traceback__),
            }
        }
    )
    if isinstance(e, ValueError):
        response.status = 400
    else:
        response.status = 500
    return response


def encode_and_get_right_type_sent_tok(prompts, lang='',min_sent_length_char=200,min_sent_length_word=50):
    if isinstance(prompts, str):
        #flush_write('prompts before sent_tokenize:' + prompts, f)
        prompts = sent_tokenize(prompts)
        #flush_write('prompts after sent_tokenize:' + str(prompts), f)
        len_sent = len(prompts)
        logging.info('len_sent'+str(len_sent))
        #flush_write('prompts before selected:'+str(prompts), f)
        cur_str_list = []
        prompts_ret = []
        for sent in prompts:
            cur_str = ' '.join(cur_str_list+[sent])
            sentsplit = cur_str.split(' ')
            if len(sentsplit) > len(cur_str_list)+1:
                if len(sentsplit) > min_sent_length_word:
                    prompts_ret.append(cur_str)
                    cur_str_list = []
                else:
                    cur_str_list.append(sent)
            else:
                if len(cur_str) > min_sent_length_char:
                    prompts_ret.append(cur_str)
                    cur_str_list = []
                else:
                    cur_str_list.append(sent)
        if len(cur_str_list) != 0:
            prompts_ret.extend(cur_str_list)
        #nflush_write('prompts after selected' + str(prompts_ret), f)
        # single string. tokenize and trn it to the single pre-tokenized case
        prompts_before_paren = []
        for prompt in prompts_ret:
            prompts_before_paren.extend(encode_fn(generator, prompt))
    # prompts = prompts_before_paren
    #print('prompts_before_paren' + str(prompts_before_paren))  # ['▁how ▁are ▁you', 'xxx', 'xxx']
    #print('len(prompts_before_paren)', len(prompts_before_paren))
    prompts_after_paren = [prompts_before_paren]  # [['▁how ▁are ▁you', 'xxx xxx', 'xxx']]
    #f.writelines('prompts after []:' + str(prompts_after_paren))
    return prompts_after_paren

def encode_and_get_right_type(prompts):
    if isinstance(prompts, str):
        prompts = prompts.replace('\n','').replace('\r','')
        # single string. tokenize and turn it to the single pre-tokenized case
        prompts = [encode_fn(generator, prompts)]
    # assert isinstance(prompts, list)
    # assert len(prompts) > 0
    # print(str(prompts[0]))  # ['▁how ▁are ▁you']
    # print(len(prompts))  # 1
    elif isinstance(prompts[0], str):
        # multi string
        prompts = [encode_fn(generator, p) for p in prompts]
    elif isinstance(prompts[0], int):
        # single pre-tokenized
        prompts = [prompts]
    # assert isinstance(prompts[0], list)
    # # final case: multi pre-tokenized
    # assert len(prompts[0]) > 0
    return prompts

@app.route("/v1/engines/<engine>/completions", methods=["POST"])
def completions(engine=None):
    # prompt can be 4 types:
    # - str. Basic case. Return one generation.
    # - list of ints. Pretokenized. Return one generation
    # - list of str. Multiple generations, one per prompt
    # - list of list of ints. Pretokenized multiple generations.

    # our approach is to turn everything into the last case

    prompts = request.json["prompt"]
    del request.json["prompt"]
    generation_args = request.json

    print(prompts)  # how are you
    #print(len(prompts))  # 11
    prompts = encode_and_get_right_type(prompts)

    if "min_tokens" in generation_args:
        generation_args["min_tokens"] = int(generation_args["min_tokens"])
    if "max_tokens" in generation_args:
        generation_args["max_tokens"] = int(generation_args["max_tokens"])
    if "stop" in generation_args:
        stop = generation_args["stop"]
        if stop is None:
            pass
        elif isinstance(stop, str):
            stop = [encode_fn(generator, stop)[0]]
        else:
            stop = [encode_fn(generator, s)[0] for s in stop]
        generation_args["stop"] = stop
    if "temperature" in generation_args:
        generation_args["temperature"] = round(float(generation_args["temperature"]), 1)
    else:
        generation_args["temperature"] = 1.0
    if "top_p" in generation_args:
        generation_args["top_p"] = round(float(generation_args["top_p"]), 1)
    else:
        generation_args["top_p"] = 1.0
    # beam search top n
    if "n" in generation_args:
        generation_args["n"] = int(generation_args["n"])
    else:
        generation_args["n"] = 1

    if "back_translate" in generation_args:
        generation_args["back_translate"] = bool(generation_args["back_translate"])
    else:
        generation_args["back_translate"] = False

    if "content_detection" in generation_args:
        generation_args["content_detection"] = bool(generation_args["content_detection"])
    else:
        generation_args["content_detection"] = True

    if "origin_prompt" in generation_args:
        generation_args["origin_prompt"] = generation_args["origin_prompt"]
    else:
        generation_args["origin_prompt"] = None

    if "generated_prompt" in generation_args:
        generation_args["generated_prompt"] = generation_args["generated_prompt"]
    else:
        generation_args["generated_prompt"] = None

    ret_queue = queue.Queue()
    for i, prompt in enumerate(prompts):
        # suggestion = content_detection(prompt[0].replace(" ", "").replace("_", " "))
        # if suggestion == "block":
        #     prompt = ["▁ 您 输 入 的 内容 中 存在 某些 词 汇 , 触 发 了 阿 里 云 的 敏感 词 检 测 , 建议 您 重新 输 入 。 若 有 误 触 情况 , 可 联系 cone @ x - vent ure . tech 寻 求 帮助 ~"]
        #     generation_args["src"] = "zh"
            # generation_args["tar"] = "zh"
        request_object = {"input": prompt, **generation_args}
        max_len = generation_args.get("max_tokens", 0)
        BATCH_QUEUE.put(WorkItem(len(prompt) + max_len, i, ret_queue, request_object))
    unordered_results = []
    for _ in prompts:
        unordered_results.append(ret_queue.get())
    # resort results by the original ordering
    # weirdly, openai returns to you a flat list if you gave multiple prompts
    reordered = sorted(unordered_results, key=lambda x: x[0])
    results = []
    for (_, generations) in reordered:
        if isinstance(generations, Exception):
            raise generations
        results += generations
    # transform the result into the openai format
    return OAIResponse(results).__dict__()


@app.route("/")
def index():
    # TODO(roller): decouple demopage.html
    # fn = pkg_resources.resource_filename("fairseq", "service/index.html")
    #fn = '/mnt/petrelfs/yuanfei/project/demo/demo-2023-02-13/CM2M/fairseq/fairseq/service/index_one_box.html'
    fn = '../fairseq/service/index_one_box.html'
    # fn='/mnt/petrelfs/zhaoguangxiang/CM2M_1216/fairseq/fairseq/service/index.html'
    with open(fn) as f:
        return f.read()


def cli_main():
    """
    Hosted version of the web UI for generation.
    """

    global port, MODE, cfg
    parser = options.get_generation_parser()
    parser.add_argument(
        "--last_ckpt_dir",
        default=None,
        help="load specific checkpoint for lang pair"
    )
    parser.add_argument(
        "--force_reload",
        default=False,
        help="reload model"
    )
    parser.add_argument(
        "--ensemble_type",
        default=4,
        type=int,
        help="combine different branch result. 1: M-L model; 2: L-M model; 4: M-M Model"
    )
    parser.add_argument(
        "--is_src",
        default=None,
        type=str
    )
    # parser = options.get_interactive_generation_parser()

    # dumb defaults overriding
    parser.set_defaults(lr_scheduler=None, criterion='label_smoothed_cross_entropy')
    flat_launch_args = []
    for s in LAUNCH_ARGS:
        flat_launch_args += s.split()
    # from fairseq import pdb; pdb.set_trace()
    args = options.parse_args_and_arch(parser, input_args=flat_launch_args)
    # args = options.parse_args_and_arch(parser)
    # args.criterion='label_smoothed_cross_entropy'
    # args.path ="s3://CM2M_bucket/xujingjing/ckpt/all-zh-new/averaged_model_10.pt"
    args.data = os.path.dirname(args.path)  # hardcode the data arg

    port = DEFAULT_PORT
    cfg = convert_namespace_to_omegaconf(args)
    cfg.distributed_training.distributed_world_size = TOTAL_WORLD_SIZE
    dist_utils.call_main(cfg, worker_main, namespace_args=args)
    # try:
    #     f.close()
    # finally:
    #     f.close()


if __name__ == "__main__":
    cli_main()
