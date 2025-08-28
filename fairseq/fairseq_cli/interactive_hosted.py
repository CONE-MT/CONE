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

import random
import threading
import traceback

import requests
import torch
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException

from fairseq import options
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import utils as dist_utils
from fairseq.service.queue import PriorityQueueRingShard
from fairseq.service.workers import WorkItem
from fairseq.service.constants import (
    MAX_SEQ_LEN,
    MAX_BATCH_TOKENS,
    DEFAULT_PORT,
    TOTAL_WORLD_SIZE,
    LAUNCH_ARGS,
)
from fairseq.service.utils import get_my_ip, encode_fn, build_logger
from fairseq.service.responses import OAIResponse

from fairseq.hub_utils import GeneratorInterface

import logging
from flask_cors import CORS

from fairseq_cli.demo_utils import content_detection

app = Flask(__name__)
CORS(app, supports_credentials=True)
import json

# global state (mutable!)
cfg = None
port = 10054
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
        #     # score = sacrebleu.sentence_bleu(translation_list, [src_list]).score
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
                # raise queue.Empty
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
                        "sample_ids": []
                    }
                    request_object["content_detection_flag"] = True
                    for sample_id, work_item in enumerate(batch):
                        ro = work_item.data
                        #print("ro, ", ro)
                        # ro, {'input': ['▁how ▁are ▁you'], 'src': 'en', 'tar': 'zh', 'temperature': 1.0, 'top_p': 1.0,
                        #      'n': 1}
                        request_object["inputs"].append(ro["input"])
                        request_object["min_tokens"].append(ro.get("min_tokens", 0))
                        request_object["max_tokens"].append(
                            ro.get("max_tokens", MAX_SEQ_LEN)
                        )
                        request_object["srcs"].append(ro.get("src"))
                        request_object["tars"].append(ro.get("tar"))
                        request_object["sample_ids"].append(sample_id)

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
                        request_object_back['sample_ids'] = request_object['sample_ids']
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
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        [model.cuda() for model in models]
    generator.build_generator()
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
        #logging.info('len_sent'+str(len_sent))
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

@app.route("/v2/engines/<engine>/completions", methods=["POST"])
def completions_v2(engine=None):
    url='http://10.140.0.32:10191/v1/engines/175b/completions'
    data=request.json
    print("request: ", data)
    headers={'Accept': '*/*',
     'Accept-Encoding': 'gzip, deflate',
     'Accept-Language': 'zh-CN,zh;q=0.9',
     'Connection': 'keep-alive',
     'Content-Length': '40',
     'Content-Type': 'application/json',
     'Host': '10.140.0.32:10191',
     'Origin': 'http://10.140.0.32:10191',
     'Referer': 'http://10.140.0.32:10191/',
     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'}

    res = requests.post(url, json=data, headers=headers)

    return json.loads(res.content)



from hashlib import md5
import time
# Generate salt and sign
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()


baidu_vs_us_code = {'ar': 'ara', 'sq': 'alb', 'an': 'arg', 'ay': 'aym', 'os': 'oss', 'or': 'ori', 'pl': 'pl', 'ba': 'bak', 'be': 'bel', 'bg': 'bul', 'bem': 'bem', 'bal': 'bal', 'bho': 'bho', 'cv': 'chv', 'da': 'dan', "nan": 'ir', 'nds': 'log', 'ru': 'ru', 'fr': 'fra', 'sa': 'san', 'fo': 'fao', 'gd': 'gla', 'km': 'hkm', 'gu': 'guj', 'gn': 'grn', 'ko': 'kor', 'hak': 'hak', 'ha': 'hau', 'ky': 'kir', 'ca': 'cat', 'kab': 'kab', 'csb': 'kah', 'co': 'cos', 'tlh': 'kli', 'ks': 'kas', 'la': 'lat', 'ltg': 'lag', 'ln': 'lin', 'rue': 'ruy', 'rm': 'roh', 'ms': 'may', 'mg': 'mg', 'mh': 'mah', 'mfe': 'mau', 'mt': 'mlt', 'no': 'nor', 'af': 'afr', 'pt': 'pt', 'ps': 'pus', 'ny': 'nya', 'ja': 'jp', 'sc': 'srd', 'sr': 'srp', 'eo': 'epo', 'sk': 'sk', 'so': 'som', 'th': 'th', 'ta': 'tam', 'te': 'tel', 'uk': 'ukr', 've': 'ven', 'es': 'spa', 'hu': 'hu', 'hil': 'hil', 'nn': 'nno', 'sn': 'sna', 'su': 'sun', 'en': 'en', 'it': 'it', 'ia': 'ina', 'ig': 'ibo', 'hy': 'arm', 'zh': 'zh', 'zhtrad': 'cht', 'zu': 'zul', 'ga': 'gle', 'arq': 'arq', 'am': 'amh', 'az': 'aze', 'et': 'est', 'om': 'orm', 'fa': 'per', 'eu': 'baq', 'se': 'sme', 'is': 'ice', 'ts': 'tso', 'de': 'de', 'tet': 'tet', 'fil': 'fil', 'fur': 'fri', 'kg': 'kon', 'kl': 'kal', 'grc': 'gra', 'nl': 'nl', 'ht': 'ht', 'gl': 'glg', 'cs': 'cs', 'kn': 'kan', 'kw': 'cor', 'cr': 'cre', 'hr': 'hrv', 'gom': 'kok', 'lo': 'lao', 'lv': 'lav', 'lg': 'lug', 'rw': 'kin', 'ro': 'ro', 'my': 'bur', 'ml': 'mal', 'mai': 'mai', 'mi': 'mao', 'mww': 'hmn', 'nap': 'nea', 'st': 'sot', 'pa': 'pan', 'tw': 'twi', 'sv': 'swe', 'sm': 'sm', 'nb': 'nob', 'sw': 'swa', 'tr': 'tr', 'tl': 'tgl', 'wa': 'wln', 'wo': 'wol', 'he': 'heb', 'fy': 'fry', 'dsb': 'los', 'ceb': 'ceb', 'hi': 'hi', 'vi': 'vie', 'ace': 'ach', 'io': 'ido', 'iu': 'iku', 'zza': 'zaz', 'jv': 'jav', 'oc': 'oci', 'ak': 'aka', 'as': 'asm', 'ast': 'ast', 'br': 'bre', 'pot': 'pot', 'pam': 'pam', 'nso': 'ped', 'bi': 'bis', 'bs': 'bos', 'tt': 'tat', 'dv': 'div', 'fi': 'fin', 'ff': 'ful', 'hsb': 'ups', 'ka': 'geo', 'ang': 'eno', 'hup': 'hup', 'kr': 'kau', 'xh': 'xho', 'crh': 'cri', 'que': 'que', 'ku': 'kur', 'rom': 'rom', 'li': 'lim', 'lb': 'ltz', 'lt': 'lit', 'jbo': 'loj', 'mr': 'mar', 'mk': 'mac', 'gv': 'glv', 'bn': 'ben', 'nr': 'nbl', 'ne': 'nep', 'pap': 'pap', 'chr': 'chr', 'sh': 'sec', 'si': 'sin', 'tg': 'tgk', 'ti': 'tir', 'tk': 'tuk', 'cy': 'wel', 'ur': 'urd', 'el': 'el', 'szl': 'sil', 'haw': 'haw', 'sd': 'snd', 'syr': 'syr', 'id': 'id', 'yi': 'yid', 'inh': 'ing', 'yo': 'yor', 'lzh': 'wyw', 'frm': 'frm'}

def baidu_translation_reptile(from_lang, to_lang, query, compare_baseline):
    from fairseq.fairseq_cli.baidu_translate import BaiDuTranslater
    if from_lang not in baidu_vs_us_code or to_lang not in baidu_vs_us_code:
        from_lang = "zh"
        to_lang = "zh"
        query = "百度不支持的语言种类"
        compare_baseline = None
    else:
        from_lang = baidu_vs_us_code[from_lang]
        to_lang = baidu_vs_us_code[to_lang]

    translater = BaiDuTranslater(query)
    # 获取sign的值
    sign = translater.make_sign()
    # 构建参数
    data = translater.make_data(sign, from_lang, to_lang)
    # 获取翻译内容
    text = translater.get_content(data)
    time.sleep(3)
    score = 0
    # TODO tmp score=0
    # translation_list = " ".join([c for c in text])
    # if compare_baseline is not None:
    #     baseline_list = " ".join([c for c in compare_baseline])
    #     score = sacrebleu.sentence_bleu(translation_list, [baseline_list]).score
    results = [{"text": text, "score": score}]
    return results, text


def baidu_translation(from_lang, to_lang, query, compare_baseline):
    # Set your own appid/appkey.
    appid = '20230221001569868'
    appkey = 'M9Wwu3QDNiVD8qT8yjVl'

    if from_lang not in baidu_vs_us_code or to_lang not in baidu_vs_us_code:
        from_lang = "zh"
        to_lang = "zh"
        query = "百度不支持的语言种类"
        compare_baseline = None
    else:
        from_lang = baidu_vs_us_code[from_lang]
        to_lang = baidu_vs_us_code[to_lang]


    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path

    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}
    print(payload)
    time.sleep(3)
    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()

    text = result['trans_result'][0]["dst"]
    score = 0
    # TODO tmp score=0
    # translation_list = " ".join([c for c in text])
    # if compare_baseline is not None:
    #     baseline_list = " ".join([c for c in compare_baseline])
    #     score = sacrebleu.sentence_bleu(translation_list, [baseline_list]).score
    results = [{"text": text, "score": score}]
    return results, text

# port for baidu
@app.route("/v3/engines/<engine>/completions", methods=["POST"])
def completions_v3(engine=None):
    # import time
    data = request.json
    print("request: ", data)

    # For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
    from_lang = data["src"]
    to_lang = data["tar"]
    query = data["prompt"].replace("\n", "")
    origin_prompt = data["origin_prompt"] if "origin_prompt" in data else None
    back_translate =bool(data['back_translate']) if "back_translate" in data else False
    generated_prompt = data["generated_prompt"] if "generated_prompt" in data else None
    has_generated = False if generated_prompt is None or len(generated_prompt) == 0 else True
    try:
        if back_translate and not has_generated:
            # original_result, text = baidu_translation(from_lang, to_lang, query, None)
            # back_result, _ = baidu_translation(to_lang, from_lang, text, query)

            original_result, text = baidu_translation_reptile(from_lang, to_lang, query, None)
            back_result, _ = baidu_translation_reptile(to_lang, from_lang, text, query)

            original_result.extend(back_result)
            return OAIResponse(original_result).__dict__()
        elif back_translate and has_generated:
            # back_result, _ = baidu_translation(to_lang, from_lang, generated_prompt, query)
            back_result, _ = baidu_translation_reptile(to_lang, from_lang, generated_prompt, query)
            return OAIResponse(back_result).__dict__()
        else:
            # original_result, text = baidu_translation(from_lang, to_lang, query, origin_prompt)
            original_result, text = baidu_translation_reptile(from_lang, to_lang, query, origin_prompt)
            return OAIResponse(original_result).__dict__()
        # Show response
    except Exception as e:
        print("current error for baidu translation: ", e)
        print(traceback.format_exc())
        text = "百度不支持的语言种类"
        score = 0
    results = [{
        "text": text,
        "score": score
    }]
    print("baidu translation: ", results)
    return OAIResponse(results).__dict__()


def generate_google_translation(src_language_code, target_language_code, sample_text, compare_baseline):
    from os import environ

    from google.cloud import translate

    project_id = environ.get("PROJECT_ID", "")
    assert project_id
    parent = f"projects/{project_id}"
    client = translate.TranslationServiceClient()

    shared_lg_codes = [
        'zh', 'he', 'jv', 'uz', 'om', 'hy', 'ht', 'mn', 'bm', 'mt', 'zu', 'mr', 'uk', 'bho', 'bs', 'tk', 'pt', 'haw', 'lo', 'kn', 'mg',
        'tg', 'az', 'qu', 'mk', 'as', 'ro', 'pl', 'be', 'ja', 'xh', 'ln', 'el', 'ig', 'ky', 'sk', 'hu', 'nso', 'ku',
        'gl', 'de', 'ckb', 'te', 'ne', 'gn', 'fr', 'ka', 'lt', 'es', 'tr', 'ny', 'ur', 'cs', 'lv', 'ak', 'da', 'sa',
        'fy', 'mi', 'ti', 'bg', 'vi', 'eu', 'am', 'ta', 'lb', 'ar', 'sn', 'ru', 'co', 'so', 'ay', 'ca', 'fil', 'sm',
        'ug', 'gom', 'ceb', 'en', 'ilo', 'sd', 'lg', 'si', 'kk', 'yi', 'st', 'ko', 'eo', 'or', 'bn', 'sl', 'ms', 'ee',
        'af', 'su', 'tt', 'hi', 'nl', 'sr', 'gu', 'ga', 'th', 'yo', 'it', 'ha', 'pa', 'hr', 'km', 'ml', 'is', 'la',
        'fa', 'no', 'rw', 'mai', 'ps', 'id', 'gd', 'sq', 'dv', 'fi', 'tl', 'my', 'sv', 'ts', 'cy', 'sw', 'et'
    ]
    replace_dict = {
        "zhtrad": "zh-TW",
    }

    if src_language_code not in shared_lg_codes and src_language_code not in replace_dict:
        src_language_code = "zh"
        sample_text = "目前 Google 不支持的语言"

    if target_language_code not in shared_lg_codes and target_language_code not in replace_dict:
        target_language_code = "zh"
        sample_text = "目前 Google 不支持的语言"

    if src_language_code in replace_dict:
        src_language_code = replace_dict[src_language_code]
    if target_language_code in replace_dict:
        target_language_code = replace_dict[target_language_code]

    time.sleep(3)
    response = client.translate_text(
        source_language_code=src_language_code,
        contents=[sample_text],
        target_language_code=target_language_code,
        parent=parent,
    )

    text = response.translations[0].translated_text
    score = 0
    # TODO tmp score=0
    # if compare_baseline is not None:
    #     translation_list = " ".join([c for c in text])
    #     src_list = " ".join([c for c in compare_baseline])
    #     score = sacrebleu.sentence_bleu(translation_list, [src_list]).score
    results = [{
        "text": text,
        "score": score
    }]
    return results, text

# port for google
@app.route("/v4/engines/<engine>/completions", methods=["POST"])
def completions_v4(engine=None):

    data = request.json
    print("request: ", data)

    src_language_code = data["src"]
    target_language_code = data["tar"]
    sample_text = data["prompt"].replace("\n", "")
    origin_prompt = data["origin_prompt"] if "origin_prompt" in data else None
    back_translate = bool(data['back_translate']) if "back_translate" in data else False
    generated_prompt = data["generated_prompt"] if "generated_prompt" in data else None
    has_generated = False if generated_prompt is None or len(generated_prompt) == 0 else True

    try:
        if back_translate and not has_generated:
            original_result, text = generate_google_translation(src_language_code, target_language_code, sample_text, origin_prompt)
            back_result, _ = generate_google_translation(target_language_code, src_language_code, text, sample_text)
            original_result.extend(back_result)
            return OAIResponse(original_result).__dict__()
        elif back_translate and has_generated:
            back_result, _ = generate_google_translation(target_language_code, src_language_code, generated_prompt, sample_text)
            return OAIResponse(back_result).__dict__()
        else:
            original_result, text = generate_google_translation(src_language_code, target_language_code, sample_text, origin_prompt)
            return OAIResponse(original_result).__dict__()
        # Show response
    except Exception as e:
        print("current error for Google translation: ", e)
        print(traceback.format_exc())
        text = "Google 不支持的语言种类"
        score = 0
    results = [{
        "text": text,
        "score": score
    }]
    print("google translation: ", results)
    return OAIResponse(results).__dict__()

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
