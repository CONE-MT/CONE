# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

MAX_SEQ_LEN = 256
BATCH_SIZE = 128  # silly high bc we dynamically batch by MAX_BATCH_TOKENS
MAX_BATCH_TOKENS = 256
DEFAULT_PORT = 10045
MODEL_PARALLEL = 1
TOTAL_WORLD_SIZE = 1

try:
    # internal logic denoting where checkpoints are in meta infrastructure
    from metaseq_internal.constants import CHECKPOINT_FOLDER
except ImportError:
    # CHECKPOINT_FOLDER should point to a shared drive (e.g. NFS) where the
    # checkpoints from S3 are stored. As an example:
    # CHECKPOINT_FOLDER = "/example/175B/reshard_no_os"
    # $ ls /example/175B/reshard_no_os
    # reshard-model_part-0.pt
    # reshard-model_part-1.pt
    # reshard-model_part-2.pt
    # reshard-model_part-3.pt
    # reshard-model_part-4.pt
    # reshard-model_part-5.pt
    # reshard-model_part-6.pt
    # reshard-model_part-7.pt
    CHECKPOINT_FOLDER = "/data/zhuo/code/cm2m/data_model/fi_en"
    # CHECKPOINT_FOLDER = "./"
    #CHECKPOINT_FOLDER = "/mnt/lustre/limukai/Efficient-LM/meta-train/eva_transformer_lm_gpt3_large_lr5e-6_possinusoidal_thepile_m4096_b4_v2"

    VOCAB_FOLDER = '/mnt/cache/limukai/Efficient-LM-Metaseq/metaseq/projects/OPT/assets/'
# tokenizer files
BPE_MERGES = os.path.join(VOCAB_FOLDER, "gpt2-merges.txt")
BPE_VOCAB = os.path.join(VOCAB_FOLDER, "gpt2-vocab.json")
MODEL_FILE = os.path.join(CHECKPOINT_FOLDER, "checkpoint_best.pt")


LAUNCH_ARGS = [
   # f"--model-parallel-size {MODEL_PARALLEL}",
    #f"--distributed-world-size {TOTAL_WORLD_SIZE}",
    "--task translation_multi_simple_epoch",
    #"--arch transformer_wmt_en_de_big",
    # f"--bpe-merges {BPE_MERGES}",
    # f"--bpe-vocab {BPE_VOCAB}",
    "--bpe sentencepiece",
    #"--arch  transformer",
    "--langs af,am,ar,ast,az,ba,be,bg,bn,br,bs,ca,ceb,cs,cy,da,de,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kk,km,kn,ko,lb,lg,ln,lo,lt,lv,mg,mk,ml,mn,mr,ms,my,ne,nl,no,ns,oc,or,pa,pl,ps,pt,ro,ru,sd,si,sk,sl,so,sq,sr,ss,su,sv,sw,ta,th,tl,tn,tr,uk,ur,uz,vi,wo,xh,yi,yo,zh,zu,pt_br,zh_cn,zh_tw,trg1,nb,ku,eo,eu",
    "--gen-subset test",
    "--remove-bpe sentencepiece",
    f"--sentencepiece-model  {CHECKPOINT_FOLDER}/spm.128k.model ",
    "-s en ",
    "-t fi" ,
    # f"--merges-filename {BPE_MERGES}",  # TODO(susanz): hack for getting interactive_hosted working on public repo
    # f"--vocab-filename {BPE_VOCAB}",  # TODO(susanz): hack for getting interactive_hosted working on public repo
    f"--path {CHECKPOINT_FOLDER}/418M_last_checkpoint.pt",
    "--beam 1",
    "--nbest 1",
    "--fp16",
    "--distributed-port -1",
    "--checkpoint-shard-count 1",
    "--encoder-langtok src ",
    "--decoder-langtok",
    #"--use-sharded-state",

    f"--batch-size {BATCH_SIZE}",
    #f"--buffer-size {BATCH_SIZE * MAX_SEQ_LEN}",
    #f"--max-tokens {BATCH_SIZE * MAX_SEQ_LEN}",
    "/tmp",  # required "data" argument.
]
