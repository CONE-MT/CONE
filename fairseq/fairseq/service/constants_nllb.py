# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

MAX_SEQ_LEN = 256
BATCH_SIZE = 128  # silly high bc we dynamically batch by MAX_BATCH_TOKENS
MAX_BATCH_TOKENS = 256
DEFAULT_PORT = 6021
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
    CHECKPOINT_FOLDER = "../fairseq/service/demo_files/"
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
    "--langs ace,ace_1,acm,acq,aeb,af,ajp,ak,am,apc,ar,ars,ary,arz,as,ast,awa,ay,az,azb,ba,ban,be,ber,bg,bi,bjn,bjn_1,bm,bn,bo,bs,bug,ca,ceb,cjk,ckb,crh,cs,cy,da,de,dik,dyu,dz,ee,el,en,eo,es,et,eu,fa,fi,fj,fo,fon,fr,fur,fuv,ga,gd,gl,gn,gu,ha,he,hi,hne,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kab,kac,kam,kbp,kea,kg,kik,kk,km,kmb,kmr,kn,ko,kr,kr_1,ks,ks_1,ky,lb,lg,li,lij,lmo,ln,lo,lt,ltg,lua,luo,lus,lv,mag,mai,mi,min,mk,ml,mn,mni,mos,mr,ms,mt,my,ne,nl,nn,no,ns,nus,ny,oc,om,or,pa,pag,pap,pl,plt,prs,ps,pt,quy,rn,ro,ru,rw,sa,sat,sc,scn,sd,sg,shn,si,sk,sl,sm,sn,so,sq,sr,ss,st,su,sv,sw,szl,ta,taq,taq_1,te,tg,th,ti,tk,tl,tn,tpi,tr,ts,tt,tum,tw,tzm,ug,uk,umb,ur,uz,vec,vi,war,wo,xh,yi,yo,zh,zhtrad,trg1",
    "--fixed-dictionary  ../fairseq/service/demo_files/data_dict.nllb.txt ",
    "--gen-subset test",
    "--remove-bpe sentencepiece",
    "--sentencepiece-model  ../fairseq/service/demo_files/flores200sacrebleuspm ",
    "-s en ",
    "-t zh" ,
    # f"--merges-filename {BPE_MERGES}",  # TODO(susanz): hack for getting interactive_hosted working on public repo
    # f"--vocab-filename {BPE_VOCAB}",  # TODO(susanz): hack for getting interactive_hosted working on public repo
    f"--path {CHECKPOINT_FOLDER}/nllb200densedst1bcheckpoint",
    "--beam 1",
    "--nbest 1",
    "--distributed-port -1",
    "--checkpoint-shard-count 1",
    "--encoder-langtok src ",
    "--decoder-langtok",
    "--is_nllb_model True",
    #"--use-sharded-state",

    f"--batch-size {BATCH_SIZE}",
    #f"--buffer-size {BATCH_SIZE * MAX_SEQ_LEN}",
    #f"--max-tokens {BATCH_SIZE * MAX_SEQ_LEN}",
    "/tmp",  # required "data" argument.
]
