
<div align="center">
<img src="png/lego-MT_logo.png" border="0" width=1200px/>
</div>

------

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="https://arxiv.org/pdf/2212.10551.pdf">Paper</a> •
  <a href="#citation">Citation</a> 
</p>

![version](https://img.shields.io/badge/version-0.1-blue)

## Installation
Note: Lego-MT requires Python 3.6+.

**Using Pip**
```
pip install -r ./requirements.txt
```


**Installation for local development:**
```
https://github.com/CONE-MT/CONE.git
cd CONE
pip install -e .
```

## What News
You can download the dataset of LegoMT2 from [here](https://huggingface.co/datasets/Lego-MT/Parallel_Dataset).


## Quick Start
Following those steps to show you how to inference with Lego-MT.

#### Step 1: Download Model
You can download the checkpoint of [LegoMT2](https://huggingface.co/Lego-MT/LegoMT2/tree/main).


#### Step 2: Prepare Your Data

```python
  python ./fairseq/scripts/spm_encode.py \
      --model /path/flores200sacrebleuspm \
      --output_format=piece \
      --inputs=/path/src_input \
      --outputs=/path/spm.src

  python ./fairseq/scripts/spm_encode.py \
    --model /path/flores200sacrebleuspm \
    --output_format=piece \
    --inputs=/path/tgt_input \
    --outputs=/path/spm.trg

  python ../fairseq/fairseq_cli/preprocess.py \
      --source-lang ${M2M_SRC} --target-lang ${M2M_TRG} \
      --testpref ${TMP_DIR}/spm \
      --workers 50 \
      --thresholdsrc 0 --thresholdtgt 0 \
      --destdir ${IDX_PATH} \
      --srcdict merge_dict_nllb --tgtdict merge_dict_nllb
```

#### Step 3: Inference
You can choose different ensemble types for inference.

ensemble_type=1: DecFlow (multilingual encoder + languages-specific decoder)

ensemble_type=2: Enc-Flow (languages-specific encoder + multilingual decoder)

ensemble_type=4: Mix-Flow (multilingual encoder + multilingual decoder)

ensemble_type=8: unseen language-specific Flow (the combination of a language-specific encoder and a language-specific decoder)

Notice: the multilingual encoder is named encoder_main.pt and the multilingual decoder is named decoder_m2m.pt in the checkpoint directory.

```python
 # M2M_FNAME: directory path of downloaded lego-MT model
  python ./fairseq/fairseq_cli/generate.py  ${IDX_PATH} \
  --batch-size 128 --path ${M2M_FNAME}/checkpoint_last.pt \
  --last_ckpt_dir ${M2M_FNAME}/${LAST_CKPT_DIR}  \
  -s $1 -t $2 --remove-bpe sentencepiece --beam 5 --task multilingual_translation_branch --lang-pairs ${lang_pairs} \
  --langs $(cat opus-langs.txt) --decoder-langtok --encoder-langtok src --gen-subset test \
  --force_reload True --ensemble_type ${ENSEMBLE_TYPE} 
```

## More Information
[中文介绍 (README in Chinese)](readme_Chinese.md)  

[README in English](readme_English.md)

## Citation
If you find this repository helpful, feel free to cite our paper:
```bibtex
@inproceedings{yuan-etal-2025-legomt2,
    title = "{L}ego{MT}2: Selective Asynchronous Sharded Data Parallel Training for Massive Neural Machine Translation",
    author = "Yuan, Fei  and
      Lu, Yinquan  and
      Li, Lei  and
      Xu, Jingjing",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1200/",
    doi = "10.18653/v1/2025.findings-acl.1200",
    pages = "23359--23376",
    ISBN = "979-8-89176-256-5",
    abstract = "It is a critical challenge to learn a single model for massive languages. Prior methods focus on increasing the model size and training data size. However, large models are difficult to optimize efficiently even with distributed parallel training and translation capacity can interfere among languages. To address the challenge, we propose LegoMT2, an efficient training approach with an asymmetric multi-way model architecture for massive multilingual neural machine translation. LegoMT2 shards 435 languages into 8 language-centric groups and attributes one local encoder for each group{'}s languages and a mix encoder-decoder for all languages. LegoMT2 trains the model through local data parallel and asynchronous distributed updating of parameters. LegoMT2 is 16.2$\times$ faster than the distributed training method for M2M-100-12B (which only for 100 languages) while improving the translation performance by an average of 2.2 BLEU on \textit{Flores-101}, especially performing better for low-resource languages ."
}
```
