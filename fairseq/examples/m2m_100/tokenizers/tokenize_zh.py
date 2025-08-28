#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import fileinput

from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh

tokenize_zh = TokenizerZh()
for line in fileinput.input():
    print(tokenize_zh(line))
