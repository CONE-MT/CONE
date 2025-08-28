#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import contextlib
import functools
import io
import sys

import sentencepiece as spm
from fairseq.file_io import PathManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="sentencepiece model to use for encoding"
    )
    parser.add_argument(
        "--inputs", nargs="+", default=["-"], help="input files to filter/encode"
    )
    parser.add_argument(
        "--outputs", nargs="+", default=["-"], help="path to save encoded outputs"
    )
    parser.add_argument("--output_format", choices=["piece", "id"], default="piece")
    parser.add_argument(
        "--min-len",
        type=int,
        metavar="N",
        help="filter sentence pairs with fewer than N tokens",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        metavar="N",
        help="filter sentence pairs with more than N tokens",
    )
    args = parser.parse_args()

    assert len(args.inputs) == len(
        args.outputs
    ), "number of input and output paths should match"

    sp = spm.SentencePieceProcessor()
    sp.Load(args.model)

    if args.output_format == "piece":

        def encode(input):
            return sp.EncodeAsPieces(input)

    elif args.output_format == "id":

        def encode(input):
            return list(map(str, sp.EncodeAsIds(input)))

    else:
        raise NotImplementedError

    if args.min_len is not None or args.max_len is not None:

        def valid(line):
            return (args.min_len is None or len(line) >= args.min_len) and (
                args.max_len is None or len(line) <= args.max_len
            )

    else:

        def valid(lines):
            return True

    def encode_line(line):
        line = line.strip()
        if len(line) > 0:
            line = encode(line)
            if valid(line):
                return line
            else:
                stats["num_filtered"] += 1
        else:
            stats["num_empty"] += 1
        return ["1"]

    is_s3_path = True if "s3" in args.outputs[0] else False
    if is_s3_path:
        stats = {
            "num_empty": 0,
            "num_filtered": 0,
        }
        for i, (input_path, output_path) in enumerate(zip(args.inputs, args.outputs)):
            with io.BytesIO() as stream:
                print(input_path)
                response = PathManager.get_ceph_manager().return_line_stream(input_path)
                for i, line in enumerate(response.iter_lines()):
                    line = line.decode('utf-8')
                    encode_res = " ".join(encode_line(line)) + "\n"
                    stream.write(encode_res.encode())
                    if i % 10000 == 0:
                        print("processed {} lines".format(i))
                PathManager.get_ceph_manager().write(output_path, stream.getvalue())
        print("skipped {} empty lines".format(stats["num_empty"]), file=sys.stderr)
        print("filtered {} lines".format(stats["num_filtered"]), file=sys.stderr)
    else:
        with contextlib.ExitStack() as stack:
            inputs = [
                stack.enter_context(open(input, "r", encoding="utf-8"))
                if input != "-"
                else sys.stdin
                for input in args.inputs
            ]
            outputs = [
                stack.enter_context(open(output, "w", encoding="utf-8"))
                if output != "-"
                else sys.stdout
                for output in args.outputs
            ]

            stats = {
                "num_empty": 0,
                "num_filtered": 0,
            }

            for i, lines in enumerate(zip(*inputs), start=1):
                enc_lines = list(map(functools.partial(encode_line,stats=stats), lines))
                if not any(enc_line is None for enc_line in enc_lines):
                    for enc_line, output_h in zip(enc_lines, outputs):
                        print(" ".join(enc_line), file=output_h)
                if i % 10000 == 0:
                    print("processed {} lines".format(i), file=sys.stderr)

            print("skipped {} empty lines".format(stats["num_empty"]), file=sys.stderr)
            print("filtered {} lines".format(stats["num_filtered"]), file=sys.stderr)


if __name__ == "__main__":
    main()
