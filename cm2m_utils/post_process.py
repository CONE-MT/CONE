import argparse
import re


def _character_bleu_language_set():
    return ["ko", "zh_cn", "ja",
            "zh_tw", "my", "ka",
            "ku", "zh", "ur"]


def _get_lines(file_path):
    lines = []
    with open(file_path, 'r', encoding="utf-8") as reader:
        for line in reader:
            line = re.sub(r"__(\w+)(-*)(_*)(\w*)__", "", line)
            lines.append(line.strip())
    return lines


def _convert_to_char_list(lines, language):
    res = []
    if language in _character_bleu_language_set():
        for line in lines:
            char_list = [c for c in line]
            res.append(" ".join(char_list))
    else:
        res = lines
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="decoding output file path")
    parser.add_argument("--save_file", required=True, help="ground truth path")
    parser.add_argument("--language", required=True, help="language")
    args = parser.parse_args()

    lines = _get_lines(args.input_file)
    lines = _convert_to_char_list(lines, args.language)
    with open(args.save_file, 'w', encoding="utf-8") as writer:
        for line in lines:
            writer.write("%s\n" % line)


