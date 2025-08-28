import sys

import pandas as pd

src_lang_col = "src_lg"
tgt_lang_col = "tgt_lg"
fairseq_score_col = "fairseq_score"
sacre_score_col = "scare_score"
multi_bleu_score_col = "multi_bleu_score"


def get_lg_pair(line):
    if "src lang" in line:
        parts = line.split()
        s_lg, t_lg = parts[2].strip(), parts[5].strip()
        return s_lg, t_lg
    return None


def get_fairseq_score(line):
    if "Generate test with beam=" in line:
        parts = line.replace(",", "").split()
        score = float(parts[6])
        return score
    return None


def get_sacre_score(line):
    if "version.1.5.1" in line:
        parts = line.split()
        score = float(parts[2])
        return score
    return None


def get_multi_bleu_score(line):
    if "BLEU = " in line:
        parts = line.replace(",", "").split()
        score = float(parts[2])
        return score
    return None


if __name__ == '__main__':
    file_path = sys.argv[1]
    # save_path = sys.argv[2]
    res_df = pd.DataFrame(columns=[src_lang_col, tgt_lang_col, fairseq_score_col, sacre_score_col, multi_bleu_score_col])
    with open(file_path, "r", encoding="utf-8") as reader:
        lines = reader.readlines()
        curr_ix = 0
        while curr_ix < len(lines):
            s_lg, t_lg = get_lg_pair(lines[curr_ix])
            fairseq_score = get_fairseq_score(lines[curr_ix+1])
            sacre_score = get_sacre_score(lines[curr_ix + 3])
            multi_bleu_score = get_multi_bleu_score(lines[curr_ix + 5])
            res_df.loc[res_df.shape[0]] = [s_lg, t_lg, fairseq_score, sacre_score, multi_bleu_score]
            curr_ix = curr_ix + 6
        print(res_df)
    # res_df.to_csv(save_path, index=False)


