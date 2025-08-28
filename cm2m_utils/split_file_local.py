import os
import sys
from tqdm import tqdm

if __name__ == '__main__':
    train_file_dir = sys.argv[1]
    src_file_name = sys.argv[2]
    tgt_file_name = sys.argv[3]
    save_dir = sys.argv[4]
    split_parts = int(sys.argv[5]) if len(sys.argv) > 5 else 10

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(train_file_dir, src_file_name), encoding="utf-8") as src_reader, \
            open(os.path.join(train_file_dir, tgt_file_name), encoding="utf-8") as tgt_reader:
        src_lines, tgt_lines = src_reader.readlines(), tgt_reader.readlines()
        assert len(src_lines) == len(tgt_lines)
        part_num = round(len(src_lines) / split_parts)
        range_list = [0 for i in range(split_parts+1)]
        for i in range(1, split_parts+1):
            range_list[i] = (part_num * i)
        range_list[-1] = len(src_lines)
        print(len(src_lines))
        i = 0
        while i < split_parts:
            start = range_list[i]
            end = range_list[i+1]
            sub_src = src_lines[start:end]
            sub_tgt = tgt_lines[start:end]
            sub_dir = os.path.join(save_dir, "split_%s" % i)
            os.makedirs(sub_dir, exist_ok=True)
            print("%s,  start: %s, end: %s, sub numbers: %s" % (i, start, end, len(sub_src)))
            with open(os.path.join(sub_dir, src_file_name), "w", encoding="utf-8") as src_writer, \
                    open(os.path.join(sub_dir, tgt_file_name), "w", encoding="utf-8") as tgt_writer:
                src_writer.writelines(sub_src)
                tgt_writer.writelines(sub_tgt)
            i += 1
    print("finish split !")