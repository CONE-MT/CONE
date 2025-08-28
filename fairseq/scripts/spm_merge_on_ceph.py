import os.path
import sys
from petrel_client.client import Client
import io
import random


data_dir = sys.argv[1]
target_dir = sys.argv[2]
split_num = int(sys.argv[3])

target_dir = os.path.join(target_dir, "split_%s" % split_num)

client = Client("/mnt/cache/yuanfei/envs/petreloss.conf")


def write_train_res_out(inputs, output_path):
    with io.BytesIO() as stream:
        for i, (input_path, lg_id) in enumerate(inputs):
            response = client.get(input_path, enable_stream=True, no_cache=True)
            for i, line in enumerate(response.iter_lines()):
                lg_str = "__%s__ " % lg_id
                stream.write(lg_str.encode() + line + "\n".encode())
                if i % 10000 == 0 and i != 0:
                    print("{}, processed {} lines".format(input_path, i))
        client.put(output_path, stream.getvalue())


def write_dev_res_out(indices, input_lines, output_path ):
    res = [input_lines[i] for i in indices]
    print("input line size: %s, sample line size: %s" % (len(input_lines), len(res)))
    with io.BytesIO() as stream:
        for i, (line, lg_id) in enumerate(res):
            lg_str = "__%s__ " % lg_id
            stream.write(lg_str.encode() + line + "\n".encode())
            if i % 10000 == 0 and i != 0:
                print("processed {} lines".format(i))
        client.put(output_path, stream.getvalue())


def down_sampling_dev(inputs):
    res = []
    for _, (input_path, lg) in enumerate(inputs):
        response = client.get(input_path, enable_stream=True, no_cache=True)
        for i, line in enumerate(response.iter_lines()):
            res.append((line, lg))
    sample_size = min(len(res), 2000)
    indices = random.sample(range(0, len(res)), sample_size)
    return indices, res


def generate_core_output(src_inputs, target_inputs, src, split):
    output_lg = "%s-trg1" % src
    src_output = os.path.join(target_dir, "%s.%s.%s" % (split, output_lg, src))
    trg1_output = os.path.join(target_dir, "%s.%s.trg1" % (split, output_lg))
    src_inputs = sorted(src_inputs, key=lambda x: x[0])
    target_inputs = sorted(target_inputs, key=lambda x: x[0])
    res = []
    for s_input, t_input in zip(src_inputs, target_inputs):
        s_input, t_input = s_input[0], t_input[0]
        s = s_input[s_input.rfind("/")+1:s_input.rfind(".")]
        t = t_input[t_input.rfind("/")+1:t_input.rfind(".")]
        res.append(s == t)
    assert all(res), "src and trg input is not parallel !"
    if not client.contains(src_output):
        if split == "train":
            write_train_res_out(src_inputs, src_output)
    else:
        print("already merged!")
    if not client.contains(trg1_output):
        if split == "train":
            write_train_res_out(target_inputs, trg1_output)
    else:
        print("already merged!")

    if split == "dev":
        indices, src_res = down_sampling_dev(src_inputs)
        _, target_res = down_sampling_dev(target_inputs)
        write_dev_res_out(indices, src_res, src_output)
        write_dev_res_out(indices, target_res, trg1_output)



def get_input_paths(shard_dir, split, lg):
    # $target_path /$filename / train.$filename.$src >> $target_merge_path / train.$lg - trg1.$lg
    # $target_path /$filename / train.$filename.$trg >> $target_merge_path / train.$lg - trg1.trg1
    src_input_paths, trg_input_paths = set(), set()
    dir_names = client.list(shard_dir)
    for dir_name in dir_names:
        dir_name = dir_name.replace("/", "")
        src, trg = dir_name.split("-")
        if src == lg:
            sub_src_dir = os.path.join(shard_dir, dir_name, "%s.%s.%s" % (split, dir_name, src))
            sub_trg_dir = os.path.join(shard_dir, dir_name, "%s.%s.%s" % (split, dir_name, trg))
            src_input_paths.add((sub_src_dir, src))
            trg_input_paths.add((sub_trg_dir, trg))
        elif trg == lg:
            sub_src_dir = os.path.join(shard_dir, dir_name, "%s.%s.%s" % (split, dir_name, trg))
            sub_trg_dir = os.path.join(shard_dir, dir_name, "%s.%s.%s" % (split, dir_name, src))
            src_input_paths.add((sub_src_dir, trg))
            trg_input_paths.add((sub_trg_dir, src))
    return src_input_paths, trg_input_paths


def get_lgs(shard_dir_name):
    lgs_set = set()
    dir_names = client.list(shard_dir_name)
    for dir_name in dir_names:
        src, trg = dir_name.replace("/", "").split("-")
        lgs_set.add(src)
        lgs_set.add(trg)
    return lgs_set


for split in ["train", "dev", "test"]:
    shard_dir_name = os.path.join(data_dir, "split_%s" % split_num)
    lg_set = get_lgs(shard_dir_name)
    if split in ["dev", "test"] and split_num != 0:
        continue
    for lg in lg_set:
        print("current lg: %s" % lg)
        src_inputs, trg_inputs = get_input_paths(shard_dir_name, split, lg)
        generate_core_output(src_inputs, trg_inputs, lg, split)