import os
import sys


from petrel_client.client import Client


def readlines(client, url):
    response = client.get(url, enable_stream=True, no_cache=True)

    lines = []
    for line in response.iter_lines():
        lines.append(line.decode('utf-8'))
    return lines


def writer_out(client, lines, output_path):
    import io
    with io.BytesIO() as stream:
        for i, line in enumerate(lines):
            stream.write((line + "\n").encode())
            if i % 10000 == 0 and i != 0:
                print("processed {} lines".format(i))
        client.put(output_path, stream.getvalue())


if __name__ == '__main__':
    file_path = sys.argv[1]
    save_dir = sys.argv[2]
    split_parts = int(sys.argv[3])
    config_path = sys.argv[4] if len(
        sys.argv) > 4 else "/mnt/cache/yuanfei/envs/petreloss.conf"

    client = Client(config_path)
    if "s3://" not in file_path:
        raise ValueError("{} not ceph path".format(file_path))

    file_name = os.path.basename(file_path)
    src_lines = readlines(client, file_path)
    src_size = len(src_lines)
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
        target_file_path = os.path.join(save_dir, "split_%s" % i, file_name)
        # print("%s,  start: %s, end: %s, sub numbers: %s" % (i, start, end, len(sub_src)))
        writer_out(client, sub_src, target_file_path)
        i += 1
    print("finish split !")
