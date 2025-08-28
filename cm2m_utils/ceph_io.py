import json
import logging

import torch

try:
    from petrel_client.client import Client
except ImportError:
    # raise ImportError('Please install petrel_client')
    logging.warning('Please install petrel_client''Please install petrel_client')

import io


class CephManager:

    def __init__(self, s2_conf_path='~/petreloss.conf'):
        self.conf_path = s2_conf_path
        self._client = Client(conf_path=s2_conf_path)

    def readlines(self, url):

        response = self.return_line_stream(url)

        lines = []
        for line in response.iter_lines():
            lines.append(line.decode('utf-8'))
        return lines

    def load_data(self, path, ceph_read=False):
        if ceph_read:
            return self.readlines(path)
        else:
            return self._client.get(path, no_cache=True)

    def return_line_stream(self, url):
        return self._client.get(url, enable_stream=True, no_cache=True)

    def get(self, file_path):
        return self._client.get(file_path, no_cache=True)

    def get_with_range(self, file_path, range):
        # range example: 0-512 note: [0, 512]
        return self._client.get(file_path, range=range)

    def load_json(self, json_url):
        return json.loads(self.load_data(json_url, ceph_read=False))

    def load_model(self, model_path, map_location):
        file_bytes = self._client.get(model_path, no_cache=True)
        buffer = io.BytesIO(file_bytes)
        return torch.load(buffer, map_location=map_location)

    def write(self, save_dir, obj):
        self._client.put(save_dir, obj)

    def put_text(self,
                 obj: str,
                 filepath,
                 encoding: str = 'utf-8') -> None:
        self.write(filepath, bytes(obj, encoding=encoding))

    def exists(self, url):
        return self._client.contains(url)
    
    def remove(self, url):
        return self._client.delete(url)
    
    def isdir(self, url):
        return self._client.isdir(url)

    def isfile(self, url):
        return self.exists(url) and not self.isdir(url)

    def listdir(self, url):
        return self._client.list(url)

    def copy(self, src_path, dst_path, overwrite):
        if not overwrite and self.exists(dst_path):
            pass
        object = self._client.get(src_path, no_cache=True)
        self._client.put(dst_path, object)
        return dst_path

    def size(self, url):
        return self._client.size(url)



