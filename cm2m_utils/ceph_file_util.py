import io
import logging

import torch

from cm2m_utils.ceph_io import CephManager
import logging

try:
    from petrel_client.client import Client
except ImportError:
    # raise ImportError('Please install petrel_client')
    logging.warning('Please install petrel_client''Please install petrel_client')

import os

logger = logging.getLogger(__name__)


class CEPHFileUtil(object):
    def __init__(self, s3cfg_path='~/petreloss.conf'):
        self.ceph_handler = CephManager(s3cfg_path)

    @staticmethod
    def _is_use_ceph(path):
        return True if "s3://" in path else False

    def make_dirs(self, dir_path, exist_ok):

        if not self._is_use_ceph(dir_path):
            os.makedirs(dir_path, exist_ok=exist_ok)

    def exists(self, file_path):

        use_ceph = self._is_use_ceph(file_path)
        return self.ceph_handler.exists(file_path) if use_ceph else os.path.exists(file_path)

    def lexists(self, file_path):

        def lexists_for_ceph(file_path):
            tmp_str = file_path[:-1] if file_path.endswith("/") else file_path
            return self.exists(tmp_str[:tmp_str.rindex("/")])

        use_ceph = self._is_use_ceph(file_path)
        return lexists_for_ceph(file_path) if use_ceph else os.path.lexists(file_path)

    def remove(self, file_path):

        use_ceph = self._is_use_ceph(file_path)
        self.ceph_handler.remove(file_path) if use_ceph else os.remove(file_path)

    def load_checkpoint(self, file_path, map_location):
        def _load_from_local(local_path, m_location):
            with open(local_path, "rb") as f:
                state = torch.load(f, map_location=m_location)
            return state

        def _load_from_ceph(url, m_location):
            return self.ceph_handler.load_model(url, map_location=m_location)

        use_ceph = self._is_use_ceph(file_path)
        return _load_from_ceph(file_path, map_location) if use_ceph else _load_from_local(file_path, map_location)

    def readlines(self, url):
        return self.ceph_handler.readlines(url)

    def get(self, url):
        return self.ceph_handler.get(url)

    def put(self, url, obj):
        self.ceph_handler.write(url, obj)
