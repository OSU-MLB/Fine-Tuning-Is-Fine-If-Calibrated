import torch
import logging
import os
from torchvision.datasets.utils import download_and_extract_archive


"""

Data structure utilities

"""


def flatten_list(l):
    flattened_list = [i for _l in l for i in _l]
    return flattened_list


def all_in(a, b):
    is_all_in = all([i in b for i in a])
    return is_all_in


def is_json(s):
    try:
        import json
        json.loads(s)
        return True
    except ValueError:
        return False


def type_or_none(T):
    def _type_or_none(value):
        try:
            return T(value)
        except ValueError:
            return None

    return _type_or_none

"""

 FS utilities

 Modified from:

@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com

Copyright (c) 2018 The Python Packaging Authority
"""


def check_exists(root, file_name):
    if not os.path.exists(os.path.join(root, file_name)):
        raise FileNotFoundError(f"Dataset directory {file_name} not found under {root}. ")


def read_list_from_file(file_name):
    l = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            l.append(line.strip())
    return l


def download_data(root: str, file_name: str, archive_name: str, url_link: str):
    if not os.path.exists(os.path.join(root, file_name)):
        logging.info("Downloading {}".format(file_name))
        try:
            download_and_extract_archive(url_link, download_root=root, filename=archive_name, remove_finished=False)
        except Exception:
            raise Exception(f"Fail to download {archive_name} from url link {url_link}. ")

def torch_load(file_name, map_location=None):
    level = logging.getLogger().level
    f = torch.load(file_name, map_location=map_location)
    return f

def torch_save(obj, file_name):
    level = logging.getLogger().level
    if level == logging.DEBUG:
        logging.debug(f'Not saving {file_name} in debug mode. ')
        return
    torch.save(obj, file_name)


def makedirs(dir):
    level = logging.getLogger().level
    if level == logging.DEBUG:
        logging.debug(f'Not creating {dir} in debug mode. ')
        return
    if not os.path.exists(dir):
        os.makedirs(dir)

"""

Other utilities

"""

class DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

def print_metrics(metrics):
    for _k, _r in metrics.items():
        logging.info(f'{_k} metrics: \n{_r}')
