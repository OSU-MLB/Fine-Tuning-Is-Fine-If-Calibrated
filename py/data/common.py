import copy
import time
import logging
import torch
from torch.utils.data import DataLoader, Dataset


class DomainInfo:
    def __init__(self, all_classes, visible_classes, invisible_classes=None, num_classes=None):
        assert isinstance(all_classes, torch.Tensor), 'all_classes must be a tensor. '
        assert isinstance(visible_classes, torch.Tensor), 'visible_classes must be a tensor. '
        assert (invisible_classes is None) or (isinstance(invisible_classes, torch.Tensor)), 'invisible_classes must be a tensor. '
        self.all_classes = all_classes
        self.visible_classes = visible_classes
        if invisible_classes is None:
            visible_clz_ind = torch.isin(all_classes, visible_classes)
            invisible_clz_ind = ~visible_clz_ind
            invisible_classes = all_classes[invisible_clz_ind]
        if num_classes is None:
            num_classes = len(all_classes)
        self.invisible_classes = invisible_classes
        self.num_classes = num_classes
    
    def to(self, device):
        self.all_classes = self.all_classes.to(device)
        self.visible_classes = self.visible_classes.to(device)
        self.invisible_classes = self.invisible_classes.to(device)
        if hasattr(self, 'visible_ind'):
            self.visible_ind = self.visible_ind.to(device)
        if hasattr(self, 'invisible_ind'):
            self.invisible_ind = self.invisible_ind.to(device)
        if hasattr(self, 'labels'):
            self.labels = self.labels.to(device)
        return self
    
    def __repr__(self):
        return f'DomainInfo(all_classes={self.all_classes}, visible_classes={self.visible_classes}, invisible_classes={self.invisible_classes}, num_classes={self.num_classes})'
    
    def __str__(self):
        return self.__repr__()


class IndexedDatasetWrapper(Dataset):
    
        def __init__(self, dataset: Dataset):
            self.dataset = dataset
    
        def __getitem__(self, index):
            return index, self.dataset[index]
    
        def __len__(self):
            return len(self.dataset)
        
        def __getattr__(self, name):
            return getattr(self.dataset, name)
        
        def __setattr__(self, name, value):
            if name == 'dataset':
                self.__dict__[name] = value
            else:
                setattr(self.dataset, name, value)

class PartialDomainDataset(Dataset):

    def __init__(self, dataset: Dataset, domain_info: DomainInfo):
        # start = time.time()
        self._check_dataset_format(dataset)

        self.dataset = IndexedDatasetWrapper(dataset)
        self.domain_info = copy.deepcopy(domain_info)
        self._iterate_ind = None
        self.visible_ind = None
        self.invisible_ind = None
        self.all_ind = None
        self._init_indices()
        # end = time.time()
        # logging.info(f'PartialDomainDataset init time: {end - start:.3f} s. ')


    def visible_mask(self, ind):
        mask = torch.isin(ind, self.visible_ind)
        return mask
    
    def invisible_mask(self, ind):
        mask = torch.isin(ind, self.invisible_ind)
        return mask

    def _check_dataset_format(self, dataset):
        _sample = dataset[0]
        assert isinstance(_sample, tuple) and len(_sample) == 2, 'PartialDomainDataset only supports dataset with (data, target) format. '
        assert hasattr(dataset, 'targets'), 'PartialDomainDataset only supports dataset with targets field. '

    def _init_indices(self):
        logging.debug('Initializing indices... ')
        domain_info = self.domain_info
        visible_classes = domain_info.visible_classes
        invisible_classes = domain_info.invisible_classes
        visible_ind = []
        invisible_ind = []
        all_ind = []
        labels = []
        for i, _c in enumerate(self.dataset.targets):
            if _c in visible_classes:
                visible_ind.append(i)
            elif _c in invisible_classes:
                invisible_ind.append(i)
            all_ind.append(i)
            labels.append(_c)
        self.visible_ind = visible_ind
        self.invisible_ind = invisible_ind
        self.all_ind = all_ind
        self._iterate_ind = self.visible_ind
        domain_info.visible_ind = torch.tensor(self.visible_ind)
        domain_info.invisible_ind = torch.tensor(self.invisible_ind)
        domain_info.labels = torch.tensor(labels)
        logging.debug('Indices initialized. ')

    def __getitem__(self, index):
        return self.dataset[self._iterate_ind[index]]

    def __len__(self):
        return len(self._iterate_ind)

    def set_scope(self, scope):
        assert scope in ['visible', 'invisible', 'all'], f'Invalid scope: {scope}. '
        if scope == 'visible':
            self._iterate_ind = self.visible_ind
        elif scope == 'invisible':
            self._iterate_ind = self.invisible_ind
        elif scope == 'all':
            self._iterate_ind = self.all_ind

    def train(self):
        if hasattr(self.dataset, 'train'):
            self.dataset.train()

    def eval(self):
        if hasattr(self.dataset, 'eval'):
            self.dataset.eval()


"""
Modified from: https://github.com/thuml/Transfer-Learning-Library 

@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com

Copyright (c) 2018 The Python Packaging Authority
"""


class ForeverDataIterator:

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iter = iter(self.dataloader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.dataloader)
