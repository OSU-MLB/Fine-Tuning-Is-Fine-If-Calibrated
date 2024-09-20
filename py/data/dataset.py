import os

from torchvision.datasets.folder import default_loader

from typing import List, Tuple, Any, Optional, Callable
import torchvision.datasets as datasets
from . import transform
from . import common
from ..util import common
from ..util import constant as C

"""
Modified from: https://github.com/thuml/Transfer-Learning-Library 

@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com

Copyright (c) 2018 The Python Packaging Authority
"""


class ImageList(datasets.VisionDataset):
    """A generic Dataset class for image classification

    Args:
        root (str): Root directory of dataset
        classes (list[str]): The names of all the classes
        data_list_file (str): File to read the image list from.
        transform (callable, optional): A function/transform that  takes in an PIL image \
            and returns a transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `data_list_file`, each line has 2 values in the following format.
        ::
            source_dir/dog_xxx.png 0
            source_dir/cat_123.png 1
            target_dir/dog_xxy.png 0
            target_dir/cat_nsdf3.png 1

        The first value is the relative path of an image, and the second value is the label of the corresponding image.
        If your data_list_file has different formats, please over-ride :meth:`~ImageList.parse_data_file`.
    """

    def __init__(self, root: str, classes: List[str], data_list_file: str,
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
                 eval_transform: Optional[Callable] = None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.samples = self.parse_data_file(data_list_file)
        self.targets = [s[1] for s in self.samples]
        self.classes = classes
        self.class_to_idx = {cls: idx
                             for idx, cls in enumerate(self.classes)}
        self.loader = default_loader
        self.data_list_file = data_list_file
        self.training_transform = transform
        self.val_transform = eval_transform

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Args:
            index (int): Index
            return (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.samples)

    def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
        """Parse file to data list

        Args:
            file_name (str): The path of data file
            return (list): List of (image path, class_index) tuples
        """
        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                split_line = line.split()
                target = split_line[-1]
                path = ' '.join(split_line[:-1])
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                target = int(target)
                data_list.append((path, target))
        return data_list

    def train(self):
        self.transform = self.training_transform

    def eval(self):
        if self.val_transform is not None:
            self.transform = self.val_transform

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)

    @classmethod
    def domains(cls):
        """All possible domain in this dataset"""
        raise NotImplemented


class OfficeHome(ImageList):
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/1b0171a188944313b1f5/?dl=1"),
        ("Art", "Art.tgz", "https://cloud.tsinghua.edu.cn/f/6a006656b9a14567ade2/?dl=1"),
        ("Clipart", "Clipart.tgz", "https://cloud.tsinghua.edu.cn/f/ae88aa31d2d7411dad79/?dl=1"),
        ("Product", "Product.tgz", "https://cloud.tsinghua.edu.cn/f/f219b0ff35e142b3ab48/?dl=1"),
        ("Real_World", "Real_World.tgz", "https://cloud.tsinghua.edu.cn/f/6c19f3f15bb24ed3951a/?dl=1")
    ]
    image_list = {
        "Ar": "Art.txt",
        "Cl": "Clipart.txt",
        "Pr": "Product.txt",
        "Rw": "Real_World.txt",
        "Ar_training": "Art_training.txt",
        "Ar_testing": "Art_testing.txt",
        "Cl_training": "Clipart_training.txt",
        "Cl_testing": "Clipart_testing.txt",
        "Pr_training": "Product_training.txt",
        "Pr_testing": "Product_testing.txt",
        "Rw_training": "Real_World_training.txt",
        "Rw_testing": "Real_World_testing.txt"
    }
    CLASSES = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
               'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
               'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
               'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
               'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
               'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
               'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker']

    def __init__(self, domain, **kwargs):
        assert domain in self.image_list, f"Task '{domain}' is not available!"
        data_list_path = os.path.join(C.OFFICEHOME_IMAGE_LIST_PATH, self.image_list[domain])

        list(map(lambda args: common.download_data(C.OFFICEHOME_DATA_PATH, *args), self.download_list))
        super(OfficeHome, self).__init__(C.OFFICEHOME_DATA_PATH, OfficeHome.CLASSES, data_list_file=data_list_path,
                                         **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())


def officehome(domain, _type):
    domain_type = f'{domain}_{_type}'
    assert domain_type in OfficeHome.domains(), f"Domain '{domain}' with type '{type}' is not available!"
    if _type == 'training':
        data = OfficeHome(domain_type, transform=transform.OFFICE_HOME_TRAINING,
                          eval_transform=transform.OFFICE_HOME_VALIDATION)
    elif _type == 'testing':
        data = OfficeHome(domain_type, transform=transform.OFFICE_HOME_VALIDATION)
    return data


# TODO: Arpita: finish this class and data retrieval function
class ImageNet(ImageList):
    pass

def imagenet(domain, _type):
    pass
