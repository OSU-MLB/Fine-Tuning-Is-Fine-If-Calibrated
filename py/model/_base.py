import torch
import torch.nn as nn

from typing import Tuple, Optional, List, Dict

"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""


class Classifier(nn.Module):
    """A generic Classifier class for domain adaptation.

    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (int): Number of classes
        bottleneck (torch.nn.Module, optional): Any bottleneck layer. Use no bottleneck by default
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: -1
        head (torch.nn.Module, optional): Any classifier head. Use :class:`torch.nn.Linear` by default
        finetune (bool): Whether finetune the classifier or train from scratch. Default: True

    .. note::
        Different classifiers are used in different domain adaptation algorithms to achieve better accuracy
        respectively, and we provide a suggested `Classifier` for different algorithms.
        Remember they are not the core of algorithms. You can implement your own `Classifier` and combine it with
        the domain adaptation algorithm in this algorithm library.

    .. note::
        The learning rate of this classifier is set 10 times to that of the feature extractor for better accuracy
        by default. If you have other optimization strategies, please over-ride :meth:`~Classifier.get_parameters`.

    Inputs:
        - x (tensor): input data fed to `backbone`

    Outputs:
        - predictions: classifier's predictions
        - features: features after `bottleneck` layer and before `head` layer

    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - predictions: (minibatch, `num_classes`)
        - features: (minibatch, `features_dim`)

    """

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=True, pool_layer=None):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        if head is None:
            self.head = nn.Linear(self._features_dim, num_classes)
        else:
            self.head = head
        self.finetune = finetune
        ##### Record parameter momentums in Adam optimizer
        for param in self.head.parameters():
            param.record_momentum = True

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        predictions = self.head(f)
        if self.training:
            return predictions, f
        else:
            return predictions

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params


class ImageClassifier(Classifier):
    """ImageClassifier specific for reproducing results of `DomainBed <https://github.com/facebookresearch/DomainBed>`_.
    You are free to freeze all `BatchNorm2d` layers and insert one additional `Dropout` layer, this can achieve better
    results for some datasets like PACS but may be worse for others.

    Args:
        backbone (torch.nn.Module): Any backbone to extract features from data
        num_classes (int): Number of classes
        freeze_bn (bool, optional): whether to freeze all `BatchNorm2d` layers. Default: False
        dropout_p (float, optional): dropout ratio for additional `Dropout` layer, this layer is only used when `freeze_bn` is True. Default: 0.1
    """

    def __init__(self, backbone: nn.Module, num_classes: int, freeze_bn=False, dropout_p=0.1, **kwargs):
        super(ImageClassifier, self).__init__(backbone, num_classes, **kwargs)
        self.freeze_bn = freeze_bn
        if freeze_bn:
            self.feature_dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        f_unpooled = self.backbone(x)
        f = self.pool_layer(f_unpooled)
        f = self.bottleneck(f)
        if self.freeze_bn:
            f = self.feature_dropout(f)
        predictions = self.head(f)
        res = [predictions]
        if self.training or return_feat:
            res.append(f)
        # if return_unpooled_feat:
        #     res.append(f_unpooled)
        return tuple(res)

    def train(self, mode=True):
        super(ImageClassifier, self).train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
