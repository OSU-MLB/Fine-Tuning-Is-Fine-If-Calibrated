import argparse
from ..util import constant as C

import logging
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import sys

from ..data import common as data_api
from ..data import dataset
from ..serialization.local import ExperimentSpace
from ..util import cmd as cmd_util
from ..util import evaluate
from ..util import model as model_util
from ..workflow import train

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

device = 'cuda:0'

class LinearClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)


def LP_accuracy(domain_info, logits, labels):
    # Unpack domain_info
    in_features = logits.shape[1]
    out_features = domain_info.num_classes
    model = LinearClassifier(in_features, out_features)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1)

    for _ in range(10):

        def closure():
            optimizer.zero_grad()
            y_pred = model(logits)
            loss = criterion(y_pred, labels)
            loss.backward()

            return loss

        optimizer.step(closure)

        loss = closure()
        running_loss += loss.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LP feature analysis')
    parser.add_argument('--base_dir', type=str, default=C.EXPERIMENT_PATH)
    args = parser.parse_args()

    base_dir = args.base_dir

    space = ExperimentSpace(base_dir)
    all_experiments = space.all()

    for experiment in all_experiments:
        print(f'Analyzing {experiment}. ')
        testing_eval_dir = experiment.get_checkpoint_dir('evaluation/testing')
        training_eval_dir = experiment.get_checkpoint_dir('evaluation/training')
        oracle_training_eval_dir = experiment.get_checkpoint_dir('evaluation/oracle_training')

        assert testing_eval_dir.epochs == oracle_training_eval_dir.epochs, 'Epochs mismatch. '

        oracle_training_cls_mean = []
        testing_cls_mean = []
        dists = []

        epochs = testing_eval_dir.epochs
        o_nmc = []

        o_nmc_all_all = []
        o_nmc_invisible_all = []
        o_nmc_invisible_invisible = []
        o_nmc_visible_all = []
        o_nmc_visible_visible = []

        tr_nmc_all_all = []
        tr_nmc_invisible_all = []
        tr_nmc_invisible_invisible = []
        tr_nmc_visible_all = []
        tr_nmc_visible_visible = []

        te_nmc_all_all = []
        te_nmc_invisible_all = []
        te_nmc_invisible_invisible = []
        te_nmc_visible_all = []
        te_nmc_visible_visible = []

        for epoch, training_eval_epoch_file, testing_eval_epoch_file, oracle_training_eval_epoch_file in zip(epochs, training_eval_dir.epoch_files, testing_eval_dir.epoch_files, oracle_training_eval_dir.epoch_files):
            
            oracle_training_eval = torch.load(oracle_training_eval_epoch_file, map_location=device)
            training_eval = torch.load(training_eval_epoch_file, map_location=device)
            testing_eval = torch.load(testing_eval_epoch_file, map_location=device)

            o_nmc = training_eval.metric['NMC Accuracy']
            tr_nmc = oracle_training_eval.metric['NMC Accuracy']
            te_nmc = testing_eval.metric['NMC Accuracy']

            # {'All/All Accuracy': OrderedDict([('Top- 1 Accuracy', 92.18750762939453),
            #                                 ('Top- 5 Accuracy', 99.55357360839844)]),
            # 'Invisible/All Accuracy': OrderedDict([('Top- 1 Accuracy', nan),
            #                                         ('Top- 5 Accuracy', nan)]),
            # 'Invisible/Invisible Accuracy': OrderedDict([('Top- 1 Accuracy', nan),
            #                                             ('Top- 5 Accuracy', nan)]),
            # 'Visible/All Accuracy': OrderedDict([('Top- 1 Accuracy', 92.18750762939453),
            #                                     ('Top- 5 Accuracy', 99.55357360839844)]),
            # 'Visible/Visible Accuracy': OrderedDict([('Top- 1 Accuracy',
            #                                         99.77679443359375),
            #                                         ('Top- 5 Accuracy', 100.0)])}

            _o_nmc_all_all = o_nmc['All/All Accuracy']['Top- 1 Accuracy']
            _o_nmc_invisible_all = o_nmc['Invisible/All Accuracy']['Top- 1 Accuracy']
            _o_nmc_invisible_invisible = o_nmc['Invisible/Invisible Accuracy']['Top- 1 Accuracy']
            _o_nmc_visible_all = o_nmc['Visible/All Accuracy']['Top- 1 Accuracy']
            _o_nmc_visible_visible = o_nmc['Visible/Visible Accuracy']['Top- 1 Accuracy']

            _tr_nmc_all_all = tr_nmc['All/All Accuracy']['Top- 1 Accuracy']
            _tr_nmc_invisible_all = tr_nmc['Invisible/All Accuracy']['Top- 1 Accuracy']
            _tr_nmc_invisible_invisible = tr_nmc['Invisible/Invisible Accuracy']['Top- 1 Accuracy']
            _tr_nmc_visible_all = tr_nmc['Visible/All Accuracy']['Top- 1 Accuracy']
            _tr_nmc_visible_visible = tr_nmc['Visible/Visible Accuracy']['Top- 1 Accuracy']

            _te_nmc_all_all = te_nmc['All/All Accuracy']['Top- 1 Accuracy']
            _te_nmc_invisible_all = te_nmc['Invisible/All Accuracy']['Top- 1 Accuracy']
            _te_nmc_invisible_invisible = te_nmc['Invisible/Invisible Accuracy']['Top- 1 Accuracy']
            _te_nmc_visible_all = te_nmc['Visible/All Accuracy']['Top- 1 Accuracy']
            _te_nmc_visible_visible = te_nmc['Visible/Visible Accuracy']['Top- 1 Accuracy']

            o_nmc_all_all.append(_o_nmc_all_all)
            o_nmc_invisible_all.append(_o_nmc_invisible_all)
            o_nmc_invisible_invisible.append(_o_nmc_invisible_invisible)
            o_nmc_visible_all.append(_o_nmc_visible_all)
            o_nmc_visible_visible.append(_o_nmc_visible_visible)

            tr_nmc_all_all.append(_tr_nmc_all_all)
            tr_nmc_invisible_all.append(_tr_nmc_invisible_all)
            tr_nmc_invisible_invisible.append(_tr_nmc_invisible_invisible)
            tr_nmc_visible_all.append(_tr_nmc_visible_all)
            tr_nmc_visible_visible.append(_tr_nmc_visible_visible)

            te_nmc_all_all.append(_te_nmc_all_all)
            te_nmc_invisible_all.append(_te_nmc_invisible_all)
            te_nmc_invisible_invisible.append(_te_nmc_invisible_invisible)
            te_nmc_visible_all.append(_te_nmc_visible_all)
            te_nmc_visible_visible.append(_te_nmc_visible_visible)

        # print(f'Oracle NMC All/All: ')
        # print()
        # for i in o_nmc_all_all:
        #     print(i, end='\t')
        # print()

        # print(f'Oracle NMC Invisible/All: ')
        # print()
        # for i in o_nmc_invisible_all:
        #     print(i, end='\t')
        # print()

        # print(f'Oracle NMC Invisible/Invisible: ')
        # print()
        # for i in o_nmc_invisible_invisible:
        #     print(i, end='\t')
        # print()

        # print(f'Oracle NMC Visible/All: ')
        # print()
        # for i in o_nmc_visible_all:
        #     print(i, end='\t')
        # print()

        # print(f'Oracle NMC Visible/Visible: ')
        # print()
        # for i in o_nmc_visible_visible:
        #     print(i, end='\t')
        # print()

        # print(f'Training NMC All/All: ')
        # print()
        # for i in tr_nmc_all_all:
        #     print(i, end='\t')
        # print()

        # print(f'Training NMC Invisible/All: ')
        # print()
        # for i in tr_nmc_invisible_all:
        #     print(i, end='\t')
        # print()

        # print(f'Training NMC Invisible/Invisible: ')
        # print()
        # for i in tr_nmc_invisible_invisible:
        #     print(i, end='\t')
        # print()

        # print(f'Training NMC Visible/All: ')
        # print()
        # for i in tr_nmc_visible_all:
        #     print(i, end='\t')
        # print()

        # print(f'Training NMC Visible/Visible: ')
        # print()
        # for i in tr_nmc_visible_visible:
        #     print(i, end='\t')
        # print()

        print(f'Testing NMC All/All: ')
        print()
        for i in te_nmc_all_all:
            print(i, end='\t')
        print()

        print(f'Testing NMC Invisible/All: ')
        print()
        for i in te_nmc_invisible_all:
            print(i, end='\t')
        print()

        print(f'Testing NMC Invisible/Invisible: ')
        print()
        for i in te_nmc_invisible_invisible:
            print(i, end='\t')
        print()

        print(f'Testing NMC Visible/All: ')
        print()
        for i in te_nmc_visible_all:
            print(i, end='\t')
        print()

        print(f'Testing NMC Visible/Visible: ')
        print()
        for i in te_nmc_visible_visible:
            print(i, end='\t')
        print()


            