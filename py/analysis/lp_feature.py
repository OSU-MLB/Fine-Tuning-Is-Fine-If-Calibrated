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
from ..util import model as model_util
from ..workflow import train

import torch
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

    device = args.device
    base_dir = args.base_dir

    space = ExperimentSpace(base_dir)
    all_experiments = space.all()

    for experiment in all_experiments:
        print(f'Analyzing {experiment}. ')
        testing_eval_dir = experiment.get_checkpoint_dir('evaluation/testing')
        training_eval_dir = experiment.get_checkpoint_dir('evaluation/training')
        oracle_training_eval_dir = experiment.get_checkpoint_dir('evaluation/oracle_training')
        
        assert testing_eval_dir.epochs == oracle_training_eval_dir.epochs, 'Epochs mismatch. '

        epochs = testing_eval_dir.epochs
        for epoch, training_eval_epoch_file, testing_eval_epoch_file, oracle_training_eval_epoch_file in zip(epochs, training_eval_dir.epoch_files, testing_eval_dir.epoch_files, oracle_training_eval_dir.epoch_files):
            training_eval = torch.load(training_eval_epoch_file)
            testing_eval = torch.load(testing_eval_epoch_file)
            oracle_training_eval = torch.load(oracle_training_eval_epoch_file)
            domain_info = testing_eval.domain_info

            training_extraction = training_eval['extraction']
            testing_extraction = testing_eval['extraction']
            oracle_training_extraction = oracle_training_eval['extraction']

            training_features = training_extraction['features']
            training_logits = training_extraction['logits']
            training_labels = training_extraction['labels']
            training_lp_accuracy = LP_accuracy(domain_info, training_logits, training_labels)

            testing_features = testing_extraction['features']
            testing_logits = testing_extraction['logits']
            testing_labels = testing_extraction['labels']
            testing_lp_accuracy = LP_accuracy(domain_info, testing_logits, testing_labels)

            oracle_training_features = oracle_training_extraction['features']
            oracle_training_logits = oracle_training_extraction['logits']
            oracle_training_labels = oracle_training_extraction['labels']
            oracle_training_lp_accuracy = LP_accuracy(domain_info, oracle_training_logits, oracle_training_labels)
