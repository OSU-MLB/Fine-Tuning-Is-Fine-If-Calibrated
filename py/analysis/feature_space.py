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
        print(f'Epochs: {epochs}')
        for epoch, training_eval_epoch_file, testing_eval_epoch_file, oracle_training_eval_epoch_file in zip(epochs, training_eval_dir.epoch_files, testing_eval_dir.epoch_files, oracle_training_eval_dir.epoch_files):
            
            print(f'Epoch {epoch}. ')
            oracle_training_eval = torch.load(oracle_training_eval_epoch_file, map_location=device)
            training_eval = torch.load(training_eval_epoch_file, map_location=device)
            testing_eval = torch.load(testing_eval_epoch_file, map_location=device)

            training_domain_info = training_eval.domain_info
            testing_domain_info = testing_eval.domain_info
            oracle_training_domain_info = oracle_training_eval.domain_info

            training_extraction = training_eval.extraction
            testing_extraction = testing_eval.extraction
            oracle_training_extraction = oracle_training_eval.extraction

            oracle_training_features = oracle_training_extraction.features
            oracle_training_logits = oracle_training_extraction.logits
            oracle_training_labels = oracle_training_extraction.labels
            oracle_training_data_ind = oracle_training_extraction.data_ind

            oracle_training_visible_mask = torch.isin(oracle_training_data_ind, oracle_training_domain_info.visible_ind)
            oracle_training_invisible_mask = torch.isin(oracle_training_data_ind, oracle_training_domain_info.invisible_ind)
            assert oracle_training_features.shape[0] == oracle_training_data_ind.shape[0]
            assert torch.sum(oracle_training_visible_mask) + torch.sum(oracle_training_invisible_mask) == oracle_training_data_ind.shape[0]
        
            testing_features = testing_extraction.features
            testing_logits = testing_extraction.logits
            testing_labels = testing_extraction.labels
            testing_data_ind = testing_extraction.data_ind

            testing_visible_mask = torch.isin(testing_data_ind, testing_domain_info.visible_ind)
            testing_invisible_mask = torch.isin(testing_data_ind, testing_domain_info.invisible_ind)
            assert testing_features.shape[0] == testing_data_ind.shape[0]
            assert torch.sum(testing_visible_mask) + torch.sum(testing_invisible_mask) == testing_data_ind.shape[0]

            oracle_training_features = F.normalize(oracle_training_features, dim=1)

            visible_oracle_training_features = oracle_training_features[oracle_training_visible_mask]
            visible_oracle_training_labels = oracle_training_labels[oracle_training_visible_mask]
            assert torch.all(torch.isin(visible_oracle_training_labels, oracle_training_domain_info.visible_classes))

            invisible_oracle_training_features = oracle_training_features[oracle_training_invisible_mask]
            invisible_oracle_training_labels = oracle_training_labels[oracle_training_invisible_mask]
            assert torch.all(torch.isin(invisible_oracle_training_labels, oracle_training_domain_info.invisible_classes))

            testing_features = F.normalize(testing_features, dim=1)

            visible_testing_features = testing_features[testing_visible_mask]
            visible_testing_labels = testing_labels[testing_visible_mask]
            assert torch.all(torch.isin(visible_testing_labels, testing_domain_info.visible_classes))

            invisible_testing_features = testing_features[testing_invisible_mask]
            invisible_testing_labels = testing_labels[testing_invisible_mask]
            assert torch.all(torch.isin(invisible_testing_labels, testing_domain_info.invisible_classes))

            _oracle_training_cls_mean = evaluate.get_class_mean(oracle_training_domain_info, oracle_training_features, oracle_training_labels)
            _oracle_training_cls_mean = F.normalize(_oracle_training_cls_mean, dim=1)

            _testing_cls_mean = evaluate.get_class_mean(testing_domain_info, testing_features, testing_labels)
            _testing_cls_mean = F.normalize(_testing_cls_mean, dim=1)

            oracle_training_cls_mean.append(_oracle_training_cls_mean)
            testing_cls_mean.append(_testing_cls_mean)

            oracle_training_features_mean = oracle_training_features.mean(dim=0)
            visible_oracle_training_features_mean = visible_oracle_training_features.mean(dim=0)
            invisible_oracle_training_features_mean = invisible_oracle_training_features.mean(dim=0)

            testing_features_mean = testing_features.mean(dim=0)
            visible_testing_features_mean = visible_testing_features.mean(dim=0)
            invisible_testing_features_mean = invisible_testing_features.mean(dim=0)

            _dists = [epoch]
            # Visible training to cls mean
            visible_oracle_training_cls_mean = _oracle_training_cls_mean[visible_oracle_training_labels]
            _dist = torch.norm(visible_oracle_training_features - visible_oracle_training_cls_mean, dim=1).mean()
            _dists.append(_dist)

            # Invisible training to cls mean
            invisible_oracle_training_cls_mean = _oracle_training_cls_mean[invisible_oracle_training_labels]
            _dist = torch.norm(invisible_oracle_training_features - invisible_oracle_training_cls_mean, dim=1).mean()
            _dists.append(_dist)

            # Visible testing to cls mean
            visible_testing_cls_mean = _testing_cls_mean[visible_testing_labels]
            _dist = torch.norm(visible_testing_features - visible_testing_cls_mean, dim=1).mean()
            _dists.append(_dist)

            # Invisible testing to cls mean
            invisible_testing_cls_mean = _testing_cls_mean[invisible_testing_labels]
            _dist = torch.norm(invisible_testing_features - invisible_testing_cls_mean, dim=1).mean()
            _dists.append(_dist)

            # Visible training to mean of visible training feature mean
            _dist = torch.norm(visible_oracle_training_features - visible_oracle_training_features_mean, dim=1).mean()
            _dists.append(_dist)

            # Invisible training to mean of invisible training feature mean
            _dist = torch.norm(invisible_oracle_training_features - invisible_oracle_training_features_mean, dim=1).mean()
            _dists.append(_dist)

            # Visible testing to mean of visible testing feature
            _dist = torch.norm(visible_testing_features - visible_testing_features_mean, dim=1).mean()
            _dists.append(_dist)

            # Invisible testing to mean of invisible testing feature
            _dist = torch.norm(invisible_testing_features - invisible_testing_features_mean, dim=1).mean()
            _dists.append(_dist)

            # Visible training to invisible training feature mean
            _dist = torch.norm(visible_oracle_training_features - invisible_oracle_training_features_mean, dim=1).mean()
            _dists.append(_dist)

            # Invisible training to visible training feature mean
            _dist = torch.norm(invisible_oracle_training_features - visible_oracle_training_features_mean, dim=1).mean()
            _dists.append(_dist)

            # Visible testing to invisible testing feature mean
            _dist = torch.norm(visible_testing_features - invisible_testing_features_mean, dim=1).mean()
            _dists.append(_dist)
            
            # Invisible testing to visible testing feature mean
            _dist = torch.norm(invisible_testing_features - visible_testing_features_mean, dim=1).mean()
            _dists.append(_dist)

            # Visible training to overall training feature mean
            _dist = torch.norm(visible_oracle_training_features - oracle_training_features_mean, dim=1).mean()
            _dists.append(_dist)
            
            # Invisible training to overall training feature mean
            _dist = torch.norm(invisible_oracle_training_features - oracle_training_features_mean, dim=1).mean()
            _dists.append(_dist)

            # Visible testing to overall training feature mean
            _dist = torch.norm(visible_testing_features - oracle_training_features_mean, dim=1).mean()
            _dists.append(_dist)

            # Invisible testing to overall training feature mean
            _dist = torch.norm(invisible_testing_features - oracle_training_features_mean, dim=1).mean()
            _dists.append(_dist)

            dists.append(torch.tensor(_dists))

        oracle_training_cls_mean = torch.stack(oracle_training_cls_mean, dim=0)
        testing_cls_mean = torch.stack(testing_cls_mean, dim=0)

        oracle_training_cls_mean_move = oracle_training_cls_mean[1:] - oracle_training_cls_mean[:-1]
        testing_cls_mean_move = testing_cls_mean[1:] - testing_cls_mean[:-1]

        oracle_training_cls_mean_move_dist = torch.norm(oracle_training_cls_mean_move, dim=-1)
        testing_cls_mean_move_dist = torch.norm(testing_cls_mean_move, dim=-1)

        overall_oracle_training_cls_mean_move_dist = oracle_training_cls_mean_move_dist.mean(axis=1)
        visible_oracle_training_cls_mean_move_dist = oracle_training_cls_mean_move_dist[:, oracle_training_domain_info.visible_classes].mean(axis=1)
        invisible_oracle_training_cls_mean_move_dist = oracle_training_cls_mean_move_dist[:, oracle_training_domain_info.invisible_classes].mean(axis=1)

        overall_testing_cls_mean_move_dist = testing_cls_mean_move_dist.mean(axis=1)
        visible_testing_cls_mean_move_dist = testing_cls_mean_move_dist[:, testing_domain_info.visible_classes].mean(axis=1)
        invisible_testing_cls_mean_move_dist = testing_cls_mean_move_dist[:, testing_domain_info.invisible_classes].mean(axis=1)

        print('Overall oracle training class mean move distance: ')
        for _dirt in overall_oracle_training_cls_mean_move_dist:
            print(f'{_dirt.item():.4f}', end='\t')
        print()

        print('\nVisible oracle training class mean move distance: ')
        for _dirt in visible_oracle_training_cls_mean_move_dist:
            print(f'{_dirt.item():.4f}', end='\t')
        print()

        print('\nInvisible oracle training class mean move distance: ')
        for _dirt in invisible_oracle_training_cls_mean_move_dist:
            print(f'{_dirt.item():.4f}', end='\t')
        print()

        print('\nOverall testing class mean move distance: ')
        for _dirt in overall_testing_cls_mean_move_dist:
            print(f'{_dirt.item():.4f}', end='\t')
        print()

        print('\nVisible testing class mean move distance: ')
        for _dirt in visible_testing_cls_mean_move_dist:
            print(f'{_dirt.item():.4f}', end='\t')
        print()

        print('\nInvisible testing class mean move distance: ')
        for _dirt in invisible_testing_cls_mean_move_dist:
            print(f'{_dirt.item():.4f}', end='\t')
        print()

        dists = torch.stack(dists, dim=0)


        dist_name = [
                     'Epoch',    
                     'Visible training to cls mean', 
                     'Invisible training to cls mean', 
                     'Visible testing to cls mean', 
                     'Invisible testing to cls mean', 
                     'Visible training to mean of visible training feature', 
                     'Invisible training to mean of invisible training feature', 
                     'Visible testing to mean of visible testing feature', 
                     'Invisible testing to mean of invisible testing feature', 
                     'Visible training to invisible training feature mean', 
                     'Invisible training to visible training feature mean', 
                     'Visible testing to invisible testing feature mean', 
                     'Invisible testing to visible testing feature mean', 
                     'Visible training to overall training feature mean', 
                     'Invisible training to overall training feature mean',
                     'Visible testing to overall training feature mean',
                     'Invisible testing to overall training feature mean', 
                     ]
        
        for dist, n in zip(dists.T, dist_name):
            print(f'\n{n}: ')
            for _dirt in dist:
                print(f'{_dirt.item():.4f}', end='\t')
            print()
