import logging
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import sys
import json

from .data import common as data_api
from .data import dataset
from .serialization.local import ExperimentSpace
from .util import cmd as cmd_util
from .util import model as model_util
from .workflow import train
from .util import common
from .util import constant as C


class HolisticTransfer:

    def __init__(self, experiment, trainer, serialization_config):
        self.experiment = experiment
        self.trainer = trainer
        self.serialization_config = serialization_config
        self.wrapped = False

    def _f_evaluate_print_save(self):
        experiment = self.experiment

        def _f(_self):
            logging.debug('In evaluate_print_save... ')
            evaluation = _self.evaluate()
            common.print_metrics(evaluation)
            for k, _evaluation in evaluation.items():
                epoch = _self.state['curr_epoch']
                logging.info(f'Saving evaluation for {k}... ')
                experiment.save_checkpoint(f'evaluation/{k}', epoch, _evaluation)
            logging.info('Saving model... ')
            experiment.save_checkpoint('model', epoch, _self.model.state_dict())

        return _f

    def _wrap_evaluate_print_save(self):
        if self.wrapped:
            return
        trainer = self.trainer
        f = self._f_evaluate_print_save()
        trainer.evaluate_print_save = f.__get__(trainer, train.PartialDomainTrainer)
        self.wrapped = True

    def fit(self):
        self._wrap_evaluate_print_save()
        self.trainer.fit()

    def evaluate(self):
        self.trainer.evaluate_print()


def main(args):
    # Unpack arguments
    base_dir = args.base_dir
    device = args.device
    dataset_name = args.dataset
    source = args.source
    target = args.target
    seed = args.seed
    arch = args.arch
    serialization_config = args.serialization_config
    model_config = args.model_config
    n_visible_classes = args.n_visible_classes
    visible_classes = args.visible_classes
    n_invisible_classes = args.n_invisible_classes
    batch_size = args.batch_size
    workers = args.workers
    optimizer_type = args.optimizer
    optimizer_parameters = args.optimizer_parameters
    training_config = args.training_config
    cross_val_config = args.cross_val_config
    debug = args.debug

    # Construct experiment space
    space = ExperimentSpace(base_dir)


    # Create experiment instance
    experiment = space.start(dataset=dataset_name,
                             arch=arch,
                             source=source,
                             target=target,
                             model_config=model_config,
                             n_visible_classes=n_visible_classes,
                             visible_classes=visible_classes,
                             n_invisible_classes=n_invisible_classes,
                             optimizer=optimizer_type,
                             optimizer_parameters=optimizer_parameters,
                             seed=seed, 
                             debug=debug)

    # Unpack model config
    loss_type = model_config['loss_type']
    loss_scope = model_config['loss_scope']
    freeze_bn = model_config['freeze_bn']
    freeze_classifier = model_config['freeze_classifier']
    freeze_backbone = model_config['freeze_backbone']
    dropout = model_config['dropout']

    # Set random seed
    if seed is not None:
        logging.info(f'Setting random seed to {seed}... ')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # Load data
    logging.info(f'Loading {dataset_name} training dataset... ')
    training_data = dataset.get_dataset(dataset_name, target, 'training')
    logging.info(f'Loading {dataset_name} testing dataset... ')
    testing_data = dataset.get_dataset(dataset_name, target, 'testing')

    # Get visible and invisible classes
    logging.info('Getting visible and invisible classes... ')
    from . import HT_HARDCODED as HC
    num_classes, all_classes, visible_classes, invisible_classes = HC.GET_HARDCODED_CLASSES(dataset_name, source, target, n_visible_classes, n_invisible_classes, visible_classes=visible_classes)

    # Create data loaders
    logging.info('Creating domain info... ')
    domain_info = data_api.DomainInfo(all_classes, visible_classes, invisible_classes=invisible_classes, num_classes=num_classes)
    logging.info(f'Domain info: {domain_info}')
    
    logging.info('Creating partial domain training dataset... ')
    training_data = data_api.PartialDomainDataset(training_data, domain_info)
    logging.info(f'Training Visible size: {len(training_data.visible_ind)}')
    
    logging.info('Creating partial domain testing dataset... ')
    testing_data = data_api.PartialDomainDataset(testing_data, domain_info)
    logging.info(f'Testing Visible size: {len(testing_data.visible_ind)}')
    
    logging.info('Creating training data loader... ')
    training_data.domain_info.to(device)
    testing_data.domain_info.to(device)
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=workers,
                                 drop_last=True)
    
    logging.info('Creating testing data loader... ')
    testing_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=False, num_workers=workers)
        
    # Init cross-validation experiments
    if cross_val_config is not None:
        logging.info('Creating cross-validation experiments... ')
        cv_evalution = {
            'source': [],
            'target': []
        }
        cross_val_config = json.load(open(cross_val_config, 'r'))
        for exp_path in cross_val_config:
            # Load experiment
            exp = space.instance_from_path(exp_path)
            assert C.SOURCE_EPOCH in exp.epochs(C.ORACLE_TRAINING_EVALUATION_CKPT_PATH), 'Source evaluation not found in experiment {exp_path}. '
            evalution_epochs = exp.epochs(C.
            ORACLE_TRAINING_EVALUATION_CKPT_PATH)
            logging.info(f'Experiment {exp_path} has evaluation epochs: {evalution_epochs}, loading {C.SOURCE_EPOCH} as source evaluation and {evalution_epochs[-1]} as target evaluation... ')

            # Load evaluation
            _cv_evaluation_source = exp.load_checkpoint(C.ORACLE_TRAINING_EVALUATION_CKPT_PATH, C.SOURCE_EPOCH)
            _cv_evaluation_target = exp.load_checkpoint(C.ORACLE_TRAINING_EVALUATION_CKPT_PATH, evalution_epochs[-1])

            # Append to list
            cv_evalution['source'].append(_cv_evaluation_source)
            cv_evalution['target'].append(_cv_evaluation_target)
    else:
        cv_evalution = None
        logging.info('No cross-validation experiments... ')

    if args.train:
        source_model_path = experiment.source_model_path

        # Init model
        logging.info(f'Creating model... ')
        source_model_path = experiment.source_model_path
        model = model_util.create_model(arch, freeze_bn, dropout, domain_info.num_classes, source_model_path)

        # Init optimizer
        logging.info(f'Creating optimizer... ')
        optimizer = model_util.build_optimizer(model, optimizer_type, optimizer_parameters, freeze_classifier,
                                            freeze_backbone)

        # Create trainer
        logging.info(f'Creating trainer... ')
        trainer = train.PartialDomainTrainer(model, optimizer, loss_type, loss_scope, device)
        trainer.set_training_loader(training_loader)
        trainer.add_val_loader('testing', testing_loader)
        trainer.set_training_config(training_config)
        trainer.set_cv_evaluation(cv_evalution)

        # Create HT instance
        logging.info(f'Creating HolisticTransfer instance... ')
        ht = HolisticTransfer(experiment, trainer, serialization_config)

        # Fit
        logging.info('Fitting... ')
        ht.fit()

    elif args.eval:
        source_model_path = args.eval_model_path
    
        # Init model
        model = model_util.create_model(arch, freeze_bn, dropout, domain_info.num_classes, source_model_path)

        # Init optimizer
        logging.info(f'Creating optimizer... ')
        optimizer = model_util.build_optimizer(model, optimizer_type, optimizer_parameters, freeze_classifier,
                                            freeze_backbone)

        # Create trainer
        logging.info(f'Creating trainer... ')
        trainer = train.PartialDomainTrainer(model, optimizer, loss_type, loss_scope, device)
        trainer.set_training_loader(training_loader)
        trainer.add_val_loader('testing', testing_loader)
        trainer.set_training_config(training_config)
        trainer.set_cv_evaluation(cv_evalution)

        # Create HT instance
        logging.info(f'Creating HolisticTransfer instance... ')
        ht = HolisticTransfer(experiment, trainer, serialization_config)

        # Evaluate
        logging.info('Evaluating... ')
        ht.evaluate()

    logging.info('Done. ')
    sys.exit(0)


if __name__ == '__main__':
    args = cmd_util.parse_arguments()
    main(args)
