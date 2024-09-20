import argparse
import json

from . import common
from . import constant as C


def _precheck_arguments(args):
    if args.dataset == 'OfficeHome':
        if args.source == 'Ar':
            assert args.target in ['Cl', 'Pr', 'Rw']
        elif args.source == 'Rw':
            assert args.target in ['Ar', 'Cl', 'Pr']
        else:
            raise NotImplementedError()
    # Check if method and optimizer_parameters are in json format
    assert common.is_json(
        args.serialization_config), f"Serialization config is not in json format: {args.serilization_config}. "
    assert common.is_json(args.model_config), f"Model config is not in json format: {args.model_config}. "
    assert common.is_json(args.training_config), f"Train config is not in json format: {args.training_config}. "
    assert common.is_json(
        args.optimizer_parameters), f"Optimizer parameters is not in json format: {args.optimizer_parameters}. "

def _process_arguments(args):
    args.serialization_config = json.loads(args.serialization_config)
    args.model_config = json.loads(args.model_config)
    args.training_config = json.loads(args.training_config)
    args.optimizer_parameters = json.loads(args.optimizer_parameters)
    if args.debug:
        args.training_config['epochs'] = 1
        args.training_config['iterations'] = 1
        args.training_config['save_every'] = 1


def _postcheck_arguments(args):
    model_config = args.model_config
    assert not (model_config['freeze_classifier'] and model_config[
        'freeze_backbone']), f"Cannot freeze both classifier and backbone"
    assert model_config['loss_type'] in ['cross-entropy'], f"Unsupported loss type: {model_config['loss_type']}. "
    assert model_config['loss_scope'] in ['all', 'seen'], f"Unsupported loss scope: {model_config['loss_scope']}. "


def parse_arguments():
    parser = argparse.ArgumentParser(description='Holistic Transfer')
    parser.add_argument('--device', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--source', type=str)
    parser.add_argument('--target', type=str)
    parser.add_argument('--base_dir', type=str, default=C.EXPERIMENT_PATH)
    parser.add_argument('--seed', type=common.type_or_none(int), default=0)
    parser.add_argument('--serialization_config', default='{}')
    parser.add_argument('--debug', action='store_true')
    args, _ = parser.parse_known_args()
    if args.dataset == 'OfficeHome':
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--workers', default=4, type=int)
        parser.add_argument('--arch', default='resnet50')
        parser.add_argument('--model_config',
                            default='{"loss_type":"cross-entropy","loss_scope":"all","dropout":0.1,"freeze_classifier":false,"freeze_bn":false,"freeze_backbone":false}')
        parser.add_argument('--training_config',
                            default='{"epochs":20,"iterations":500,"save_every":1,"evaluate_freq":0.2}')
        parser.add_argument('--n_seen_classes', type=int, default=30)
        parser.add_argument('--optimizer', type=str, default='SGD')
        parser.add_argument('--optimizer_parameters', type=str,
                            default='{"lr":1e-3,"weight_decay":5e-4,"momentum":0.9,"nesterov":true}')
    else:
        raise NotImplementedError(f"Unsupported dataset: {args.dataset}. ")
    args = parser.parse_args()
    _precheck_arguments(args)
    _process_arguments(args)
    _postcheck_arguments(args)
    return args
