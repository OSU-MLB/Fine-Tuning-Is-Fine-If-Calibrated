import copy
import torch
import logging
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..data import common as data_util
from ..util import evaluate
from ..util import common
import time
import json

class PartialDomainTrainer:

    def __init__(self, model, optimizer, loss_type, loss_scope, device):
        self.model = model.to(device)
        self.src_model = copy.deepcopy(model)
        self.optimizer = optimizer
        self.device = device
        self.loss_type = loss_type
        self.loss_scope = loss_scope
        self.state = None
        self.f_loss = None
        self.training_loader = None
        self.training_iterator = None
        self.val_loaders = {}
        self.training_config = None
        self.cv_evaluation = None
        self._inited = False


    def _f_loss(self, loss_type, loss_scope):
        # Define loss function
        if loss_type == 'cross-entropy':
            _f = F.cross_entropy

        # Define loss scope
        if loss_scope == 'all':
            def _loss(logits, y):
                return _f(logits, y)
        return _loss

    def set_training_config(self, training_config):
        # Check if the trainer has been initialized
        assert not self._inited, 'Trainer has already been initialized. '
        assert self.training_loader is not None, 'training_loader must be set before setting training_config. '
        self.training_config = training_config

        # Initialize state
        state = {}
        epochs = training_config['epochs']
        iterations = training_config['iterations']
        state['epochs'] = epochs
        state['n_data_epoch'] = 0
        state['iterations'] = iterations
        state['curr_epoch'] = -1
        state['curr_iteration'] = 0
        lr_scheduler = CosineAnnealingLR(self.optimizer, epochs * iterations)
        state['lr_scheduler'] = lr_scheduler
        self.state = state

        # Initialize loss function
        _loss = self._f_loss(self.loss_type, self.loss_scope)
        self.f_loss = _loss

        # Set _inited flag
        self._inited = True

    def set_cv_evaluation(self, cv_evaluation):
        self.cv_evaluation = cv_evaluation

    def add_val_loader(self, k, loader):
        dataset = loader.dataset
        assert isinstance(dataset, data_util.PartialDomainDataset), 'Only PartialDomainDataset is supported. '
        self.val_loaders[k] = loader

    def set_training_loader(self, loader):
        # Check if the training loader has already been set
        assert self.training_loader is None, 'training_loader has already been set. '

        # Set training loader
        dataset = loader.dataset
        assert isinstance(dataset, data_util.PartialDomainDataset), 'Only PartialDomainDataset is supported. '
        self.training_loader = loader
        self.training_iterator = data_util.ForeverDataIterator(loader)

        # # TODO: Refactor this part
        # # Get source model training invisible accuracy
        # logging.info('Getting source model training invisible accuracy... ')
        # self.src_model.eval()
        # dataset.set_scope('all')
        # dataset.eval()

        # src_training_pred, training_labels, _ = self.extract_pred(loader, model=self.src_model)

        # domain_info = dataset.domain_info
        # invisible_mask = torch.isin(training_labels, domain_info.invisible_classes)

        # src_training_invisible_acc = (src_training_pred[invisible_mask] == training_labels[invisible_mask]).float().mean().item() * 100
        # logging.info(f'Source model training invisible accuracy: {src_training_invisible_acc}. ')
        # self.src_unseen_acc = src_training_invisible_acc


    def _training_iteration(self):
        start = time.time()

        # Set model, dataset and optimizer
        logging.debug('Setting model, dataset and optimizer... ')
        state = self.state
        self.model.train()
        dataset = self.training_loader.dataset
        dataset.train()
        dataset.set_scope('visible')
        optimizer = self.optimizer
        training_iterator = self.training_iterator
        t1 = time.time()
        logging.debug(f'Setup time: {t1 - start:.3f} s. ')

        # Set device
        logging.debug('Setting device... ')
        _, (X, y) = next(training_iterator)
        X = X.to(self.device)
        y = y.to(self.device)
        logging.debug(f'X shape: {X.shape}, y shape: {y.shape}. ')
        logging.debug(f'Label of training iteration: {y}. ')
        state['n_data_epoch'] += len(y)
        t2 = time.time()
        logging.debug(f'Device setup time: {t2 - t1:.3f} s. ')
        
        # Extract batch
        logging.debug('Extracting batch... ')
        logits, _, labels = self.extract_batch(X, y)
        t3 = time.time()
        logging.debug(f'Extraction time: {t3 - t2:.3f} s. ')

        # Compute loss
        logging.debug('Computing loss... ')
        loss = self.f_loss(logits, labels)
        t4 = time.time()
        logging.debug(f'Loss computation time: {t4 - t3:.3f} s. ')

        # Backward and optimize
        logging.debug('Backward and optimize... ')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t5 = time.time()
        logging.debug(f'Backward and optimize time: {t5 - t4:.3f} s. ')
        logging.debug(f'Training iteration time: {t5 - start:.3f} s. ')


    def extract_batch(self, X, y, model=None):
        # Set model
        if model is None:
            model = self.model

        # Extract batch
        X = X.to(self.device)
        y = y.to(self.device)
        logits, features, = model(X, return_feat=True)

        # Postprocess
        return logits, features, y

    def extract_pred(self, loader, model=None):
        pred = []
        labels = []
        data_ind = []

        # Extract features
        for _data_ind, (_X, _y) in loader:
            # logging.info(_data_ind)
            _logits, _, _labels = self.extract_batch(_X, _y, model=model)
            _pred = _logits.argmax(dim=1)
            pred.append(_pred)
            labels.append(_labels)
            data_ind.append(_data_ind)

        pred = torch.cat(pred, dim=0)
        labels = torch.cat(labels, dim=0)
        data_ind = torch.cat(data_ind, dim=0).to(self.device)
        return pred, labels, data_ind

    def extract(self, loader, model=None):

        logging.debug('Extracting features... ')
        start = time.time()
        # Initialize extraction
        features = []
        logits = []
        labels = []
        data_ind = []

        # Extract features
        for _data_ind, (_X, _y) in loader:
            _logits, _features, _labels = self.extract_batch(_X, _y, model=model)
            features.append(_features)
            logits.append(_logits)
            labels.append(_labels)
            data_ind.append(_data_ind)
        
        # Concatenate features
        features = torch.cat(features, dim=0)
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)
        data_ind = torch.cat(data_ind, dim=0).to(self.device)

        # Ensemble extraction
        extraction = evaluate.Extraction(features, logits, labels, data_ind)
        end = time.time()
        logging.debug(f'Extraction time: {end - start:.3f} s. ')
        return extraction

    def evaluate(self):
        evaluation_result = {}
        
        # Validation on training data
        training_loader = self.training_loader
        training_dataset = training_loader.dataset
        domain_info = training_dataset.domain_info
        model = self.model
        model.eval()
        training_dataset.eval()

        # Extract oracle training features
        training_dataset.set_scope('all')
        logging.debug('Validating on training oracle data...')
        with torch.no_grad():
            logging.debug('Extracting oracle training features... ')
            oracle_training_extraction = self.extract(training_loader)
            logging.debug('Evaluating oracle training features... ')
            oracle_training_evaluation = evaluate.evaluate(domain_info, 
                                                           oracle_training_extraction, 
                                                           oracle_training_extraction, self.cv_evaluation)
        evaluation_result['oracle_training'] = oracle_training_evaluation

        # Extract training features
        # Validation on validation data
        val_loaders = self.val_loaders
        
        # Extract validation features
        for k, val_loader in val_loaders.items():
            logging.debug(f'Validating on {k} data... ')
            val_dataset = val_loader.dataset
            val_domain_info = val_dataset.domain_info
            val_dataset.eval()
            val_dataset.set_scope('all')

            # Extract validation features
            with torch.no_grad():
                val_extraction = self.extract(val_loader)
                val_evaluation = evaluate.evaluate(val_domain_info, val_extraction, oracle_training_extraction, self.cv_evaluation)
            evaluation_result[k] = val_evaluation
        
        return evaluation_result

    def evaluate_and_save(self):
        # Overwrite this method to save evaluation results
        return self.evaluate()
    

    def evaluate_print(self):
        eval_result = self.evaluate()
        common.print_metrics(eval_result)

    def fit(self):
        assert self._inited, 'Trainer has not been initialized. '
        logging.info(f'Starting training, training_config: {self.training_config}... ')
        logging.info('Initializing... ')
        
        # Prepare variables
        state = self.state
        epochs = state['epochs']
        iterations = state['iterations']
        training_config = self.training_config
        evaluate_freq = training_config['evaluate_freq']
        if evaluate_freq == -1:
            eval_every = iterations + 1
        else:
            eval_every = max(1, int(iterations * evaluate_freq))
        logging.debug(f'Evaluation frequency: {evaluate_freq}. ')
        logging.debug(f'Evaluation every: {eval_every}. ')
        
        # Training loop
        # for epoch in range(state['next_epoch'], epochs):
        while state['curr_epoch'] <= epochs:

            # Evaluation
            logging.info(f"Evaluating after epoch {state['curr_epoch']}...")
            self.evaluate_print_save()

            # Set state
            state['curr_epoch'] += 1
            state['curr_iteration'] = 0
            state['n_data_epoch'] = 0

            if state['curr_epoch'] == epochs:
                break

            while state['curr_iteration'] < iterations:
                # Evaluation
                if state['curr_iteration'] > 0 and state['curr_iteration'] % eval_every == 0:
                    logging.info(f"Epoch {state['curr_epoch']}, iteration {state['curr_iteration']}, pre-evaluation... ")
                    self.evaluate_print()
                    
                logging.debug(f"Epoch {state['curr_epoch']}, iteration {state['curr_iteration']}... ")
                
                # Training iteration
                self._training_iteration()
                
                # Post iteration
                state['curr_iteration'] += 1
                state['lr_scheduler'].step()
            
            logging.info(f"Epoch {state['curr_epoch']} finished. Number of data seen: {state['n_data_epoch']}. ")

