import copy
import torch
import logging
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..data import common as data_util
from ..util import evaluate

class PartialDomainTrainer:

    def __init__(self, model, optimizer, loss_type, loss_scope, device):
        self.model = model.to(device)
        self.src_training_extraction = None
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
        self._inited = False


    def _f_loss(self, loss_type, loss_scope, domain_info):
        # Unpack domain info
        visible_classes = domain_info.visible_classes
        num_classes = domain_info.num_classes
        
        # Visible and invisible inds
        dataset = self.training_loader.dataset

        # Define loss function
        if loss_type == 'cross-entropy':
            _f = F.cross_entropy

        # Define loss scope
        if loss_scope == 'all':
            def _loss(logits, y):
                return _f(logits, y)
        # TODO: Rewrite this part
        # elif loss_scope == 'seen':
        #     def _loss(logits, y, ind_order):
        #         visible_mask = dataset.visible_mask(ind_order)
        #         logits = logits[visible_mask]
        #         y = y[visible_mask]
        #         return _f(logits, y)
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
        state['next_epoch'] = 0
        state['next_iteration'] = 0
        lr_scheduler = CosineAnnealingLR(self.optimizer, epochs * iterations)
        state['lr_scheduler'] = lr_scheduler
        self.state = state

        # Initialize data and domain info
        training_loader = self.training_loader
        training_data = training_loader.dataset
        domain_info = training_data.domain_info

        # Initialize loss function
        _loss = self._f_loss(self.loss_type, self.loss_scope, domain_info)
        self.f_loss = _loss

        # Set _inited flag
        self._inited = True

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

        # TODO: Refactor this part
        # Get source model training invisible accuracy
        logging.info('Getting source model training invisible accuracy... ')
        self.src_model.eval()
        dataset.set_scope('all')
        dataset.eval()

        src_training_pred, training_labels, training_data_ind = self.extract_pred(loader, model=self.src_model)

        domain_info = dataset.domain_info
        invisible_mask = torch.isin(training_labels, domain_info.invisible_classes)

        src_training_invisible_acc = (src_training_pred[invisible_mask] == training_labels[invisible_mask]).float().mean().item() * 100
        logging.info(f'Source model training invisible accuracy: {src_training_invisible_acc}. ')
        self.src_unseen_acc = src_training_invisible_acc


    def _training_iteration(self):
        state = self.state
        self.model.train()
        dataset = self.training_loader.dataset
        dataset.train()
        dataset.set_scope('visible')
        optimizer = self.optimizer
        training_iterator = self.training_iterator
        _, (X, y) = next(training_iterator)
        X = X.to(self.device)
        y = y.to(self.device)
        logging.debug(f'X shape: {X.shape}, y shape: {y.shape}. ')
        logging.debug(f'Label of training iteration: {y}. ')
        state['n_data_epoch'] += len(y)
        # _, logits, labels = self.extract_batch(X, y)
        logits, _, labels = self.extract_batch(X, y)
        loss = self.f_loss(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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
        return extraction

    def evaluate(self):
        evaluation_result = {}
        
        # Validation on training data
        training_loader = self.training_loader
        training_dataset = training_loader.dataset
        domain_info = training_dataset.domain_info
        src_training_extraction = self.src_training_extraction
        model = self.model
        model.eval()
        training_dataset.eval()

        # Extract training oracle features
        training_dataset.set_scope('all')
        logging.debug('Validating on training oracle data...')
        with torch.no_grad():
            oracle_training_extraction = self.extract(training_loader)
            oracle_training_evaluation = evaluate.evaluate(domain_info, 
                                                           oracle_training_extraction, 
                                                           oracle_training_extraction, src_unseen_acc=
                                                           self.src_unseen_acc)
        evaluation_result['oracle_training'] = oracle_training_evaluation

        # Extract training features

        # training_dataset.set_scope('visible')
        # logging.debug('Validating on training data... ')
        # with torch.no_grad():
        #     training_extraction = self.extract(training_loader)
        #     training_evaluation = evaluate.evaluate(domain_info, training_extraction, oracle_training_extraction, src_unseen_acc=self.src_unseen_acc)
        # evaluation_result['training'] = training_evaluation

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
                val_evaluation = evaluate.evaluate(val_domain_info, val_extraction, oracle_training_extraction, src_unseen_acc=self.src_unseen_acc)
            evaluation_result[k] = val_evaluation
        
        return evaluation_result

    def evaluate_and_save(self):
        # Overwrite this method to save evaluation results
        return self.evaluate()

    def print_evaluate(self):
        eval_result = self.evaluate()
        for _k, _r in eval_result.items():
            logging.info(f'{_k} metrics: \n{_r}')

    def fit(self):
        assert self._inited, 'Trainer has not been initialized. '
        logging.info(f'Starting training, training_config: {self.training_config}... ')
        logging.info('Initializing... ')
        
        # Prepare variables
        state = self.state
        epochs = state['epochs']
        iterations = state['iterations']
        training_config = self.training_config
        eval_freq = training_config['evaluate_freq']
        eval_every = max(1, int(iterations * eval_freq))
        logging.debug(f'Evaluation frequency: {eval_every}. ')
        
        for epoch in range(state['next_epoch'], epochs):
            state['n_data_epoch'] = 0
            for iteration in range(state['next_iteration'], iterations):
                # Evaluation
                if iteration % eval_every == 0:
                    logging.info(f'Epoch {epoch}, iteration {iteration}, pre-evaluation... ')
                    self.print_evaluate()
                    
                logging.debug(f'Epoch {epoch}, iteration {iteration}... ')
                
                # Training iteration
                self._training_iteration()
                
                # Post iteration
                state['next_iteration'] = iteration + 1
                state['lr_scheduler'].step()
            
            logging.info(f'Epoch {epoch} finished. Number of data seen: {state["n_data_epoch"]}. ')

            # TODO: Refactor this line
            self.evaluate_and_save()
            state['next_iteration'] = 0
            state['next_epoch'] = epoch + 1
