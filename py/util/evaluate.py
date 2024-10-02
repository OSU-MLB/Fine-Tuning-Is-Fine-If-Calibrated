import copy
import torch
import torch.nn.functional as F
import prettytable
import numpy as np
from sklearn import metrics
import logging
import time

from . import math


class Extraction:

    def __init__(self, features, logits, labels, data_ind):
        self.features = features
        self.logits = logits
        self.labels = labels
        self.data_ind = data_ind


class Evaluation:

    def __init__(self, domain_info, extraction, metric):
        self.domain_info = domain_info
        self.extraction = extraction
        self.metric = metric
        self.tables = self._generate_tables()

    def _generate_tables(self):
        tables = []
        for category, metrics in self.metric.items():
            if metrics is None:
                continue             
            table = prettytable.PrettyTable()
            header = [category]
            sub_metric_set = set()
            for sub_metrics in metrics.values():
                if isinstance(sub_metrics, dict):
                    sub_metric_set.update(sub_metrics.keys())
            header.extend(sub_metric_set)
            table.field_names = header
            for metric_name, sub_metrics in metrics.items():
                row = [metric_name]
                for sub_metric in header[1:]:
                    row.append(sub_metrics.get(sub_metric, "N/A"))
                table.add_row(row)
            tables.append(str(table))
        return tables

    def __str__(self):
        return '\n'.join([table for table in self.tables])


def get_class_mean(domain_info, features, labels):
    all_classes = domain_info.all_classes
    num_classes = domain_info.num_classes
    dim_features = features.shape[1]
    clz_mean = torch.full((num_classes, dim_features), fill_value=-torch.inf, dtype=features.dtype,
                          device=features.device)
    for clz in all_classes:
        clz_mask = labels == clz
        if torch.sum(clz_mask) == 0:
            continue
        clz_features = features[clz_mask]
        _clz_mean = torch.mean(clz_features, dim=0)
        clz_mean[clz] = _clz_mean

    return clz_mean


def mask_column_score(score, column_mask):
    score = copy.deepcopy(score)
    # score[~row_mask] = -torch.inf
    score[:, ~column_mask] = -torch.inf
    # filter nan values
    score[torch.isnan(score)] = -torch.inf
    return score

def mask_predict_score(score, label, row_mask=None, column_mask=None):
    if row_mask is None:
        row_mask = torch.ones(score.shape[0], dtype=torch.bool)
    if column_mask is None:
        column_mask = torch.ones(score.shape[1], dtype=torch.bool)
    masked_score = mask_column_score(score, column_mask)
    masked_score = masked_score[row_mask]
    masked_label = label[row_mask]
    accuracy = math.topk_accuracy(masked_score, masked_label)
    return accuracy

def get_accuracies(domain_info, score, label, data_ind):
    # Unpack domain_info
    visible_classes = domain_info.visible_classes.type(torch.int64)
    invisible_classes = domain_info.invisible_classes.type(torch.int64)
    remaining_classes = domain_info.remaining_classes.type(torch.int64)
    dim_score = score.shape[1]

    # Masks
    visible_row_mask = torch.isin(data_ind, domain_info.visible_ind)
    invisible_row_mask = torch.isin(data_ind, domain_info.invisible_ind)
    remaining_row_mask = torch.isin(data_ind, domain_info.remaining_ind)
    visible_column_mask = torch.zeros(dim_score, dtype=torch.bool)
    visible_column_mask[visible_classes] = 1
    invisible_column_mask = torch.zeros(dim_score, dtype=torch.bool)
    invisible_column_mask[invisible_classes] = 1
    remaining_column_mask = torch.zeros(dim_score, dtype=torch.bool)
    remaining_column_mask[remaining_classes] = 1
    
    # Calculate metric
    #  From all classes / Over all classes
    all_all_accuracy = mask_predict_score(score, label, row_mask=None, column_mask=None)
    
    #  From visible classes / Over all classes
    visible_all_accuracy = mask_predict_score(score, label, row_mask=visible_row_mask, column_mask=None)
    
    #  From invisible classes / Over all classes
    invisible_all_accuracy = mask_predict_score(score, label, row_mask=invisible_row_mask, column_mask=None)

    #  From remaining classes / Over all classes
    remaining_all_accuracy = mask_predict_score(score, label, row_mask=remaining_row_mask, column_mask=None)
    
    #  From visible classes / Over visible classes
    visible_visible_accuracy = mask_predict_score(score, label, row_mask=visible_row_mask, column_mask=visible_column_mask)
    
    #  From invisible classes / Over invisible classes
    invisible_invisible_accuracy = mask_predict_score(score, label, row_mask=invisible_row_mask, column_mask=invisible_column_mask)

    #  From remaining classes / Over remaining classes
    remaining_remaining_accuracy = mask_predict_score(score, label, row_mask=remaining_row_mask, column_mask=remaining_column_mask)
    
    # Package accuracies
    accuracies = {
        'All/All Accuracy': all_all_accuracy,
        'Visible/All Accuracy': visible_all_accuracy,
        'Invisible/All Accuracy': invisible_all_accuracy,
        'Remaining/All Accuracy': remaining_all_accuracy,
        'Visible/Visible Accuracy': visible_visible_accuracy,
        'Invisible/Invisible Accuracy': invisible_invisible_accuracy,
        'Remaining/Remaining Accuracy': remaining_remaining_accuracy
    }
    return accuracies


def evaluate_clsf(domain_info, extraction, oracle_extraction):
    logits = extraction.logits
    labels = extraction.labels
    data_ind = extraction.data_ind
    # Calculate metric
    metric = get_accuracies(domain_info, logits, labels, data_ind)
    return metric

def evaluate_nmc(domain_info, extraction, oracle_extraction):
    # Unpack extraction
    features = extraction.features
    labels = extraction.labels
    data_ind = extraction.data_ind
    # Unpack oracle_extraction
    oracle_features = oracle_extraction.features
    oracle_labels = oracle_extraction.labels
    # Calculate nmc similarity
    normed_oracle_features = F.normalize(oracle_features, dim=1)
    normed_features = F.normalize(features, dim=1)
    clz_mean = get_class_mean(domain_info, normed_oracle_features, oracle_labels)
    normed_clz_mean = F.normalize(clz_mean, dim=1)
    similarity = torch.matmul(normed_features, normed_clz_mean.T)
    # Calculate metric
    metric = get_accuracies(domain_info, similarity, labels, data_ind)
    return metric


################################################################################
# TODO: Refactor this part

def _compute_accuracy(all_logits, all_labels, visible_mask, invisible_mask,
                           chopped_out_classes=None):
    new_all_logits = all_logits.clone()
    if chopped_out_classes is not None:
        new_all_logits[:, chopped_out_classes] = float('-inf')
    
    overall_acc = (new_all_logits.argmax(dim=1) == all_labels).sum().item() / all_labels.shape[0]
    
    visible_acc = (new_all_logits[visible_mask].argmax(dim=1) == all_labels[visible_mask]).sum().item() / visible_mask.sum().item()
    invisible_acc = (new_all_logits[invisible_mask].argmax(dim=1) == all_labels[invisible_mask]).sum().item() / invisible_mask.sum().item()
    
    return [overall_acc * 100., visible_acc * 100., invisible_acc * 100.]


def _compute_shifting(all_logits, visible_classes, invisible_classes, mode='positive'):
    assert mode in ['positive', 'negative']

    # Compute maximum logits for seen and unseen classes
    max_seen_logits = all_logits[:, visible_classes].max(dim=1)[0]
    max_unseen_logits = all_logits[:, invisible_classes].max(dim=1)[0]

    # Determine valid indices based on mode
    if mode == 'positive':
        valid = max_seen_logits >= max_unseen_logits
        diff = (max_seen_logits[valid] - max_unseen_logits[valid]).sort()[0]
    else:
        valid = max_seen_logits <= max_unseen_logits
        diff = (max_unseen_logits[valid] - max_seen_logits[valid]).sort()[0]

    # Check for invalid differences
    assert (diff < 0).sum() == 0

    # Handle different cases based on diff shape
    if diff.shape[0] == 0:
        return 0., True
    elif diff.shape[0] == 1:
        return diff[0] + 1., True
    else:
        first = diff[0]
        for d in diff[1:]:
            if d != first:
                second = d
                break
            else:
                second = d
        if first != second:
            return (first + second) / 2., False
        else:
            return first + 1., True


def _get_curve_results(domain_info, extraction):
    start = time.time()
    curve_results = torch.empty((0, 4), dtype=torch.float64, device=extraction.logits.device)

    # Compute seen and unseen sample masks
    visible_class_mask = torch.tensor([l.item() in domain_info.visible_classes for l in extraction.labels], dtype=torch.bool, device=extraction.labels.device)
    invisible_class_mask = torch.tensor([l.item() in domain_info.invisible_classes for l in extraction.labels], dtype=torch.bool, device=extraction.labels.device)
    
    t1 = time.time()
    logging.debug(f'Compute masks: {t1 - start:.4f} seconds')

    # Increase unseen accuracy
    logits_copy = extraction.logits.clone().to(torch.float64)
    final = False
    accumulate_shifting = 0.
    logging.debug('Increasing unseen accuracy...')
    while not final:
        unseen_shifting, final = _compute_shifting(
                logits_copy, domain_info.visible_classes, domain_info.invisible_classes,
                mode='positive')
        logits_copy[:, domain_info.invisible_classes] += unseen_shifting
        accumulate_shifting += unseen_shifting
        overall_acc, visible_acc, invisible_acc = _compute_accuracy(logits_copy, extraction.labels, visible_class_mask, invisible_class_mask)
        new_entry = torch.tensor([[overall_acc, visible_acc, invisible_acc, accumulate_shifting]], device=extraction.logits.device, dtype=torch.float64)
        curve_results = torch.cat((curve_results, new_entry), dim=0)


    t2 = time.time()
    logging.debug(f'Increase unseen accuracy: {t2 - t1:.4f} seconds')
    logging.debug('Increasing seen accuracy...')
    # Increase seen accuracy
    logits_copy = extraction.logits.clone().to(torch.float64)
    final = False
    accumulate_shifting = 0.
    while not final:
        shifting_start = time.time()
        unseen_shifting, final = _compute_shifting(
                logits_copy, domain_info.visible_classes, domain_info.invisible_classes,
                mode='negative')
        shifting_end = time.time()
        logits_copy[:, domain_info.invisible_classes] -= unseen_shifting
        accumulate_shifting -= unseen_shifting
        logging.debug(f'Compute shifting: {shifting_end - shifting_start:.4f} seconds, unseen_shifting: {unseen_shifting}, accumulate_shifting: {accumulate_shifting}')
        overall_acc, visible_acc, invisible_acc = _compute_accuracy(logits_copy, extraction.labels, visible_class_mask, invisible_class_mask)

        new_entry = torch.tensor([[overall_acc, visible_acc, invisible_acc, accumulate_shifting]], device=extraction.logits.device, dtype=torch.float64)
        curve_results = torch.cat((curve_results, new_entry), dim=0)
    
    t3 = time.time()
    logging.debug(f'Increase seen accuracy: {t3 - t2:.4f} seconds')

    # Get trade-off curve
    curve_results = curve_results[torch.argsort(curve_results[:, 1])]  # Sort by seen acc.
    trade_off_curve = curve_results[:, 1:3]

    t4 = time.time()
    logging.debug(f'Get trade-off curve: {t4 - t3:.4f} seconds')

    end = time.time()
    logging.debug(f'Total time: {end - start:.4f} seconds')

    return curve_results, trade_off_curve


def evaluate_baseline_calibration(domain_info, extraction):
    wrong_visible_logits = []
    invisible_logits = []

    for idx in range(extraction.logits.shape[0]):
        logit = extraction.logits[idx]
        label = extraction.labels[idx]

        invisible_logits.append(logit[domain_info.invisible_classes].mean().item())

        wrong_visible_classes = [x.item() for x in domain_info.visible_classes if x.item() != label.item()]
        wrong_visible_logits.append(logit[wrong_visible_classes].mean().item())

    baseline_calib_factor= torch.tensor(wrong_visible_logits).mean().item() - torch.tensor(invisible_logits).mean().item()

    calib_all_logits = extraction.logits.clone()
    calib_all_logits[:, domain_info.invisible_classes] += baseline_calib_factor

    # Compute seen and unseen sample masks    

    # generate_accu_metric(domain_info, score, label, data_ind)
    metric = get_accuracies(domain_info, calib_all_logits, extraction.labels, extraction.data_ind)

    return metric


def compute_best_idx_for_better_calibration(curve_results, src_unseen_acc=None):
    if src_unseen_acc is None:
         valid = torch.ones(curve_results.shape[0], dtype=torch.bool)
    else:
        unseen_accs = curve_results[:, 2]
        valid = unseen_accs >= src_unseen_acc
    if not torch.any(valid):
        best_idx=-1
    else:
        masked_overall_accs = torch.where(valid, curve_results[:, 0], torch.tensor(float('-inf'), dtype=curve_results.dtype,device=curve_results.device))
        best_idx = torch.argmax(masked_overall_accs)

    return best_idx 

def evaluate_better_calibration(domain_info, extraction, curve_results, cv_evaluation):
    average_calib_factor = 0.
    calib_count = 0

    # for _cross_val_extraction in cross_val_extraction:
    for _cv_evaluation_source, _cv_evaluation_target in zip(cv_evaluation['source'], cv_evaluation['target']):
        
        src_unseen_acc = _cv_evaluation_source.metric['Classifier Accuracy']['Remaining/All Accuracy']['Top- 1 Accuracy']
        _cv_evaluation_source.metric
        curve_results, _ = _get_curve_results(_cv_evaluation_target.domain_info, _cv_evaluation_target.extraction)
        best_idx = compute_best_idx_for_better_calibration(curve_results, src_unseen_acc=src_unseen_acc) #Need unseen accuracy of source model on cross  validation unseen classes
        if best_idx == -1:
            continue
        else:
            best_calib_factor = curve_results[best_idx, 3]
            average_calib_factor += best_calib_factor
            calib_count += 1
    #calculate average calibration factor for three cross validation models trained on half of seen classes.        
    average_calib_factor = average_calib_factor / calib_count
            
    score = extraction.logits.clone()
    score[:, domain_info.invisible_classes] += average_calib_factor #add calibration factor to all invisible classes of target model 

    #get accuracy after adding calibration factor to all invisible classes of target model
    metric = get_accuracies(domain_info, score, extraction.labels, extraction.data_ind) 

    return metric
################################################################################

def evaluate(domain_info, extraction, oracle_extraction, cv_evaluation=None):
    # Evaluate the features through the classifier (regular cnn model)
    logging.info('Evaluating classifier...')
    clsf_metric = evaluate_clsf(domain_info, extraction, oracle_extraction)
    
    # Evaluate the features through the nearest mean classifier
    logging.info('Evaluating NMC...')
    nmc_metric = evaluate_nmc(domain_info, extraction, oracle_extraction)
    
    # Get curve results for calibration
    logging.info('Evaluating calibration...')
    logging.debug('Getting curve results...')
    curve_results, trade_off_curve = _get_curve_results(domain_info, extraction)

    # Evaluate AUC
    logging.info('Calculating AUC...')
    auc_score = metrics.auc(trade_off_curve[:, 0].cpu().numpy() / 100., trade_off_curve[:, 1].cpu().numpy() / 100.)
    auc_metric = {
        'AUC': {
            '-': auc_score
        }
    }

    # Evaluate baseline calibration
    logging.info('Evaluating baseline calibration...')
    baseline_calibration_metric = evaluate_baseline_calibration(domain_info, extraction)

    # Evaluate better calibration
    if cv_evaluation is not None:
        logging.info('Evaluating better calibration...')
        better_calibration_metric = evaluate_better_calibration(domain_info, extraction, curve_results, cv_evaluation)
    else:
        better_calibration_metric = None
        logging.info('Better calibration evaluation skipped...')

    # Package evaluation
    logging.debug('Packaging evaluation...')
    evaluation_metric = {
        'Classifier Accuracy': clsf_metric,
        'NMC Accuracy': nmc_metric,
        'AUC': auc_metric,
        'Baseline Calibration': baseline_calibration_metric,
        'Better Calibration': better_calibration_metric
    }
    evaluation = Evaluation(domain_info, extraction, evaluation_metric)

    return evaluation
