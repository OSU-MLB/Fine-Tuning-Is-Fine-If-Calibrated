import copy
import torch
import torch.nn.functional as F
import prettytable
import numpy as np
from sklearn import metrics

from . import math


class Extraction:

    def __init__(self, features, features_unpooled, logits, labels, data_ind):
        self.features = features
        self.features_unpooled = features_unpooled
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


def mask_similarity(similarity, row_mask, column_mask):
    similarity = copy.deepcopy(similarity)
    similarity[~row_mask] = -torch.inf
    similarity[:, ~column_mask] = -torch.inf
    return similarity

def mask_predict_similarity(similarity, label, row_mask=None, column_mask=None):
    if row_mask is None:
        row_mask = torch.ones(similarity.shape[0], dtype=torch.bool)
    if column_mask is None:
        column_mask = torch.ones(similarity.shape[1], dtype=torch.bool)
    masked_similarity = mask_similarity(similarity, row_mask, column_mask)
    masked_similarity = masked_similarity[row_mask]
    masked_label = label[row_mask]
    accuracy = math.topk_accuracy(masked_similarity, masked_label)
    return accuracy

def generate_metric(domain_info, similarity, label, data_ind):
    # Unpack domain_info
    visible_classes = domain_info.visible_classes
    invisible_classes = domain_info.invisible_classes
    dim_similarity = similarity.shape[1]
    # Masks
    visible_row_mask = torch.isin(data_ind, domain_info.visible_ind)
    invisible_row_mask = torch.isin(data_ind, domain_info.invisible_ind)
    visible_column_mask = torch.zeros(dim_similarity, dtype=torch.bool)
    visible_column_mask[visible_classes] = 1
    invisible_column_mask = torch.zeros(dim_similarity, dtype=torch.bool)
    invisible_column_mask[invisible_classes] = 1
    # Calculate metric
    #  From all classes / Over all classes
    all_all_accuracy = mask_predict_similarity(similarity, label, row_mask=None, column_mask=None)
    #  From visible classes / Over all classes
    visible_all_accuracy = mask_predict_similarity(similarity, label, row_mask=visible_row_mask, column_mask=None)
    #  From invisible classes / Over all classes
    invisible_all_accuracy = mask_predict_similarity(similarity, label, row_mask=invisible_row_mask, column_mask=None)
    #  From visible classes / Over visible classes
    visible_visible_accuracy = mask_predict_similarity(similarity, label, row_mask=visible_row_mask, column_mask=visible_column_mask)
    #  From invisible classes / Over invisible classes
    invisible_invisible_accuracy = mask_predict_similarity(similarity, label, row_mask=invisible_row_mask, column_mask=invisible_column_mask)
    # Package accuracies
    accuracies = {
        'All/All Accuracy': all_all_accuracy,
        'Visible/All Accuracy': visible_all_accuracy,
        'Invisible/All Accuracy': invisible_all_accuracy,
        'Visible/Visible Accuracy': visible_visible_accuracy,
        'Invisible/Invisible Accuracy': invisible_invisible_accuracy
    }
    return accuracies


def evaluate_clsf(domain_info, extraction, oracle_extraction):
    logits = extraction.logits
    labels = extraction.labels
    data_ind = extraction.data_ind
    # Calculate metric
    metric = generate_metric(domain_info, logits, labels, data_ind)
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
    metric = generate_metric(domain_info, similarity, labels, data_ind)
    return metric


def evaluate_lp():
    pass

# TODO: Arpita: finish this methods following my other evalute methods
def evaluate_auc(domain_info, extraction, oracle_extraction):
    curve_results=[]

    # Compute seen and unseen sample masks
    visible_class_mask = torch.tensor([l.item() in domain_info.visible_classes for l in extraction.labels], dtype=torch.bool, device=extraction.labels.device)
    invisible_class_mask = torch.tensor([l.item() in domain_info.invisible_classes for l in extraction.labels], dtype=torch.bool, device=extraction.labels.device)
    
    # Increase unseen accuracy
    logits_copy = extraction.logits.clone().to(torch.float64)
    final = False
    accumulate_shifting = 0.
    while not final:
        unseen_shifting, final = compute_shifting(
                logits_copy, domain_info.visible_classes, domain_info.invisible_classes,
                mode='positive')
        logits_copy[:, domain_info.invisible_classes] += unseen_shifting
        accumulate_shifting += unseen_shifting
        curve_results.append(list(compute_accuracy(logits_copy, extraction.labels,
                            visible_class_mask, invisible_class_mask)) + [accumulate_shifting])

    #Increase seen accuracy
    logits_copy = extraction.logits.clone().to(torch.float64)
    final = False
    accumulate_shifting = 0.
    while not final:
        unseen_shifting, final = compute_shifting(
                logits_copy, domain_info.visible_classes, domain_info.invisible_classes,
                mode='negative')
        logits_copy[:, domain_info.invisible_classes] -= unseen_shifting
        accumulate_shifting -= unseen_shifting
        curve_results.append(list(compute_accuracy(logits_copy, extraction.labels,
                            visible_class_mask, invisible_class_mask)) + [accumulate_shifting])
        

    # Get trade-off curve
    curve_results = torch.tensor(curve_results)
    curve_results = curve_results[torch.argsort(curve_results[:, 1])]  # Sort by seen acc.
    trade_off_curve = curve_results[:, 1:3]

    auc_score = metrics.auc(
        trade_off_curve[:, 0].cpu().numpy() / 100.,
        trade_off_curve[:, 1].cpu().numpy() / 100.)
    
    return curve_results, auc_score


def compute_shifting(all_logits, visible_classes, invisible_classes, mode='positive'):
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

def compute_accuracy(all_logits, all_labels, visible_mask, invisible_mask,
                           chopped_out_classes=None):
    new_all_logits = all_logits.clone()
    if chopped_out_classes is not None:
        new_all_logits[:, chopped_out_classes] = float('-inf')
    
    overall_acc = (new_all_logits.argmax(dim=1) == all_labels).sum().item() / all_labels.shape[0]
    
    seen_acc = (new_all_logits[visible_mask].argmax(dim=1) == all_labels[visible_mask]).sum().item() / visible_mask.sum().item()
    
    unseen_acc = (new_all_logits[invisible_mask].argmax(dim=1) == all_labels[invisible_mask]).sum().item() / invisible_mask.sum().item()
    
    return overall_acc * 100., seen_acc * 100., unseen_acc * 100.


def evaluate_baseline_calibration(domain_info, extraction, oracle_extraction, method='baseline'):
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
    visible_class_mask = torch.tensor([l.item() in domain_info.visible_classes for l in extraction.labels], dtype=torch.bool, device=extraction.labels.device)
    invisible_class_mask = torch.tensor([l.item() in domain_info.invisible_classes for l in extraction.labels], dtype=torch.bool, device=extraction.labels.device)
    

    calib_all_preds = calib_all_logits.argmax(dim=1)
    
    overall_acc = 100 * (calib_all_preds == extraction.labels).sum().item() / extraction.labels.shape[0]
    seen_acc = 100 * (calib_all_preds[visible_class_mask] == extraction.labels[visible_class_mask]).sum().item() / visible_class_mask.sum().item()
    unseen_acc = 100 * (calib_all_preds[invisible_class_mask] == extraction.labels[invisible_class_mask]).sum().item() / invisible_class_mask.sum().item()
    
    return {'overall_acc':overall_acc,'seen_acc': seen_acc,'unseen_acc': unseen_acc}

#Ping: Input is needed: Src model unseen accuracy, but work will be made on target model. 
def evaluate_better_calibration(domain_info, extraction, oracle_extraction, curve_results, method='better',src_unseen_acc=None):
    if src_unseen_acc is None:
        valid = torch.ones(curve_results.shape[0], dtype=torch.bool, device=curve_results.device)
    else:
        unseen_accs = curve_results[:, 2]
        valid = unseen_accs >= src_unseen_acc

    # Mask the overall accuracies based on the valid mask
    masked_overall_accs = torch.where(valid, curve_results[:, 0], torch.tensor(float('-inf'), dtype=curve_results.dtype, device=curve_results.device))


    # Find the index of the maximum masked value
    best_idx = torch.argmax(masked_overall_accs)

    return curve_results[best_idx]

def evaluate(domain_info, extraction, oracle_extraction):
    # Evaluate the features through the classifier (regular cnn model)
    clsf_metric = evaluate_clsf(domain_info, extraction, oracle_extraction)
    
    # Evaluate the features through the nearest mean classifier
    nmc_metric = evaluate_nmc(domain_info, extraction, oracle_extraction)
    
    # Evaluate the features through the linear probing
    lp_metric = evaluate_lp()

    # Evaluate AUC
    curve_results, auc_metric = evaluate_auc(domain_info, extraction, oracle_extraction)

    # Evaluate baseline calibration
    baseline_calibration_metric = evaluate_baseline_calibration(domain_info, extraction, oracle_extraction)

    # Evaluate better calibration
    better_calibration_metric = evaluate_better_calibration(domain_info, extraction, oracle_extraction, curve_results, method='better')

    # Package evaluation
    evaluation_metric = {
        'Classifier Accuracy': clsf_metric,
        'NMC Accuracy': nmc_metric,
        'LP Metric': lp_metric,
        'AUC': auc_metric,
        'Baseline Calibration': baseline_calibration_metric,
        'Better Calibration': better_calibration_metric
    }
    #PING: Have to see what to return from other functions so that we can generate tables.
    evaluation = Evaluation(domain_info, extraction, evaluation_metric)
    return evaluation
