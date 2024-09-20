import copy
import tabulate
import torch
import torch.nn.functional as F
import prettytable

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
    pass

def evaluate_calibration(domain_info, extraction, oracle_extraction, method='baseline'):
    pass

def evaluate(domain_info, extraction, oracle_extraction):
    # Evaluate the features through the classifier (regular cnn model)
    clsf_metric = evaluate_clsf(domain_info, extraction, oracle_extraction)
    
    # Evaluate the features through the nearest mean classifier
    nmc_metric = evaluate_nmc(domain_info, extraction, oracle_extraction)
    
    # Evaluate the features through the linear probing
    lp_metric = evaluate_lp()

    # Evaluate AUC
    auc_metric = evaluate_auc(domain_info, extraction, oracle_extraction)

    # Evaluate baseline calibration
    baseline_calibration_metric = evaluate_calibration(domain_info, extraction, oracle_extraction)

    # Evaluate better calibration
    better_calibration_metric = evaluate_calibration(domain_info, extraction, oracle_extraction, method='better')

    # Package evaluation
    evaluation_metric = {
        'Classifier Accuracy': clsf_metric,
        'NMC Accuracy': nmc_metric,
        'LP Metric': lp_metric,
        'AUC': auc_metric,
        'Baseline Calibration': baseline_calibration_metric,
        'Better Calibration': better_calibration_metric
    }
    evaluation = Evaluation(domain_info, extraction, evaluation_metric)
    return evaluation
