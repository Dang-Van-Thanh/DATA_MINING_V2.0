"""
Module đánh giá metrics
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report, silhouette_score,
                           davies_bouldin_score)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Tính toán các metrics đánh giá"""
    
    @staticmethod
    def classification_metrics(y_true, y_pred, y_pred_proba=None):
        """Tính metrics cho phân lớp"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Thêm classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics['report'] = report
        
        return metrics
    
    @staticmethod
    def regression_metrics(y_true, y_pred):
        """Tính metrics cho hồi quy"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
        
        return metrics
    
    @staticmethod
    def clustering_metrics(X, labels):
        """Tính metrics cho phân cụm"""
        metrics = {}
        
        if len(np.unique(labels)) > 1:
            metrics['silhouette'] = silhouette_score(X, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
        else:
            metrics['silhouette'] = -1
            metrics['davies_bouldin'] = -1
        
        # Phân bố cụm
        unique, counts = np.unique(labels, return_counts=True)
        metrics['cluster_distribution'] = dict(zip(unique, counts))
        
        return metrics
    
    @staticmethod
    def format_metrics_table(metrics_dict, decimals=4):
        """Format metrics thành bảng đẹp"""
        df = pd.DataFrame(metrics_dict).T
        return df.round(decimals)