"""
Module huấn luyện mô hình supervised
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier  # Thêm Decision Tree
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix)
import yaml
import logging
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupervisedModel:
    """Huấn luyện và đánh giá mô hình supervised"""
    
    def __init__(self, config_path="configs/params.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.seed = self.config['project']['seed']
        self.test_size = self.config['project']['test_size']
        self.cv_folds = self.config['project']['cv_folds']
        
        self.models = {}
        self.results = {}
        self.baselines = []
        self.improved_model = None
        
    def get_model(self, model_name, class_weight=None, scale_pos_weight=None):
        """Lấy mô hình theo tên"""
        
        # ✅ BASELINE 1: Logistic Regression
        if model_name == 'logistic_regression':
            params = self.config['models']['logistic_regression']['params'].copy()
            if class_weight is not None:
                params['class_weight'] = class_weight
            model = LogisticRegression(**params)
            model.is_baseline = True
            return model
        
        # ✅ BASELINE 2: Decision Tree
        elif model_name == 'decision_tree':
            params = self.config['models']['decision_tree']['params'].copy()
            if class_weight is not None:
                params['class_weight'] = class_weight
            model = DecisionTreeClassifier(**params)
            model.is_baseline = True
            return model
        
        # 🔥 IMPROVED: XGBoost
        elif model_name == 'xgboost':
            params = self.config['models']['xgboost']['params'].copy()
            if scale_pos_weight is not None:
                params['scale_pos_weight'] = scale_pos_weight
            model = XGBClassifier(**params)
            model.is_baseline = False
            return model
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def train_test_split(self, X, y):
        """Chia train/test"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed, stratify=y
        )
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def calculate_scale_pos_weight(self, y):
        """Tính scale_pos_weight cho XGBoost"""
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        return neg_count / pos_count
    
    def train_model(self, model_name, X_train, y_train, X_test, y_test):
        """Huấn luyện một mô hình"""
        logger.info(f"Training {model_name}...")
        
        # Chuẩn bị tham số đặc biệt cho từng model
        if model_name == 'xgboost':
            scale_pos_weight = self.calculate_scale_pos_weight(y_train)
            model = self.get_model(model_name, scale_pos_weight=scale_pos_weight)
        else:
            # Logistic Regression và Decision Tree dùng class_weight='balanced'
            model = self.get_model(model_name, class_weight='balanced')
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Đánh giá
        metrics = self.evaluate_model(y_test, y_pred, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring='roc_auc')
        metrics['cv_roc_auc_mean'] = cv_scores.mean()
        metrics['cv_roc_auc_std'] = cv_scores.std()
        
        # Thời gian train (ước lượng)
        metrics['training_time'] = "N/A"
        
        # Phân loại model
        metrics['model_type'] = 'baseline' if hasattr(model, 'is_baseline') and model.is_baseline else 'improved'
        
        logger.info(f"{model_name} - F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f} ({metrics['model_type']})")
        
        return model, metrics
    
    def evaluate_model(self, y_true, y_pred, y_pred_proba):
        """Đánh giá mô hình với nhiều metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        # PR-AUC (quan trọng cho imbalance)
        from sklearn.metrics import precision_recall_curve, auc
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        metrics['pr_auc'] = auc(recall, precision)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['tn'] = tn
        metrics['fp'] = fp
        metrics['fn'] = fn
        metrics['tp'] = tp
        
        return metrics
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Huấn luyện tất cả các mô hình"""
        for model_name, config in self.config['models'].items():
            if config.get('enabled', False):
                try:
                    model, metrics = self.train_model(
                        model_name, X_train, y_train, X_test, y_test
                    )
                    self.models[model_name] = model
                    self.results[model_name] = metrics
                    
                    # Phân loại
                    if config.get('is_baseline', False):
                        self.baselines.append(model_name)
                    else:
                        self.improved_model = model_name
                        
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
        
        logger.info(f"\nBaselines: {self.baselines}")
        logger.info(f"Improved model: {self.improved_model}")
        
        return self.models, self.results
    
    def create_comparison_table(self):
        """Tạo bảng so sánh các mô hình - LÀM NỔI BẬT BASELINE VS IMPROVED"""
        comparison = pd.DataFrame(self.results).T
        
        # Chọn các metrics chính
        metrics_to_show = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'model_type']
        comparison = comparison[metrics_to_show]
        
        # Làm tròn
        for col in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
            if col in comparison.columns:
                comparison[col] = pd.to_numeric(comparison[col], errors="coerce").round(4)
        
        # Thêm cột improvement so với baseline tốt nhất
        baseline_results = comparison[comparison['model_type'] == 'baseline']
        if not baseline_results.empty:
            best_baseline_f1 = baseline_results['f1'].max()
            
            def calc_improvement(row):
                if row['model_type'] == 'improved':
                    return f"+{(row['f1'] - best_baseline_f1) * 100:.1f}%"
                return "-"
            
            comparison['improvement_vs_best_baseline'] = comparison.apply(calc_improvement, axis=1)
        
        return comparison
    
    def compare_baselines(self):
        """So sánh chi tiết giữa 2 baselines"""
        baseline_results = {name: self.results[name] for name in self.baselines}
        
        comparison = {}
        for name, metrics in baseline_results.items():
            comparison[name] = {
                'F1 Score': metrics['f1'],
                'ROC-AUC': metrics['roc_auc'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'TP (True Positive)': metrics['tp'],
                'FN (False Negative)': metrics['fn']
            }
        
        df = pd.DataFrame(comparison).T
        return df
    
    def analyze_improvement(self):
        """Phân tích sự cải thiện của improved model so với baselines"""
        if not self.improved_model or not self.baselines:
            return None
        
        improved_metrics = self.results[self.improved_model]
        
        analysis = {
            'improved_model': self.improved_model,
            'baselines': self.baselines,
            'improvements': {}
        }
        
        for baseline in self.baselines:
            baseline_metrics = self.results[baseline]
            
            # Tính % cải thiện cho từng metric
            improvements = {}
            for metric in ['f1', 'roc_auc', 'precision', 'recall']:
                if metric in improved_metrics and metric in baseline_metrics:
                    base_val = baseline_metrics[metric]
                    impr_val = improved_metrics[metric]
                    if base_val > 0:
                        pct_improve = (impr_val - base_val) / base_val * 100
                        improvements[metric] = f"{pct_improve:+.1f}%"
            
            analysis['improvements'][baseline] = improvements
        
        return analysis
    
    def save_models(self, preprocessor=None, builder=None, output_dir="outputs/models/"):
        """Lưu pipeline hoàn chỉnh (feature + preprocess + model)"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            if preprocessor is not None and builder is not None:
                pipeline = Pipeline([
                    ("feature_builder", builder),
                    ("preprocessor", preprocessor),
                    ("model", model)
                ])
            else:
                pipeline = model

            path = Path(output_dir) / f"{name}_pipeline.joblib"
            joblib.dump(pipeline, path)
            logger.info(f"Saved pipeline {name} to {path}")