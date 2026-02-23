"""
Module bán giám sát cho phân lớp
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.semi_supervised import SelfTrainingClassifier
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemiSupervisedLearning:
    """Huấn luyện bán giám sát với self-training"""
    
    def __init__(self, config_path="configs/params.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.seed = self.config['project']['seed']
        self.label_ratios = self.config['semi_supervised']['label_ratios']
        self.confidence_threshold = self.config['semi_supervised']['confidence_threshold']
        self.max_iterations = self.config['semi_supervised']['max_iterations']
        
        self.results = {}
    
    def simulate_label_scarcity(self, X, y, label_ratio):
        """
        Giả lập tình huống thiếu nhãn
        Giữ lại p% nhãn, phần còn lại coi là unlabeled
        
        X: numpy array
        y: numpy array
        """
        n_samples = len(y)
        n_labeled = int(n_samples * label_ratio)
        
        # Tạo masked labels (-1 cho unlabeled) - numpy array
        y_masked = np.full_like(y, -1, dtype=int)
        
        # Chọn ngẫu nhiên các mẫu có nhãn
        np.random.seed(self.seed)
        labeled_indices = np.random.choice(
            n_samples, size=n_labeled, replace=False
        )
        
        # Gán nhãn cho các mẫu được chọn
        y_masked[labeled_indices] = y[labeled_indices]
        
        logger.info(f"Label ratio {label_ratio}: {n_labeled} labeled samples, {n_samples - n_labeled} unlabeled")
        
        return y_masked, labeled_indices
    
    def train_supervised_only(self, X_train, y_train_masked, X_test, y_test):
        """
        Huấn luyện chỉ với dữ liệu có nhãn (supervised-only)
        """
        from sklearn.linear_model import LogisticRegression
        
        # Lọc chỉ lấy mẫu có nhãn
        labeled_mask = y_train_masked != -1
        X_labeled = X_train[labeled_mask]
        y_labeled = y_train_masked[labeled_mask]
        
        logger.info(f"Supervised-only: {len(X_labeled)} labeled samples")
        
        if len(X_labeled) == 0:
            return None, {'f1': 0, 'roc_auc': 0}
        
        # Huấn luyện mô hình
        base_model = LogisticRegression(
            C=1.0, max_iter=1000, class_weight='balanced', random_state=self.seed
        )
        
        base_model.fit(X_labeled, y_labeled)
        
        # Đánh giá
        y_pred = base_model.predict(X_test)
        y_pred_proba = base_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return base_model, metrics
    
    def train_self_training(self, X_train, y_train_masked, X_test, y_test):
        """
        Huấn luyện với self-training (semi-supervised)
        """
        from sklearn.linear_model import LogisticRegression
        
        # Base model
        base_model = LogisticRegression(
            C=1.0, max_iter=1000, class_weight='balanced', random_state=self.seed
        )
        
        # Self-training
        self_training_model = SelfTrainingClassifier(
            base_model,
            threshold=self.confidence_threshold,
            criterion='threshold',
            max_iter=self.max_iterations,
            verbose=False
        )
        
        self_training_model.fit(X_train, y_train_masked)
        
        # Đánh giá
        y_pred = self_training_model.predict(X_test)
        y_pred_proba = self_training_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'iterations': self_training_model.n_iter_,
        }
        
        # Đếm số pseudo labels được thêm vào
        if hasattr(self_training_model, 'transduction_'):
            pseudo_labels_count = np.sum(self_training_model.transduction_ != -1) - np.sum(y_train_masked != -1)
            metrics['pseudo_labels'] = int(pseudo_labels_count)
            
            # Phân tích pseudo-label
            pseudo_analysis = self.analyze_pseudo_labels(
                X_train, y_train_masked, self_training_model
            )
            metrics.update(pseudo_analysis)
        else:
            metrics['pseudo_labels'] = 0
        
        return self_training_model, metrics
    
    def analyze_pseudo_labels(self, X, y_masked, model):
        """Phân tích rủi ro của pseudo-labels"""
        if not hasattr(model, 'transduction_'):
            return {'pseudo_correct_rate': 0, 'pseudo_risk': 0}
        
        # Lấy pseudo-labels
        pseudo_labels = model.transduction_
        
        # Xác định các mẫu được pseudo-label
        originally_labeled = y_masked != -1
        pseudo_labeled = (pseudo_labels != -1) & ~originally_labeled
        
        if not np.any(pseudo_labeled):
            return {
                'pseudo_count': 0,
                'pseudo_avg_confidence': 0,
                'pseudo_low_confidence_ratio': 0
            }
        
        # Ước lượng độ tin cậy qua xác suất dự đoán
        pseudo_proba = model.predict_proba(X[pseudo_labeled])
        pseudo_confidence = np.max(pseudo_proba, axis=1)
        
        # Rủi ro ước lượng: tỷ lệ pseudo-labels có độ tin cậy thấp
        low_confidence = np.mean(pseudo_confidence < 0.9)
        
        return {
            'pseudo_count': int(np.sum(pseudo_labeled)),
            'pseudo_avg_confidence': float(np.mean(pseudo_confidence)),
            'pseudo_low_confidence_ratio': float(low_confidence)
        }
    
    def run_experiment(self, X_train, y_train, X_test, y_test):
        """
        Chạy thực nghiệm với các tỷ lệ nhãn khác nhau
        So sánh supervised-only vs semi-supervised
        """
        results = []
        
        for ratio in self.label_ratios:
            logger.info(f"\n=== Experiment with label ratio: {ratio} ===")
            
            # Giả lập thiếu nhãn
            y_masked, labeled_indices = self.simulate_label_scarcity(
                X_train, y_train, ratio
            )
            
            # Supervised-only
            _, sup_metrics = self.train_supervised_only(
                X_train, y_masked, X_test, y_test
            )
            sup_metrics['method'] = 'supervised_only'
            sup_metrics['label_ratio'] = ratio
            
            # Semi-supervised
            _, semi_metrics = self.train_self_training(
                X_train, y_masked, X_test, y_test
            )
            semi_metrics['method'] = 'semi_supervised'
            semi_metrics['label_ratio'] = ratio
            
            results.append(sup_metrics)
            results.append(semi_metrics)
            
            logger.info(f"Supervised-only: F1={sup_metrics['f1']:.4f}, ROC-AUC={sup_metrics['roc_auc']:.4f}")
            logger.info(f"Semi-supervised: F1={semi_metrics['f1']:.4f}, ROC-AUC={semi_metrics['roc_auc']:.4f}")
            logger.info(f"Pseudo-labels added: {semi_metrics.get('pseudo_count', 0)}")
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def plot_learning_curve(self, save_path=None):
        """Vẽ learning curve theo % nhãn"""
        if self.results.empty:
            logger.error("No results to plot")
            return
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # F1 Score
        for method in ['supervised_only', 'semi_supervised']:
            data = self.results[self.results['method'] == method]
            if not data.empty:
                data = data.sort_values('label_ratio')
                axes[0].plot(
                    data['label_ratio'] * 100, 
                    data['f1'], 
                    marker='o', 
                    label=method.replace('_', ' ').title()
                )
        
        axes[0].set_xlabel('Label Ratio (%)')
        axes[0].set_ylabel('F1 Score')
        axes[0].set_title('Learning Curve - F1 Score')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # ROC-AUC
        for method in ['supervised_only', 'semi_supervised']:
            data = self.results[self.results['method'] == method]
            if not data.empty:
                data = data.sort_values('label_ratio')
                axes[1].plot(
                    data['label_ratio'] * 100, 
                    data['roc_auc'], 
                    marker='o', 
                    label=method.replace('_', ' ').title()
                )
        
        axes[1].set_xlabel('Label Ratio (%)')
        axes[1].set_ylabel('ROC-AUC')
        axes[1].set_title('Learning Curve - ROC-AUC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Learning curve saved to {save_path}")
        
        plt.show()