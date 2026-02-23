"""
Module vẽ biểu đồ
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Plotter:
    """Vẽ các biểu đồ phân tích"""
    
    def __init__(self, output_dir="outputs/figures/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def save_figure(self, fig, filename, dpi=100):
        """Lưu figure"""
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        logger.info(f"Figure saved to {filepath}")
        plt.close(fig)
    
    def plot_target_distribution(self, df, target_col='y', save=True):
        """Biểu đồ 1: Phân bố target"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Count plot
        target_counts = df[target_col].value_counts()
        labels = ['No', 'Yes'] if 0 in target_counts.index else target_counts.index
        
        axes[0].bar(labels, target_counts.values, color=['skyblue', 'salmon'])
        axes[0].set_title('Phân bố Target (Đăng ký term deposit)')
        axes[0].set_xlabel('Đăng ký')
        axes[0].set_ylabel('Số lượng')
        
        # Thêm số liệu
        for i, v in enumerate(target_counts.values):
            axes[0].text(i, v + 50, str(v), ha='center')
        
        # Pie chart
        axes[1].pie(target_counts.values, labels=labels, autopct='%1.1f%%',
                   colors=['skyblue', 'salmon'], startangle=90)
        axes[1].set_title('Tỷ lệ target')
        
        plt.tight_layout()
        
        if save:
            self.save_figure(fig, '01_target_distribution.png')
        
        return fig
    
    def plot_age_balance_distribution(self, df, save=True):
        """Biểu đồ 2: Phân bố age và balance theo target"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Age distribution
        for i, target in enumerate([0, 1]):
            target_name = 'No' if target == 0 else 'Yes'
            data = df[df['y'] == target]['age']
            axes[0, i].hist(data, bins=30, alpha=0.7, edgecolor='black')
            axes[0, i].set_title(f'Phân bố tuổi - Target: {target_name}')
            axes[0, i].set_xlabel('Tuổi')
            axes[0, i].set_ylabel('Tần số')
            axes[0, i].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.1f}')
            axes[0, i].legend()
        
        # Balance distribution (log scale)
        for i, target in enumerate([0, 1]):
            target_name = 'No' if target == 0 else 'Yes'
            data = df[df['y'] == target]['balance']
            # Xử lý giá trị âm
            data_positive = data[data > 0]
            if len(data_positive) > 0:
                axes[1, i].hist(data_positive, bins=50, alpha=0.7, edgecolor='black', log=True)
                axes[1, i].set_title(f'Phân bố số dư (log scale) - Target: {target_name}')
                axes[1, i].set_xlabel('Số dư (EUR)')
                axes[1, i].set_ylabel('Tần số (log)')
                axes[1, i].axvline(data_positive.mean(), color='red', linestyle='--', 
                                  label=f'Mean: {data_positive.mean():.0f}')
                axes[1, i].legend()
        
        plt.tight_layout()
        
        if save:
            self.save_figure(fig, '02_age_balance_distribution.png')
        
        return fig
    
    def plot_categorical_features(self, df, save=True):
        """Biểu đồ 3: Phân tích các biến categorical"""
        categorical_cols = ['job', 'marital', 'education', 'housing', 'loan', 'contact']
        
        n_cols = 2
        n_rows = (len(categorical_cols) + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
        axes = axes.flatten()
        
        for idx, col in enumerate(categorical_cols):
            if col not in df.columns:
                continue
                
            # Tính tỷ lệ thành công theo từng category
            success_rate = df.groupby(col)['y'].mean().sort_values(ascending=False) * 100
            
            # Vẽ bar chart
            colors = ['#2ecc71' if v > 20 else '#e74c3c' for v in success_rate.values]
            success_rate.plot(kind='bar', ax=axes[idx], color=colors)
            axes[idx].set_title(f'Tỷ lệ thành công theo {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Tỷ lệ thành công (%)')
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Thêm giá trị
            for i, v in enumerate(success_rate.values):
                axes[idx].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)
        
        # Ẩn các subplot thừa
        for idx in range(len(categorical_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            self.save_figure(fig, '03_categorical_analysis.png')
        
        return fig
    
    def plot_correlation_heatmap(self, df, save=True):
        """Biểu đồ 4: Heatmap tương quan"""
        # Chọn các biến numeric
        numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous', 'y']
        numeric_df = df[numeric_cols].copy()
        
        # Xử lý pdays
        if 'pdays' in numeric_df.columns:
            numeric_df['pdays_adj'] = numeric_df['pdays'].replace(999, 365)
            numeric_df = numeric_df.drop('pdays', axis=1)
        
        # Tính correlation
        corr = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Vẽ heatmap
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, square=True, linewidths=1, ax=ax)
        
        ax.set_title('Ma trận tương quan giữa các biến số', fontsize=14)
        
        plt.tight_layout()
        
        if save:
            self.save_figure(fig, '04_correlation_heatmap.png')
        
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save=True):
        """Biểu đồ confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {model_name}')
        
        # Thêm thông tin
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        metrics_text = f'Accuracy: {accuracy:.2f}\n'
        metrics_text += f'Precision: {precision:.2f}\n'
        metrics_text += f'Recall: {recall:.2f}\n'
        metrics_text += f'F1: {f1:.2f}'
        
        ax.text(1.5, 2, metrics_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
        
        plt.tight_layout()
        
        if save:
            self.save_figure(fig, f'05_confusion_matrix_{model_name}.png')
        
        return fig
    
    def plot_roc_curves(self, y_test, predictions_dict, save=True):
        """Biểu đồ ROC curves so sánh các model"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        for idx, (model_name, y_pred_proba) in enumerate(predictions_dict.items()):
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Xác định line style dựa trên loại model
            if 'logistic' in model_name.lower():
                linestyle = '--'  # Baseline 1
                linewidth = 2
            elif 'decision' in model_name.lower() or 'tree' in model_name.lower():
                linestyle = '-.'  # Baseline 2
                linewidth = 2
            else:
                linestyle = '-'   # Improved
                linewidth = 3
            
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', 
                   color=colors[idx % len(colors)], linestyle=linestyle, linewidth=linewidth)
        
        # Đường baseline
        ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)', linewidth=1)
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - 2 Baselines vs 1 Improved Model', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            self.save_figure(fig, '06_roc_curves_comparison.png')
        
        return fig
    
    def plot_feature_importance(self, feature_names, importance, model_name, top_n=15, save=True):
        """Biểu đồ feature importance"""
        # Tạo DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Màu sắc dựa trên loại model
        if 'decision' in model_name.lower():
            color = 'green'
        elif 'xgb' in model_name.lower():
            color = 'red'
        else:
            color = 'steelblue'
        
        # Vẽ horizontal bar chart
        bars = ax.barh(importance_df['feature'], importance_df['importance'], color=color)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance - {model_name}')
        
        # Thêm giá trị
        for bar, val in zip(bars, importance_df['importance']):
            ax.text(val + 0.01 * importance_df['importance'].max(), 
                   bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center')
        
        plt.tight_layout()
        
        if save:
            self.save_figure(fig, f'07_feature_importance_{model_name}.png')
        
        return fig
    
    def plot_cluster_profiles(self, df, cluster_labels, save=True):
        """Biểu đồ hồ sơ các cụm"""
        df_plot = df.copy()
        df_plot['Cluster'] = cluster_labels
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        plot_cols = ['age', 'balance', 'duration', 'campaign', 'previous']
        
        for i, col in enumerate(plot_cols):
            if col in df_plot.columns:
                # Box plot
                df_plot.boxplot(column=col, by='Cluster', ax=axes[i])
                axes[i].set_title(f'Phân bố {col} theo cụm')
                axes[i].set_xlabel('Cụm')
                
                # Thêm mean
                means = df_plot.groupby('Cluster')[col].mean()
                for j, mean_val in enumerate(means):
                    if not np.isnan(mean_val):
                        axes[i].text(j+1, df_plot[col].max() * 1.05, f'μ={mean_val:.1f}', 
                                   ha='center', fontsize=9)
        
        # Success rate by cluster
        if 'y' in df_plot.columns:
            success_rate = df_plot.groupby('Cluster')['y'].mean() * 100
            axes[5].bar(success_rate.index, success_rate.values, color='teal')
            axes[5].set_title('Tỷ lệ thành công theo cụm (%)')
            axes[5].set_xlabel('Cụm')
            axes[5].set_ylabel('Tỷ lệ thành công (%)')
            
            for j, val in enumerate(success_rate.values):
                axes[5].text(j, val + 1, f'{val:.1f}%', ha='center')
        
        plt.suptitle('Cluster Profiles Analysis', fontsize=16)
        plt.tight_layout()
        
        if save:
            self.save_figure(fig, '08_cluster_profiles.png')
        
        return fig
    
    def plot_semi_supervised_learning_curve(self, results_df, save=True):
        """Biểu đồ learning curve cho semi-supervised"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for method in ['supervised_only', 'semi_supervised']:
            data = results_df[results_df['method'] == method]
            if not data.empty:
                # Sắp xếp theo label_ratio
                data = data.sort_values('label_ratio')
                
                # Vẽ F1
                axes[0].plot(data['label_ratio'] * 100, data['f1'], 
                            marker='o', linewidth=2, label=method.replace('_', ' ').title())
                
                # Vẽ ROC-AUC
                axes[1].plot(data['label_ratio'] * 100, data['roc_auc'], 
                            marker='o', linewidth=2, label=method.replace('_', ' ').title())
        
        axes[0].set_xlabel('Tỷ lệ nhãn (%)')
        axes[0].set_ylabel('F1 Score')
        axes[0].set_title('Learning Curve - F1 Score')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Tỷ lệ nhãn (%)')
        axes[1].set_ylabel('ROC-AUC')
        axes[1].set_title('Learning Curve - ROC-AUC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            self.save_figure(fig, '09_semi_supervised_learning_curve.png')
        
        return fig