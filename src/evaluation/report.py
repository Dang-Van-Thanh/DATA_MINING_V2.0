"""
Module tổng hợp báo cáo kết quả
"""
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportGenerator:
    """Tổng hợp báo cáo kết quả"""
    
    def __init__(self, config_path="configs/params.yaml", output_dir="outputs/reports/"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_results(self, results, filename):
        """Lưu kết quả dạng JSON"""
        # Chuyển đổi numpy types thành Python types
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            else:
                return obj
        
        results_serializable = json.loads(
            json.dumps(results, default=convert_to_serializable)
        )
        
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {filepath}")
    
    def generate_classification_report(self, model_results, feature_importance=None):
        """Tạo báo cáo phân lớp"""
        report = {
            'model_comparison': {},
            'baselines': [],
            'improved_model': None,
            'best_model': None,
            'feature_importance': feature_importance
        }
        
        best_f1 = -1
        best_model = None
        
        for model_name, metrics in model_results.items():
            report['model_comparison'][model_name] = {
                'accuracy': float(metrics.get('accuracy', 0)),
                'precision': float(metrics.get('precision', 0)),
                'recall': float(metrics.get('recall', 0)),
                'f1': float(metrics.get('f1', 0)),
                'roc_auc': float(metrics.get('roc_auc', 0)),
                'pr_auc': float(metrics.get('pr_auc', 0)),
                'cv_roc_auc_mean': float(metrics.get('cv_roc_auc_mean', 0)),
                'cv_roc_auc_std': float(metrics.get('cv_roc_auc_std', 0)),
                'model_type': metrics.get('model_type', 'unknown')
            }
            
            if metrics.get('model_type') == 'baseline':
                report['baselines'].append(model_name)
            else:
                report['improved_model'] = model_name
            
            if metrics.get('f1', 0) > best_f1:
                best_f1 = metrics['f1']
                best_model = model_name
        
        report['best_model'] = best_model
        report['best_model_f1'] = best_f1
        
        return report
    
    def generate_clustering_report(self, cluster_profile, cluster_insights, metrics):
        """Tạo báo cáo phân cụm"""
        report = {
            'clustering_metrics': metrics,
            'cluster_profiles': cluster_profile.to_dict() if isinstance(cluster_profile, pd.DataFrame) else cluster_profile,
            'insights': cluster_insights
        }
        
        return report
    
    def generate_association_report(self, rules, insights):
        """Tạo báo cáo luật kết hợp"""
        if isinstance(rules, pd.DataFrame):
            rules_dict = rules.head(10).to_dict() if len(rules) > 0 else {}
        else:
            rules_dict = {}
            
        report = {
            'total_rules': len(rules) if isinstance(rules, pd.DataFrame) else 0,
            'top_rules': rules_dict,
            'insights': insights
        }
        
        return report
    
    def generate_semi_supervised_report(self, semi_results):
        """Tạo báo cáo bán giám sát"""
        if isinstance(semi_results, pd.DataFrame):
            results_dict = semi_results.to_dict()
            
            # Tính toán cải thiện
            summary = {
                'best_label_ratio': None,
                'best_improvement': 0
            }
            
            for ratio in semi_results['label_ratio'].unique():
                sup = semi_results[(semi_results['label_ratio'] == ratio) & 
                                   (semi_results['method'] == 'supervised_only')]
                semi = semi_results[(semi_results['label_ratio'] == ratio) & 
                                    (semi_results['method'] == 'semi_supervised')]
                
                if len(sup) > 0 and len(semi) > 0:
                    improvement = float(semi['f1'].values[0] - sup['f1'].values[0])
                    if improvement > summary['best_improvement']:
                        summary['best_improvement'] = improvement
                        summary['best_label_ratio'] = float(ratio)
        else:
            results_dict = {}
            summary = {}
        
        report = {
            'learning_curve': results_dict,
            'summary': summary
        }
        
        return report
    
    def generate_full_report(self, all_results):
        """Tạo báo cáo đầy đủ"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'config': self.config,
            'results': all_results
        }
        
        return report