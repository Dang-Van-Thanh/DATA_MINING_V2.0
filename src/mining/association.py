"""
Module khai phá luật kết hợp
"""
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AssociationMiner:
    """Khai phá luật kết hợp"""
    
    def __init__(self, config_path="configs/params.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.min_support = self.config['association_rules']['min_support']
        self.min_confidence = self.config['association_rules']['min_confidence']
        self.min_lift = self.config['association_rules']['min_lift']
        self.max_len = self.config['association_rules']['max_len']
    
    def discretize_numeric(self, df, col, bins=3):
        """Rời rạc hóa biến numeric"""
        if col not in df.columns:
            return df
            
        labels = [f"{col}_{i}" for i in range(bins)]
        try:
            df[f"{col}_bin"] = pd.cut(df[col], bins=bins, labels=labels)
        except:
            # Nếu có lỗi, dùng quantile-based binning
            df[f"{col}_bin"] = pd.qcut(df[col], q=bins, labels=labels, duplicates='drop')
        return df
    
    def prepare_basket(self, df):
        """
        Chuẩn bị dữ liệu dạng basket cho Apriori
        Mỗi dòng là một giao dịch (khách hàng) với các items là đặc trưng
        """
        df_basket = df.copy()
        
        # Rời rạc hóa các biến numeric quan trọng
        df_basket = self.discretize_numeric(df_basket, 'age', bins=4)
        df_basket = self.discretize_numeric(df_basket, 'balance', bins=5)
        df_basket = self.discretize_numeric(df_basket, 'duration', bins=3)
        
        # Tạo items từ categorical columns
        categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 
                           'loan', 'contact', 'month', 'poutcome']
        
        # Thêm các bin columns
        bin_cols = [col for col in df_basket.columns if col.endswith('_bin')]
        categorical_cols.extend(bin_cols)
        
        # Chỉ lấy các cột tồn tại
        categorical_cols = [col for col in categorical_cols if col in df_basket.columns]
        
        # Thêm target
        if 'y' in df_basket.columns:
            categorical_cols.append('y')
        
        # One-hot encoding cho từng item
        baskets = []
        for idx, row in df_basket[categorical_cols].iterrows():
            items = []
            for col in categorical_cols:
                val = row[col]
                if pd.notna(val) and val != 'unknown':
                    items.append(f"{col}={val}")
            baskets.append(items)
        
        # Tạo DataFrame one-hot
        from mlxtend.preprocessing import TransactionEncoder
        te = TransactionEncoder()
        te_ary = te.fit(baskets).transform(baskets)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        logger.info(f"Basket prepared: {df_encoded.shape[1]} items, {df_encoded.shape[0]} transactions")
        return df_encoded
    
    def mine_rules(self, df_encoded):
        """Khai phá luật kết hợp"""
        logger.info(f"Mining association rules with min_support={self.min_support}")
        
        # Tìm frequent itemsets
        try:
            frequent_itemsets = apriori(
                df_encoded, 
                min_support=self.min_support,
                max_len=self.max_len,
                use_colnames=True
            )
        except Exception as e:
            logger.error(f"Error in apriori: {e}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(frequent_itemsets)} frequent itemsets")
        
        if len(frequent_itemsets) == 0:
            logger.warning("No frequent itemsets found")
            return pd.DataFrame()
        
        # Sinh luật kết hợp
        try:
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=self.min_confidence
            )
        except Exception as e:
            logger.error(f"Error generating rules: {e}")
            return pd.DataFrame()
        
        # Lọc theo lift
        rules = rules[rules['lift'] >= self.min_lift]
        
        # Sắp xếp theo lift giảm dần
        rules = rules.sort_values('lift', ascending=False)
        
        logger.info(f"Found {len(rules)} association rules")
        
        return rules
    
    def analyze_rules(self, rules):
        """Phân tích luật kết hợp và rút insight"""
        if len(rules) == 0:
            return ["Không tìm thấy luật kết hợp đáng kể"]
        
        insights = []
        
        # Luật liên quan đến target (y=yes)
        target_rules = rules[rules['consequents'].apply(lambda x: 'y=1' in str(x) or 'y=yes' in str(x))]
        if len(target_rules) > 0:
            insights.append(f"Có {len(target_rules)} luật liên quan đến đăng ký thành công")
            
            # Top luật có confidence cao nhất
            top_target = target_rules.nlargest(5, 'confidence')
            for _, rule in top_target.iterrows():
                antecedents = ', '.join([str(item) for item in list(rule['antecedents'])])
                insights.append(f"- Nếu {antecedents} → đăng ký thành công (conf={rule['confidence']:.2f}, lift={rule['lift']:.2f})")
        
        # Luật có lift cao nhất
        high_lift_rules = rules.nlargest(5, 'lift')
        insights.append("\nTop 5 luật có lift cao nhất:")
        for _, rule in high_lift_rules.iterrows():
            antecedents = ', '.join([str(item) for item in list(rule['antecedents'])])
            consequents = ', '.join([str(item) for item in list(rule['consequents'])])
            insights.append(f"- {antecedents} → {consequents} (lift={rule['lift']:.2f})")
        
        # Cross-sell insights
        housing_loan_rules = rules[
            rules['antecedents'].apply(lambda x: any('housing=yes' in str(item) or 'loan=yes' in str(item) for item in x))
        ]
        if len(housing_loan_rules) > 0:
            insights.append("\nGợi ý cross-sell cho khách có khoản vay:")
            for _, rule in housing_loan_rules.head(3).iterrows():
                consequents = ', '.join([str(item) for item in list(rule['consequents'])])
                insights.append(f"- Khách có khoản vay thường {consequents} (conf={rule['confidence']:.2f})")
        
        return insights