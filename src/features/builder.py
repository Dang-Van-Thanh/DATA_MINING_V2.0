"""
Module xây dựng đặc trưng
LƯU Ý: Đã loại bỏ các feature gây data leakage (duration)
"""
import pandas as pd
import numpy as np
import yaml
import logging

from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Xây dựng các đặc trưng mới - KHÔNG DATA LEAKAGE"""
    
    def __init__(self, config_path="configs/params.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def create_rfm_features(self, df):
        """
        Tạo các đặc trưng RFM (Recency, Frequency, Monetary)
        Các feature này AN TOÀN, không gây leakage
        """
        df_rfm = df.copy()
        
        # R: Recency - dựa trên pdays (thông tin quá khứ)
        df_rfm['R_score'] = pd.cut(
            df_rfm['pdays'].replace(999, 365),
            bins=[-1, 30, 90, 180, 365],
            labels=[4, 3, 2, 1],
            include_lowest=True
        ).astype(float)
        
        # F: Frequency - dựa trên previous (thông tin quá khứ)
        df_rfm['F_score'] = pd.cut(
            df_rfm['previous'],
            bins=[-1, 0, 2, 5, 100],
            labels=[1, 2, 3, 4],
            include_lowest=True
        ).astype(float)
        
        # M: Monetary - dựa trên balance (thông tin hiện tại)
        df_rfm['M_score'] = pd.cut(
            df_rfm['balance'],
            bins=[-float('inf'), 0, 500, 2000, 10000, float('inf')],
            labels=[1, 2, 3, 4, 5]
        ).astype(float)
        
        # RFM tổng hợp
        df_rfm['RFM_score'] = df_rfm['R_score'] + df_rfm['F_score'] + df_rfm['M_score']
        
        logger.info("✅ RFM features created (safe)")
        return df_rfm
    
    def create_engagement_features(self, df):
        """
        Tạo đặc trưng về mức độ tương tác
        ĐÃ LOẠI BỎ duration để tránh data leakage
        """
        df_eng = df.copy()
        
        # Tỷ lệ thành công của các chiến dịch trước (thông tin quá khứ)
        df_eng['prev_success_rate'] = 0.0
        mask_success = df_eng['poutcome'] == 'success'
        df_eng.loc[mask_success, 'prev_success_rate'] = 1.0
        
        # ❌ ĐÃ LOẠI BỎ - không dùng duration
        # df_eng['contact_intensity'] = df_eng['campaign'] / (df_eng['duration'] + 1)
        
        # Có từng được liên lạc chưa (thông tin quá khứ)
        df_eng['was_contacted_before'] = (df_eng['previous'] > 0).astype(int)
        
        logger.info("✅ Engagement features created (safe - no duration)")
        return df_eng
    
    def create_interaction_features(self, df):
        """
        Tạo đặc trưng tương tác giữa các biến
        KHÔNG dùng duration
        """
        df_inter = df.copy()
        
        # Tương tác giữa age và balance (an toàn)
        df_inter['age_balance'] = df_inter['age'] * (df_inter['balance'] / 1000)
        
        # ❌ ĐÃ LOẠI BỎ - campaign*duration gây leakage
        # df_inter['campaign_duration'] = df_inter['campaign'] * df_inter['duration']
        
        # Tương tác giữa housing và loan (an toàn)
        df_inter['has_debt'] = ((df_inter['housing'] == 'yes') | (df_inter['loan'] == 'yes')).astype(int)
        
        logger.info("✅ Interaction features created (safe)")
        return df_inter
    
    def create_time_features(self, df):
        """
        Tạo đặc trưng thời gian
        Các feature này AN TOÀN, dựa trên lịch
        """
        df_time = df.copy()
        
        # Mùa hè (tháng 6-8)
        df_time['is_summer'] = df_time['month'].isin(['jun', 'jul', 'aug']).astype(int)
        
        # Cuối tuần (dựa trên ngày)
        weekend_days = [6, 7, 13, 14, 20, 21, 27, 28]
        df_time['is_weekend'] = df_time['day'].isin(weekend_days).astype(int)
        
        # Đầu tháng (ngày 1-5)
        df_time['is_month_start'] = (df_time['day'] <= 5).astype(int)
        
        # Cuối tháng (ngày 25-31)
        df_time['is_month_end'] = (df_time['day'] >= 25).astype(int)
        
        logger.info("✅ Time features created (safe)")
        return df_time
    
    def build_all_features(self, df):
        """
        Xây dựng tất cả đặc trưng
        
        LƯU Ý QUAN TRỌNG:
        - Đã loại bỏ hoàn toàn 'duration' khỏi feature engineering
        - Tất cả features đều an toàn, không gây data leakage
        - Có thể áp dụng trực tiếp cho train và test
        
        Returns:
            DataFrame với các features mới (không bao gồm duration)
        """
        logger.info("=" * 50)
        logger.info("BUILDING FEATURES - SAFE MODE (NO DURATION)")
        logger.info("=" * 50)
        
        df_features = df.copy()
        
        # Lưu duration riêng để kiểm tra (nếu có)
        if 'duration' in df_features.columns:
            duration_stats = {
                'mean': df_features['duration'].mean(),
                'median': df_features['duration'].median(),
                'min': df_features['duration'].min(),
                'max': df_features['duration'].max()
            }
            logger.info(f"⚠️  'duration' exists in data but will NOT be used in features")
            logger.info(f"   Duration stats: mean={duration_stats['mean']:.1f}, "
                       f"median={duration_stats['median']:.1f}")
        
        # Tạo các nhóm đặc trưng (KHÔNG dùng duration)
        df_features = self.create_rfm_features(df_features)
        df_features = self.create_engagement_features(df_features)
        df_features = self.create_interaction_features(df_features)
        df_features = self.create_time_features(df_features)
        
        # Log danh sách features đã tạo
        new_features = [col for col in df_features.columns if col not in df.columns]
        logger.info(f"✅ Created {len(new_features)} new features: {new_features}")
        logger.info(f"✅ Total features: {df_features.shape[1]} (all safe)")
        
        return df_features
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.build_all_features(X)