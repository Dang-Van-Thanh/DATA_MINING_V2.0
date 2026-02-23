"""
Module tiền xử lý dữ liệu
ĐÃ LOẠI BỎ HOÀN TOÀN 'duration' để tránh data leakage
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """Xử lý làm sạch và tiền xử lý dữ liệu - KHÔNG DATA LEAKAGE"""
    
    def __init__(self, config_path="configs/params.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.numeric_cols = self.config['features']['numeric_cols']
        self.categorical_cols = self.config['features']['categorical_cols']
        self.target_col = self.config['data']['target_col']
        self.target_mapping = self.config['features']['target_mapping']
        
        # KIỂM TRA VÀ LOẠI BỎ duration
        self._check_and_remove_duration()
    
    def _check_and_remove_duration(self):
        """Kiểm tra và loại bỏ duration khỏi numeric_cols"""
        if 'duration' in self.numeric_cols:
            logger.warning("=" * 60)
            logger.warning("⚠️  DANGER: 'duration' found in numeric_cols!")
            logger.warning("⚠️  This causes SEVERE data leakage in real-world scenarios")
            logger.warning("⚠️  Automatically removing 'duration' from features")
            logger.warning("=" * 60)
            
            # Loại bỏ duration
            self.numeric_cols = [col for col in self.numeric_cols if col != 'duration']
            logger.info(f"✅ Safe numeric columns: {self.numeric_cols}")
    
    def clean_data(self, df):
        """Làm sạch dữ liệu cơ bản - KHÔNG ẢNH HƯỞNG ĐẾN LEAKAGE"""
        logger.info("Starting data cleaning...")
        df_clean = df.copy()
        
        # 1. Xử lý giá trị thiếu
        missing_before = df_clean.isnull().sum().sum()
        if self.config['preprocessing']['handle_missing'] == 'drop':
            df_clean = df_clean.dropna()
        
        logger.info(f"Missing values: {missing_before} -> {df_clean.isnull().sum().sum()}")
        
        # 2. Xử lý duplicate
        dup_before = df_clean.duplicated().sum()
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Duplicates: {dup_before} -> {df_clean.duplicated().sum()}")
        
        # 3. Xử lý outlier cho numeric columns (CHỈ CÁC CỘT AN TOÀN)
        safe_numeric = [col for col in self.numeric_cols if col in df_clean.columns]
        for col in safe_numeric:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            if outliers > 0:
                logger.info(f"Column {col}: {outliers} outliers detected (capped)")
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        # 4. Xử lý pdays = -1 (chưa từng liên lạc)
        if 'pdays' in df_clean.columns:
            df_clean['never_contacted'] = (df_clean['pdays'] == -1).astype(int)
            df_clean['pdays'] = df_clean['pdays'].replace(-1, 999)
        
        # 5. Mã hóa target
        if self.target_col in df_clean.columns:
            df_clean[self.target_col] = df_clean[self.target_col].map(self.target_mapping)
        
        # 6. CẢNH BÁO NẾU VẪN CÒN duration
        if 'duration' in df_clean.columns:
            logger.warning("⚠️  'duration' still exists in dataframe but will be excluded from modeling")
            logger.warning("    (kept only for EDA purposes)")
        
        logger.info("Data cleaning completed")
        return df_clean
    
    def create_preprocessing_pipeline(self):
        """
        Tạo pipeline tiền xử lý cho features
        ĐÃ LOẠI BỎ HOÀN TOÀN DURATION
        """
        # Lọc chỉ lấy các cột numeric an toàn
        safe_numeric_cols = [col for col in self.numeric_cols if col in self.numeric_cols]
        
        if len(safe_numeric_cols) == 0:
            logger.warning("⚠️  No numeric columns left after removing duration!")
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, safe_numeric_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ])
        
        logger.info(f"✅ Preprocessing pipeline created")
        logger.info(f"   Numeric features: {safe_numeric_cols}")
        logger.info(f"   Categorical features: {self.categorical_cols}")
        
        return preprocessor
    
    def prepare_features(self, df, preprocessor=None, fit=True):
        """
        Chuẩn bị features cho modeling
        ĐẢM BẢO KHÔNG CÓ DURATION TRONG FEATURES
        """
        # KIỂM TRA NGHIÊM NGẶT - KHÔNG CHO PHÉP duration
        if 'duration' in df.columns:
            logger.warning("⚠️  Removing 'duration' from features to prevent data leakage")
            df = df.drop(columns=['duration'])
        
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col] if self.target_col in df.columns else None
        
        if fit:
            if preprocessor is None:
                preprocessor = self.create_preprocessing_pipeline()
            X_processed = preprocessor.fit_transform(X)
            return X_processed, y, preprocessor
        else:
            X_processed = preprocessor.transform(X)
            return X_processed, y
    
    def apply_smote(self, X, y):
        """Áp dụng SMOTE để cân bằng lớp - CHỈ DÙNG TRÊN TRAIN"""
        if self.config['preprocessing']['handle_imbalance'] and \
           self.config['preprocessing']['imbalance_method'] == 'smote':
            
            logger.info("Applying SMOTE for class imbalance")
            smote = SMOTE(random_state=self.config['project']['seed'])
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            before = dict(zip(*np.unique(y, return_counts=True)))
            after = dict(zip(*np.unique(y_resampled, return_counts=True)))
            
            logger.info(f"Before SMOTE: {before}")
            logger.info(f"After SMOTE: {after}")
            
            return np.array(X_resampled), np.array(y_resampled)
        
        return X, y
    
    def get_feature_names(self, preprocessor):
        """Lấy tên các features sau preprocessing (hữu ích cho interpretability)"""
        if preprocessor is None:
            return []
        
        feature_names = []
        
        # Lấy tên cho numeric features
        numeric_features = [col for col in self.numeric_cols if col != 'duration']
        feature_names.extend(numeric_features)
        
        # Lấy tên cho categorical features (one-hot encoded)
        for i, cat_col in enumerate(self.categorical_cols):
            # Lấy các categories từ onehot encoder
            cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
            if hasattr(cat_encoder, 'categories_'):
                categories = cat_encoder.categories_[i][1:]  # drop='first'
                for cat in categories:
                    feature_names.append(f"{cat_col}_{cat}")
        
        return feature_names