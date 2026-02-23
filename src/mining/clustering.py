"""
Module phân cụm khách hàng
ĐÃ LOẠI BỎ duration để tránh insights sai lệch
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomerClustering:
    """Phân cụm khách hàng dựa trên hồ sơ tài chính - KHÔNG DÙNG duration"""
    
    def __init__(self, config_path="configs/params.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.n_clusters = self.config['clustering']['n_clusters']
        self.random_state = self.config['clustering']['random_state']
    
    def _remove_duration_from_features(self, feature_list):
        """Loại bỏ duration khỏi danh sách features"""
        if 'duration' in feature_list:
            logger.warning("⚠️  Removing 'duration' from clustering features")
            logger.warning("   (duration causes misleading insights)")
            return [f for f in feature_list if f != 'duration']
        return feature_list
    
    def prepare_clustering_features(self, df):
        """
        Chuẩn bị features cho phân cụm
        ĐÃ LOẠI BỎ duration
        """
        # Chọn các features liên quan đến hồ sơ tài chính (KHÔNG duration)
        financial_features = ['age', 'balance', 'campaign', 'previous']
        #                                 ^ ĐÃ LOẠI BỎ duration
        
        # Thêm các đặc trưng đã được xây dựng
        if 'RFM_score' in df.columns:
            financial_features.extend(['R_score', 'F_score', 'M_score', 'RFM_score'])
        
        if 'has_debt' in df.columns:
            financial_features.append('has_debt')
        
        if 'was_contacted_before' in df.columns:
            financial_features.append('was_contacted_before')
        
        # Loại bỏ duration (an toàn)
        financial_features = self._remove_duration_from_features(financial_features)
        
        # Lọc các features có trong dataframe
        available_features = [f for f in financial_features if f in df.columns]
        
        logger.info(f"Clustering features (safe, no duration): {available_features}")
        
        X_cluster = df[available_features].copy()
        
        # Xử lý giá trị vô hạn hoặc NaN
        X_cluster = X_cluster.replace([np.inf, -np.inf], np.nan)
        X_cluster = X_cluster.fillna(X_cluster.mean())
        
        # Chuẩn hóa
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        return X_scaled, scaler, available_features
    
    def find_optimal_k(self, X, max_k=10):
        """Tìm số cụm tối ưu bằng elbow method và silhouette score"""
        inertias = []
        silhouettes = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            try:
                sil = silhouette_score(X, labels)
                silhouettes.append(sil)
                logger.info(f"K={k}: inertia={kmeans.inertia_:.2f}, silhouette={sil:.4f}")
            except:
                silhouettes.append(-1)
                logger.info(f"K={k}: inertia={kmeans.inertia_:.2f}, silhouette= N/A")
        
        return inertias, silhouettes
    
    def perform_clustering(self, X, n_clusters=None):
        """Thực hiện phân cụm K-means"""
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        logger.info(f"Performing K-means clustering with {n_clusters} clusters")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Đánh giá
        try:
            silhouette = silhouette_score(X, cluster_labels)
        except:
            silhouette = -1
            logger.warning("Could not compute silhouette score")
        
        try:
            davies_bouldin = davies_bouldin_score(X, cluster_labels)
        except:
            davies_bouldin = -1
            logger.warning("Could not compute Davies-Bouldin score")
        
        logger.info(f"Silhouette Score: {silhouette:.4f}")
        logger.info(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
        
        return cluster_labels, kmeans, silhouette, davies_bouldin
    
    def profile_clusters(self, df, cluster_labels):
        """
        Tạo hồ sơ cho từng cụm
        KHÔNG DÙNG duration trong profile
        """
        df_profile = df.copy()
        df_profile['Cluster'] = cluster_labels
        
        # Tính toán các đặc trưng cho từng cụm (KHÔNG duration)
        numeric_cols = ['age', 'balance', 'campaign', 'previous']
        #                                 ^ ĐÃ LOẠI BỎ duration
        
        if 'RFM_score' in df.columns:
            numeric_cols.extend(['R_score', 'F_score', 'M_score', 'RFM_score'])
        
        if 'was_contacted_before' in df.columns:
            numeric_cols.append('was_contacted_before')
        
        if 'has_debt' in df.columns:
            numeric_cols.append('has_debt')
        
        # Loại bỏ duration nếu còn sót
        numeric_cols = self._remove_duration_from_features(numeric_cols)
        
        # Lọc các cột tồn tại
        numeric_cols = [col for col in numeric_cols if col in df_profile.columns]
        
        cluster_profile = df_profile.groupby('Cluster')[numeric_cols].agg(['mean', 'std', 'count'])
        
        # Tính tỷ lệ đăng ký thành công theo cụm
        if 'y' in df_profile.columns:
            success_rate = df_profile.groupby('Cluster')['y'].mean()
            cluster_profile[('y_success_rate', 'mean')] = success_rate
        
        # Thống kê categorical
        cat_cols = ['job', 'marital', 'education', 'housing', 'loan']
        cat_profiles = {}
        for col in cat_cols:
            if col in df_profile.columns:
                try:
                    cat_profiles[col] = df_profile.groupby('Cluster')[col].value_counts(normalize=True).unstack().fillna(0)
                except:
                    pass
        
        return cluster_profile, cat_profiles
    
    def get_cluster_insights(self, df, cluster_labels):
        """
        Rút insight từ các cụm
        KHÔNG DÙNG duration trong insights
        """
        df_insight = df.copy()
        df_insight['Cluster'] = cluster_labels
        
        insights = []
        total_customers = len(df_insight)
        
        # Cảnh báo nếu vẫn còn duration
        if 'duration' in df_insight.columns:
            logger.warning("⚠️  'duration' exists but excluded from clustering insights")
        
        # Phân tích từng cụm
        for cluster in sorted(df_insight['Cluster'].unique()):
            cluster_data = df_insight[df_insight['Cluster'] == cluster]
            
            # Đặc điểm cơ bản (KHÔNG duration)
            size = len(cluster_data)
            age_mean = cluster_data['age'].mean()
            balance_mean = cluster_data['balance'].mean()
            campaign_mean = cluster_data['campaign'].mean()
            previous_mean = cluster_data['previous'].mean()
            
            insight = f"\nCụm {cluster} ({size} khách hàng - {size/total_customers*100:.1f}%):"
            insights.append(insight)
            insights.append(f"- Tuổi trung bình: {age_mean:.1f}")
            insights.append(f"- Số dư trung bình: {balance_mean:.0f} EUR")
            insights.append(f"- Số lần liên lạc TB: {campaign_mean:.1f}")
            insights.append(f"- Số lần liên lạc trước TB: {previous_mean:.1f}")
            
            # RFM scores nếu có
            if 'RFM_score' in cluster_data.columns:
                rfm_mean = cluster_data['RFM_score'].mean()
                insights.append(f"- RFM score TB: {rfm_mean:.1f}")
            
            # Tỷ lệ thành công
            if 'y' in cluster_data.columns:
                success_rate = cluster_data['y'].mean() * 100
                insights.append(f"- Tỷ lệ đăng ký thành công: {success_rate:.2f}%")
            
            # Đặc trưng nổi bật
            if 'has_debt' in cluster_data.columns:
                debt_rate = cluster_data['has_debt'].mean() * 100
                insights.append(f"- Có khoản vay: {debt_rate:.1f}%")
            
            if 'was_contacted_before' in cluster_data.columns:
                contacted_rate = cluster_data['was_contacted_before'].mean() * 100
                insights.append(f"- Đã từng liên lạc: {contacted_rate:.1f}%")
            
            # Nghề nghiệp phổ biến
            if 'job' in cluster_data.columns:
                top_jobs = cluster_data['job'].value_counts()
                if len(top_jobs) > 0:
                    top_job = top_jobs.index[0]
                    top_job_pct = top_jobs.iloc[0] / size * 100
                    insights.append(f"- Nghề phổ biến: {top_job} ({top_job_pct:.1f}%)")
        
        # Thêm insight tổng hợp
        insights.append("\n📊 INSIGHTS TỔNG HỢP:")
        insights.append("• Các cụm được phân dựa trên hành vi khách hàng (KHÔNG dùng duration)")
        insights.append("• Có thể xây dựng chiến lược marketing riêng cho từng cụm")
        
        return insights
    
    def analyze_cluster_separability(self, X, labels, feature_names):
        """Phân tích khả năng phân tách giữa các cụm"""
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        # Giảm chiều để visualize
        tsne = TSNE(n_components=2, random_state=self.random_state)
        X_tsne = tsne.fit_transform(X)
        
        # Vẽ biểu đồ
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization of Clusters (No duration)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        
        # Lưu figure
        from pathlib import Path
        output_dir = Path('outputs/figures')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'cluster_tsne.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ t-SNE cluster visualization saved")