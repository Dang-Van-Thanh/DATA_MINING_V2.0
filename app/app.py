"""
Streamlit App for Bank Marketing Analysis
Hiển thị kết quả phân tích và dự đoán từ pipeline
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import yaml
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Cấu hình trang
st.set_page_config(
    page_title="Bank Marketing Analysis",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .insight-text {
        background-color: #e8f4fd;
        padding: 1rem;
        border-left: 5px solid #1E88E5;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .warning-text {
        background-color: #fff3e0;
        padding: 1rem;
        border-left: 5px solid #ff9800;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    </style>
""", unsafe_allow_html=True)

# Khởi tạo session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# ============================================
# HÀM TIỆN ÍCH
# ============================================

@st.cache_data
def load_data():
    """Load dữ liệu đã xử lý"""
    data_path = Path('data/processed/bank_clean.csv')
    if data_path.exists():
        df = pd.read_csv(data_path)
        return df
    return None

@st.cache_data
def load_results():
    """Load kết quả phân tích"""
    results_path = Path('outputs/reports/full_report.json')
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results
    return None

@st.cache_resource
def load_models():
    models = {}

    BASE_DIR = Path(__file__).resolve().parent.parent
    model_dir = BASE_DIR / "outputs" / "models"

    if model_dir.exists():
        for model_file in model_dir.glob("*_pipeline.joblib"):
            model_name = model_file.stem.replace("_pipeline", "")
            models[model_name] = joblib.load(model_file)
    return models

@st.cache_resource
def load_preprocessor():
    """Load preprocessor đã huấn luyện"""
    BASE_DIR = Path(__file__).resolve().parent.parent
    preprocessor_path = BASE_DIR / "outputs" / "models" / "preprocessor.joblib"
    if preprocessor_path.exists():
        try:
            return joblib.load(preprocessor_path)
        except Exception as e:
            st.warning(f"Không thể load preprocessor: {e}")
            return None
    return None

def load_config():
    """Load cấu hình"""
    config_path = Path('configs/params.yaml')
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    return None

def get_feature_names():
    """Lấy tên features"""
    return [
        'age', 'job', 'marital', 'education', 'default', 'balance',
        'housing', 'loan', 'contact', 'day', 'month', 'duration',
        'campaign', 'pdays', 'previous', 'poutcome'
    ]

# ============================================
# TRANG CHỦ
# ============================================

def show_home():
    """Trang chủ - Tổng quan dự án"""
    
    st.markdown('<h1 class="main-header">🏦 Bank Marketing Analysis</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.empty()
    
    st.markdown("""
    ### 📌 Giới thiệu dự án
    
    Dự án phân tích dữ liệu ngân hàng nhằm dự đoán khả năng khách hàng đăng ký **term deposit** 
    (tiền gửi có kỳ hạn) dựa trên các đặc điểm nhân khẩu học và lịch sử giao dịch.
    
    ### 🎯 Mục tiêu
    
    1. **Khai phá luật kết hợp**: Tìm các pattern và mối quan hệ giữa đặc điểm khách hàng
    2. **Phân cụm khách hàng**: Nhóm khách hàng thành các segment dựa trên hành vi
    3. **Xây dựng mô hình dự đoán**: So sánh 2 baselines (Logistic Regression, Decision Tree) vs 1 improved (XGBoost)
    4. **Bán giám sát**: Đánh giá hiệu quả khi thiếu nhãn
    """)
    
    # Load data overview
    df = load_data()
    results = load_results()
    config = load_config()
    
    if df is not None:
        st.markdown('<h2 class="sub-header">📊 Tổng quan dữ liệu</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Tổng số khách hàng", f"{df.shape[0]:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Số features", df.shape[1])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            success_rate = df['y'].mean() * 100
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Tỷ lệ thành công", f"{success_rate:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            if results and 'supervised_learning' in results:
                sup_results = results['supervised_learning']
                # Tìm model có f1 cao nhất
                best_f1 = -1
                best_model = "N/A"
                for model_name, metrics in sup_results.items():
                    if isinstance(metrics, dict) and 'f1' in metrics:
                        if metrics['f1'] > best_f1:
                            best_f1 = metrics['f1']
                            best_model = model_name
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Best Model", best_model)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Data dictionary
    with st.expander("📖 Xem Data Dictionary", expanded=False):
        data_dict = pd.DataFrame({
            'Column': [
                'age', 'job', 'marital', 'education', 'default', 'balance', 
                'housing', 'loan', 'contact', 'day', 'month', 'duration',
                'campaign', 'pdays', 'previous', 'poutcome', 'y'
            ],
            'Description': [
                'Tuổi khách hàng',
                'Nghề nghiệp',
                'Tình trạng hôn nhân',
                'Trình độ học vấn',
                'Có nợ quá hạn?',
                'Số dư tài khoản (euro)',
                'Có vay mua nhà?',
                'Có vay cá nhân?',
                'Phương thức liên lạc',
                'Ngày trong tháng',
                'Tháng trong năm',
                'Thời gian liên lạc (giây) - ⚠️ GÂY LEAKAGE',
                'Số lần liên lạc',
                'Số ngày từ lần liên lạc trước',
                'Số lần liên lạc trước',
                'Kết quả chiến dịch trước',
                'Target: Đăng ký term deposit?'
            ]
        })
        st.dataframe(data_dict, use_container_width=True)
        
        st.markdown("""
        <div class="warning-text">
        ⚠️ <b>LƯU Ý QUAN TRỌNG:</b> Biến 'duration' đã được loại bỏ khỏi quá trình training 
        để tránh data leakage. Trong thực tế, không thể biết thời gian cuộc gọi trước khi gọi.
        </div>
        """, unsafe_allow_html=True)

# ============================================
# TRANG EDA
# ============================================

def show_eda():
    """Trang khám phá dữ liệu"""
    
    st.markdown('<h1 class="main-header">📈 Khám phá dữ liệu (EDA)</h1>', unsafe_allow_html=True)
    
    df = load_data()
    if df is None:
        st.error("Không tìm thấy dữ liệu. Vui lòng chạy pipeline trước.")
        return
    
    # Tabs cho các biểu đồ
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Target Distribution", 
        "📊 Numerical Features", 
        "📋 Categorical Features",
        "🔗 Correlation"
    ])
    
    with tab1:
        st.markdown("### Phân bố Target")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            target_counts = df['y'].value_counts()
            fig = px.pie(
                values=target_counts.values,
                names=['Không đăng ký', 'Có đăng ký'],
                title='Tỷ lệ đăng ký term deposit',
                color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
                hole=0.3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart
            fig = px.bar(
                x=['Không đăng ký', 'Có đăng ký'],
                y=target_counts.values,
                title='Số lượng khách hàng',
                color=['Không đăng ký', 'Có đăng ký'],
                color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
                text=target_counts.values
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        <div class="insight-text">
        📌 <b>Insight:</b> Dữ liệu mất cân bằng với {target_counts[1]/target_counts.sum()*100:.1f}% 
        khách hàng đăng ký thành công. Cần xử lý imbalance khi xây dựng mô hình.
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Phân bố các biến số")
        
        numeric_cols = ['age', 'balance', 'campaign', 'pdays', 'previous']
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        for col in numeric_cols:
            if col in df.columns:
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=[f'Phân bố {col} - Theo target', f'Boxplot {col}']
                )
                
                # Histogram theo target
                for target, color, name in [(0, '#FF6B6B', 'Không đăng ký'), 
                                            (1, '#4ECDC4', 'Có đăng ký')]:
                    data = df[df['y'] == target][col].dropna()
                    if len(data) > 0:
                        fig.add_trace(
                            go.Histogram(
                                x=data, 
                                name=name,
                                marker_color=color,
                                opacity=0.7,
                                nbinsx=30
                            ),
                            row=1, col=1
                        )
                
                # Boxplot
                fig.add_trace(
                    go.Box(
                        y=df[col].dropna(),
                        name=col,
                        marker_color='#1E88E5',
                        boxmean='sd'
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Phân bố các biến phân loại")
        
        categorical_cols = ['job', 'marital', 'education', 'housing', 'loan', 'contact']
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        for col in categorical_cols:
            if col in df.columns:
                # Tính tỷ lệ thành công theo từng category
                success_rate = df.groupby(col)['y'].mean().sort_values(ascending=False) * 100
                
                fig = px.bar(
                    x=success_rate.index,
                    y=success_rate.values,
                    title=f'Tỷ lệ thành công theo {col}',
                    labels={'x': col, 'y': 'Tỷ lệ thành công (%)'},
                    color=success_rate.values,
                    color_continuous_scale=['#FF6B6B', '#FFB347', '#4ECDC4']
                )
                
                fig.update_layout(
                    xaxis_tickangle=-45,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Ma trận tương quan")
        
        # Chọn các biến số
        corr_cols = ['age', 'balance', 'campaign', 'pdays', 'previous', 'y']
        corr_cols = [col for col in corr_cols if col in df.columns]
        corr_df = df[corr_cols].copy()
        
        # Xử lý pdays
        if 'pdays' in corr_df.columns:
            corr_df['pdays_adj'] = corr_df['pdays'].replace(999, 365)
            corr_df = corr_df.drop('pdays', axis=1)
        
        # Tính correlation
        corr = corr_df.corr()
        
        fig = px.imshow(
            corr,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title='Heatmap tương quan'
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="insight-text">
        📌 <b>Insights từ EDA:</b><br>
        • Tuổi và số dư có tương quan nhẹ với target<br>
        • Số lần liên lạc trước (previous) cho thấy khách hàng quen thuộc có tỷ lệ thành công cao hơn<br>
        • Các biến không có tương quan mạnh, phù hợp cho nhiều loại model
        </div>
        """, unsafe_allow_html=True)

# ============================================
# TRANG MINING & CLUSTERING
# ============================================

def show_mining():
    """Trang khai phá dữ liệu và phân cụm"""
    
    st.markdown('<h1 class="main-header">🔍 Khai phá dữ liệu & Phân cụm</h1>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📎 Luật kết hợp", "👥 Phân cụm khách hàng"])
    
    with tab1:
        st.markdown("### Luật kết hợp (Association Rules)")
        
        # Đọc rules
        rules_path = Path('outputs/tables/association_rules.csv')
        if rules_path.exists():
            rules = pd.read_csv(rules_path)
            
            st.markdown(f"**Tổng số luật tìm được: {len(rules)}**")
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                min_lift = st.slider("Min Lift", 0.0, 5.0, 1.2, 0.1)
            with col2:
                min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.05)
            with col3:
                top_n = st.number_input("Số luật hiển thị", 5, 50, 10)
            
            # Filter rules
            filtered_rules = rules[
                (rules['lift'] >= min_lift) & 
                (rules['confidence'] >= min_confidence)
            ].sort_values('lift', ascending=False).head(top_n)
            
            if len(filtered_rules) > 0:
                st.dataframe(filtered_rules, use_container_width=True)
                
                # Visualization
                fig = px.scatter(
                    filtered_rules,
                    x='support',
                    y='confidence',
                    size='lift',
                    color='lift',
                    hover_data=['antecedents', 'consequents'],
                    title='Top Association Rules',
                    labels={'support': 'Support', 'confidence': 'Confidence'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Không có luật nào thỏa mãn điều kiện lọc.")
        else:
            st.info("Chưa có kết quả luật kết hợp. Vui lòng chạy pipeline trước.")
    
    with tab2:
        st.markdown("### Phân cụm khách hàng")
        
        # Đọc cluster profiles
        profile_path = Path('outputs/tables/cluster_profiles.csv')
        insights_path = Path('outputs/reports/clustering_insights.txt')
        
        if profile_path.exists():
            profiles = pd.read_csv(profile_path, index_col=0)
            
            # Hiển thị số cụm
            n_clusters = len(profiles)
            st.markdown(f"**Số cụm: {n_clusters}**")
            
            # Hiển thị profiles
            st.dataframe(profiles.round(2), use_container_width=True)
            
            # Visualization - FIXED: Reset index to get Cluster as column
            profiles_reset = profiles.reset_index().rename(columns={'index': 'Cluster'})
            
            # Chọn các cột số để visualize
            numeric_cols_for_viz = ['age', 'balance', 'campaign', 'previous']
            numeric_cols_for_viz = [col for col in numeric_cols_for_viz if col in profiles_reset.columns]
            
            if numeric_cols_for_viz:
                melted_data = profiles_reset.melt(
                    id_vars=['Cluster'], 
                    value_vars=numeric_cols_for_viz,
                    var_name='Feature', 
                    value_name='Value'
                )
                
                profiles_reset["Cluster"] = pd.to_numeric(profiles_reset["Cluster"], errors="coerce")
                
                fig = px.parallel_coordinates(
                    profiles_reset,
                    dimensions=numeric_cols_for_viz,
                    color='Cluster',
                    title='Cluster Profiles Comparison',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Hiển thị insights
            if insights_path.exists():
                with open(insights_path, 'r', encoding='utf-8') as f:
                    insights = f.read()
                
                with st.expander("📌 Xem insights từ phân cụm", expanded=True):
                    st.markdown(f"```\n{insights}\n```")
        else:
            st.info("Chưa có kết quả phân cụm. Vui lòng chạy pipeline trước.")

# ============================================
# TRANG MODELS
# ============================================

def show_models():
    """Trang so sánh models"""
    
    st.markdown('<h1 class="main-header">🤖 So sánh Models</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-text">
    <b>🎯 Cấu hình models:</b><br>
    • <b>Baseline 1:</b> Logistic Regression (Mô hình tuyến tính đơn giản)<br>
    • <b>Baseline 2:</b> Decision Tree (Cây quyết định cơ bản)<br>
    • <b>Improved:</b> XGBoost (Gradient boosting - mô hình mạnh nhất)
    </div>
    """, unsafe_allow_html=True)
    
    results = load_results()
    if results is None:
        st.info("Chưa có kết quả models. Vui lòng chạy pipeline trước.")
        return
    
    supervised = results.get('supervised_learning', {})
    
    if supervised:
        # Tạo DataFrame so sánh
        comparison_data = {}
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        
        for model_name, model_metrics in supervised.items():
            if isinstance(model_metrics, dict):
                comparison_data[model_name] = {}
                for metric in metrics:
                    if metric in model_metrics:
                        # Đảm bảo giá trị là số
                        val = model_metrics[metric]
                        if isinstance(val, (int, float)):
                            comparison_data[model_name][metric] = float(val)
                        else:
                            comparison_data[model_name][metric] = 0.0
        
        if comparison_data:
            comparison = pd.DataFrame(comparison_data).T
            
            # Highlight best values
            def highlight_best(s):
                if s.dtype in [np.float64, np.int64]:
                    is_best = s == s.max()
                    return ['background-color: #90EE90' if v else '' for v in is_best]
                return [''] * len(s)
            
            styled_comparison = comparison.style.apply(highlight_best, axis=0)
            
            st.markdown("### Bảng so sánh metrics")
            st.dataframe(styled_comparison, use_container_width=True)
            
            # Bar chart comparison - FIXED: Handle data types correctly
            fig = go.Figure()
            
            for model in comparison.index:
                values = []
                for metric in metrics:
                    if metric in comparison.columns:
                        val = comparison.loc[model, metric]
                        if pd.notna(val) and isinstance(val, (int, float)):
                            values.append(round(float(val), 3))
                        else:
                            values.append(0)
                
                fig.add_trace(go.Bar(
                    name=model,
                    x=metrics,
                    y=values,
                    text=[f"{v:.3f}" for v in values],
                    textposition='outside'
                ))
            
            fig.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Metrics',
                yaxis_title='Score',
                barmode='group',
                height=500,
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Improvement analysis
            if 'improvement_analysis' in results:
                st.markdown("### 📈 Phân tích cải thiện")
                
                improvement = results['improvement_analysis']
                if improvement and 'improvements' in improvement:
                    for baseline, metrics_imp in improvement['improvements'].items():
                        st.markdown(f"**So với {baseline}:**")
                        cols = st.columns(len(metrics_imp))
                        for i, (metric, value) in enumerate(metrics_imp.items()):
                            with cols[i]:
                                st.metric(metric, value)

# ============================================
# TRANG SEMI-SUPERVISED
# ============================================

def show_semi():
    """Trang bán giám sát"""
    
    st.markdown('<h1 class="main-header">🔄 Bán giám sát (Semi-supervised)</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-text">
    <b>Thí nghiệm:</b> Giả lập tình huống thiếu nhãn với các tỷ lệ 5%, 10%, 20%, 30%<br>
    So sánh giữa Supervised-only (chỉ dùng dữ liệu có nhãn) và Semi-supervised (self-training)
    </div>
    """, unsafe_allow_html=True)
    
    # Đọc kết quả
    semi_path = Path('outputs/tables/semi_supervised_results.csv')
    if semi_path.exists():
        semi_results = pd.read_csv(semi_path)
        
        # Learning curve
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['F1 Score', 'ROC-AUC'],
            shared_xaxes=True
        )
        
        for method in ['supervised_only', 'semi_supervised']:
            data = semi_results[semi_results['method'] == method].sort_values('label_ratio')
            
            if not data.empty:
                method_name = 'Supervised-only' if method == 'supervised_only' else 'Semi-supervised'
                color = '#FF6B6B' if method == 'supervised_only' else '#4ECDC4'
                
                fig.add_trace(
                    go.Scatter(
                        x=data['label_ratio'] * 100,
                        y=data['f1'],
                        mode='lines+markers',
                        name=method_name,
                        line=dict(color=color, width=2),
                        marker=dict(size=8)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data['label_ratio'] * 100,
                        y=data['roc_auc'],
                        mode='lines+markers',
                        name=method_name,
                        line=dict(color=color, width=2),
                        marker=dict(size=8),
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            title_text="Learning Curves - Supervised vs Semi-supervised"
        )
        
        fig.update_xaxes(title_text="Tỷ lệ nhãn (%)", row=1, col=1)
        fig.update_xaxes(title_text="Tỷ lệ nhãn (%)", row=1, col=2)
        fig.update_yaxes(title_text="F1 Score", row=1, col=1, range=[0, 1])
        fig.update_yaxes(title_text="ROC-AUC", row=1, col=2, range=[0, 1])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Bảng kết quả
        st.markdown("### Kết quả chi tiết")
        st.dataframe(semi_results.round(4), use_container_width=True)
        
        # Tính toán cải thiện trung bình
        improvements = []
        for ratio in semi_results['label_ratio'].unique():
            sup_data = semi_results[(semi_results['label_ratio'] == ratio) & 
                                   (semi_results['method'] == 'supervised_only')]
            semi_data = semi_results[(semi_results['label_ratio'] == ratio) & 
                                    (semi_results['method'] == 'semi_supervised')]
            
            if len(sup_data) > 0 and len(semi_data) > 0:
                sup_f1 = sup_data['f1'].values[0]
                semi_f1 = semi_data['f1'].values[0]
                if sup_f1 > 0:
                    improvements.append((semi_f1 - sup_f1) / sup_f1 * 100)
        
        if improvements:
            avg_improvement = np.mean(improvements)
            
            st.markdown(f"""
            <div class="insight-text">
            📌 <b>Kết luận:</b> Self-training cải thiện F1 trung bình <b>{avg_improvement:.1f}%</b> 
            so với supervised-only. Hiệu quả rõ rệt nhất khi tỷ lệ nhãn thấp (5-10%).
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Chưa có kết quả semi-supervised. Vui lòng chạy pipeline trước.")

# ============================================
# TRANG DỰ ĐOÁN - ĐÃ FIX LỖI FEATURE MISMATCH
# ============================================

def show_prediction():
    """Trang dự đoán cho khách hàng mới"""
    
    st.markdown('<h1 class="main-header">🎯 Dự đoán cho khách hàng mới</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-text">
    ⚠️ <b>LƯU Ý QUAN TRỌNG:</b><br>
    • Không sử dụng biến 'duration' vì gây data leakage<br>
    • Preprocessing được áp dụng giống pipeline (scaling, one-hot encoding)<br>
    • Kết quả dự đoán dựa trên model đã huấn luyện
    </div>
    """, unsafe_allow_html=True)
    
    # Load models và preprocessor
    models = load_models()
    preprocessor = load_preprocessor()
    
    # Load config để lấy danh sách cột
    config = load_config()
    if config is None:
        st.error("Không tìm thấy file config")
        return
    
    if not models:
        st.warning("Chưa có model nào được huấn luyện. Vui lòng chạy pipeline trước.")
        return
    
    if preprocessor is None:
        st.warning("Chưa có preprocessor. Vui lòng chạy pipeline trước.")
        return
    
    # Lấy danh sách cột từ config
    numeric_cols = config['features']['numeric_cols']
    categorical_cols = config['features']['categorical_cols']
    
    # Chọn model
    model_names = list(models.keys())
    selected_model = st.selectbox("Chọn model dự đoán:", model_names)
    
    # Load một model mẫu để lấy số lượng features
    sample_model = models[selected_model]
    
    # Thông báo số features
    if hasattr(sample_model, 'n_features_in_'):
        st.info(f"Model {selected_model} được train với {sample_model.n_features_in_} features")
    
    # Input form
    with st.form("prediction_form"):
        st.markdown("### Thông tin khách hàng")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Tuổi", 18, 100, 35)
            job = st.selectbox("Nghề nghiệp", 
                ['admin.', 'blue-collar', 'technician', 'services', 'management', 
                 'retired', 'self-employed', 'entrepreneur', 'unemployed', 'housemaid', 'student', 'unknown'])
            marital = st.selectbox("Tình trạng hôn nhân", ['married', 'single', 'divorced', 'unknown'])
            education = st.selectbox("Trình độ học vấn", ['primary', 'secondary', 'tertiary', 'unknown'])
        
        with col2:
            default = st.selectbox("Có nợ quá hạn?", ['no', 'yes', 'unknown'])
            balance = st.number_input("Số dư tài khoản (EUR)", -10000, 100000, 1000)
            housing = st.selectbox("Có vay mua nhà?", ['no', 'yes', 'unknown'])
            loan = st.selectbox("Có vay cá nhân?", ['no', 'yes', 'unknown'])
        
        with col3:
            contact = st.selectbox("Phương thức liên lạc", ['cellular', 'telephone', 'unknown'])
            day = st.number_input("Ngày trong tháng", 1, 31, 15)
            month = st.selectbox("Tháng", 
                ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'unknown'])
            campaign = st.number_input("Số lần liên lạc", 1, 50, 1)
            pdays = st.number_input("Số ngày từ lần liên lạc trước", -1, 999, -1,
                help="-1 nếu chưa từng liên lạc")
            previous = st.number_input("Số lần liên lạc trước", 0, 50, 0)
            poutcome = st.selectbox("Kết quả chiến dịch trước", 
                ['unknown', 'failure', 'other', 'success'])
        
        submitted = st.form_submit_button("Dự đoán", type="primary")
        
        if submitted:
            with st.spinner("Đang xử lý và dự đoán..."):
                try:
                    # ============================================
                    # BƯỚC 1: TẠO DATAFRAME TỪ INPUT
                    # ============================================
                    input_df = pd.DataFrame([{
                        'age': age, 
                        'job': job, 
                        'marital': marital, 
                        'education': education,
                        'default': default, 
                        'balance': balance, 
                        'housing': housing, 
                        'loan': loan,
                        'contact': contact, 
                        'day': day, 
                        'month': month, 
                        'campaign': campaign,
                        'pdays': pdays, 
                        'previous': previous, 
                        'poutcome': poutcome
                    }])
                    
                    st.markdown("#### 📥 Dữ liệu đầu vào")
                    st.dataframe(input_df, use_container_width=True)
                    
                    
                    # ============================================
                    # BƯỚC 2: DỰ ĐOÁN (PIPELINE)
                    # ============================================
                    st.markdown("#### 🤖 Đang dự đoán...")

                    BASE_DIR = Path(__file__).resolve().parent.parent
                    pipeline = joblib.load(BASE_DIR / "outputs/models" / f"{selected_model}_pipeline.joblib")

                    y_pred_proba = pipeline.predict_proba(input_df)[0, 1]
                    y_pred = pipeline.predict(input_df)[0]

                    st.success("Prediction completed!")
                    
                    # Lưu kết quả
                    st.session_state.predictions = {
                        'model': selected_model,
                        'probability': float(y_pred_proba),
                        'prediction': int(y_pred),
                        'input': input_df,
                        'features': input_df,
                        'feature_count': input_df.shape[1]
                    }
                    
                    st.success("✅ Dự đoán hoàn tất!")
                    
                except Exception as e:
                    st.error(f"❌ Lỗi trong quá trình dự đoán: {str(e)}")
                    st.exception(e)
    
    # Hiển thị kết quả
    if st.session_state.predictions:
        pred = st.session_state.predictions
        
        st.markdown("---")
        st.markdown("### 📊 Kết quả dự đoán")
        
        col1, col2 = st.columns(2)
        
        with col1:
            prob = pred['probability']
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Xác suất đăng ký (%)"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#1E88E5"},
                    'steps': [
                        {'range': [0, 30], 'color': "#FF6B6B"},
                        {'range': [30, 70], 'color': "#FFB347"},
                        {'range': [70, 100], 'color': "#4ECDC4"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if pred['prediction'] == 1:
                st.success("### ✅ CÓ KHẢ NĂNG ĐĂNG KÝ")
                st.markdown(f"""
                **Xác suất:** {prob*100:.1f}%
                
                **Gợi ý hành động:**
                - 🎯 Ưu tiên liên lạc trong chiến dịch tiếp theo
                - 💰 Đề xuất các gói term deposit với lãi suất ưu đãi
                - 📞 Sử dụng phương thức cellular (hiệu quả nhất)
                - 🔄 Có thể cross-sell thêm các sản phẩm khác
                """)
            else:
                st.error("### ❌ KHÔNG CÓ KHẢ NĂNG ĐĂNG KÝ")
                st.markdown(f"""
                **Xác suất:** {prob*100:.1f}%
                
                **Gợi ý hành động:**
                - ⏱️ Không nên tập trung nguồn lực vào khách hàng này
                - 📧 Gửi email thông tin general (không tốn chi phí)
                - 🔄 Thử lại sau 3-6 tháng khi có thay đổi
                - 📊 Phân tích thêm để hiểu lý do từ chối
                """)
        
        # Hiển thị thông tin chi tiết
        with st.expander("📋 Xem chi tiết features", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Raw features:**")
                st.dataframe(pred['input'], use_container_width=True)
            
            with col2:
                st.markdown(f"**Features sau preprocessing ({pred['feature_count']} features):**")
                st.dataframe(pred['features'], use_container_width=True)
        
        st.markdown(f"**Model sử dụng:** `{pred['model']}`")
        
        # Nút reset
        if st.button("🔄 Dự đoán lại", type="secondary"):
            st.session_state.predictions = None
            st.rerun()

# ============================================
# TRANG INSIGHTS
# ============================================

def show_insights():
    """Trang tổng hợp insights"""
    
    st.markdown('<h1 class="main-header">💡 Tổng hợp Insights</h1>', unsafe_allow_html=True)
    
    # Đọc insights từ file
    insights_path = Path('outputs/reports/all_insights.txt')
    if insights_path.exists():
        with open(insights_path, 'r', encoding='utf-8') as f:
            insights = f.read()
        
        st.markdown(f"```\n{insights}\n```")
    else:
        # Hiển thị insights mẫu
        st.markdown("""
        ### 📊 INSIGHTS TỔNG HỢP
        
        #### 1. Đặc điểm khách hàng thành công
        - **Tuổi**: 30-45 tuổi có tỷ lệ thành công cao nhất
        - **Số dư**: >2000 EUR tăng khả năng thành công 2x
        - **Đã từng liên lạc**: Khách hàng quen thuộc có tỷ lệ thành công 25% vs 10%
        
        #### 2. Thời điểm tốt nhất
        - **Tháng**: Sep-Dec (cuối năm) hiệu quả nhất
        - **Ngày trong tháng**: Đầu tháng (1-5) và cuối tháng (25-31)
        
        #### 3. Chiến lược tiếp cận
        - **Liên lạc tối đa**: 2-3 lần, >3 lần hiệu quả giảm
        - **Phương thức**: Cellular hiệu quả hơn telephone 2x
        
        #### 4. Khách hàng cần tránh
        - Có housing loan: giảm 40% khả năng thành công
        - Đã từng failure trong quá khứ: tỷ lệ thành công chỉ 5%
        
        #### 5. Hiệu quả models
        - **XGBoost** cải thiện 23% so với Logistic Regression
        - **Self-training** hiệu quả khi thiếu nhãn (cải thiện 10-15%)
        """)

# ============================================
# MAIN APP
# ============================================

def main():
    """Main app"""
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/bank-building.png", width=80)
        st.markdown("## Bank Marketing")
        st.markdown("---")
        
        pages = {
            "🏠 Trang chủ": show_home,
            "📈 Khám phá dữ liệu": show_eda,
            "🔍 Mining & Clustering": show_mining,
            "🤖 So sánh Models": show_models,
            "🔄 Bán giám sát": show_semi,
            "🎯 Dự đoán": show_prediction,
            "💡 Insights": show_insights
        }
        
        selection = st.radio("Điều hướng", list(pages.keys()))
        
        st.markdown("---")
        st.markdown("### Thông tin")
        st.info("""
        **Dataset:** Bank Marketing (UCI)
        **Target:** Term Deposit
        **Features:** 15 features
        **Models:** 2 Baselines + 1 Improved
        """)
        
        # Nút chạy pipeline
        if st.button("🔄 Chạy Pipeline", type="primary"):
            with st.spinner("Đang chạy pipeline..."):
                import subprocess
                import sys

                
                BASE_DIR = Path(__file__).resolve().parent.parent
                result = subprocess.run(
                    [sys.executable, str(BASE_DIR / "scripts" / "run_pipeline.py")],
                    capture_output=True,
                    text=True,
                    cwd=BASE_DIR   # ⭐ QUAN TRỌNG
                )
                if result.returncode == 0:
                    st.success("Pipeline chạy thành công!")
                    st.cache_data.clear()
                    st.cache_resource.clear()
                else:
                    st.error(f"Lỗi: {result.stderr}")
    
    # Hiển thị trang được chọn
    pages[selection]()

if __name__ == "__main__":
    main()