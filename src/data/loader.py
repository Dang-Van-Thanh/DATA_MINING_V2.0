"""
Module load dữ liệu
"""
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Load dữ liệu từ file"""
    
    def __init__(self, config_path="configs/params.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def load_raw_data(self):
        """Load dữ liệu gốc"""
        raw_path = Path(self.config['data']['raw_path'])
        logger.info(f"Loading data from {raw_path}")
        
        df = pd.read_csv(raw_path, sep=';')
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
    
    def get_data_dictionary(self):
        """Tạo data dictionary"""
        data_dict = {
            'Column': [
                'age', 'job', 'marital', 'education', 'default', 'balance', 
                'housing', 'loan', 'contact', 'day', 'month', 'duration',
                'campaign', 'pdays', 'previous', 'poutcome', 'y'
            ],
            'Description': [
                'Tuổi khách hàng',
                'Nghề nghiệp (management, blue-collar, technician, admin., services, retired, self-employed, entrepreneur, unemployed, housemaid, student)',
                'Tình trạng hôn nhân (married, single, divorced)',
                'Trình độ học vấn (primary, secondary, tertiary, unknown)',
                'Có nợ quá hạn? (yes, no)',
                'Số dư tài khoản hàng năm (euro)',
                'Có vay mua nhà? (yes, no)',
                'Có vay cá nhân? (yes, no)',
                'Phương thức liên lạc (cellular, telephone, unknown)',
                'Ngày trong tháng liên lạc lần cuối',
                'Tháng trong năm liên lạc lần cuối (jan, feb, ...)',
                'Thời gian liên lạc lần cuối (giây)',
                'Số lần liên lạc trong chiến dịch này',
                'Số ngày từ lần liên lạc trước (-1 nếu chưa từng)',
                'Số lần liên lạc trước đây',
                'Kết quả chiến dịch trước (unknown, failure, other, success)',
                'Khách hàng có đăng ký term deposit? (yes: target)'
            ],
            'Data Type': [
                'int', 'categorical', 'categorical', 'categorical', 'binary',
                'int', 'binary', 'binary', 'categorical', 'int', 'categorical',
                'int', 'int', 'int', 'int', 'categorical', 'binary'
            ]
        }
        
        df_dict = pd.DataFrame(data_dict)
        return df_dict