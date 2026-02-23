#!/usr/bin/env python
"""
Script chạy tất cả notebooks bằng papermill
"""
import papermill as pm
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_notebooks():
    """Chạy tất cả notebooks theo thứ tự"""
    
    notebooks = [
        '01_eda.ipynb',
        '02_preprocess_feature.ipynb',
        '03_mining_clustering.ipynb',
        '04_modeling.ipynb',
        '04b_semi_supervised.ipynb',
        '05_evaluation_report.ipynb'
    ]
    
    output_dir = Path('outputs/notebooks')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    parameters = {
        'seed': 42,
        'test_size': 0.2,
        'random_state': 42
    }
    
    for notebook in notebooks:
        input_path = f'notebooks/{notebook}'
        output_path = output_dir / f"executed_{notebook}"
        
        logger.info(f"Running {notebook}...")
        
        try:
            pm.execute_notebook(
                input_path,
                str(output_path),
                parameters=parameters,
                kernel_name='python3'
            )
            logger.info(f"✅ {notebook} completed successfully")
        except Exception as e:
            logger.error(f"❌ {notebook} failed: {e}")
            raise

if __name__ == "__main__":
    run_notebooks()