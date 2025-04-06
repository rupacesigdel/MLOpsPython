from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ClassificationQualityMetric
import pandas as pd
import logging

def generate_report(current_data: pd.DataFrame, 
                   reference_data: pd.DataFrame,
                   target_col: str = "target"):
    """Generate data drift and quality report"""
    column_mapping = ColumnMapping(
        target=target_col,
        numerical_features=current_data.select_dtypes(include='number').columns.tolist()
    )
    
    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationQualityMetric()
    ])
    
    report.run(
        current_data=current_data,
        reference_data=reference_data,
        column_mapping=column_mapping
    )
    
    return report