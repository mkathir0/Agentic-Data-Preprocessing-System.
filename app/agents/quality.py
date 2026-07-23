import pandas as pd
import pandera as pa
from typing import List

from app.agents.schemas import QualityOutput, QualityAnomaly

def run_quality_check(csv_path: str) -> QualityOutput:
    """
    Runs quality checks using Pandera and Pandas heuristics.
    Returns a structured QualityOutput detailing anomalies.
    """
    df = pd.read_csv(csv_path)
    anomalies: List[QualityAnomaly] = []
    
    num_rows = len(df)
    if num_rows == 0:
        return QualityOutput(
            is_valid=False,
            anomalies=[QualityAnomaly(anomaly_type="empty_file", description="The dataset has 0 rows.", severity="high")],
            summary="Dataset is entirely empty."
        )

    # Check 1: Duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        anomalies.append(QualityAnomaly(
            anomaly_type="duplicate_rows",
            description=f"Found {duplicate_count} exact duplicate rows.",
            severity="medium",
            affected_count=int(duplicate_count),
            affected_percentage=float(duplicate_count) / num_rows
        ))
        
    # Check 2: Missing values per column
    missing_counts = df.isnull().sum()
    for col_name, count in missing_counts.items():
        if count > 0:
            percentage = float(count) / num_rows
            severity = "high" if percentage > 0.5 else ("medium" if percentage > 0.05 else "low")
            anomalies.append(QualityAnomaly(
                column=col_name,
                anomaly_type="missing_values",
                description=f"Column '{col_name}' has {count} missing values.",
                severity=severity,
                affected_count=int(count),
                affected_percentage=percentage
            ))

    # Check 3: Mixed data types (e.g., numbers and strings in a column)
    # Pandas infers mixed types as 'object'
    for col_name in df.select_dtypes(include=['object']).columns:
        # A simple heuristic for mixed types: apply type to all non-null values and check uniqueness of types
        types = df[col_name].dropna().apply(type).unique()
        if len(types) > 1:
            type_names = [t.__name__ for t in types]
            anomalies.append(QualityAnomaly(
                column=col_name,
                anomaly_type="mixed_types",
                description=f"Column '{col_name}' contains mixed data types: {type_names}.",
                severity="medium"
            ))

    # Summary
    is_valid = len([a for a in anomalies if a.severity in ["high"]]) == 0
    summary = f"Detected {len(anomalies)} anomalies. "
    if is_valid:
        summary += "The dataset is generally of good quality."
    else:
        summary += "The dataset requires significant cleaning due to high-severity issues."

    return QualityOutput(
        is_valid=is_valid,
        anomalies=anomalies,
        summary=summary
    )
