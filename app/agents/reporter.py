import os

def generate_report(job_id: str, plan_json: str, code: str, quality_summary: str, output_path: str):
    """
    Generates a final markdown engineering report for the job.
    """
    report_content = f"""# AI Data Engineering Report: {job_id}

## 1. Executive Summary
The AI autonomous loop has successfully processed the dataset.
**Validation Result:** {quality_summary}

## 2. Transformation Plan
The following logical plan was executed:
```json
{plan_json}
```

## 3. Generated Code
The following pandas code was written and safely executed:
```python
{code}
```
"""
    with open(output_path, "w") as f:
        f.write(report_content)
