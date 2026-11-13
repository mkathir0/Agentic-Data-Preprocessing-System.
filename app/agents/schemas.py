from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field

# ── Phase 1: Profiling ──────────────────────────────────────────────────────

class ProfilerOutput(BaseModel):
    num_rows: int
    num_columns: int
    memory_usage_bytes: int
    missing_cells: int
    missing_cells_percentage: float
    duplicate_rows: int
    duplicate_rows_percentage: float
    column_types: Dict[str, str]       # e.g. {"age": "Numeric", "name": "Text"}
    columns_summary: Dict[str, Any]    # Detailed summary per column
    candidate_transformations: List[Dict[str, Any]] = Field(default_factory=list)

# ── Quality anomaly (light) ─────────────────────────────────────────────────

class QualityAnomaly(BaseModel):
    column: Optional[str] = None
    anomaly_type: str
    description: str
    severity: str                      # "high", "medium", "low"
    affected_count: Optional[int] = None
    affected_percentage: Optional[float] = None

class QualityOutput(BaseModel):
    is_valid: bool
    anomalies: List[QualityAnomaly]
    summary: str

# ── Schema inference ────────────────────────────────────────────────────────

class SchemaOutput(BaseModel):
    primary_keys: List[str]
    candidate_foreign_keys: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    inferred_types: Dict[str, str]

# ── Transformation planning ─────────────────────────────────────────────────

class TransformationAction(BaseModel):
    column: Optional[str] = Field(None, description="The specific column this action applies to, or null if it applies to the whole dataset")
    action_type: str = Field(..., description="The type of transformation (e.g. 'drop_duplicates', 'fill_missing', 'cast_type', 'drop_column')")
    explanation: str = Field(..., description="Explanation of why this action is being taken based on the data profile")
    confidence: Literal["high", "medium", "low"] = Field("medium", description="Confidence that this transformation is correct: high=well-supported by statistics, medium=reasonable heuristic, low=uncertain/may alter data meaning")
    assumption: Optional[str] = Field(None, description="Any assumption the LLM made to justify this action, in plain English (e.g. 'assumed missing station data = dockless rides')")
    estimated_rows_affected_pct: Optional[float] = Field(None, description="Estimated percentage of rows affected by this action, 0.0-1.0")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Explicit parser or normalization parameters derived from deterministic evidence")

class TransformationPlan(BaseModel):
    dataset_summary: str = Field(..., description="A short summary of the dataset state based on the profiling data")
    actions: List[TransformationAction] = Field(..., description="The sequence of deterministic actions to clean the dataset")
    global_assumptions: List[str] = Field(default_factory=list, description="High-level assumptions about this dataset that apply across multiple steps")

# ── Code generation ─────────────────────────────────────────────────────────

class CodeGenerationOutput(BaseModel):
    code: str = Field(..., description="The executable Python pandas code that performs the actions. Do not include markdown code block syntax like ```python, just the raw code.")
    dependencies: List[str] = Field(default_factory=lambda: ["pandas", "numpy"], description="List of pip packages required to run the code, e.g. ['pandas']")

# ── Rich validation report ──────────────────────────────────────────────────

class ColumnStats(BaseModel):
    """Per-column statistics snapshot."""
    column: str
    dtype: str
    null_pct: float
    unique_count: Optional[int] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    skew: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    outlier_count_iqr: Optional[int] = None  # values outside 1.5×IQR
    outlier_count_zscore: Optional[int] = None  # |z| > 3

class StatsDiff(BaseModel):
    """Before/after statistics for one column."""
    column: str
    before: ColumnStats
    after: ColumnStats
    imputation_pct: float = 0.0  # % of values that were imputed (synthetic)

class MissingnessPattern(BaseModel):
    """Pattern analysis for missing data."""
    columns: List[str]
    pattern_type: Literal["MCAR", "MAR", "MNAR", "structural"]
    co_null_pct: float  # % of rows where ALL columns in the group are null together
    description: str    # plain-English explanation

class BusinessRuleViolation(BaseModel):
    """A violated cross-field or domain business rule."""
    rule: str           # e.g. "ended_at > started_at"
    violation_count: int
    violation_pct: float
    disposition: Literal["dropped", "flagged", "corrected", "not_handled"]
    example: Optional[str] = None  # One example row (truncated string)

class ValidationCheck(BaseModel):
    """Result of a single named validation check."""
    category: Literal[
        "schema", "range_bounds", "cross_field",
        "distributional", "duplicates", "format_encoding", "regression"
    ]
    name: str
    passed: bool
    violations: int = 0
    details: str

class EscalationFlag(BaseModel):
    """A column or issue flagged for human review."""
    column: Optional[str]
    reason: str
    severity: Literal["high", "medium", "low"]

class ValidationReport(BaseModel):
    integrity_passed: bool = False
    review_required: bool = False
    review_reasons: List[str] = Field(default_factory=list)

    """Full 7-layer validation report for a single pipeline run."""
    # 1. Schema
    schema_checks: List[ValidationCheck] = Field(default_factory=list)
    columns_before: List[str] = Field(default_factory=list)
    columns_after: List[str] = Field(default_factory=list)
    columns_dropped: List[str] = Field(default_factory=list)
    columns_added: List[str] = Field(default_factory=list)

    # 2. Range & bounds
    range_checks: List[ValidationCheck] = Field(default_factory=list)

    # 3. Cross-field / relational
    cross_field_checks: List[ValidationCheck] = Field(default_factory=list)
    business_rule_violations: List[BusinessRuleViolation] = Field(default_factory=list)
    missingness_patterns: List[MissingnessPattern] = Field(default_factory=list)

    # 4. Distributional
    stats_diff: List[StatsDiff] = Field(default_factory=list)
    distributional_checks: List[ValidationCheck] = Field(default_factory=list)

    # 5. Duplicates
    duplicate_checks: List[ValidationCheck] = Field(default_factory=list)
    key_based_duplicate_count: int = 0

    # 6. Format / encoding
    format_checks: List[ValidationCheck] = Field(default_factory=list)

    # 7. Regression (re-profiler on cleaned output)
    regression_checks: List[ValidationCheck] = Field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0
    cols_before: int = 0
    cols_after: int = 0
    new_nulls_introduced: Dict[str, int] = Field(default_factory=dict)  # col -> count

    # Summary
    escalation_flags: List[EscalationFlag] = Field(default_factory=list)
    total_checks: int = 0
    total_passed: int = 0
    total_failed: int = 0
    overall_passed: bool = True

    # Verifiable Data Quality Metrics (0.0 to 100.0)
    data_quality_score_before: float = 0.0
    data_quality_score_after: float = 0.0
    score_improvement: float = 0.0
    completeness_score: float = 0.0
    validity_score: float = 0.0
    uniqueness_score: float = 0.0
    consistency_score: float = 0.0

    # Pandera Contract Validation
    pandera_schema_passed: bool = True
    pandera_failure_count: int = 0
    pandera_failure_cases: List[Dict[str, Any]] = Field(default_factory=list)

    # Fuzzy & Key Integrity Metrics
    fuzzy_duplicate_clusters: int = 0
    fuzzy_duplicate_examples: List[str] = Field(default_factory=list)
    key_conflict_count: int = 0
    key_conflict_details: List[str] = Field(default_factory=list)
    circuit_breaker_retries: int = 0
    exact_duplicates_removed: int = 0

    # Interoperability Suites
    great_expectations_suite: Optional[Dict[str, Any]] = None
