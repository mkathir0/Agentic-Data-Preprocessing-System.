"""Every job owns a separate set of output artifacts."""
from pathlib import Path


def artifact_paths(output_dir: str, job_id: str, filename: str) -> dict[str, Path]:
    directory = Path(output_dir) / job_id
    stem = Path(filename).stem
    return {
        'directory': directory,
        'cleaned': directory / f'cleaned_{stem}.csv',
        'report': directory / f'report_{stem}.md',
        'script': directory / f'clean_{stem}.py',
        'suite': directory / f'quality_suite_{stem}.json',
        'validation': directory / f'validation_{stem}.json',
    }
