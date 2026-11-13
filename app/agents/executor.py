"""Execute generated transformations inside an OS-enforced sandbox.

macOS uses Seatbelt via ``sandbox-exec``. Linux uses a locked-down Docker
container. Unsupported hosts or missing sandbox runtimes fail closed.
"""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from typing import Dict, Any

from langsmith import traceable


def _sbpl_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _mac_sandbox_profile(workspace: Path) -> str:
    allowed_read = {
        "/System",
        "/usr/lib",
        "/usr/share",
        "/Library/Apple/System/Library",
        str(Path(sys.prefix).resolve()),
        str(Path(sys.base_prefix).resolve()),
        str(workspace.resolve()),
    }
    clauses = ["(version 1)", "(deny default)", "(allow sysctl-read)"]
    clauses.append("(allow file-read-metadata (subpath \"/\"))")
    clauses.append("(allow file-read* (literal \"/\"))")
    clauses.extend(f"(allow file-read* (subpath {_sbpl_quote(path)}))" for path in sorted(allowed_read))
    clauses.append(f"(allow file-read* (literal {_sbpl_quote(str(Path(sys.executable).resolve()))}))")
    clauses.append(f"(allow process-exec (literal {_sbpl_quote(sys.executable)}))")
    clauses.append(f"(allow process-exec (literal {_sbpl_quote(str(Path(sys.executable).resolve()))}))")
    clauses.append(f"(allow file-write* (subpath {_sbpl_quote(str(workspace.resolve()))}))")
    return "\n".join(clauses) + "\n"


def _sandbox_command(workspace: Path, script: Path, source: Path, candidate: Path) -> tuple[list[str], str | None]:
    if sys.platform == "darwin":
        sandbox_exec = shutil.which("sandbox-exec")
        if not sandbox_exec:
            raise RuntimeError("OS sandbox unavailable: macOS sandbox-exec was not found.")
        profile = workspace / "executor.sb"
        profile.write_text(_mac_sandbox_profile(workspace), encoding="utf-8")
        return [sandbox_exec, "-f", str(profile), sys.executable, str(script), str(source), str(candidate)], None

    if sys.platform.startswith("linux"):
        docker = shutil.which("docker")
        if not docker:
            raise RuntimeError("OS sandbox unavailable: install Docker to execute generated code on Linux.")
        container_name = f"data-cleaner-{uuid.uuid4().hex}"
        image = os.environ.get("DATA_EXECUTOR_IMAGE", "data-engineer-executor:latest")
        command = [
            docker, "run", "--rm", "--name", container_name,
            "--network=none", "--read-only", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--pids-limit=64",
            "--memory=2g", "--cpus=1", "--user=65534:65534",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=128m",
            "--env", "HOME=/tmp", "--env", "TMPDIR=/tmp", "--env", "MPLCONFIGDIR=/tmp",
            "--mount", f"type=bind,src={workspace.resolve()},dst=/work,rw",
            "--workdir", "/work", image,
            "python", "/work/run.py", f"/work/{source.name}", f"/work/{candidate.name}",
        ]
        return command, container_name

    raise RuntimeError(f"OS sandbox unavailable on platform {sys.platform!r}; generated code execution is disabled.")


def _stop_process(process: subprocess.Popen, container_name: str | None) -> None:
    if container_name:
        docker = shutil.which("docker")
        if docker:
            subprocess.run([docker, "rm", "-f", container_name], capture_output=True, timeout=10)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()


@traceable
def execute_cleaning_code(code_str: str, input_csv_path: str, output_csv_path: str, timeout: int = 60) -> Dict[str, Any]:
    output = Path(output_csv_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output == Path(input_csv_path).resolve():
        return {"success": False, "logs": "", "error": "Input and output paths must differ."}

    output.unlink(missing_ok=True)
    result = {"success": False, "logs": "", "error": ""}
    with tempfile.TemporaryDirectory(prefix="cleaning-") as workspace_name:
        work = Path(workspace_name)
        work.chmod(0o777)  # Docker's unprivileged UID must access the bind mount.
        (work / "tmp").mkdir()
        (work / "mpl").mkdir()
        source = work / ("input" + Path(input_csv_path).suffix)
        shutil.copyfile(input_csv_path, source)
        source.chmod(0o444)
        candidate = work / "candidate.csv"
        script = work / "run.py"

        # Stage only the CSV loader needed by the wrapper. The sandbox cannot
        # read the application repository, its .env, or other job files.
        staged_app = work / "app"
        (staged_app / "utils").mkdir(parents=True)
        (staged_app / "__init__.py").write_text("", encoding="utf-8")
        (staged_app / "utils" / "__init__.py").write_text("", encoding="utf-8")
        loader = Path(__file__).resolve().parents[1] / "utils" / "data_loader.py"
        shutil.copyfile(loader, staged_app / "utils" / "data_loader.py")

        script.write_text(
            "import sys\nimport pandas as pd\n"
            "sys.path.insert(0, '/work' if sys.platform.startswith('linux') else "
            f"{str(work)!r})\n"
            "from app.utils.data_loader import load_dataframe\n"
            + code_str + "\n"
            'if __name__ == "__main__":\n'
            "    df = load_dataframe(sys.argv[1])\n"
            "    cleaned = clean_data(df)\n"
            "    if not isinstance(cleaned, pd.DataFrame):\n"
            '        raise TypeError("clean_data must return a DataFrame")\n'
            "    if not cleaned.columns.is_unique:\n"
            '        raise ValueError("Duplicate output columns are not allowed")\n'
            "    cleaned.to_csv(sys.argv[2], index=False)\n",
            encoding="utf-8",
        )

        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "TMPDIR": str(work / "tmp"),
            "TEMP": str(work / "tmp"),
            "TMP": str(work / "tmp"),
            "MPLCONFIGDIR": str(work / "mpl"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        container_name = None
        process = None
        try:
            command, container_name = _sandbox_command(work, script, source, candidate)
            process = subprocess.Popen(
                command,
                cwd=work,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _stop_process(process, container_name)
                stdout, stderr = process.communicate()
                result["logs"] = stdout
                result["error"] = f"Execution timed out after {timeout} seconds. {stderr[-2000:]}"
                return result

            result["logs"] = stdout
            if process.returncode != 0:
                result["error"] = stderr[-4000:] or f"Sandbox exited with status {process.returncode}."
            elif not candidate.is_file() or candidate.stat().st_size == 0:
                result["error"] = "Execution produced no output."
            else:
                with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stage:
                    staging = Path(stage.name)
                try:
                    shutil.copyfile(candidate, staging)
                    os.replace(staging, output)
                    result["success"] = True
                finally:
                    staging.unlink(missing_ok=True)
        except Exception as exc:
            result["error"] = f"Sandboxed execution failed: {exc}"
        finally:
            if container_name and process and process.poll() is not None:
                docker = shutil.which("docker")
                if docker:
                    subprocess.run([docker, "rm", "-f", container_name], capture_output=True, timeout=10)
    return result
