import logging
import os
import re
import sys
from pathlib import Path

from jupyterlab.labapp import main


def find_jupyterlab_path(runfile_exec: Path) -> str:
    """
    Guesses the Jupyter Lab package path from the runfile executable.
    This is a necessary hack because Bazel has the wrong AppDir by default.
    """
    base_path = runfile_exec
    while base_path.name != "jupyterlab.runfiles":
        base_path = base_path.parent

    if not base_path:
        raise RuntimeError(
            "Could not locate the base path for pip dependencies in runfiles."
        )

    for root, dirs, _ in os.walk(base_path):
        if "jupyterlab" in dirs:
            return os.path.join(root, "jupyterlab")
    raise RuntimeError("Could not locate the jupyterlab package in runfiles.")


def get_notebook_dir(bwd: Path) -> str:
    """
    Infers the notebook directory from any path inside the repo
    by first finding the workspace root.
    """
    while not os.path.exists(os.path.join(bwd, "WORKSPACE")):
        bwd = bwd.parent

    return str(bwd / "notebooks")


if __name__ == "__main__":
    sys.argv[0] = re.sub(r"(-script\.pyw?|\.exe)?$", "", sys.argv[0])

    jupyterlab_path = find_jupyterlab_path(Path(sys.argv[0]))
    sys.argv.append(f"--LabApp.app_dir={jupyterlab_path}")

    notebook_dir = get_notebook_dir(Path(os.environ["BUILD_WORKING_DIRECTORY"]))
    sys.argv.append(f"--notebook-dir={notebook_dir}")

    sys.exit(main())
