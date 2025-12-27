import os
import sys
from typing import Dict, List, Optional

import pytest
import sh


def run_sh_command(command: List[str], env: Optional[Dict[str, str]] = None) -> None:
    """Default method for executing shell commands with `pytest` and `sh` package.

    :param command: A list of shell commands as strings.
    :param env: Optional dict of environment variables to add to the current environment.
    """
    # Merge provided env with current environment
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    
    try:
        # Use sys.executable to ensure we use the same Python interpreter as the test runner
        sh.Command(sys.executable)(*command, _out=sys.stdout, _err=sys.stderr, _env=run_env)
    except sh.ErrorReturnCode as e:
        # Always fail on non-zero exit code
        # Note: stderr might be empty if it was redirected to sys.stderr
        msg = e.stderr.decode().strip()
        if msg:
            pytest.fail(reason=msg)
        else:
            pytest.fail(reason=f"Command failed with exit code {e.exit_code}")
