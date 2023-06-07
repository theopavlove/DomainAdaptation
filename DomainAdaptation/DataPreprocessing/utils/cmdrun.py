import subprocess  # nosec
from typing import Optional


def cmdrun(
    cmd: str,
    input: Optional[bytes] = None,
    stdout=None,
    shell: bool = False,
    capture_output: bool = True,
):
    return subprocess.run(
        cmd if shell else cmd.split(),
        input=input,
        stdout=stdout,
        check=True,
        shell=shell,  # nosec
        capture_output=capture_output and (stdout is None),
    )
