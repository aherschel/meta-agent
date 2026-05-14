"""RunContext — runtime context passed to candidate config and harness modules."""

from dataclasses import dataclass


@dataclass
class RunContext:
    """Runtime values available when building candidate config or harness specs.

    Candidate modules receive this and use it to set fields like cwd and model
    that are not known until execution time.
    """

    cwd: str
    model: str
    task_instruction: str
