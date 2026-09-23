"""duplicate-side-effect-desk.

`load_environment` is imported lazily on purpose. The world, the grader, the
dataset and the reference agents all run with nothing but the standard library,
and `scripts/run_report.py` and `scripts/run_attacks.py` are meant to stay that
way — importing `verifiers` eagerly here would make every offline check require
the full RL stack.
"""

__all__ = ["load_environment"]


def __getattr__(name: str):
    if name == "load_environment":
        from .environment import load_environment

        return load_environment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
