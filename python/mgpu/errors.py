"""Simulator (not architectural) errors shared by every backend."""


class SimTimeout(Exception):
    """The kernel exceeded max_steps ([FLT-3]); not an architectural fault."""
