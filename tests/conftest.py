"""Pytest configuration and shared fixtures."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom command-line options.

    Args:
        parser: The pytest argument parser.
    """
    parser.addoption(
        "--agents",
        default=None,
        metavar="AGENT[,AGENT,...]",
        help=(
            "Limit regression tests to a comma-separated list of agent(s)"
            " (e.g. --agents random,gp_ucb). Default: all registered agents."
        ),
    )
