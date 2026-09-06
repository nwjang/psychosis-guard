"""Smoke test: the package installs and imports, and carries its disclaimer."""

import psychosis_guard


def test_package_imports():
    assert psychosis_guard.__version__


def test_disclaimer_in_docstring():
    # The domain disclaimer is load-bearing  — keep it in the
    # package docstring at all times.
    assert "NOT a medical device" in psychosis_guard.__doc__
