"""
Unit and regression test for the tdmdpy package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import tdmdpy


def test_tdmdpy_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "tdmdpy" in sys.modules
