# Python

# Local
from compress import CompressTest

# External
import pytest


def test_compress_create():
    abc = CompressTest()
    assert isinstance(abc, CompressTest)
