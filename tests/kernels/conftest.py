import pytest

from kernutil import BACKENDS


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param
