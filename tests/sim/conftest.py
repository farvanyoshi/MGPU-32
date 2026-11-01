import pytest

from simutil import BACKEND_PARAMS, Run


@pytest.fixture(params=BACKEND_PARAMS)
def backend(request):
    return request.param


@pytest.fixture
def run(backend):
    return lambda src, **kw: Run(backend, src, **kw)
