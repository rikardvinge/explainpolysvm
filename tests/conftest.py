import pytest
import matplotlib

@pytest.fixture(autouse=True)
def set_matplotlib_backend():
    matplotlib.use("Agg")