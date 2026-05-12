"""Root test conftest — provides --backend option to override FHE backend.

Usage:
    # Run oracle tests with DeSiLo backend:
    .venv/bin/python -m pytest tests/oracle/ -v -s --backend=desilo

    # Run with Lattigo (default, no override needed):
    .venv/bin/python -m pytest tests/oracle/ -v
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        default=None,
        help="Override FHE backend (e.g. 'desilo', 'lattigo')",
    )


@pytest.fixture(scope="session", autouse=True)
def _patch_backend(request):
    """If --backend is given, monkeypatch Scheme.setup_backend to force it."""
    backend = request.config.getoption("--backend")
    if not backend:
        return

    import orion.core.orion as orion_mod

    original_setup = orion_mod.Scheme.setup_backend

    def patched_setup(self, params):
        params.orion_params.backend = backend
        return original_setup(self, params)

    orion_mod.Scheme.setup_backend = patched_setup
