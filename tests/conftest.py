import jax
import pytest

from molecular_compiler.fixtures import synthetic_system

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="session")
def system():
    return synthetic_system()
