"""
Inicialização de ambiente para testes	(conftest.py)
"""
import pytest
from object_detection import config


@pytest.fixture()
def sample_tracking_config():
    """cria um dicionário com as configurações de tracking"""
    tracking_config = config.create_and_validate_config(
        parsed_config_path="./tests/test_inputs/config.yml"
    )
    return tracking_config.tracking_config
