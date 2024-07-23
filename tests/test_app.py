import pytest
import toml
import os


@pytest.fixture
def read_config():
    """Fixture to read perceptron configuration from a TOML file in .streamlit."""
    config = toml.load('../.streamlit/config.toml')
    return config


def test_cfg_logger(read_config):
    assert read_config['logger']['level'] == 'INFO'
    fmt = '{time:YYYY MM DD HH:MM:SS} | {level} | {file} | {function} | Line {line} | {message}'
    assert read_config['logger']['format'] == fmt


def test_cfg_data(read_config):
    assert read_config['data']['data'] == [[0, 0],[0, 1],[1, 0],[1, 1]]
    assert read_config['data']['steps'] == 2


def test_cfg_and(read_config):
    assert read_config['AND']['plot_title1'] == "Error Convergence for logical AND"
    assert read_config['AND']['plot_title2'] == "Decision Boundary for logical AND"
    assert read_config['AND']['output'] == [0, 0, 0, 1]


def test_cfg_or(read_config):
    assert read_config['OR']['plot_title1'] == "Error Convergence for logical OR"
    assert read_config['OR']['plot_title2'] == "Decision Boundary for logical OR"
    assert read_config['OR']['output'] == [0, 1, 1, 1]


def test_cfg_xor(read_config):
    assert read_config['XOR']['plot_title1'] == "Error Convergence for logical XOR"
    assert read_config['XOR']['plot_title2'] == "Decision Boundary for logical XOR"
    assert read_config['XOR']['output'] == [0, 1, 1, 0]
