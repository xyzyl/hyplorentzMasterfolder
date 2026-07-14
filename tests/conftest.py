import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running Phase 2 training harness")


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m"):
        return  # explicit -m selection: run what was asked
    skip = pytest.mark.skip(reason="slow harness; run with -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
