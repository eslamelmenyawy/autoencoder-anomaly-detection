import pytest

from aead.synthetic import generate_dataset


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("data") / "synthetic"
    return generate_dataset(root, n_train=24, n_test_good=6, n_per_defect=3, size=96, seed=1)

