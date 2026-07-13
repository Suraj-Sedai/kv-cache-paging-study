from src.metrics.hardware import (
    get_cuda_version,
    get_driver_version,
    get_git_commit_hash,
    get_gpu_name,
    get_hardware_info,
    raw_csv_hardware_fields,
)


def test_git_commit_hash_is_string():
    commit_hash = get_git_commit_hash()

    assert isinstance(commit_hash, str)
    assert len(commit_hash) > 0


def test_cuda_version_is_string():
    cuda_version = get_cuda_version()

    assert isinstance(cuda_version, str)
    assert len(cuda_version) > 0


def test_gpu_name_is_string():
    gpu_name = get_gpu_name()

    assert isinstance(gpu_name, str)
    assert len(gpu_name) > 0


def test_driver_version_is_string():
    driver_version = get_driver_version()

    assert isinstance(driver_version, str)
    assert len(driver_version) > 0


def test_get_hardware_info_contains_expected_fields():
    info = get_hardware_info()

    expected_fields = {
        "gpu_name",
        "cuda_version",
        "pytorch_version",
        "driver_version",
        "commit_hash",
        "python_version",
        "platform",
    }

    assert expected_fields.issubset(info.keys())

    for field in expected_fields:
        assert isinstance(info[field], str)
        assert len(info[field]) > 0


def test_raw_csv_hardware_fields_contains_only_logger_fields():
    fields = raw_csv_hardware_fields()

    assert set(fields.keys()) == {
        "gpu_name",
        "cuda_version",
        "pytorch_version",
        "driver_version",
        "commit_hash",
    }

    for value in fields.values():
        assert isinstance(value, str)
        assert len(value) > 0
