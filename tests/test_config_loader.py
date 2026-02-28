from pathlib import Path

import pytest

from config_loader import load_config


def test_load_config_success_with_relative_paths(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        """
directories:
  - ./data
host: 0.0.0.0
port: 9000
index_file: ./cache/index.pkl
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(config_file)

    assert cfg.host == "0.0.0.0"
    assert cfg.port == 9000
    assert cfg.directories == [(tmp_path / "data").resolve()]
    assert cfg.index_file == (tmp_path / "cache/index.pkl").resolve()


def test_load_config_success_with_absolute_paths(tmp_path: Path) -> None:
    data_dir = (tmp_path / "abs_data").resolve()
    index_file = (tmp_path / "abs_index.pkl").resolve()
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        f"""
directories:
  - {data_dir}
index_file: {index_file}
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(config_file)

    assert cfg.directories == [data_dir]
    assert cfg.index_file == index_file


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yml")


def test_load_config_invalid_directories_type_raises(tmp_path: Path) -> None:
    file = tmp_path / "config.yml"
    file.write_text("directories: bad", encoding="utf-8")

    with pytest.raises(ValueError, match="directories"):
        load_config(file)


def test_load_config_invalid_directory_value_raises(tmp_path: Path) -> None:
    file = tmp_path / "config.yml"
    file.write_text("directories:\n  - ''", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty string"):
        load_config(file)


def test_load_config_invalid_host_raises(tmp_path: Path) -> None:
    file = tmp_path / "config.yml"
    file.write_text("directories:\n  - ./data\nhost: ''", encoding="utf-8")

    with pytest.raises(ValueError, match="host"):
        load_config(file)


def test_load_config_invalid_port_raises(tmp_path: Path) -> None:
    file = tmp_path / "config.yml"
    file.write_text("directories:\n  - ./data\nport: 99999", encoding="utf-8")

    with pytest.raises(ValueError, match="port"):
        load_config(file)


def test_load_config_invalid_index_file_raises(tmp_path: Path) -> None:
    file = tmp_path / "config.yml"
    file.write_text("directories:\n  - ./data\nindex_file: ''", encoding="utf-8")

    with pytest.raises(ValueError, match="index_file"):
        load_config(file)
