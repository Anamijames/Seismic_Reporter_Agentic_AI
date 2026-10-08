from pathlib import Path


def test_streamlit_config_disables_file_watcher():
    config_path = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
    assert config_path.exists()

    config = config_path.read_text(encoding="utf-8")
    assert "[server]" in config
    assert 'fileWatcherType = "none"' in config
