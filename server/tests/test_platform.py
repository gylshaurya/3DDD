import pytest

from threeddd import local


@pytest.mark.parametrize("system,machine,target", [
    ("Darwin", "arm64", "aarch64-apple-darwin"),
    ("Windows", "AMD64", "x86_64-pc-windows-msvc"),
    ("Linux", "x86_64", "x86_64-unknown-linux-gnu"),
    ("Linux", "aarch64", None),  # Brush ships no ARM Linux build: cloud trainer only
    ("Darwin", "x86_64", None),  # Intel Macs: likewise
])
def test_brush_target(system, machine, target):
    assert local.brush_target(system, machine) == target


def test_brush_path_finds_platform_build(monkeypatch, tmp_path):
    monkeypatch.delenv("THREEDDD_BRUSH", raising=False)
    monkeypatch.setattr(local, "TOOLS_DIR", tmp_path)
    monkeypatch.setattr(local.shutil, "which", lambda _: None)
    monkeypatch.setattr(local, "brush_target", lambda: "x86_64-pc-windows-msvc")
    assert local.brush_path() is None and not local.available()
    exe = tmp_path / "brush-app-x86_64-pc-windows-msvc" / "brush_app.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    assert local.brush_path() == exe


def test_brush_path_env_override(monkeypatch, tmp_path):
    exe = tmp_path / "custom_brush"
    exe.write_bytes(b"")
    monkeypatch.setenv("THREEDDD_BRUSH", str(exe))
    assert local.brush_path() == exe
