"""Checks prerequisites and installs Brush for this OS.

    uv run python -m threeddd.setup_tools          # check and install Brush if missing
    uv run python -m threeddd.setup_tools --check  # only check
"""

from __future__ import annotations

import argparse
import hashlib
import io
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from .config import TOOLS_DIR
from .local import BRUSH_VERSION, brush_path, brush_target

RELEASE_URL = f"https://github.com/ArthurBrussee/brush/releases/download/{BRUSH_VERSION}/"
SHA256 = {
    "aarch64-apple-darwin": "65b2631398c839be3c1d4d7160fe2326389dec87830aac0710985e6690a1048c",
    "x86_64-pc-windows-msvc": "b68e3e9cf052d51bf3ee30776fa5a364de7f2ba13b58443128ff797bb7bcfcd6",
    "x86_64-unknown-linux-gnu": "4f0f9a8785d1951c62df26aae247c02c5bba32b00f40b06df4e1c9b867399e20",
}

FFMPEG_HINT = {
    "darwin": "brew install ffmpeg",
    "windows": "winget install --id Gyan.FFmpeg -e   (then open a new terminal)",
    "linux": "sudo apt install ffmpeg   (or your distro's package manager)",
}


def _system() -> str:
    return platform.system().lower()


def install_brush(target: str) -> Path:
    archive = f"brush-app-{target}" + (".zip" if target.endswith("windows-msvc") else ".tar.xz")
    print(f"Downloading {archive} ({BRUSH_VERSION})...", flush=True)
    with urllib.request.urlopen(RELEASE_URL + archive, timeout=300) as r:
        data = r.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256[target]:
        raise SystemExit(f"Checksum mismatch for {archive}: got {digest}. Refusing to install.")

    dest = TOOLS_DIR / f"brush-app-{target}"
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    if archive.endswith(".zip"):
        # The Windows zip has no top-level folder.
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                if Path(name).name and ".." not in Path(name).parts:
                    (dest / Path(name).name).write_bytes(z.read(name))
        exe = dest / "brush_app.exe"
    else:
        # Flatten the tarball in case it has a top-level folder.
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as t:
            for m in t.getmembers():
                if m.isfile() and ".." not in Path(m.name).parts:
                    f = t.extractfile(m)
                    if f:
                        (dest / Path(m.name).name).write_bytes(f.read())
        exe = dest / "brush_app"
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        if _system() == "darwin":
            subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(dest)], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Installed Brush at {exe}")
    return exe


def check(install: bool) -> bool:
    ok = True
    system = _system()
    print(f"Platform: {platform.system()} {platform.machine()} · Python {platform.python_version()}")

    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool)
        print(f"  [{'ok' if found else 'missing'}] {tool}: {found or 'not found'}")
        if not found:
            ok = False
            print(f"      install: {FFMPEG_HINT.get(system, 'install ffmpeg and put it on PATH')}")

    try:
        import pycolmap

        print(f"  [ok] pycolmap {pycolmap.__version__}")
    except ImportError as e:
        ok = False
        print(f"  [missing] pycolmap: {e}. Supported on macOS arm64, Windows x64 and Linux x64 (uv sync installs it).")

    brush = brush_path()
    target = brush_target()
    if brush:
        print(f"  [ok] Brush (offline trainer): {brush}")
    elif target is None:
        print(f"  [skip] Brush: no build for {platform.system()} {platform.machine()}; use the cloud GPU trainer")
    elif install:
        brush = install_brush(target)
    else:
        print("  [skip] Brush (offline trainer): not installed, run without --check to install")

    modal_token = Path.home() / ".modal.toml"
    print(f"  [{'ok' if modal_token.exists() else 'skip'}] Modal token: "
          f"{'found' if modal_token.exists() else 'missing (cloud GPU disabled until you run: uv run modal token new)'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only report, don't install anything")
    args = ap.parse_args()
    sys.exit(0 if check(install=not args.check) else 1)


if __name__ == "__main__":
    main()
