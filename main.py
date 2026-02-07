"""MusicVis - Electric Energy Ball Music Visualizer."""

import sys
import os

# Conda's ALSA can't find PipeWire plugins, so PortAudio only sees raw hardware
# devices and grabs them exclusively. Point ALSA at the system plugin dir so the
# 'default' and 'pipewire' virtual devices appear and audio shares nicely.
if sys.platform == "linux":
    _sys_plugin_dir = "/usr/lib/x86_64-linux-gnu/alsa-lib"
    if os.path.isdir(_sys_plugin_dir):
        os.environ.setdefault("ALSA_PLUGIN_DIR", _sys_plugin_dir)

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from src.app import App


def main():
    app = App(width=1280, height=720)
    app.run()


if __name__ == "__main__":
    main()
