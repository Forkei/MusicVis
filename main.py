"""MusicVis - Electric Energy Ball Music Visualizer."""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from src.app import App


def main():
    app = App(width=1280, height=720)
    app.run()


if __name__ == "__main__":
    main()
