"""Compare every persona response with every matched baseline response."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.multi_turn_response_analysis import main

if __name__ == '__main__':
    sys.exit(main('pairing'))
