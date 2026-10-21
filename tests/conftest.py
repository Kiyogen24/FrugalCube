import os
import sys

# Les modules de src/ s'importent entre eux à plat (import fast_cube as fc)
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
