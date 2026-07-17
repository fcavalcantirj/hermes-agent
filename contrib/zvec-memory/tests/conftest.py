# Make the sibling `zvec_memory_core.py` importable without installing the package:
# the core lives one directory up from tests/. Insert-at-front so a same-named
# module elsewhere on sys.path cannot shadow the unit under test.
import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
