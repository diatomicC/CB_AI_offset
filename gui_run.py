#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launch the registration analyzer GUI (same app as cmyk_analyzer_gui / run_gui).
Ensures project root is on sys.path when executed as `python gui_run.py`.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cmyk_analyzer_gui import main

if __name__ == "__main__":
    main()
