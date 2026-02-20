"""Streamlit Cloud entry point.

Adds the repository root to ``sys.path`` so that all internal imports
(``core``, ``apps``, ``knowledge``, etc.) resolve correctly without
requiring an editable install.
"""

import sys
from pathlib import Path

# Ensure the repo root is on the path when running from Streamlit Cloud
_root = Path(__file__).parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from apps.ui.app import main  # noqa: E402

main()
