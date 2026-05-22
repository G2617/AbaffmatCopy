# Derived from Aureka-AIntibody-Challenges (MIT License; Copyright 2025 Jixian Zhang).
# Modifications Copyright 2026 Eamon Byrne.
# AbAffMat modifications are licensed under Apache-2.0; see LICENSE and THIRD_PARTY_NOTICES.md.

"""Compatibility wrapper for the renamed scoring heads module.

Formerly this file contained the scoring head implementations directly.
Use `models.scoring_heads` for new imports.
"""

from .scoring_heads import *  # noqa: F401,F403
