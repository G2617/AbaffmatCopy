# Derived from Aureka-AIntibody-Challenges (MIT License; Copyright 2025 Jixian Zhang).
# Modifications Copyright 2026 Eamon Byrne.
# AbAffMat modifications are licensed under Apache-2.0; see LICENSE and THIRD_PARTY_NOTICES.md.

"""Compatibility wrapper for the renamed scoring model module.

Formerly this file contained `DistilledBinder` directly; the primary class is now
`ComplexScoringModel`.
Use `models.scoring_model` for new imports.
"""

from .scoring_model import *  # noqa: F401,F403
