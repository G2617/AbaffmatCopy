# Derived from Aureka-AIntibody-Challenges (MIT License; Copyright 2025 Jixian Zhang).
# Modifications Copyright 2026 Eamon Byrne.
# AbAffMat modifications are licensed under Apache-2.0; see LICENSE and THIRD_PARTY_NOTICES.md.

"""Compatibility wrapper for the renamed complex scoring runner module.

Use `runners.score_complex` for new imports and CLI entrypoints.
"""

from runners.score_complex import *  # noqa: F401,F403


if __name__ == "__main__":
    run()
