"""Face-restoration adapters.

CodeFormer is fully integrated (detect -> align -> restore -> blend back);
GFPGAN is declared but not integrated. See :mod:`restoration.codeformer.model`.
"""

from restoration.codeformer.model import register_codeformer

__all__ = ["register_codeformer"]
