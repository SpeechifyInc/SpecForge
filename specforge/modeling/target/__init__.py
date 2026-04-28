# Eagle3 target models depend on sglang at import time
# (eagle3_target_model.py does unconditional `import sglang.srt.managers.mm_utils`).
# DFlash training (scripts/train_dflash.py) does not use any Eagle3 class,
# but imports auto.py which transitively imports this package — so an
# unconditional import here forces sglang to be installed even for DFlash-only
# use. Wrap in try/except so the package stays loadable without sglang;
# Eagle3 paths still work when sglang is installed, and a clear error fires
# only if a caller actually tries to use an Eagle3 class without it.
try:
    from .eagle3_target_model import (
        CustomEagle3TargetModel,
        Eagle3TargetModel,
        HFEagle3TargetModel,
        SGLangEagle3TargetModel,
        get_eagle3_target_model,
    )
except ImportError as _eagle3_err:
    def _eagle3_unavailable(*_args, **_kwargs):
        raise ImportError(
            "Eagle3 target models require sglang to be installed. "
            f"Original import error: {_eagle3_err}"
        ) from _eagle3_err

    CustomEagle3TargetModel = _eagle3_unavailable  # type: ignore[assignment]
    Eagle3TargetModel = _eagle3_unavailable        # type: ignore[assignment]
    HFEagle3TargetModel = _eagle3_unavailable      # type: ignore[assignment]
    SGLangEagle3TargetModel = _eagle3_unavailable  # type: ignore[assignment]
    get_eagle3_target_model = _eagle3_unavailable  # type: ignore[assignment]

from .target_head import TargetHead

__all__ = [
    "Eagle3TargetModel",
    "SGLangEagle3TargetModel",
    "HFEagle3TargetModel",
    "CustomEagle3TargetModel",
    "get_eagle3_target_model",
    "TargetHead",
]
