# from .auto import AutoDistributedTargetModel, AutoDraftModelConfig, AutoEagle3DraftModel
from .auto import AutoDraftModelConfig, AutoEagle3DraftModel
from .draft.llama3_eagle import LlamaForCausalLMEagle3
# Re-export Eagle3 target classes through this package's namespace, but
# tolerate sglang being absent — DFlash training (scripts/train_dflash.py)
# imports specforge.modeling transitively and never touches Eagle3 classes.
# See specforge/modeling/target/__init__.py for the same pattern + rationale.
try:
    from .target.eagle3_target_model import (
        CustomEagle3TargetModel,
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

    CustomEagle3TargetModel = _eagle3_unavailable    # type: ignore[assignment]
    HFEagle3TargetModel = _eagle3_unavailable        # type: ignore[assignment]
    SGLangEagle3TargetModel = _eagle3_unavailable    # type: ignore[assignment]
    get_eagle3_target_model = _eagle3_unavailable    # type: ignore[assignment]

__all__ = [
    "LlamaForCausalLMEagle3",
    "SGLangEagle3TargetModel",
    "HFEagle3TargetModel",
    "CustomEagle3TargetModel",
    "get_eagle3_target_model",
    "AutoDraftModelConfig",
    "AutoEagle3DraftModel",
]
