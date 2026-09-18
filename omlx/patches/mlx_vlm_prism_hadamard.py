"""Adapter module for Prism ML Bonsai 2 Hadamard VLM packs.

Extracts model and processor loading from the pack's bundled runtime
(runtime/vision_artifact.py) to integrate with omlx.
"""
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Tuple

# Suppress known non-fatal warnings from transformers and tokenizers
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def is_supported_config(config: dict) -> bool:
    """Return True if the model configuration matches a Prism Hadamard pack."""
    model_type = config.get("model_type", "")
    if model_type == "prism_hadamard_qwen35":
        return True

    quant_config = config.get("quantization_config", {})
    quant_method = quant_config.get("quant_method", "")
    return "prism" in str(quant_method).lower() or "hadamard" in str(quant_method).lower()


def load(model_name: str) -> Tuple[Any, Any]:
    """Load the model and processor using the pack's bundled runtime."""
    pack = Path(model_name).resolve()
    runtime = pack / "runtime"

    if not runtime.is_dir():
        raise FileNotFoundError(
            f"Prism Hadamard pack at {pack} is missing the 'runtime/' directory."
        )

    loader_file = runtime / "vision_artifact.py"
    if not loader_file.is_file():
        raise FileNotFoundError(
            f"Missing {loader_file}. This Bonsai 2 pack does not contain vision support."
        )

    runtime_str = str(runtime)
    if runtime_str not in sys.path:
        sys.path.insert(0, runtime_str)

    warnings.filterwarnings("ignore")

    import mlx.core as mx

    mx.set_default_device(mx.gpu)

    try:
        from vision_artifact import load_vl_model
    except ImportError as e:
        raise ImportError(
            f"Failed to import 'load_vl_model' from {loader_file}: {e}"
        ) from e

    # load_vl_model returns (model, processor, config); omlx requires (model, processor)
    model, processor, _ = load_vl_model(str(pack))
    return model, processor