"""
Model Manager - Download and cache MediaPipe model files.

Downloads model bundles from official Google storage and caches them locally.
"""

import os
import logging
from pathlib import Path
from typing import Optional
import hashlib

logger = logging.getLogger(__name__)

# Model URLs from official MediaPipe storage
MODEL_URLS = {
    "face_landmarker": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}

# Expected file sizes (approximate, for validation)
MODEL_SIZES = {
    "face_landmarker": 4_000_000,  # ~4MB
    "hand_landmarker": 10_000_000,  # ~10MB
}


def get_models_dir() -> Path:
    """Get the models directory path."""
    # Try relative to this file first (for development)
    module_dir = Path(__file__).parent.parent.parent
    models_dir = module_dir / "assets" / "models"

    # If running from PyInstaller bundle
    if hasattr(os, "_MEIPASS"):
        models_dir = Path(os._MEIPASS) / "assets" / "models"

    return models_dir


def get_model_path(model_name: str) -> Path:
    """Get path to a model file."""
    models_dir = get_models_dir()
    return models_dir / f"{model_name}.task"


def is_model_valid(model_path: Path, model_name: str) -> bool:
    """Check if a downloaded model file is valid."""
    if not model_path.exists():
        return False

    # Check file size (should be at least some minimum)
    min_size = MODEL_SIZES.get(model_name, 1_000_000)
    if model_path.stat().st_size < min_size * 0.5:  # Allow 50% variance
        return False

    return True


def download_model(model_name: str, force: bool = False) -> Optional[Path]:
    """
    Download a model file if not already cached.

    Args:
        model_name: Name of the model (e.g., "face_landmarker")
        force: Force re-download even if cached

    Returns:
        Path to the model file, or None if download failed
    """
    try:
        import requests
    except ImportError:
        logger.error("requests package not installed. Run: pip install requests")
        return None

    if model_name not in MODEL_URLS:
        logger.error(f"Unknown model: {model_name}")
        return None

    model_path = get_model_path(model_name)

    # Check if already cached and valid
    if not force and is_model_valid(model_path, model_name):
        logger.info(f"Model {model_name} already cached at {model_path}")
        return model_path

    # Create models directory
    model_path.parent.mkdir(parents=True, exist_ok=True)

    url = MODEL_URLS[model_name]
    logger.info(f"Downloading {model_name} from {url}...")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        # Download with progress
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) < 8192:  # Log every ~1MB
                            logger.info(f"  {progress:.1f}% ({downloaded // 1024}KB)")

        logger.info(f"Downloaded {model_name} to {model_path}")
        return model_path

    except requests.RequestException as e:
        logger.error(f"Failed to download {model_name}: {e}")
        if model_path.exists():
            model_path.unlink()  # Remove partial download
        return None


def ensure_models() -> bool:
    """
    Ensure all required models are downloaded.

    Returns:
        True if all models are available, False otherwise
    """
    all_ok = True

    for model_name in MODEL_URLS:
        model_path = get_model_path(model_name)

        if is_model_valid(model_path, model_name):
            logger.info(f"Model {model_name} OK: {model_path}")
        else:
            path = download_model(model_name)
            if path is None:
                all_ok = False

    return all_ok


def download_models_cli():
    """CLI entry point for downloading models."""
    import argparse

    parser = argparse.ArgumentParser(description="Download MediaPipe model files")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--model", choices=list(MODEL_URLS.keys()), help="Download specific model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.model:
        path = download_model(args.model, force=args.force)
        if path:
            print(f"✓ {args.model}: {path}")
        else:
            print(f"✗ {args.model}: Failed")
            exit(1)
    else:
        print("Downloading all models...")
        if ensure_models():
            print("✓ All models downloaded successfully")
        else:
            print("✗ Some models failed to download")
            exit(1)


if __name__ == "__main__":
    download_models_cli()
