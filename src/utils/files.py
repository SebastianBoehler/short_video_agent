"""File utility functions."""

from pathlib import Path
from typing import Optional


def load_speaker_images(
    speaker_dir: str | Path,
    max_images: int = 4,
) -> list[str]:
    """
    Load speaker images from a directory.
    
    Args:
        speaker_dir: Path to speaker image directory
        max_images: Maximum number of images to return
    
    Returns:
        List of absolute paths to speaker images
    """
    dir_path = Path(speaker_dir)
    if not dir_path.exists():
        return []
    
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
    images = []
    
    for ext in extensions:
        images.extend(sorted(dir_path.glob(ext)))
    
    return [str(p.resolve()) for p in images[:max_images]]


def load_product_images(
    product_dir: str | Path,
    max_images: int = 4,
) -> list[str]:
    """
    Load product images from a directory.
    
    Args:
        product_dir: Path to product image directory
        max_images: Maximum number of images to return
    
    Returns:
        List of absolute paths to product images
    """
    return load_speaker_images(product_dir, max_images)


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, create if not."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_output_path(
    base_dir: str | Path,
    scene_id: str,
    suffix: str,
    extension: str = "mp4",
) -> Path:
    """Generate output path for a scene artifact."""
    base = Path(base_dir)
    return base / scene_id / f"{scene_id}_{suffix}.{extension}"
