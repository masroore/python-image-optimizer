from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import yaml
from PIL import Image, ImageEnhance, ImageFilter, ExifTags
from PIL.Image import Transpose, Resampling


@dataclass
class PipelineConfig:
    orientation_enabled: bool
    scaling_enabled: bool
    adjustments_enabled: bool
    max_dimensions: Tuple[int, int]
    adjustments: Dict[str, float]
    watermark: Dict[str, Any]
    webp_settings: Dict[str, Any]
    thumbnail: Dict[str, Any]
    blur: Dict[str, Any]
    input_dir: Path
    output_dir: Path


def load_config(config_path: Path) -> PipelineConfig:
    """Load configuration from YAML file."""
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    return PipelineConfig(
        max_dimensions=(
            config["scaling"]["max_width"],
            config["scaling"]["max_height"],
        ),
        adjustments=config["adjustments"],
        watermark=config["watermark"],
        webp_settings=config["webp"],
        thumbnail=config["thumbnail"],
        blur=config["blur"],
        orientation_enabled=config["orientation_enabled"],
        scaling_enabled=config["scaling_enabled"],
        adjustments_enabled=config["adjustments_enabled"],
        input_dir=Path(config["input_dir"]),
        output_dir=Path(config["output_dir"]),
    )


def fix_orientation(image: Image.Image, config: PipelineConfig) -> Image.Image:
    """Fix image orientation based on EXIF data."""
    if not config.orientation_enabled:
        return image

    if not hasattr(image, "_getexif") or image._getexif() is None:
        return image

    exif = dict(image._getexif().items())

    # Find the orientation tag
    orientation_tag = None
    for tag, value in ExifTags.TAGS.items():
        if value == "Orientation":
            orientation_tag = tag
            break

    if orientation_tag is None or orientation_tag not in exif:
        return image

    orientation = exif[orientation_tag]

    # Apply orientation fixes
    if orientation == 2:
        # Horizontal flip
        image = image.transpose(Transpose.FLIP_LEFT_RIGHT)
    elif orientation == 3:
        # 180 degree rotation
        image = image.transpose(Transpose.ROTATE_180)
    elif orientation == 4:
        # Vertical flip
        image = image.transpose(Transpose.FLIP_TOP_BOTTOM)
    elif orientation == 5:
        # Horizontal flip + 90 degree counterclockwise rotation
        image = image.transpose(Transpose.FLIP_LEFT_RIGHT).transpose(
            Transpose.ROTATE_90
        )
    elif orientation == 6:
        # 90 degree counterclockwise rotation
        image = image.transpose(Transpose.ROTATE_270)
    elif orientation == 7:
        # Horizontal flip + 90 degree clockwise rotation
        image = image.transpose(Transpose.FLIP_LEFT_RIGHT).transpose(
            Transpose.ROTATE_270
        )
    elif orientation == 8:
        # 90 degree clockwise rotation
        image = image.transpose(Transpose.ROTATE_90)

    return image


def scale_image(image: Image.Image, config: PipelineConfig) -> Image.Image:
    """Scale down image if it exceeds maximum dimensions."""
    if not config.scaling_enabled:
        return image

    max_width, max_height = config.max_dimensions
    width, height = image.size

    # Check if scaling is needed
    if width <= max_width and height <= max_height:
        return image

    # Calculate scaling factor
    scale_width = max_width / width
    scale_height = max_height / height
    scale_factor = min(scale_width, scale_height)

    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    return image.resize((new_width, new_height), Resampling.LANCZOS)


def adjust_image(image: Image.Image, config: PipelineConfig) -> Image.Image:
    """Apply brightness, contrast, and sharpness adjustments."""
    if not config.adjustments_enabled:
        return image

    adjustments = config.adjustments

    if adjustments.get("brightness", 1.0) != 1.0:
        image = ImageEnhance.Brightness(image).enhance(adjustments["brightness"])

    if adjustments.get("contrast", 1.0) != 1.0:
        image = ImageEnhance.Contrast(image).enhance(adjustments["contrast"])

    if adjustments.get("sharpness", 1.0) != 1.0:
        image = ImageEnhance.Sharpness(image).enhance(adjustments["sharpness"])

    return image


def add_watermark(image: Image.Image, config: PipelineConfig) -> Image.Image:
    """Add watermark to the image."""
    watermark_config = config.watermark

    if not watermark_config.get("enabled", False):
        return image

    # Load watermark image
    watermark_path = Path(watermark_config["path"])
    if not Path.exists(watermark_path):
        return image

    watermark = Image.open(watermark_path).convert("RGBA")

    # Resize watermark if needed
    if watermark_config.get("resize_percentage", 0) > 0:
        percent = watermark_config["resize_percentage"] / 100
        wm_width = int(image.width * percent)
        wm_height = int(watermark.height * (wm_width / watermark.width))
        watermark = watermark.resize((wm_width, wm_height), Resampling.LANCZOS)

    # Create transparent layer for the watermark
    transparent = Image.new("RGBA", image.size, (0, 0, 0, 0))

    # Calculate position
    position = watermark_config.get("position", "bottom-right")
    padding = watermark_config.get("padding", 10)

    if position == "top-left":
        pos = (padding, padding)
    elif position == "top-right":
        pos = (image.width - watermark.width - padding, padding)
    elif position == "bottom-left":
        pos = (padding, image.height - watermark.height - padding)
    elif position == "bottom-right":
        pos = (
            image.width - watermark.width - padding,
            image.height - watermark.height - padding,
        )
    elif position == "center":
        pos = (
            (image.width - watermark.width) // 2,
            (image.height - watermark.height) // 2,
        )
    else:
        pos = (padding, padding)  # Default to top-left

    # Apply opacity
    opacity = int(255 * watermark_config.get("opacity", 0.5))
    watermark.putalpha(opacity)

    # Paste the watermark onto the transparent layer
    transparent.paste(watermark, pos, watermark)

    # Convert the original image to RGBA if it's not already
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Combine the original image with the watermark layer
    result = Image.alpha_composite(image, transparent)

    return result.convert("RGB")  # Convert back to RGB for compatibility


def save_webp(image: Image.Image, image_path: Path, config: PipelineConfig) -> Path:
    """Save the image as WebP format with all EXIF data stripped."""
    webp_settings = config.webp_settings
    output_dir = config.output_dir

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output path
    base_name = image_path.stem
    output_path = output_dir / f"{base_name}.webp"

    # Create a new image without EXIF data
    clean_img = Image.new(image.mode, image.size)
    clean_img.paste(image)

    # Save as WebP
    clean_img.save(
        output_path,
        format="WEBP",
        quality=webp_settings.get("quality", 80),
        method=webp_settings.get("method", 4),
        lossless=webp_settings.get("lossless", False),
    )

    return output_path


def create_thumbnail(
    image: Image.Image, image_path: Path, config: PipelineConfig
) -> Optional[Path]:
    """Create thumbnail version of the image with EXIF data stripped."""
    thumbnail_config = config.thumbnail

    if not thumbnail_config.get("enabled", False):
        return None

    # Get thumbnail dimensions
    width = thumbnail_config.get("width", 200)
    height = thumbnail_config.get("height", 200)

    # Create a copy of the image for thumbnail (this also strips EXIF)
    thumb = Image.new(image.mode, image.size)
    thumb.paste(image)
    thumb.thumbnail((width, height), Resampling.LANCZOS)

    # Create thumbnail directory if it doesn't exist
    thumbnail_dir = config.output_dir / thumbnail_config.get("subfolder", "thumbnails")
    thumbnail_dir.mkdir(parents=True, exist_ok=True)

    # Generate output path
    base_name = image_path.stem
    output_path = thumbnail_dir / f"{base_name}_thumb.webp"

    # Save as WebP
    thumb.save(
        output_path,
        format="WEBP",
        quality=thumbnail_config.get("quality", 70),
        method=4,
    )

    return output_path


def create_blurred(
    image: Image.Image, image_path: Path, config: PipelineConfig
) -> Optional[Path]:
    """Create blurred version of the image with EXIF data stripped."""
    blur_config = config.blur

    if not blur_config.get("enabled", False):
        return None

    # Get blur dimensions
    width = blur_config.get("width", 800)
    height = blur_config.get("height", 600)

    # Create a new image without EXIF data
    blurred = Image.new(image.mode, image.size)
    blurred.paste(image)
    blurred = blurred.resize((width, height), Image.LANCZOS)

    # Apply Gaussian blur
    blur_radius = blur_config.get("radius", 10)
    blurred = blurred.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Create blur directory if it doesn't exist
    blur_dir = config.output_dir / blur_config.get("subfolder", "blurred")
    blur_dir.mkdir(parents=True, exist_ok=True)

    # Generate output path
    base_name = image_path.stem
    output_path = blur_dir / f"{base_name}_blurred.webp"

    # Save as WebP
    blurred.save(
        output_path, format="WEBP", quality=blur_config.get("quality", 70), method=4
    )

    return output_path


def process_image(image_path_str: str, config: PipelineConfig) -> Dict[str, str]:
    """Main function to process an image through the pipeline."""
    # Convert string paths to Path objects
    image_path = Path(image_path_str)

    # Load image
    image = Image.open(image_path)

    # Main pipeline
    image = fix_orientation(image, config)
    image = scale_image(image, config)
    image = adjust_image(image, config)
    image = add_watermark(image, config)
    output_path = save_webp(image, image_path, config)

    # Additional pipelines
    thumbnail_path = create_thumbnail(image, image_path, config)
    blurred_path = create_blurred(image, image_path, config)

    # Return paths to all generated files (converting Path objects to strings)
    result = {"main": str(output_path)}
    if thumbnail_path:
        result["thumbnail"] = str(thumbnail_path)
    if blurred_path:
        result["blurred"] = str(blurred_path)

    return result
