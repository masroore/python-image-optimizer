from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import yaml
from PIL import Image, ImageEnhance, ImageFilter, ExifTags
from PIL.Image import Transpose, Resampling


@dataclass
class PipelineConfig:
    orientation_enabled: bool
    colormode: str
    scaling_enabled: bool
    adjustments_enabled: bool
    colormode_enabled: bool
    scaling: Tuple[int, int]
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
        scaling=(
            config["scaling"]["width"],
            config["scaling"]["height"],
        ),
        adjustments=config["adjustments"],
        watermark=config["watermark"],
        webp_settings=config["webp"],
        colormode=config["colormode"],
        thumbnail=config["thumbnail"],
        blur=config["blur"],
        orientation_enabled=config["orientation_enabled"],
        scaling_enabled=config["scaling_enabled"],
        colormode_enabled=config["colormode_enabled"],
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

    max_width, max_height = config.scaling
    width, height = image.size

    if width > max_width or height > max_height:
        image.thumbnail((max_width, max_height), Resampling.LANCZOS)

    return image


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
    if not watermark_path.exists():
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
    target_opacity = watermark_config.get("opacity")
    if target_opacity < 1.0:
        # Adjust watermark opacity
        alpha = watermark.split()[3]
        alpha = alpha.point(lambda p: p * target_opacity)
        watermark.putalpha(alpha)

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
    webp_config = config.webp_settings
    output_path = generate_output_path(webp_config, image_path, config.output_dir)

    # Create a new image without EXIF data
    clean_img = Image.new(image.mode, image.size)
    clean_img.paste(image)

    # Save as WebP
    clean_img.save(
        output_path,
        format="WEBP",
        quality=webp_config.get("quality"),
        method=webp_config.get("method"),
        lossless=webp_config.get("lossless"),
    )

    return output_path


def recreate_resize_image(image: Image.Image, config: Dict[str, Any]) -> Image.Image:
    target_width = config.get("width")
    target_height = config.get("height")

    # Create a copy of the image for thumbnail (this also strips EXIF)
    resized = Image.new(image.mode, image.size)
    resized.paste(image)

    # Resize the image
    resized.thumbnail((target_width, target_height), Resampling.LANCZOS)

    return resized


def generate_output_path(
    config: Dict[str, Any], file_path: Path, output_dir: Path | None = None
) -> Path:
    """Create output directory if it doesn't exist and generate output path."""
    if not output_dir:
        output_dir = Path(config.get("path"))
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = file_path.stem
    suffix = config.get("suffix", "")
    return output_dir / f"{base_name}{suffix}.webp"


def create_thumbnail(
    image: Image.Image, image_path: Path, config: PipelineConfig
) -> Optional[Path]:
    """Create thumbnail version of the image with EXIF data stripped."""
    thumbnail_config = config.thumbnail

    if not thumbnail_config.get("enabled", False):
        return None

    thumb = recreate_resize_image(image, thumbnail_config)

    thumb.save(
        generate_output_path(thumbnail_config, image_path),
        format="WEBP",
        quality=thumbnail_config.get("quality"),
        method=thumbnail_config.get("method"),
    )

    return generate_output_path(thumbnail_config, image_path)


def create_blurred(
    image: Image.Image, image_path: Path, config: PipelineConfig
) -> Optional[Path]:
    """Create blurred version of the image with EXIF data stripped."""
    blur_config = config.blur

    if not blur_config.get("enabled", False):
        return None

    blurred = recreate_resize_image(image, blur_config)

    # Apply Gaussian blur
    blur_radius = blur_config.get("radius")
    blurred = blurred.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    output_path = generate_output_path(blur_config, image_path)

    # Save as WebP
    blurred.save(
        output_path,
        format="WEBP",
        quality=blur_config.get("quality"),
        method=blur_config.get("method"),
    )

    return output_path


def fix_colormode(image: Image.Image, config: PipelineConfig) -> Image.Image:
    """Convert image to specified color mode."""
    if not config.colormode_enabled:
        return image

    colormode = config.colormode

    if colormode == "RGB" or colormode == "RGBA":
        if image.mode == "RGBA":
            return image
        if image.mode == "LA":
            return image.convert("RGBA")

        return image.convert(colormode)
    elif colormode == "GRAY":
        return image.convert("L")

    return image.convert(colormode)


def process_image(image_path_str: str, config: PipelineConfig) -> Dict[str, str]:
    """Main function to process an image through the pipeline."""
    # Convert string paths to Path objects
    image_path = Path(image_path_str)
    result = {"source": str(image_path), "source_size": image_path.stat().st_size}

    image = Image.open(image_path)

    # Apply basic fixes
    image = fix_colormode(image, config)
    image = fix_orientation(image, config)

    # Create thumbnail and blurred versions
    thumbnail_path = create_thumbnail(image, image_path, config)
    blurred_path = create_blurred(image, image_path, config)

    # Primary pipeline
    image = scale_image(image, config)
    image = adjust_image(image, config)
    image = add_watermark(image, config)
    output_path = save_webp(image, image_path, config)

    result["optimized"] = str(output_path)
    result["optimized_size"] = output_path.stat().st_size

    if thumbnail_path:
        result["thumbnail"] = str(thumbnail_path)
        result["thumbnail_size"] = thumbnail_path.stat().st_size

    if blurred_path:
        result["blurred"] = str(blurred_path)
        result["blurred_size"] = blurred_path.stat().st_size

    return result
