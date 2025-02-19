from pathlib import Path

from PIL import Image

from src.optimize import process_image, load_config

if __name__ == "__main__":
    config_path = Path("config.yaml")
    config = load_config(config_path)

    watermark_path = Path(config.watermark["path"])
    if not watermark_path.exists():
        # Simple red square watermark if not existing
        watermark_img = Image.new("RGBA", (64, 64), color=(255, 0, 0, 128))
        watermark_img.save(watermark_path)

    if config.input_dir.exists():
        for file in config.input_dir.glob("*"):
            if file.is_file():
                result = process_image(file, config)
                print(f"Processed files: {result}")

    result = process_image("input.jpg", config)
    print(f"Processed files: {result}")
