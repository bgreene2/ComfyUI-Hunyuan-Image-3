from .hunyuan_image_3_gen import HunyuanImage3  # Import your actual node class

NODE_CLASS_MAPPINGS = {
    "HunyuanImage3": HunyuanImage3  # The key is what will show up in ComfyUI interface
}

# Optional: If your node has display names different from class names
NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanImage3": "Hunyuan Image 3.0"  # Optional, for better UI display
}

# Required: To properly register your node
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
