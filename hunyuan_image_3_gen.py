import numpy as np
import torch
from transformers import AutoModelForCausalLM

class HunyuanImage3:
    """ComfyUI node for generating images with Hunyuan Image 3.0"""
    
    # Use class variable to store the loaded model so it persists between method calls
    # This prevents the model from being unloaded and reserved RAM from being released
    model = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "use_dimensions": ("BOOLEAN", {"default": False, "tooltip": "If false, the model will choose the image dimensions, and the user-provided width and height will be ignored."}),
                "width": ("INT", {"default": 1024, "min": 0, "max": 4096}),
                "height": ("INT", {"default": 1024, "min": 0, "max": 4096}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 200, "tooltip": "If 0, it will use the model's default of 50."}),
                "guidance_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "tooltip": "If 0, it will use the model's default of 7.5."}),
                "attn_implementation": ("STRING", {"multiline": False, "default": "sdpa", "tooltip": "Use sdpa. If FlashAttention is installed, you may try flash_attention_2."}),
                "moe_impl": ("STRING", {"multiline": False, "default": "eager", "tooltip": "Use eager. If FlashInfer is installed, you may try flashinfer."}),
                "weights_folder": ("STRING", {"multiline": False, "default": "./HunyuanImage-3", "tooltip": "The path to the Hunyuan Image 3.0 weights on disk. Note: the path cannot have a \".\"."}),
                "use_offload": ("BOOLEAN", {"default": True, "tooltip": "If true, a custom device_map will be created. Otherwise, device_map will be auto."}),
                "disk_offload_layers": ("INT", {"default": 32, "min": 0, "max": 32, "tooltip": "The number of layers to offload from memory to disk. IMPORTANT: Read the README."}),
                "keep_model_in_memory": ("BOOLEAN", {"default": False, "tooltip": "If true, the model will be kept in memory between runs."}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Image",)
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "Image Generation"

    def generate(
        self,
        prompt: str,
        seed: int,
        use_dimensions: bool,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        attn_implementation: str,
        moe_impl: str,
        weights_folder: str,
        use_offload: bool,
        disk_offload_layers: int,
        keep_model_in_memory: bool,
    ):
        """Generate an image"""
        model_id = weights_folder

        # Load model
        if keep_model_in_memory:
            # If the model is already loaded, use it. Otherwise, load it.
            if self.model is not None:
                model = self.model
            else:
                model = self._load_model(model_id, attn_implementation, moe_impl, use_offload, disk_offload_layers)
                self.model = model
        else:
            self.model = None
            model = self._load_model(model_id, attn_implementation, moe_impl, use_offload, disk_offload_layers)

        # Generate image
        image_kwargs = {'seed': seed}
        if steps > 0:
            image_kwargs['diff_infer_steps'] = steps
        if guidance_scale > 0.0:
            image_kwargs['diff_guidance_scale'] = guidance_scale
        if use_dimensions:
            image_kwargs["image_size"] = f"{height}x{width}"
        image = model.generate_image(prompt=prompt, stream=True, **image_kwargs)

        # Convert image from PIL format
        if image.mode == 'I':
            image = image.point(lambda i: i * (1 / 255))
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image,)

    def _load_model(self, model_id, attn_implementation, moe_impl, use_offload, disk_offload_layers):
        if use_offload:
            device_map = {'vae': 0, 'vision_model': 'cpu', 'vision_aligner': 'cpu', 'timestep_emb': 'cpu', 'patch_embed': 'cpu', 'time_embed': 'cpu', 'final_layer': 'cpu', 'time_embed_2': 'cpu', 'model.wte': 'cpu', 'model.layers.0': 'cpu', 'model.layers.1': 'cpu', 'model.layers.2': 'cpu', 'model.layers.3': 'cpu', 'model.layers.4': 'cpu', 'model.layers.5': 'cpu', 'model.layers.6': 'cpu', 'model.layers.7': 'cpu', 'model.layers.8': 'cpu', 'model.layers.9': 'cpu', 'model.layers.10': 'cpu', 'model.layers.11': 'cpu', 'model.layers.12': 'cpu', 'model.layers.13': 'cpu', 'model.layers.14': 'cpu', 'model.layers.15': 'cpu', 'model.layers.16': 'cpu', 'model.layers.17': 'cpu', 'model.layers.18': 'cpu', 'model.layers.19': 'cpu', 'model.layers.20': 'cpu', 'model.layers.21': 'cpu', 'model.layers.22': 'cpu', 'model.layers.23': 'cpu', 'model.layers.24': 'cpu', 'model.layers.25': 'cpu', 'model.layers.26': 'cpu', 'model.layers.27': 'cpu', 'model.layers.28': 'cpu', 'model.layers.29': 'cpu', 'model.layers.30': 'cpu', 'model.layers.31': 'cpu', 'model.ln_f': 'cpu', 'lm_head': 'cpu'}

            top_layer_num = 31
            if disk_offload_layers > 0:
                for i in range(disk_offload_layers):
                    layer_num = top_layer_num - i
                    device_map[f"model.layers.{layer_num}"] = "disk"
        else:
            device_map = 'auto'

        model_kwargs = dict(
            attn_implementation=attn_implementation,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype="auto",
            moe_impl=moe_impl,
        )

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        model.load_tokenizer(model_id)

        return model

