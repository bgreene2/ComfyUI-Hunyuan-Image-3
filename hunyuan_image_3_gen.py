import numpy as np
import gc
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

class HunyuanImage3:
    """ComfyUI node for generating images with Hunyuan Image 3.0"""
    
    # Use class variable to store the loaded model so it persists between method calls
    # This prevents the model from being unloaded and reserved RAM from being released
    model = None
    model_settings = None
    
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
                "guidance_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "If 0, it will use the model's default of 7.5."}),
                "keep_model_in_memory": ("BOOLEAN", {"default": False, "tooltip": "If true, the model will be kept in memory between runs."}),
                "model_loading_config": ("model_loading_config", {"tooltip": "This is another custom node, called \"Hunyuan Image 3 - Model Loading Config\"."}),
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
        keep_model_in_memory: bool,
        model_loading_config=None,
    ):
        """Generate an image"""
        # Load model
        # If the model is already loaded, but the settings have changed, unload the model.
        model_settings = model_loading_config['hash_string']
        if self.model is not None and self.model_settings != model_settings:
            self.model = None
            gc.collect()

        # Save the current model settings
        self.model_settings = model_settings

        # If the model is already loaded, use it. Otherwise, load it.
        if self.model is not None:
            model = self.model
        else:
            model = self._load_model(model_loading_config)

        # If we are going to keep the model in memory, save it. Otherwise, make sure it's not saved.
        if keep_model_in_memory:
            self.model = model
        else:
            self.model = None

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

    def _load_model(self, model_loading_config):
        model_kwargs = model_loading_config['model_kwargs']

        if 'bits_and_bytes_config' in model_loading_config:
            quantization_config = BitsAndBytesConfig(**model_loading_config['bits_and_bytes_config'])
            model_kwargs['quantization_config'] = quantization_config

        self._log(f"Using model args: {model_kwargs}")

        model = AutoModelForCausalLM.from_pretrained(model_loading_config['model_id'], **model_kwargs)
        model.load_tokenizer(model_loading_config['model_id'])

        return model

    def _log(self, string):
        print(f"HunyuanImage3: {string}")

class HunyuanImage3ModelLoadingConfig:
    """ComfyUI node for configuring the model loading behavior of the HunyuanImage3 node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "weights_folder": ("STRING", {"multiline": False, "default": "./HunyuanImage-3", "tooltip": "The path to the Hunyuan Image 3.0 weights on disk."}),

                "use_offload": ("BOOLEAN", {"default": True, "tooltip": "If true, a custom device_map will be created. Otherwise, device_map will be auto."}),
                "disk_offload_layers": ("INT", {"default": 32, "min": 0, "max": 32, "tooltip": "The number of layers to offload from memory to disk. IMPORTANT: Read the README."}),
                "device_map_overrides": ("STRING", {"multiline": False, "default": "", "tooltip": "Device map key=value overrides, comma-separated. e.g. \"model.layers.0=0,model.layers.1=1\"."}),

                "attn_implementation": ("STRING", {"multiline": False, "default": "sdpa", "tooltip": "Use sdpa. If FlashAttention is installed, you may try flash_attention_2."}),
                "moe_impl": ("STRING", {"multiline": False, "default": "eager", "tooltip": "Use eager. If FlashInfer is installed, you may try flashinfer."}),
                "torch_dtype": ("STRING", {"multiline": False, "default": "auto", "tooltip": "This should be set to auto in most cases. If you specify it, omit the \"torch\", e.g. for \"torch.float16\", just enter \"float16\"."}),
                "trust_remote_code": ("BOOLEAN", {"default": True, "tooltip": "Must be set to True."}),
                "moe_drop_tokens": ("BOOLEAN", {"default": True, "tooltip": "Sets moe_drop_tokens=True on model loading."}),

                "load_in_8_bit": ("BOOLEAN", {"default": False, "tooltip": "A BitsAndBytes parameter."}),
                "load_in_4_bit": ("BOOLEAN", {"default": False, "tooltip": "A BitsAndBytes parameter."}),
                "bnb_4bit_use_double_quant": ("BOOLEAN", {"default": False, "tooltip": "A BitsAndBytes parameter."}),
                "bnb_4bit_compute_dtype": ("STRING", {"multiline": False, "default": "float16", "tooltip": "A BitsAndBytes parameter. Omit the \"torch\", e.g. for \"torch.float16\", just enter \"float16\"."}),
                "bnb_4bit_quant_type": ("STRING", {"multiline": False, "default": "nf4", "tooltip": "A BitsAndBytes parameter."}),
                "llm_int8_skip_modules": ("STRING", {"multiline": False, "default": "", "tooltip": "A BitsAndBytes parameter."}),
                "llm_int8_enable_fp32_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "A BitsAndBytes parameter."}),
            }
        }
    
    RETURN_TYPES = ("model_loading_config",)
    RETURN_NAMES = ("Hunyuan Image 3.0 model loading config",)
    FUNCTION = "return_config"
    OUTPUT_NODE = False
    CATEGORY = "Image Generation"

    def return_config(
        self,

        weights_folder: str,

        use_offload: bool,
        disk_offload_layers: int,
        device_map_overrides: str,

        attn_implementation: str,
        moe_impl: str,
        torch_dtype: str,
        trust_remote_code: bool,
        moe_drop_tokens: bool,

        load_in_8_bit: bool,
        load_in_4_bit: bool,
        bnb_4bit_use_double_quant: bool,
        bnb_4bit_compute_dtype: str,
        bnb_4bit_quant_type: str,
        llm_int8_skip_modules: str,
        llm_int8_enable_fp32_cpu_offload: bool,
    ):
        """ Return configuration """
        model_config = dict(
            model_id=weights_folder
        )

        # build device map
        if use_offload or device_map_overrides:
            device_map = {'vae': 0, 'vision_model': 'cpu', 'vision_aligner': 'cpu', 'timestep_emb': 'cpu', 'patch_embed': 'cpu', 'time_embed': 'cpu', 'final_layer': 'cpu', 'time_embed_2': 'cpu', 'model.wte': 'cpu', 'model.layers.0': 'cpu', 'model.layers.1': 'cpu', 'model.layers.2': 'cpu', 'model.layers.3': 'cpu', 'model.layers.4': 'cpu', 'model.layers.5': 'cpu', 'model.layers.6': 'cpu', 'model.layers.7': 'cpu', 'model.layers.8': 'cpu', 'model.layers.9': 'cpu', 'model.layers.10': 'cpu', 'model.layers.11': 'cpu', 'model.layers.12': 'cpu', 'model.layers.13': 'cpu', 'model.layers.14': 'cpu', 'model.layers.15': 'cpu', 'model.layers.16': 'cpu', 'model.layers.17': 'cpu', 'model.layers.18': 'cpu', 'model.layers.19': 'cpu', 'model.layers.20': 'cpu', 'model.layers.21': 'cpu', 'model.layers.22': 'cpu', 'model.layers.23': 'cpu', 'model.layers.24': 'cpu', 'model.layers.25': 'cpu', 'model.layers.26': 'cpu', 'model.layers.27': 'cpu', 'model.layers.28': 'cpu', 'model.layers.29': 'cpu', 'model.layers.30': 'cpu', 'model.layers.31': 'cpu', 'model.ln_f': 'cpu', 'lm_head': 'cpu'}

            if use_offload:
                top_layer_num = 31
                if disk_offload_layers > 0:
                    for i in range(disk_offload_layers):
                        layer_num = top_layer_num - i
                        device_map[f"model.layers.{layer_num}"] = "disk"

            if device_map_overrides:
                overrides = device_map_overrides.strip().split(',')
                for override in overrides:
                    split_override = override.strip().split('=')
                    if len(split_override) != 2:
                        raise 'device_map_overrides is invalid'
                    key = split_override[0]
                    value = split_override[-1]

                    # Support `0` as integer and `cuda:0` as string
                    try:
                        value = int(value)
                    except:
                        pass

                    device_map[key] = value
        else:
            device_map = 'auto'

        # make model keyword args
        model_kwargs = dict(
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            torch_dtype=torch_dtype,
            moe_impl=moe_impl,
        )

        if moe_drop_tokens:
            model_kwargs['moe_drop_tokens'] = True

        model_config['model_kwargs'] = model_kwargs

        # make bitsandbytes config
        if load_in_8_bit or load_in_4_bit:
            bits_and_bytes_config = dict(
                    load_in_8_bit=load_in_8_bit,
                    load_in_4_bit=load_in_4_bit,
                    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                    llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload,
            )

            if bnb_4bit_compute_dtype:
                bits_and_bytes_config['bnb_4bit_compute_dtype'] = bnb_4bit_compute_dtype
            if bnb_4bit_quant_type:
                bits_and_bytes_config['bnb_4bit_quant_type'] = bnb_4bit_quant_type
            if llm_int8_skip_modules is not None or llm_int8_skip_modules.strip() != '':
                bits_and_bytes_config['llm_int8_skip_modules'] = [x.strip() for x in llm_int8_skip_modules.strip().split(',')]

            model_config["bits_and_bytes_config"] = bits_and_bytes_config

        # build a string that will change if the configuration is changed
        # TODO: this is used to tell the generation node whether it needs to reload the model. There is probably a built-in ComfyUI way for a node to know if its input has changed, and we should use that instead.
        hash_string = f"{weights_folder}-{load_in_8_bit}-{load_in_4_bit}-{bnb_4bit_use_double_quant}-{bnb_4bit_compute_dtype}-{bnb_4bit_quant_type}-{llm_int8_skip_modules}-{llm_int8_enable_fp32_cpu_offload}-{attn_implementation}-{moe_impl}-{torch_dtype}-{trust_remote_code}-{use_offload}-{disk_offload_layers}-{device_map_overrides}-{moe_drop_tokens}"

        model_config['hash_string'] = hash_string

        return (model_config,)

