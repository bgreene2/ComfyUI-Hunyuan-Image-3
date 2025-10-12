# ComfyUI Hunyuan Image 3.0 custom node

This is a custom node that allows for basic image generation using Hunyuan Image 3.0.

![Node screenshot](assets/node_screenshot.png)

## Features

- Supports CPU and disk offload to allow generation on consumer setups
    - When using CPU offload, weights are stored in system RAM and transferred to the GPU as needed for processing
    - When using disk offload, weights are stored in system RAM and on disk and transferred to the GPU as needed for processing

## Installation

You can find the node on the ComfyUI registry, or you can install it manually.

Manual installation steps:

1. Navigate to your custom_nodes folder
2. Clone this repo: `git clone https://github.com/bgreene2/ComfyUI-HunyuanImage3.git`
3. Change to the directory `cd ComfyUI-HunyuanImage3`
4. Assuming the correct Python environemnt is loaded, install dependencies `pip install -r requirements.txt`
5. If pytorch is not already installed, install pytorch: `pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128`
6. (optional) Add the flag `--disable-cuda-malloc` to your ComfyUI startup script.

## Post-installation steps

- Download [the model weights](https://huggingface.co/tencent/HunyuanImage-3.0)

## Usage Guide

The node has the following inputs:

- prompt - Your text prompt

The node has the following parameters:

- seed - The seed for the random number generator
- control after generate - This is attached to the 'seed' parameter
- use_dimensions - Whether to use the below user-specified width and height values. If false, the model will automatically choose the image dimensions. Note: the model has a few discrete image sizes that it supports. So even if you specify a width and height, the actual image dimensions used may be different, but should have a similar ratio.
- width - Image width
- height - Image height
- steps - Number of steps. The recommended value is 50.
- guidance_scale - CFG. The recommended value is 7.5.
- attn_implementation - Valid values are `sdpa` and `flash_attention_2`
- moe_impl - Valid values are `eager` and `flashinfer`
- weights_folder - The path to the model's weights on disk.
- use_offload - Whether to use CPU / disk offload.
- disk_offload_layers - The number of layers (out of 32) to offload to disk, rather than hold in memory.
- keep_model_in_memory - If enabled, the model will be kept in memory between generations, instead of being unloaded.
- device_map_overrides - You can modify the custom device_map using this. Overrides are expressed as key=value pairs, comma-separated. See the [Performance Tuning](#performance-tuning) section for more details.
- moe_drop_tokens - Enables the moe_drop_tokens parameter on model loading. See the [Memory Troubleshooting](#memory-troubleshooting) section for more details.

Basic usage: Connect a String (Multiline) input to the `prompt` input, and connect the `Image` output to a Save Image node. An [example workflow](workflows/hunyuan_image_3_example.json) is provided.
![example workflow](assets/workflow_screenshot.png)

## Recommended Usage

This model works best with detailed prompts. See the [HuggingFace page](https://huggingface.co/tencent/HunyuanImage-3.0) for a prompting guide.

## Performance Tuning

If you can fit the entire model in VRAM, you can run with use_offload set to `false`. This should give you the highest speed possible.

If you can't fit the model in VRAM, you can enable offload by setting use_offload to `true`. In this case, you will be bottlenecked by the transfer of data from system memory to VRAM to process on your GPU. You will want to have as fast a PCIe connection as possible.

If you can't fit the model in system ram, you can enable disk offload by setting disk_offload_layers to a value above `0`. In this case, you will be additionally bottlenecked by the transfer of data from disk. You will want to have as fast a drive as possible.

If you are using disk offload, you need to choose the number of layers to offload such that you still have some physical memory left over, so that the system does not use swap. This may require some trial-and-error while monitoring memory usage. On a system with 128GB of RAM, 10 layers is a good starting point.

When using disk offload with 10 layers, on a system with PCIe 4.0 and an NVMe drive, an image will take on the order of 1-1.5 hours to generate using 50 steps.

Tips:

If you wish to use more of your VRAM, you can set device_map_overrides.

- To put everything except for the model's layers onto GPU 0, you would set `vision_model=0,vision_aligner=0,timestamp_emb=0,patch_embed=0,time_embed=0,final_layer=0,time_embed_2=0,model.wte=0,model.ln_f=0,lm_head=0`.
- To start putting layers onto GPU0, you would add `model.layers.0=0,model.layers.1=0`,...
- You can also spread the load across GPUs, e.g. `model.layers.2=1,model.layers.3=1` would put layers 2 and 3 onto GPU 1.

## Memory Troubleshooting

If you are getting crashes due to running out of GPU memory, there are some things you could try:

- Enable moe_drop_tokens.

This stabilizes memory usage dramatically. It does alter the output image, but quality is not affected much. See [this comment](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/issues/39#issuecomment-3384699854).

If you don't wish to do this, or if you are still running out of memory, you could try the following:

1. Set Sysmem Fallback Policy in the Nvidia control panel (only available on Windows platforms - Nvidia's Linux drivers don't have this feature)
2. Run ComfyUI with `--disable-cuda-malloc`

This is where the option is found in the Nvidia control panel:
![CUDA Sysmem Fallback](assets/cuda_sysmem_fallback_screenshot.png)
