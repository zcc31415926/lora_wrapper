# Customized Low-Rank Adaptation Wrapper for Efficient Fine-Tuning

Refer to [the official implementation of LoRA](https://github.com/microsoft/LoRA) for details of the LoRA efficient fine-tuning technique

## Usage

Clone the repository and place it in the root directory of the target project. For example:
```
git clone https://github.com/CompVis/stable-diffusion.git
cd stable-diffusion
git clone https://github.com/zcc31415926/lora_wrapper.git
```

Prepare a valid configuration file for the wrapper.

Exemplary LoRA configuration file: [`example_config.json`](example_config.json). Set attributes for each module-to-wrap in the format `$MODULE: {"r": $R, "scale": $SCALE, "trainable_scale": $TRAINABLE_SCALE, "hard": $HARD}`
- `trainable_model`: whether to set the backbone model trainable
- `$MODULE`: name of the module-to-wrap. Hierarchical names supported
- `$R`: rank of the matrices in LoRA
- `$SCALE`: scale of the matrix product (to be added to the original weight matrix)
- `$TRAINABLE_SCALE`: whether to set `$SCALE` trainable
- `$HARD`: whether to use hard LoRA wrappers for convolutional layers
> There are two possible implementations of LoRA on CONVOLUTIONAL layers. The first one, named HARD wrapping, extracts kernel weight matrices and applies matrix decomposition. For simplicity, we conduct hard wrapping only on layers with `kernel_size == 1`. The other one, named SOFT wrapping, constructs two extra convolutional layers with the same attribute set except channel numbers, and conducts two paralleled forward processes. Soft wrapping has no restriction on convolutional layers, but has inference latency, and cannot be converted to checkpoint files compatible with LoRA-free models

To wrap a pre-defined model with LoRA according to the configuration file `lora_config`:
```python
from lora_wrapper import LoRA
import json
model = my_model(params)
with open(lora_config, 'r') as f:
    lora_config = json.load(f)
lora = LoRA(lora_config)
model = lora.wrap(model)
```

Run `python cvtckpt.py` to make conversions between checkpoint files with and without LoRA after training
- Set `mode = 'embed'` to embed LoRA weights into Stable Diffusion weights and save the ensemble. The LoRA structure is PRESERVED in the saved checkpoint file for training or sampling with the wrapped model
- Set `mode = 'extract'` to extract and save LoRA weights from a wrapped model. The saved checkpoint file is much smaller than the ensemble
- Set `mode = 'merge'` to merge LoRA weights into Stable Diffusion weights and save the ensemble. The LoRA structure is ERASED in the saved checkpoint file for training or sampling with the LoRA-free model. Merging with multiple LoRA weights supported

## Bugs & Limitations

### Bugs

- To be tested...

### Limitations

- Incompatibility between LoRA and gradient checkpointing (intrinsic)

## Acknowledgements

- [LoRA](https://github.com/microsoft/LoRA): an efficient parameter fine-tuning technique based on low-rank matrix decomposition
- [cloneofsimo/lora](https://github.com/cloneofsimo/lora): an implementation of LoRA on encapsulated diffusion models

