from lora_wrapper.ckpt_utils import embedLoRAWeights, extractLoRAWeights, mergeLoRAWeights


mode = 'embed' # embed, extract, merge
model_ckpt = 'sd.ckpt'
lora_ckpt = 'lora.ckpt'
# lora_ckpt = ['lora1.ckpt', 'lora2.ckpt']

funcs = {
    'embed': embedLoRAWeights,
    'extract': extractLoRAWeights,
    'merge': mergeLoRAWeights,
}
assert mode in funcs.keys(), f'[ERROR] Checkpoint conversion mode {mode} not supported'
funcs[mode](model_ckpt, lora_ckpt)

