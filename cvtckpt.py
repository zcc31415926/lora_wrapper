import os
import sys
from ckpt_utils import embedLoRAWeights, extractLoRAWeights, mergeLoRAWeights


model_ckpt = '../../Projects/sd-retail/logs/v1-txt2img/checkpoints/last-003.ckpt'
lora_ckpt = './lora.ckpt'

extractLoRAWeights(model_ckpt, out_ckpt=lora_ckpt)
embedLoRAWeights(model_ckpt, lora_ckpt=lora_ckpt, out_ckpt='./embedded.ckpt')
mergeLoRAWeights(model_ckpt, lora_ckpts=[lora_ckpt], out_ckpt='./merged.ckpt')

