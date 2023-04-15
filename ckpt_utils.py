def embedLoRA(model_weights, lora_weights):
    for k in lora_weights.keys():
        if k.endswith('lora_A'):
            parent_key = k[: -7]
            model_weights[f'{parent_key}.base_layer.weight'] = model_weights[f'{parent_key}.weight']
            model_weights[f'{parent_key}.base_layer.bias'] = model_weights[f'{parent_key}.bias']
            model_weights[f'{parent_key}.lora_A'] = lora_weights[f'{parent_key}.lora_A']
            model_weights[f'{parent_key}.lora_B'] = lora_weights[f'{parent_key}.lora_B']
            del model_weights[f'{parent_key}.weight']
            del model_weights[f'{parent_key}.bias']
        elif k.endswith('lora_A.weight'):
            parent_key = k[: -14]
            model_weights[f'{parent_key}.base_layer.weight'] = model_weights[f'{parent_key}.weight']
            model_weights[f'{parent_key}.base_layer.bias'] = model_weights[f'{parent_key}.bias']
            model_weights[f'{parent_key}.lora_A.weight'] = lora_weights[f'{parent_key}.lora_A.weight']
            model_weights[f'{parent_key}.lora_A.bias'] = lora_weights[f'{parent_key}.lora_A.bias']
            model_weights[f'{parent_key}.lora_B.weight'] = lora_weights[f'{parent_key}.lora_B.weight']
            model_weights[f'{parent_key}.lora_B.bias'] = lora_weights[f'{parent_key}.lora_B.bias']
            del model_weights[f'{parent_key}.weight']
            del model_weights[f'{parent_key}.bias']
    return model_weights


def extractLoRA(model_weights):
    lora_weights = {}
    for k in model_weights.keys():
        if 'lora_A' in k or 'lora_B' in k:
            lora_weights[k] = model_weights[k]
    return lora_weights


def mergeLoRA(model_weights, lora_weights=[]):
    if len(lora_weights) == 0:
        has_soft_wrappers = checkSoftWrappers(model_weights)
        if has_soft_wrappers:
            print('[ERROR] Weights with soft wrappers cannot be merged')
            return model_weights
        for k in model_weights.keys():
            if k.endswith('lora_A'):
                parent_key = k[: -7]
                lora_A = model_weights[f'{parent_key}.lora_A']
                lora_B = model_weights[f'{parent_key}.lora_B']
                base_weights = model_weights[f'{parent_key}.base_layer.weight']
                base_bias = model_weights[f'{parent_key}.base_layer.bias']
                weight = model_weights[f'{parent_key}.scale'] * lora_B @ lora_A
                weight = weight.contiguous().view(*base_weights.size())
                model_weights[f'{parent_key}.weight'] = base_weights + weight
                model_weights[f'{parent_key}.bias'] = base_bias
                del model_weights[f'{parent_key}.lora_A']
                del model_weights[f'{parent_key}.lora_B']
                del model_weights[f'{parent_key}.base_layer.weight']
                del model_weights[f'{parent_key}.base_layer.bias']
    else:
        for lora_weight in lora_weights:
            has_soft_wrappers = checkSoftWrappers(lora_weight)
            if has_soft_wrappers:
                print('[ERROR] Weights with soft wrappers cannot be merged')
                continue
            for k in lora_weight.keys():
                if k.endswith(f'lora_A'):
                    parent_key = k[: -7]
                    lora_A = lora_weight[f'{parent_key}.lora_A']
                    lora_B = lora_weight[f'{parent_key}.lora_B']
                    base_weights = model_weights[f'{parent_key}.weight']
                    weight = lora_weight[f'{parent_key}.scale'] * lora_B @ lora_A
                    weight = weight.contiguous().view(*base_weights.size())
                    model_weights[f'{parent_key}.weight'] += weight
    return model_weights


def checkSoftWrappers(weights):
    for k in weights.keys():
        if k.endswith('lora_A.weight') or k.endswith('lora_A.bias') or \
            k.endswith('lora_B.weight') or k.endswith('lora_B.bias'):
            return True
    return False

