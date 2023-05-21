import os
import sys
import argparse

import torch
import numpy as np

import transformers
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers.activations import SiLUActivation

from ptflops import get_model_complexity_info
from ptflops.pytorch_ops import bn_flops_counter_hook, pool_flops_counter_hook

from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP
from LLMPruner.peft import PeftModel

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
torch_version = int(torch.__version__.split('.')[1])

def LlamaAttention_counter_hook(module, input, output):
    # (1) Ignore past-key values
    # (2) Assume there is no attention mask
    # Input will be empty in some pytorch version. use output here since input.shape == output.shape
    flops = 0
    q_len = output[0].shape[1]
    linear_dim = output[0].shape[-1]
    num_heads = module.num_heads
    head_dim = module.head_dim

    rotary_flops = 2 * (q_len * num_heads * head_dim) * 2
    attention_flops = num_heads * (q_len * q_len * head_dim + q_len * q_len + q_len * q_len * head_dim) #QK^T + softmax + AttentionV
    linear_flops = 4 * (q_len * linear_dim * linear_dim) # 4 for q, k, v, o. 
    flops += rotary_flops + attention_flops + linear_flops
    module.__flops__ += int(flops)

def rmsnorm_flops_counter_hook(module, input, output):
    input = input[0]

    batch_flops = np.prod(input.shape)
    batch_flops *= 2
    module.__flops__ += int(batch_flops)

def main(args):
    if args.model_type == 'pretrain':
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
        model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            low_cpu_mem_usage=True if torch_version >=9 else False
        )
    elif args.model_type == 'pruneLLM':
        pruned_dict = torch.load(args.ckpt, map_location='cpu')
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    else:
        raise NotImplementedError

    if device == "cuda":
        model.half()
        model = model.cuda()
    
    forward_prompts = torch.tensor([
        [    1,   306,  4658,   278,  6593,   310,  2834,   338],
        [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
    ])
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (64,), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True,
                                                 custom_modules_hooks={
                                                    LlamaAttention: LlamaAttention_counter_hook,
                                                    LlamaRMSNorm: rmsnorm_flops_counter_hook,
                                                    SiLUActivation: pool_flops_counter_hook,
                                                 },)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    print("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLaMA (huggingface version)')

    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--model_type', type=str, required=True, help = 'choose from ')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lora_ckpt', type=str, default=None)
    
    args = parser.parse_args()
    main(args)
