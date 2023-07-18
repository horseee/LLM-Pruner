import os
import gc
import sys
import time
import json
import copy
import random
import argparse
from typing import Tuple

import torch
import numpy as np
from transformers import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from LLMPruner.models.hf_baichuan.baichuan7B.modeling_baichuan_7B import BaiChuanForCausalLM as BaiChuan7B
from LLMPruner.models.hf_baichuan.baichuan13B.modeling_baichuan_13B import BaichuanForCausalLM as BaiChuan13B

#from LLMPruner.models.hf_llama.modeling_llama import LlamaRMSNorm, LlamaAttention, LlamaMLP

import LLMPruner.torch_pruning as tp 
from LLMPruner.pruner import hf_baichuan_pruner as baichuan_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def main(args):
    set_random_seed(args.seed)

    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name), 
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        'baichuan-inc/Baichuan-13B-chat', use_fast=False, trust_remote_code=True
    ) 
    if '7B' in args.base_model:
        model = BaiChuan7B.from_pretrained(
            args.base_model, trust_remote_code=True,
            low_cpu_mem_usage=True if args.torch_version >=1.9 else False
        )
    elif '13B' in args.base_model:
        model = BaiChuan13B.from_pretrained(
            args.base_model, trust_remote_code=True,
            low_cpu_mem_usage=True if args.torch_version >=1.9 else False
        )
        model.generation_config = GenerationConfig.from_pretrained(args.base_model)
    else:
        raise NotImplementedError("Only support 7B and 13B model")

    if args.device == "cuda":
        model.half()
    model.to(args.device)

    if args.test_before_train:
        logger.log("\n==================Generation Results before Pruning================\n")
        model.eval()
        with torch.no_grad():
            inputs = tokenizer('登鹳雀楼->王之涣\n夜雨寄北->', return_tensors='pt').to(args.device)
            pred = model.generate(**inputs, max_new_tokens=64,repetition_penalty=1.1)
            response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
            logger.log(response)
    
    
    
    pruner_type = args.pruner_type.lower()
    assert pruner_type in ['random', 'l2', 'l1', 'taylor']

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    forward_prompts = torch.tensor([
        [    1,   306,  4658,   278,  6593,   310,  2834,   338],
        [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
    ]).to(args.device) # Only for building the dependency graph. Any input will be fine since the computation result are not taken into consideration.

    if pruner_type == 'random':
        imp = tp.importance.RandomImportance()
    elif pruner_type == 'l1':
        imp = baichuan_pruner.MagnitudeImportance(p=1)
    elif pruner_type == 'l2':
        imp = baichuan_pruner.MagnitudeImportance(p=2)
    elif pruner_type == 'taylor':
        imp = baichuan_pruner.TaylorImportance(group_reduction=args.grouping_strategy, taylor=args.taylor)
    else:
        raise NotImplementedError

    logger.log("Use {} pruner...".format(pruner_type))
    
    if args.block_wise:
        def forward_fn(model, example_inputs):
            return model(example_inputs, build_dp=True)

        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio, 
            "ch_sparsity_dict": {
                model.model.layers[i].self_attn.W_pack: args.pruning_ratio / 3 for i in range(args.block_attention_layer_start, args.block_attention_layer_end)
            },
            "forward_fn": forward_fn,
            "ignored_layers":[],
            "channel_groups": {
                #layer.self_attn.W_pack: 3 for layer in model.model.layers
            },
            "consecutive_groups": {
                layer.self_attn.W_pack: layer.self_attn.head_dim for layer in model.model.layers
            },
            "customized_pruners": {
                model.model.layers[0].input_layernorm.__class__: baichuan_pruner.hf_rmsnorm_pruner,
            },
            "root_module_types": None, 
            "root_instances": [model.model.layers[i].mlp.gate_proj for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end)] +
                              [model.model.layers[i].self_attn.W_pack for i in range(args.block_attention_layer_start, args.block_attention_layer_end)],
            "enable_index_mapping": True
                              
        }
        logger.log("Pruning Attention Layer = {}".format(list(range(args.block_attention_layer_start, args.block_attention_layer_end))))
        logger.log("Pruning MLP Layer = {}".format(list(range(args.block_mlp_layer_start, args.block_mlp_layer_end))))

        pruner = tp.pruner.MetaPruner(
            model,
            forward_prompts,
            **kwargs
        )

        model.zero_grad()

        logger.log("Start Pruning")
        for i in range(args.iterative_steps):

            if pruner_type in ['taylor']:
                example_prompts = get_examples('bookcorpus', tokenizer, args.num_examples, seq_len = 64).to(args.device)
                logger.log("Start Backwarding in iterative steps = {}...".format(i))
                if args.taylor in ['param_mix', 'param_second']:
                    for j in range(args.num_examples):
                        batch_input = example_prompts[j].unsqueeze(0)
                        loss = model(batch_input, labels=batch_input).loss
                        logger.log("Loss = {}".format(loss))
                        loss.backward()

                        for module_param in model.parameters():
                            module_param.grad = module_param.grad * module_param.grad / args.num_examples
                            if hasattr(module_param, 'acc_grad'):
                                module_param.acc_grad += module_param.grad
                            else:
                                module_param.acc_grad = copy.deepcopy(module_param.grad)
                        model.zero_grad()
                        del loss.grad
                    
                loss = model(example_prompts, labels=example_prompts).loss
                logger.log("Loss = {}".format(loss))
                loss.backward()

            pruner.step()

            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log("After Iter {}/{}, #parameters: {}".format(i+1, args.iterative_steps, after_pruning_parameters))

            # Store the pruning history for the attention head
            buffer = None
            if '13B' in args.base_model:
                buffer = {}
                for module in pruner.pruning_history():
                    if 'W_pack' in module[0]:
                        layer_idx = int(module[0].split('.')[2])
                        head_dim = model.model.layers[layer_idx].self_attn.head_dim
                        num_head = model.model.layers[layer_idx].self_attn.num_heads

                        sort_idx = sorted(module[-1])
                        sort_idx = sort_idx[:len(sort_idx)//3:head_dim]
                        head_idx = [idx // head_dim for idx in sort_idx]
                        buffer[layer_idx] = [i for i in range(num_head) if i not in head_idx]

            # modify inferece-related attributes
            for layer in model.model.layers:
                layer.self_attn.hidden_size = layer.self_attn.W_pack.weight.shape[0] // 3
                layer.self_attn.num_heads = layer.self_attn.hidden_size // layer.self_attn.head_dim

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if 'weight' in name:
                module.grad = None

        del pruner
    elif args.layer_wise:
        model.model.layers = model.model.layers[:args.layer]
        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        raise NotImplementedError
    logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))
    
    gc.collect()
    torch.cuda.empty_cache()

    if args.save_model:
        torch.save({
            'model': model, 
            'tokenizer': tokenizer,
            'buffer': buffer
        }, logger.best_checkpoint_path)
    
    if args.eval_device != "cpu":
        model.half()
    model.to(args.eval_device)

    # pass the pruning idx for head to model
    model.model.mask_head_idx = buffer

    if args.test_after_train:
        logger.log("\n==================Generation Results After Pruning================\n")
        
        model.eval()
        with torch.no_grad():
            inputs = tokenizer('登鹳雀楼->王之涣\n夜雨寄北->', return_tensors='pt').to(args.eval_device)
            pred = model.generate(**inputs, max_new_tokens=64,repetition_penalty=1.1)
            response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
            logger.log(response)
        
        logger.log("\n==================Finish================\n")
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune", help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='pruning ratio')
    parser.add_argument('--pruner_type', type=str, default='l2', help='pruner type')

    # argument for generation
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')

    # argument for layer-wise pruning/column-wise pruning
    parser.add_argument('--channel_wise', action='store_true', help='channel wise')
    parser.add_argument('--block_wise', action='store_true', help='block wise')
    parser.add_argument('--layer_wise', action='store_true', help='layer wise')
    parser.add_argument('--layer', type=int, default=12, help='remain the previous n layers')

    parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers', default=3)
    parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=31)
    parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=3)
    parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=31)

    parser.add_argument('--iterative_steps', type=int, default=1, help="Iteration step for pruning. Default=1")
    parser.add_argument('--grouping_strategy', type=str, default='sum', help='Reduce method for grouping')
    parser.add_argument('--global_pruning', action='store_true', help='whether global pruning')
    parser.add_argument('--taylor', type=str, default='param_first', help='choose from [vectorize, param_second, param_first, param_mix]')
    parser.add_argument('--num_examples', type=int, default=10)

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--test_before_train', action='store_true', help='whether test after train')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--test_after_train', action='store_true', help='whether test before train')

    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_true', help='if save model')
    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
