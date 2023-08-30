# More LLMs

* [BLOOM](#cherry_blossom-bloom)
* [Baichuan](#llama-baichuan)

## :cherry_blossom: BLOOM

* For Pruning:

```
python examples/bloom.py  \
    --base_model YOUR_BLOOM_MODEL \
    --pruning_ratio 0.25 \
    --block_wise \
    --block_mlp_layer_start 4 --block_mlp_layer_end 20 \
    --block_attention_layer_start 4 --block_attention_layer_end 20 \
    --pruner_type taylor \
    --device cuda  --eval_device cuda \
    --test_after_train --test_before_train \
    --save_ckpt_log_name bloom_prune
```

Here, replace `YOUR_BLOOM_MODEL` with the BLOOM model you want to prune. See the full list [here](https://huggingface.co/docs/transformers/model_doc/bloom). For example, for the 7B1 model, replace `YOUR_BLOOM_MODEL` with `bigscience/bloom-7b1`.

## :llama: Baichuan

* For pruning:
1. Baichuan-7B:
```
python examples/baichuan.py \
        --base_model baichuan-inc/Baichuan-7B \
        --pruning_ratio 0.15 \
        --block_wise \
        --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
        --block_attention_layer_start 4 --block_attention_layer_end 30 \
        --pruner_type taylor \
        --test_before_train --test_after_train \
        --device cpu  --eval_device cuda \
        --save_ckpt_log_name baichuan_prune  
```

2. Baichuan-13B:
```
python examples/baichuan.py \
        --base_model baichuan-inc/Baichuan-13B-chat  \
        --pruning_ratio 0.15 \
        --block_wise \
        --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
        --block_attention_layer_start 4 --block_attention_layer_end 30 \
        --pruner_type taylor \
        --test_before_train --test_after_train \
        --device cpu  --eval_device cuda \
        --save_ckpt_log_name baichuan_13b_prune  
```

* For Post-training:
1. Add `model.enable_input_require_grads()` before `model = get_peft_model(model, config)` in `post_training.py` (refer to the [issue](https://github.com/baichuan-inc/Baichuan-13B/issues/14)).
2. Run the following command:
```
export PYTHONPATH='YOUR_PATH_TO_HUGGINGFACE_CACHE/.cache/huggingface/modules:$PYTHONPATH'
python post_training.py \
       --prune_model YOUR_PATH_TO_PRUNED_MODEL  \
       --data_path YOUR_DATA \
       --lora_r 8 --lora_target_modules gate_proj,down_proj,up_proj,W_pack,o_proj \
       --num_epochs 2 --learning_rate 1e-5 --batch_size 64 \
       --output_dir tune_log/baichuan --wandb_project baichuan_tune 
```

The minimum example for pruning Baichuan models is provided in the above instruction. For improved performance, the following modifications are essential:

1. Adjust the calibration data used for calculating the pruning metric ([Line 143](https://github.com/horseee/LLM-Pruner/blob/2d60e00c86d72788a182b505ce42334f42fcb933/examples/baichuan.py#L143) in baichuan.py), such as using dialogues or some Chinese corpus.
2. Try different Chinese corpora during the recovery stage of the pruned model.