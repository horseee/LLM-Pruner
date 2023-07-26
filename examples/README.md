# More LLMs

* [Baichuan](#llama-baichuan-pruning)
## :llama: Baichuan Pruning

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

The minimum example for pruning Baichuan models are provided in the above instruction. For improved performance, the following modifications are essential:

1. Adjust the calibration data used for calculating the pruning metric (Line 144 in baichuan.py), such as using a dialogue or some Chinese corpus.
2. Try different chinese corpora during the recovery stage of the pruned model.