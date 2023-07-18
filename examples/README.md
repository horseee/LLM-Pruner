# More LLMs

* [Baichuan](#llama-baichuan-pruning)
## :llama: Baichuan Pruning

For Baichuan-7B:
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

for Baichuan-13B:
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
Note: For `baichuan-inc/Baichuan-13B-chat`, the registered buffer `future_mask` in `BaichuanModel` would cause misalighment in the attention mask in the class `BaichuanAttention`. Please refer to the storing and the reloading process of `Baichuan-13B` in `baichuan.py` to make the model works. 
