# More LLMs

* [Baichuan](#llama-baichuan-pruning)
## :llama: Baichuan Pruning

```
python examples/baichuan.py \
        --base_model baichuan-inc/Baichuan-7B \
        --pruning_ratio 0.24 \
        --block_wise \
        --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
        --block_attention_layer_start 4 --block_attention_layer_end 30 \
        --pruner_type taylor \
        --test_before_train --test_after_train \
        --device cpu  --eval_device cuda \
        --save_ckpt_log_name baichuan_prune  
```