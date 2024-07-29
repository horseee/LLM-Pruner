## More Results

The results here are all without post-training. 

### Llama-3.1-Instruct

| Method | Base | L2 | Taylor (param_first) | Base | L2 | Taylor (param_first) | 
| -- |  -- | -- | -- | -- | -- | -- |
| Sequence Length | 2048 | 2048 | 2048 | 128 | 128 | 128 |
| WikiText2 (PPL) |  7.21 | 51.06 | 13.65 | 17.37 | 97.54 | 28.66 |
| PTB  (PPL) | 12.33 | 76.92 | 22.19 | 32.91 | 160.54 | 52.76 |

#Param before: 8030261248, #Param after: 6612586496, Ratio = 82.3458%

### Llama-3.1

| Method | Base | L2 | Taylor (param_first) | Base | L2 | Taylor (param_first) | 
| -- |  -- | -- | -- | -- | -- | -- |
| Sequence Length | 2048 | 2048 | 2048 | 128 | 128 | 128 |
| WikiText2  (PPL) | 6.24 | 49.09 | 12.71 | 14.31 | 88.33 | 25.93 |
| PTB  (PPL) | 10.57 | 64.36 | 20.61 | 27.74 | 148.98 | 47.98 |

#Param before: 8030261248, #Param after: 6612586496, Ratio = 82.3458%

### Llama-3-Instruct

| Method | Base | L2 | Taylor (param_first) | Base | L2 | Taylor (param_first) | 
| -- |  -- | -- | -- | -- | -- | -- |
| Sequence Length | 2048 | 2048 | 2048 | 128 | 128 | 128 |
| WikiText2  (PPL) | 8.28 | 45.63 | 14.50| 19.65 | 101.63 | 30.99 |
| PTB  (PPL) | 14.10 | 78.93 | 24.34 | 35.17 | 188.79 | 53.00 |

#Param before: 8030261248, #Param after: 6612586496, Ratio = 82.3458%

### Llama-3

| Method | Base | L2 | Taylor (param_first) | Base | L2 | Taylor (param_first) | 
| -- |  -- | -- | -- | -- | -- | -- |
| Sequence Length | 2048 | 2048 | 2048 | 128 | 128 | 128 |
| WikiText2  (PPL) | 6.14 | 34.13 | 12.86 | 14.13 | 64.22 | 25.60 |
| PTB  (PPL) | 10.58 | 51.53 | 20.83 | 27.96 | 120.84 | 47.18 |

#Param before: 8030261248, #Param after: 6612586496, Ratio = 82.3458%

### TinyLlama

| Method | Base | L2 | Taylor (param_first) | Base | L2 | Taylor (param_first) | 
| -- |  -- | -- | -- | -- | -- | -- |
| Sequence Length | 2048 | 2048 | 2048 | 128 | 128 | 128 |
| WikiText2  (PPL) | 7.71 | 76.44 | 17.22 | 16.78 | 66.56 | 33.19 | 
| PTB  (PPL) | 24.59 | 288.07 | 71.93 | 61.09 | 177.43 | 113.75 | 

#Param before: 1100048384, #Param after: 945907712, Ratio = 85.9878%

Command:
```
python llama3.py --base_model TinyLlama/TinyLlama_v1.1 \
  --pruning_ratio 0.25 \
  --device cuda --eval_device cuda \
  --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 18 \
  --block_attention_layer_start 4 --block_attention_layer_end 18 \
  --save_ckpt_log_name tinyllama_prune_log \
  --pruner_type taylor  --taylor param_first \
  --save_model  --max_seq_len 2048 \
  --test_before_train --test_after_train
```
