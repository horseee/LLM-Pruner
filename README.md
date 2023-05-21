<p align="center">
<img src="figures/logo.png" width="20%"> <br>
</p>

<div align="center">
<h1>LLM-Pruner</h1>
<h3>On the Structural Pruning of Large Language Models<h3>
:llama: :llama: :llama: :llama: :llama: Compress your LLMs to any size! :llama: :llama: :llama: :llama: :llama:
</div>
<div align="center">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache%202.0-4E94CE.svg">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-%3E=v1.8.1-EE4C2C.svg?style=flat-square" alt="PyTorch>=v1.8.1">
  </a>
  <a href="https://github.com/facebookresearch/llama">
    <img src="https://img.shields.io/badge/LLMs-LLaMA-FFB000.svg?style=flat-square" alt="LLaMA">
  </a>
  <a href="https://github.com/vicuna">
    <img src="https://img.shields.io/badge/LLMs-Vicuna-924E7D.svg?style=flat-square" alt="Vicuna">
  </a>
  <a href="https://github.com/chatGLM">
    <img src="https://img.shields.io/badge/LLMs-chatGLM-6082B6.svg?style=flat-square" alt="chatGLM">
  </a>
</div>
  

<p align="center">
<img width="100%" alt="image" src="figures/intro.png">    
<img src="figures/LLaMA_example.png" width="100%"> <br>
</p>


> **LLM-Pruner: On the Structural Pruning of Large Language Models** [[Paper]](https://drive.google.com/file/d/1mJyNkNZphoOw6OUl1caBKN54xflGFIoy/view?usp=share_link)   
> *Xinyin Ma, Gongfan Fang, Xinchao Wang*   
> *National University of Singapore*  

The arxiv version will be released soon.

    
## Why LLM-Pruner
    
* **Task-agnostic compression**: The compressed LLM should retains its original ability as a multi-task solver. 
* **Less training corpus**: In this work, we use only 50k publicly available samples (alpaca) to post-train the LLM.  
* **Efficient compression**: 3 minutes for pruning and 3 hours for post-training. (You can make it longer)
* **Automatic structural pruning**: Pruning new LLMs with minimal human effort (In progress).

#### Supported LLMs:
- [x] [LLaMA-7B HuggingFace](https://huggingface.co/docs/transformers/main/model_doc/llama)
- [x] [Vicuna-7B Official](https://github.com/lm-sys/FastChat)

#### Release Soon:
- [ ] Code for the [Official LLaMA-7B](https://github.com/facebookresearch/llama)
- [ ] Code for [ChatGLM](https://github.com/THUDM/ChatGLM-6B)
- [ ] A tutorial for pruning new LLMs.


## Instructions

### 1. Installation
```
pip install -r requirement.txt
```

### 2. Minimal Example
```
bash script/llama_prune.sh
```
This script would compress the LLaMA-7B model with ～20\% parameters pruned. All the pre-trained models and the dataset would be automatically downloaded, so you do not need to manually download the resource. When running this script for the first time, it will require some time to download the model and the dataset.

    
### 3. Step-by-step Instructions  
    
It takes three steps to prune an LLM:
* <u>Discovery Stage</u>: Discover the complicated inter-dependency in LLMs and find the minimally-removable unit, **group**.
* <u>Estimation Stage</u>: Estimate the contribution of each group to the overall performance of the model and decide which group to prune. 
* <u>Recover Stage</u>: Fast post-training to recover model performance.

After pruning and post-training, we follow <a href="https://github.com/EleutherAI/lm-evaluation-harness">lm-evaluation-harness</a> for evaluation.
    
#### 3.1 Pruning (Discovery Stage + Estimation Stage)
    
:llama: **LLaMA-7B pruning with ~20% parameters pruned:**
```
python hf_prune.py --pruning_ratio 0.25 \
      --block_wise \
      --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
      --block_attention_layer_start 4 --block_attention_layer_end 30 \
      --pruner_type taylor \
      --test_after_train \
      --device cpu  --eval_device cuda \
      --save_ckpt_log_name llama_prune 
```
Arguments:
- ``Pruning Strategy``: Choose between block-wise, channel-wise, or layer-wise pruning using the respective command options: {--block_wise}, {--channel_wise}, {--layer_wise --layer NUMBER_OF_LAYERS}. For block-wise pruning, specify the start and end layers to be pruned. Channel-wise pruning does not require extra arguments. For layer pruning, use --layer NUMBER_OF_LAYERS to specify the desired number of layers to be kept after pruning.
- ``Importance Criterion``: Select from l1, l2, random, or taylor using the --pruner_type argument. For the taylor pruner, choose one of the following options: vectorize, param_second, param_first, param_mix. By default, param_mix is used, which combines approximated second-order hessian and first-order gradient. If using l1, l2, or random, no extra arguments are required.
- ``Pruning Ratio``: Specifies the pruning ratio of groups. It differs from the pruning rate of parameters, as groups are removed as the minimal units.
- ``Device`` and ``Eval_device``: Pruning and evaluation can be performed on different devices. Taylor-based methods require backward computation during pruning, which may require significant GPU RAM. Our implementation uses the CPU for importance estimation. eval_device is used to test the pruned model.

#### :llama: Vicuna Pruning
If you want to try Vicuna, please specify the argument `--base_model` to the path to vicuna weight. Please follow <a href="https://github.com/lm-sys/FastChat">https://github.com/lm-sys/FastChat</a> to get Vicuna weights.
```
python hf_prune.py --pruning_ratio 0.25 \
      --block_wise \
      --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
      --block_attention_layer_start 4 --block_attention_layer_end 30 \
      --pruner_type taylor \
      --test_after_train \
      --device cpu  --eval_device cuda \
      --save_ckpt_log_name llama_prune \
      --base_model PATH_TO_VICUNA_WEIGHTS
```
    
#### :llama: ChatGLM Pruning
Comming Soon...
    
### 3.2. Post-Training (Recover Stage)

```
python post_training.py --prune_model prune_log/PATH_TO_PRUNE_MODEL/pytorch_model.bin \
      --data_path yahma/alpaca-cleaned \
      --lora_r 8 \
      --num_epochs 2 \ 
      --learning_rate 1e-4 \ 
      --batch_size 64 \
      --output_dir tune_log/PATH_TO_SAVE_TUNE_MODEL \ 
      --wandb_project llama_tune
```
Make sure to replace `PATH_TO_PRUNE_MODEL` with the path to the pruned model in step 3.1, and replace `PATH_TO_SAVE_TUNE_MODEL` with the desired location where you want to save the tuned model.


### 3.3. Generation

Geneate texts with pre-trained or pruned models.
    
* LLaMA-7B Pre-trained
```
python generate.py --model_type pretrain
```
* Pruned Model without Post-Training
```
python generate.py --model_type pruneLLM --ckpt <YOUR_MODEL_PATH_FOR_PRUNE_MODEL>
```
* Pruned Model with Post-Training 
```
python generate.py --model_type tune_prune_LLM --ckpt <YOUR_CKPT_PATH_FOR_PRUNE_MODEL> --lora_ckpt <YOUR_CKPT_PATH_FOR_LORA_WEIGHT>
```

### 3.4. Testing MACs, Params and Memory

* Pre-trained
```
python test_speedup.py --model_type pretrain
```
* Pruned Model
```
python test_speedup.py --model_type pruneLLM --ckpt <YOUR_MODEL_PATH_FOR_PRUNE_MODEL>
```

## Zero-shot Evaluation Results
A brief quantitative results of LLM-Pruner for LLaMA-7B:

<p align="center">
<img src="figures/LLaMAResults.png" width="100%"> <br>
</p>
    
The results for Vicuna-7B:
    
<p align="center">
<img src="figures/VicunaResults.png" width="100%"> <br>
</p>
    
The results for ChatGLM-6B:

<p align="center">
<img src="figures/ChatGLMResults.png" width="80%"> <br>
</p>

Statistics for these pruned models:

<p align="center">
<img src="figures/statistic.png" width="50%"> <br>
</p>

More results can be found in the paper. 


## More Examples

| Original Model | Output | Pruned Model |  Output |
|--|--|--|--| 
| LLaMa-7B | <u>The universe is the entirety of space, time, matter, and energy that exists.</u> It includes all matter and energy, whether visible or not, and in fact the whole universe is invisible. There are many theories regarding its origin and ultimate fate... | LLaMa-5.4B | <u>The universe is the entirety of space, time, matter, and energy that exists.</u> The laws that make up physics — electrons orbiting atoms, magnets attracting other objects, planets orbiting the Sun—have been consistent over ...|
| Vicuna-7B | <u>10 steps to build an ios app:</u> 1. Research and develop the idea. 2. Decide on the platform you want to use (iOS, Android, or both). 3. Create a plan and budget. 4. Hire a team of developers. 5. Design the user interface (UI). 6. Develop the user interface (UI). 7. Test and iterate on the app. 8. Submit the app to the app store. 9. Optimize the app for performance and user experience (UX). 10. Market and promote the app. | Vicuna-5.4B | <u>10 steps to build an ios app:</u> 1. Plan your app: decide what it should do, what it will look like, and what it will cost to build it. 2. Hire a developer: A good iOS developer will help you design the user interface, create the app, and make sure it’s bug-free. 3. Create a user interface: This is where the magic happens. Your app’s user interface will be designed and mocked up at this point... |
| ChatGLM-6B | <u>Write a poetry about love and peace.</u> In the midst of life's hustle and bustle, I find your love a soothing peace. It is a feeling that cannot be conveyed, A connection that cannot be distanced. With your smile, your eyes, your embrace, I feel a sense of belonging that only you can give. It is a love that lasts forever, A love that cannot be defeat... | ChatGLM-5.6B | <u>Write a poetry about love and peace.</u> Love and peace, two such wonderful things\ A pure and endless desire in my heart\ And both of them, I must seek for\ A long, long time, I know..\ Love, I know, is a feeling of being\ A perfect partner, in every sense\ And peace, I need it, so much, one day\ A long, long way, my heart will go..|

## Limitations
* Although we only used 50K data and trained for three hours, more data would definitely be better. We are testing on this.
* The current compressed model still has several issues, such as generating repetitive tokens or producing nonsensical sentences. We believe there is significant room for improvement in the quality of the compressed model.
* There are still some models for which we cannot automatically identify the mapping of indexes after concatenation and view operations. Therefore, we need to perform additional manual operations. 


## Acknowledgement
* Logo is generated by <a href="https://dreamstudio.ai/generate">Stable Diffusion</a>
* The evaluation of the LLM:  <a href="https://github.com/EleutherAI/lm-evaluation-harness">lm-evaluation-harness</a>
* LLaMA: <a href="https://github.com/facebookresearch/llama"> https://github.com/facebookresearch/llama</a>
* Vicuna: <a href="https://github.com/lm-sys/FastChat">https://github.com/lm-sys/FastChat</a>
* Peft: <a href="https://github.com/huggingface/peft">https://github.com/huggingface/peft</a>
* Alpaca-lora: <a href="https://github.com/tloen/alpaca-lora">https://github.com/tloen/alpaca-lora</a>

## Citation
If you find this project useful, please cite
```
@article{ma2023llm_pruner,
  title={LLM-Pruner: On the Structural Pruning of Large Language Models},
  author={Ma, Xinyin, Fang, Gongfan and Wang, Xinchao},
  journal={arXiv preprint},
  year={2023}
}
```
```
@article{fang2023depgraph,
  title={DepGraph: Towards Any Structural Pruning},
  author={Fang, Gongfan and Ma, Xinyin and Song, Mingli and Mi, Michael Bi and Wang, Xinchao},
  journal={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
