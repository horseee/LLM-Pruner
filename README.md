<p align="center">
<img src="figures/logo.png" width="20%"> <br>
</p>

<div align="center">
<h1>LLM-Pruner</h1>
<h3>On the Structural Pruning of Large Language Models<h3>
:llama: :llama: :llama: :llama: :llama: Compress your LLMs to any size! :llama: :llama: :llama: :llama: :llama:
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
* **Automatic structural pruning**: Pruning new LLMs with minimal human efforts (In progress).

    
## Features
**Supported Models:**
- [x] LLaMA-7B:  the HuggingFace Version
- [x] Vicuna-7B: Official Version

**Release Soon:** 
- [ ] Code for the Official version LLaMA-7B
- [ ] Code for ChatGLM
- [ ] Code for post-training
- [ ] The tutorial of customizing the LLM-Pruner for new model: If you want to use it in your models, please try to follow this instruction


    
    
## Instructions
    
It takes three steps to prune an LLM:
* <u>Discovery Stage</u>: Discover the complicated inter-dependency in LLMs and find the minimally-removable unit, **group**.
* <u>Estimation Stage</u>: Estimating the contribution of each group to the overall performance of the model and deciding which group to be pruned. 
* <u>Recover Stage</u>: Fast post-training to recover model performance.

After pruning and post-training, we follow <a href="https://github.com/EleutherAI/lm-evaluation-harness">lm-evaluation-harness</a> for evaluation.

### 1. Installation
```
pip install -r requirement.txt
```

### 2. Minimal Example
```
bash script/llama_prune.sh
```
This script would compress the LLaMA-7B model with 20\% parameters pruned. All the pre-trained models, the dataset would be automatically downloaded, so you do not need to manually download the resource. After the model pruned and post-trained, the compressed model and its 

### 3. Pruning (Discovery Stage + Estimation Stage)
    
#### 3.1 LLaMA-7B pruning with ~20% parameters pruned:
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
- **Pruning Strategy:** Block-wise, Channel-wise, Layer-wise Pruning: place {--block_wise}/{--channel_wise}/{--layer_wise --layer your_desired_layer_size}. If you use Block-wise, please specify the start and end layer paticipate in pruning. If you channel-wise, no extra argument is needed. If you use layer-wise, please specify `--layer YOUR_LAYER_SIZE` 
- **Importance Criterion:** l1, l2, random, taylor. Use the argument --pruner_type to specify the pruner. If you use the taylor pruner, than you have the following four choice: `vectorize, param_second, param_first, param_mix`. The `param_mix` is used by default (containing both the approximated second-order hessian and first-order gradient). If you use l1, l2 or random, no extra arguments need to be specified.
- **Pruning Ratio**: The pruning ratio of groups. It is different from the **pruning rate of parameters** as we remove groups as the minimal units. 
- **device and eval_device**: Pruning and evluation can be done on different devices. Taylor-based methods requires backward during pruning, which may requires huge GPU RAMs. Our implementation uses cpu for importance estimation. Similarly, eval_device is used to test the pruned model.

#### 3.2 Vicuna Pruning
If you want to try Vicuna, please specify the argument `--base_model` to the path to vicuna weight. Please follow <a href="https://github.com/lm-sys/FastChat">https://github.com/lm-sys/FastChat</a> to get Vicuna weights.
    
#### 3.3 ChatGLM Pruning
Comming Soon...
    
### 4. Post-Training (Recover Stage)

### 5. Generation

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

## Zero-shot Evaluation Results
A brief quantitative results of LLM-Pruner of LLaMA-7B:

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
