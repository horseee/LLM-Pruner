<p align="center">
<img src="figures/logo.png" width="20%"> <br>
</p>

<div align="center">
<h1>LLM-Pruner</h1>
<h3>On the Structural Pruning of Large Language Models<h3>
:llama: :llama: :llama: :llama: :llama: Compress your LLMs to any size! :llama: :llama: :llama: :llama: :llama:
</div>
    

<p align="center">
<img width="100%" alt="image" src="https://github.com/horseee/LLM-Pruner/assets/18592211/d68595e6-5688-4602-a662-45885a166a9b">    
<img src="figures/LLaMA_example.png" width="100%"> <br>
</p>


## Why LLM-Pruner
    
* **Task-agnostic compression**: The compressed LLM should retains its original ability as a multi-task solver. 
* **Less training corpus**: In this work, we use only 50k publicly available samples (alpaca) to post-train the LLM.  
* **Efficient compression**: 3 minutes for pruning and 3 hours for post-training. (You can make it longer)
* **Automatic structural pruning**: Pruning new LLMs with minimal human efforts (In progress).

> **LLM-Pruner: On the Structural Pruning of Large Language Models** [[Paper]](https://drive.google.com/file/d/1mJyNkNZphoOw6OUl1caBKN54xflGFIoy/view?usp=share_link)   
> *Xinyin Ma, Gongfan Fang, Xinchao Wang*   
> *National University of Singapore*        
    
The arxiv version will be released soon.

## Features
**Supported Models:**
- [x] LLaMA-7B:  the HuggingFace Version
- [x] Vicuna-7B: Official Version

**Release Soon:** 
- [ ] Code for the Official version LLaMA-7B
- [ ] Code for ChatGLM
- [ ] Code for post-training
- [ ] The tutorial of customizing the LLM-Pruner for new model: If you want to use it in your models, please try to follow this instruction

## Instruction

### QuickStart
Three steps to prune an LLM:
* <u>Discovery Stage</u>: Discover the complicated inter-dependency in LLMs and find the minimally-removable unit, **group**.
* <u>Estimation Stage</u>: Estimating the contribution of each group to the overall performance of the model and deciding which group to be pruned. 
* <u>Recover Stage</u>: Fast post-training to recover model performance.

For the evaluation, we follow <a href="https://github.com/EleutherAI/lm-evaluation-harness">lm-evaluation-harness</a>.

### Installation
```
pip install -r requirement.txt
```

### Pruning (Discovery Stage + Estimation Stage)
    
An example for LLaMA-7B pruning with ~20% parameters pruned:
```
python hf_prune.py --pruning_ratio 0.25 --device cpu  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name llama_prune --pruner_type taylor --test_after_train
```

If you want to prune the Vicuna, please specify the argument `--base_model` to your path for the vicuna (see <a href="https://github.com/lm-sys/FastChat">https://github.com/lm-sys/FastChat</a> for how to get Vicuna weights)

Supported Pruning:
- [x] Block-wise, Channel-wise, Layer-wise Pruning: place {--block_wise}/{--channel_wise}/{--layer_wise --layer your_desired_layer_size}
- [x] Multiple Pruning Strategy: l1, l2, random, taylor. Use the argument --pruner_type to specify the pruner. If you use the taylor pruner, than you have the following four choice: `vectorize, param_second, param_first, param_mix`. The `param_mix` is used by default (containing both the approximated hessian and gradient). 

### Post-Training (Recover Stage)
Release Soon

## Zero-shot Evaluation Results
A brief quantitative results of LLM-Pruner of LLaMA-7B is shown in the below table. More results can be found in the paper.

<p align="center">
<img src="figures/LLaMAResults.png" width="100%"> <br>
</p>


## More Examples

| Original Model | Output | Pruned Model |  Output |
|--|--|--|--| 
| LLaMa-7B | <u>The universe is the entirety of space, time, matter, and energy that exists.</u> It includes all matter and energy, whether visible or not, and in fact the whole universe is invisible. There are many theories regarding its origin and ultimate fate... | LLaMa-5.4B | <u>The universe is the entirety of space, time, matter, and energy that exists.</u> The laws that make up physics — electrons orbiting atoms, magnets attracting other objects, planets orbiting the Sun—have been consistent over ...|
| Vicuna-7B | <u>10 steps to build an ios app:</u> 1. Research and develop the idea. 2. Decide on the platform you want to use (iOS, Android, or both). 3. Create a plan and budget. 4. Hire a team of developers. 5. Design the user interface (UI). 6. Develop the user interface (UI). 7. Test and iterate on the app. 8. Submit the app to the app store. 9. Optimize the app for performance and user experience (UX). 10. Market and promote the app. | Vicuna-5.4B | <u>10 steps to build an ios app:</u> 1. Plan your app: decide what it should do, what it will look like, and what it will cost to build it. 2. Hire a developer: A good iOS developer will help you design the user interface, create the app, and make sure it’s bug-free. 3. Create a user interface: This is where the magic happens. Your app’s user interface will be designed and mocked up at this point... |
| ChatGLM-6B | <u>Write a poetry about love and peace.</u> In the midst of life's hustle and bustle, I find your love a soothing peace. It is a feeling that cannot be conveyed, A connection that cannot be distanced. With your smile, your eyes, your embrace, I feel a sense of belonging that only you can give. It is a love that lasts forever, A love that cannot be defeat... | LLaMa-5.4B | <u>Write a poetry about love and peace.</u> Love and peace, two such wonderful things\ A pure and endless desire in my heart\ And both of them, I must seek for\ A long, long time, I know..\ Love, I know, is a feeling of being\ A perfect partner, in every sense\ And peace, I need it, so much, one day\ A long, long way, my heart will go..|

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
@article{fang2023depgraph,
  title={DepGraph: Towards Any Structural Pruning},
  author={Fang, Gongfan and Ma, Xinyin and Song, Mingli and Mi, Michael Bi and Wang, Xinchao},
  journal={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
