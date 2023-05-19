<p align="center">
<img src="figures/logo.png" width="20%"> <br>
</p>

<div align="center">
<h1>LLM-Pruner</h1>
<h3>On the structural pruning of Large Language Models<h3>
</div>
    
Use our LLM-Pruner to customize and compress your own LLM in any size! We show an example on LLaMA about the automatically detected coupled structures and the generated sentences under the same prompt.
<p align="center">
<img src="figures/LLaMA_example.png" width="100%"> <br>
</p>

The paper will be released really soon!


## Introduction
The advantage of the LLM-Pruner is: 
* **Task-agnostic compression**. The compressed language model retains its ability to
serve as a multi-task solver. 
* **No need for downloading the training corpus of the LLM**. Reduced demand for the original training corpus, where temporarily, we use only 50k publicly available samples (Alpaca).  
* **quick compression**. The compression process ends up in three hours (3 minutes on pruning and 3 hours on tuning).
* **An automatic structural pruning framework.** We hope that this pruning framework can be used to various LLMs with minimal effort to write the code for finding the coupled pruning structure and estimating the importance. We are still working on this, and we will give an tutorial on how to quickly extend this framework to a new LLM.

We are gradually organizing and releasing the code on GitHub. Please refer to the checklist below:
**Supported Models:**
- [x] LLaMA-7B:  the HuggingFace Version
- [x] Vicuna-7B: Official Version

**Features that will come out soon:** 
- [ ] Code for the Official version LLaMA-7B
- [ ] Code for ChatGLM
- [ ] Code for post-training
- [ ] The tutorial of customizing the LLM-Pruner for new model: If you want to use it in your models, please try to follow this instruction

## Instruction


### Quick look
LLM-Pruner consists of three steps: 
*  <u>Discovery Stage</u>. This step focuses on identifying groups of interdependent structures within LLMs. 
* <u>Estimation Stage</u>. Once the coupled structures are grouped, the second step entails estimating the contribution of each group to the overall performance of the model and deciding which group to be pruned. 
* <u>Recover Stage</u>. This step involves fast post-training that alleviates potential
performance degradation caused by the removal of structures

The first two steps will be accomplished in Step One, and the recovery stage will be in Step Two. For the evaluation, we follow <a href="https://github.com/EleutherAI/lm-evaluation-harness">lm-evaluation-harness</a>.

### Install


### Step One:



### Step Two:





## Quantitative Results
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



## Acknowledgement
* Logo is generated by <a href="https://dreamstudio.ai/generate">Stable Diffusion</a>
* The evaluation of the LLM:  <a href="https://github.com/EleutherAI/lm-evaluation-harness">lm-evaluation-harness</a>
* LLaMA: <a href="https://github.com/facebookresearch/llama"> https://github.com/facebookresearch/llama</a>
* Vicuna: <a href="https://github.com/lm-sys/FastChat">https://github.com/lm-sys/FastChat</a>


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