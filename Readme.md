<p align="center">
<img src="figures/logo.png" width="15%"> <br>
</p>

<div align="center">
<h1>LLM-Pruner</h1>
<h3>On the structural pruning of Large Language Models<h3>
</div>
    
Use our LLM-Pruner to customize and compress your own LLM in any size! 

The paper will be released really soon!


## Introduction

Structural Pruning offers a potential solution to this issue by removing parameters from models. To this end, this project aims to build a straightforward and general pipeline for the pruning of LLaMA and other LLMs.

The advantage of the LLM-Pruner is: 
* **Task-agnostic compression**. The compressed language model retains its ability to
serve as a multi-task solver. 
* **No need for downloading the training corpus of the LLM**. Reduced demand for the original training corpus, where temporarily, we use only 50k publicly available samples (Alpaca).  Thus, we can achieve quick compression, where the compression process ends up in three hours (3 minutes on pruning and 3 hours on tuning).
* **An automatic structural pruning framework.** We hope that this pruning framework can be used to various LLMs with minimal effort to write the code for finding the coupled pruning structure and estimating the importance. We are still working on this, and we will give an tutorial on how to quickly extend this framework to a new LLM.

Here we show an example on LLaMA about the automatically detected coupled structures and the generated sentences under the same prompt.
<p align="center">
<img src="figures/LLaMA_example.png" width="100%"> <br>
</p>

**Available Models:**
- [x] LLaMA-7B: the HuggingFace Version
- [x] Vicuna-7B


**Features will come out soon:**
- [] 
- []



## Instruction



## Quantitative Results
The quantitative results of LLaMA-7B is shown in the below table. More results can be found in the paper.

| Pruning Ratio        | Method                   | WikiText2 | PTB | BoolQ | PIQA  | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA  | Average |
|----------------------|--------------------------|-----------------------------------|-----------------------------|-------|-------|-----------|------------|-------|-------|-------|---------|
| Ratio = 0            | LLaMA-7B                 | -                                 | -                           | 76.5  | 79.8  | 76.1      | 70.1       | 72.8  | 47.6  | 57.2  | 68.59   |
|                      | LLaMA-7B*                | 12.62                             | 22.14                       | 73.18 | 78.35 | 72.99     | 67.01      | 67.45 | 41.38 | 42.40 | 63.25   |
| Ratio = 20% w/o tune | l2                       | 582.41                            | 1022.17                     | 59.66 | 58.00 | 37.04     | 52.41      | 33.12 | 28.58 | 29.80 | 42.65   |
|                      | random                   | 27.51                             | 43.19                       | 61.83 | 71.33 | 56.26     | 54.46      | 57.07 | 32.85 | 35.00 | 52.69   |
|                      | Channel Group            | 74.63                             | 153.75                      | 62.75 | 62.73 | 41.40     | 51.07      | 41.38 | 27.90 | 30.40 | 45.38   |
|                      | $\text{Parameter}^{1+2}$ | 19.24                             | 34.09                       | 62.54 | 75.41 | 65.99     | 60.30      | 61.57 | 36.69 | 39.20 | 57.39   |
| Ratio = 20% w/ tune  | Channel Group            | 22.02                             | 38.67                       | 59.08 | 73.39 | 64.02     | 60.54      | 57.95 | 35.58 | 38.40 | 55.57   |
|                      | Weight                   | 18.84                             | 33.05                       | 65.75 | 74.70 | 64.52     | 59.35      | 60.65 | 36.26 | 39.40 | 57.23   |
|                      | $\text{Parameter}^{1+2}$ | 17.39                             | 30.20                       | 66.79 | 77.58 | 68.48     | 64.96      | 64.06 | 37.88 | 39.00 | 59.82   |


## More Examples

| Original Model | Output | Pruned Model |  Output |
|--|--|--|--| 
| LLaMa-7B | <u>The universe is the entirety of space, time, matter, and energy that exists.</u> It includes all matter and energy, whether visible or not, and in fact the whole universe is invisible. There are many theories regarding its origin and ultimate fate... | LLaMa-5.4B | <u>The universe is the entirety of space, time, matter, and energy that exists.</u> The laws that make up physics — electrons orbiting atoms, magnets attracting other objects, planets orbiting the Sun—have been consistent over ...|
| Vicuna-7B | <u>10 steps to build an ios app:</u> 1. Research and develop the idea. 2. Decide on the platform you want to use (iOS, Android, or both). 3. Create a plan and budget. 4. Hire a team of developers. 5. Design the user interface (UI). 6. Develop the user interface (UI). 7. Test and iterate on the app. 8. Submit the app to the app store. 9. Optimize the app for performance and user experience (UX). 10. Market and promote the app. | Vicuna-5.4B | <u>10 steps to build an ios app:</u> 1. Plan your app: decide what it should do, what it will look like, and what it will cost to build it. 2. Hire a developer: A good iOS developer will help you design the user interface, create the app, and make sure it’s bug-free. 3. Create a user interface: This is where the magic happens. Your app’s user interface will be designed and mocked up at this point... |
| ChatGLM-6B | <u>Write a poetry about love and peace.</u> In the midst of life's hustle and bustle, I find your love a soothing peace. It is a feeling that cannot be conveyed, A connection that cannot be distanced. With your smile, your eyes, your embrace, I feel a sense of belonging that only you can give. It is a love that lasts forever, A love that cannot be defeat... | LLaMa-5.4B | <u>Write a poetry about love and peace.</u> Love and peace, two such wonderful things\textbackslash\ A pure and endless desire in my heart\textbackslash\ And both of them, I must seek for\textbackslash\ A long, long time, I know..\textbackslash\ Love, I know, is a feeling of being\textbackslash\ A perfect partner, in every sense\textbackslash\ And peace, I need it, so much, one day\textbackslash\ A long, long way, my heart will go..|


## Acknowledgement
* Logo is generated by <a href="https://dreamstudio.ai/generate">Stable Diffusion</a>


## 
