import os
import sys
import argparse
import gradio as gr
import torch
import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from LLMPruner.peft import PeftModel

#from utils.callbacks import Iteratorize, Stream
#from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
torch_version = int(torch.__version__.split('.')[1])

def main(args):
    if args.model_type == 'pretrain':
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            low_cpu_mem_usage=True if torch_version >=9 else False
        )
        description = "Model Type: {}\n Base Model: {}".format(args.model_type, args.base_model)
    elif args.model_type == 'pruneLLM':
        pruned_dict = torch.load(args.ckpt, map_location='cpu')
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
        description = "Model Type: {}\n Pruned Model: {}".format(args.model_type, args.ckpt)
    elif args.model_type == 'tune_prune_LLM':
        pruned_dict = torch.load(args.ckpt, map_location='cpu')
        tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
        model = PeftModel.from_pretrained(
            model,
            args.lora_ckpt,
            torch_dtype=torch.float16,
        )
        description = "Model Type: {}\n Pruned Model: {}\n LORA ckpt: {}".format(args.model_type, args.ckpt, args.lora_ckpt)
    else:
        raise NotImplementedError

    if device == "cuda":
        model.half()
        model = model.cuda()
    
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()

    def evaluate(
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        inputs = tokenizer(input, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                do_sample=True,
                top_k=50,
                top_p=top_p,
                temperature=temperature,
                max_length=max_new_tokens,
                return_dict_in_generate=True,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield output

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(
                minimum=0, maximum=1, value=1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.95, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=50, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
            gr.components.Checkbox(label="Stream output"),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="Evaluate Pruned Model",
        description=description,
    ).queue().launch(server_name="0.0.0.0", share=args.share_gradio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLaMA (huggingface version)')

    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--model_type', type=str, required=True, help = 'choose from ')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lora_ckpt', type=str, default=None)
    parser.add_argument('--share_gradio', action='store_true')

    args = parser.parse_args()
    main(args)


