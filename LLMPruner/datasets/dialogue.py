import random
import numpy as np
import torch
from tqdm import tqdm

from datasets import load_dataset
from torch.utils.data.dataset import Dataset
from LLMPruner.models.hf_chatglm import ChatGLMForConditionalGeneration, ChatGLMTokenizer


def load_daily_dialog(tokenizer, n_samples, max_source_len, max_target_len):
    train_datasets = load_dataset('daily_dialog', split='train')
    model_inputs = preprocess_function_train(train_datasets, tokenizer, n_samples, max_source_len, max_target_len)
    return model_inputs

def self_prune_dataset(tokenizer, n_samples, max_source_len, max_target_len):
    train_datasets = [
        {'dialog': ['你好，可以帮我推荐北京一个4.5分以上的景点吗？', '推荐你去北京的故宫博物院，它是中国最重要的博物馆之一，也是世界上规模最大、保存最完整的古代宫殿建筑群之一。故宫博物院内收藏了大量的中国古代文物和艺术品，包括陶瓷、玉器、书画、钟表等，非常值得一游。同时，故宫博物院作为明清两朝的皇宫，还展示了中国古代皇家的生活和文化。']},#'北京有很多值得一看的景点,以下是一些评分较高的4.5分以上的景点:1.北京八达岭长城世博会主题馆(4.5分):位于八达岭长城脚下,通过VR技术展示各种中国展品,包括古代建筑、科技等,非常有趣 2.古北水镇(4.5分):一个以明清古镇为主题的景点,有很多小店铺']},
        {'dialog': ['什么是高斯函数？', '高斯函数，又称为正态分布函数或高斯分布函数，是一种常见的数学函数，通常用来描述自然界中许多现象的分布规律。它的图像呈钟形曲线，以均值为中心，标准差越大曲线越平缓，标准差越小曲线越陡峭。']},
        {'dialog': ['描述一副山水画', '这幅山水画中，天空湛蓝，几朵白云飘浮在高山之间。远处山脉苍翠挺拔，连绵起伏，云雾缭绕其间，令人眼前一片迷离。在画面中央，一座高耸入云的山峰屹立在众山之巅，宛如一座不可逾越的屏障。山脚下是一片绿色的林海，茂密的树木在微风中摇曳生姿，沿着山坡缓缓向下延伸。']}
    ]
    model_inputs = preprocess_function_train(train_datasets, tokenizer, n_samples, max_source_len, max_target_len)
    return model_inputs


def preprocess_function_train(examples, tokenizer, n_samples, max_source_len, max_target_len, if_full_dialogue=False, ignore_pad_token_for_loss=True):
    max_seq_length = max_source_len + max_target_len

    model_inputs = {
        "input_ids": [],
        "labels": [],
    }
    history_column = []
    for i in tqdm(range(len(examples))):
        turn_idx = 0
        history = ''
        while turn_idx <= len(examples[i]['dialog']) - 2:
            query, answer = examples[i]['dialog'][turn_idx], examples[i]['dialog'][turn_idx+1]

            if len(history) == 0:
                prompt = query
            else:
                prompt = history
                prompt += "[Round {}]\n问：{}\n答：".format(turn_idx+1, query)

            a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

            if len(a_ids) > max_source_len - 1:
                a_ids = a_ids[: max_source_len - 1]

            if len(b_ids) > max_target_len - 2:
                b_ids = b_ids[: max_target_len - 2]

            input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

            context_length = input_ids.index(tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position+1:]
            
            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            if ignore_pad_token_for_loss:
                labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)

            if if_full_dialogue:
                history += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, query, answer)
                turn_idx += 1
            else:
                break

        if len(model_inputs['input_ids']) == n_samples:
            break
    return model_inputs

def load_dialog_dataset(dataset_name, tokenizer, n_samples, max_source_len, max_target_len, device="cuda"):
    if dataset_name == 'daily_dialog':
        model_inputs = load_daily_dialog(tokenizer, n_samples, max_source_len, max_target_len)
    elif dataset_name == 'selfprune':
        model_inputs = self_prune_dataset(tokenizer, n_samples, max_source_len, max_target_len)
    else:
        raise NotImplementedError

    model_inputs['input_ids'] = torch.tensor(model_inputs['input_ids']).to(device)
    model_inputs['labels'] = torch.tensor(model_inputs['labels']).to(device)
    print(model_inputs['input_ids'].shape, model_inputs['labels'].shape)
    return model_inputs

def print_dataset_example(example):
    for i in range(len(example["input_ids"])):
        print("input_ids:",example["input_ids"][i])
        print("inputs:", tokenizer.decode(example["input_ids"][i]))
        print("label_ids:", example["labels"][i])
        print("labels:", tokenizer.decode(example["labels"][i]))
        print()

if __name__ == '__main__':
    tokenizer = ChatGLMTokenizer.from_pretrained('THUDM/chatglm-6b')
    inputs = load_dialog_dataset('selfprune', tokenizer, 
                n_samples = 10, max_source_len =30, max_target_len = 20)

    print_dataset_example(inputs)