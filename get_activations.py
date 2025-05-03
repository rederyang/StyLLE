import os
import json
import torch
import argparse
import transformers
from tqdm import tqdm

from utils import prepare_tqa_dataset, format_tqa_DRC, format_tqa_Shakespeare, get_activations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    return parser.parse_args()


def main(args):
    # load model
    print("Loading model...")
    config = transformers.AutoConfig.from_pretrained(args.model_dir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_dir,
                                                             low_cpu_mem_usage=True,
                                                             torch_dtype=config.torch_dtype,
                                                             device_map="auto")

    # load data
    if args.dataset == "DRC":
        with open("dataset/Train_DRC.json", 'r', encoding='utf-8') as file:
            dataset = json.load(file)
            format_func = format_tqa_DRC
    elif args.dataset == "Shakespeare": 
        with open("dataset/Train_Shakespeare.json", 'r', encoding='utf-8') as file:
            dataset = json.load(file)
            format_func = format_tqa_Shakespeare
    else: 
        raise ValueError("Invalid dataset name")

    print("Tokenizing dataset...")
    qa_prefix_tokens, source_qa_tokens, target_qa_tokens = \
        prepare_tqa_dataset(dataset, tokenizer, format_func)
    assert len(qa_prefix_tokens) == len(source_qa_tokens) == len(target_qa_tokens)
    print("Number of samples: ", len(qa_prefix_tokens))

    # get activations
    print("Getting activations...")
    source_activations = []
    target_activations = []
    for i in tqdm(range(len(qa_prefix_tokens))):
        src_act, _ = get_activations(model, source_qa_tokens[i])
        src_act = src_act[:, -1, :].cpu().clone()
        tgt_act, _ = get_activations(model, target_qa_tokens[i])
        tgt_act = tgt_act[:, -1, :].cpu().clone()
        source_activations.append(src_act)
        target_activations.append(tgt_act)
    # stack activations, get [num_samples, num_layers, num_heads * head_dim]
    source_activations = torch.stack(source_activations, dim=0)
    target_activations = torch.stack(target_activations, dim=0)

    # save activations
    print("Saving activations...")
    os.makedirs(f'{args.save_dir}', exist_ok=True)
    torch.save({
        "source_activations": source_activations,
        "target_activations": target_activations,
    }, os.path.join(args.save_dir, f'act.pt'))


if __name__ == "__main__":
    args = parse_args()
    main(args)
