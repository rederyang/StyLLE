import os
import json
import argparse
import torch
import torch.nn.functional as F
import time
import transformers
from einops import rearrange

from utils import format_tqa_DRC, format_tqa_Shakespeare, get_activations

# global variables to share across functions
SRC_ACTIVATIONS = None  # source activations, shape: [num_samples, num_layers, num_heads, head_dim]
TGT_ACTIVATIONS = None  # target activations, shape: [num_samples, num_layers, num_heads, head_dim]
SS_RANK = {}  # style subspace rank
SS_VH = {}  # style subspace Vh
SS_PROJ_SRC_ACT = {}  # style subspace projection of source activations
SS_PROJ_TGT_ACT = {}  # style subspace projection of target activations
SS_PROJ_SRC_MEAN_ACT = {}  # style subspace projection of source activations mean
SS_PROJ_TGT_MEAN_ACT = {}  # style subspace projection of target activations mean
SELECTED_HEADS_BY_LAYER = {}  # layer -> head, selected heads


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--activations_path", type=str, required=True)
    parser.add_argument("--selected_heads_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    # head number
    parser.add_argument("--head_num", type=int, default=64)
    # rank selection
    rank_group = parser.add_mutually_exclusive_group()
    rank_group.add_argument("--rank", type=int)
    rank_group.add_argument("--adaRank", action="store_true")
    parser.add_argument("--var_threshold", type=float, default=None)
    # global style scaling factor
    parser.add_argument("--global_scaling_factor", type=float, default=3.0)
    # KNN
    parser.add_argument("--KNN_neighbor_num", type=int, default=None)
    # acceleration
    parser.add_argument("--generation_method", type=str, choices=["baseline", "fast", "faster"], required=True)
    return parser.parse_args()

def search_rank_BIC(X, U, s, Vh, var_threshold=None):
    N, D = X.shape
    # prevent overflow
    X, U, s, Vh = X.to(torch.float32), U.to(torch.float32), s.to(torch.float32), Vh.to(torch.float32)

    # calculate explained variance
    var_explained = (s**2) / torch.sum(s**2)
    cumulative_var_explained = torch.cumsum(var_explained, dim=0)
    
    # define search range
    r_min, r_max = 1, s.shape[0] - 1
    if var_threshold is not None:
        # find the minimum rank value that satisfies the threshold
        r_min = torch.argmax((cumulative_var_explained >= var_threshold).int()) + 1
    
    # search for the optimal rank
    best_r = r_min
    best_BIC = float('inf')
    best_r_var = cumulative_var_explained[r_min]

    for r in range(r_min, r_max + 1):
        # use the first r singular values and vectors to reconstruct the data
        X_reconstructed = U[:, :r] @ torch.diag(s[:r]) @ Vh[:r, :]
        
        # calculate the mean square error
        MSE = torch.mean((X - X_reconstructed) ** 2)
        
        # calculate the BIC
        # BIC = N * D * log(MSE) + r * (N + D + 1) * log(N * D)
        BIC = N * D * torch.log(MSE) + r * (N + D + 1) * torch.log(torch.tensor(N * D * 1.0))
        
        if BIC < best_BIC:
            best_BIC = BIC
            best_r = r
            best_r_var = cumulative_var_explained[r - 1]

    return best_r, best_BIC, best_r_var

def svd_decomposition(rank=None, adaRank=False, var_threshold=None):
    # either rank or adaRank
    assert rank is not None or adaRank
    if adaRank:
        print("adaRank with var_threshold =", var_threshold)
    else:
        print("rank =", rank)

    global SS_RANK, SS_VH, SS_PROJ_SRC_ACT, SS_PROJ_TGT_ACT, SS_PROJ_SRC_MEAN_ACT, SS_PROJ_TGT_MEAN_ACT

    for layer_idx in SELECTED_HEADS_BY_LAYER:
        for head_idx in SELECTED_HEADS_BY_LAYER[layer_idx]:
            src_activations = SRC_ACTIVATIONS[:, layer_idx, head_idx, :]
            tgt_activations = TGT_ACTIVATIONS[:, layer_idx, head_idx, :]

            # SVD
            delta_activations = tgt_activations - src_activations
            U, s, Vh = torch.linalg.svd(delta_activations.float(), full_matrices=False)
            U, s, Vh = U.to(src_activations.dtype), s.to(src_activations.dtype), Vh.to(src_activations.dtype)

            # projection of activations in the style subspace
            proj_src_activations = torch.matmul(src_activations, Vh.T)
            proj_tgt_activations = torch.matmul(tgt_activations, Vh.T)

            if adaRank:
                rank, BIC, r_var = search_rank_BIC(delta_activations, U, s, Vh, var_threshold)
                print(f"Layer {layer_idx}, head {head_idx}, rank: {rank}, BIC: {BIC}, Var explained: {r_var}")

            SS_RANK[(layer_idx, head_idx)] = rank
            SS_VH[(layer_idx, head_idx)] = Vh
            SS_PROJ_SRC_ACT[(layer_idx, head_idx)] = proj_src_activations
            SS_PROJ_TGT_ACT[(layer_idx, head_idx)] = proj_tgt_activations
            SS_PROJ_SRC_MEAN_ACT[(layer_idx, head_idx)] = torch.mean(proj_src_activations, dim=0)
            SS_PROJ_TGT_MEAN_ACT[(layer_idx, head_idx)] = torch.mean(proj_tgt_activations, dim=0)

def get_steering_vector(layer_idx, head_idx, cur_activations, global_scaling_factor=3.0, KNN_neighbor_num=None):
    # read from global variables
    rank = SS_RANK[(layer_idx, head_idx)]
    Vh = SS_VH[(layer_idx, head_idx)][:rank, :]
    proj_cur_act = torch.matmul(Vh, cur_activations)

    if KNN_neighbor_num is None:  # global mean
        proj_src_mean_act = SS_PROJ_SRC_MEAN_ACT[(layer_idx, head_idx)][:rank]
        proj_tgt_mean_act = SS_PROJ_TGT_MEAN_ACT[(layer_idx, head_idx)][:rank]
    else:  # use KNN to get local mean
        proj_src_all_act = SS_PROJ_SRC_ACT[(layer_idx, head_idx)][:, :rank]
        proj_tgt_all_act = SS_PROJ_TGT_ACT[(layer_idx, head_idx)][:, :rank]
        dist = torch.norm(proj_cur_act[None, ...] - proj_src_all_act, dim=1)
        knn_dist, knn_idx = torch.topk(dist, KNN_neighbor_num, largest=False)
        proj_src_mean_act = torch.mean(proj_src_all_act[knn_idx], dim=0)  # KNN mean
        proj_tgt_mean_act = torch.mean(proj_tgt_all_act[knn_idx], dim=0)  # KNN mean

    # base strength, determined by the dataset
    base_strength = proj_tgt_mean_act - proj_src_mean_act

    # diff strength, determined by the current activations
    diff_strength = proj_tgt_mean_act - proj_cur_act

    # combine and apply global scaling factor
    strength = global_scaling_factor * base_strength * (1 + 0.5 * torch.sign(base_strength) * diff_strength)

    steering_vector = torch.matmul(Vh.T, strength)

    return steering_vector

def edit_model_bias(model, cur_activations, **kwargs):
    """
    model: model to edit
    cur_activations: torch tensor, shape: (num_layers, seq_len, num_heads * head_dim)
    layer_head_dict: dict, key is layer_idx, value is a list of head_idx
    """
    cur_activations = rearrange(cur_activations, 'l s (h d) -> l s h d', h=model.config.num_attention_heads)

    for layer_idx, head_idx_list in SELECTED_HEADS_BY_LAYER.items():
        displacement = torch.zeros((int(model.config.num_attention_heads), int(model.config.hidden_size / model.config.num_attention_heads)),
                                   device=model.device, dtype=model.dtype)
        for head_idx in head_idx_list:
            cur_head_activations = cur_activations[layer_idx, -1, head_idx]  # vector of shape (head_dim,)
            steering_vector = get_steering_vector(layer_idx, head_idx, cur_head_activations, **kwargs)
            displacement[head_idx] = steering_vector

        displacement = rearrange(displacement, 'h d -> (h d)')
        bias_tobe = F.linear(displacement, model.model.layers[layer_idx].self_attn.o_proj.weight)
        model.model.layers[layer_idx].self_attn.o_proj.bias = torch.nn.parameter.Parameter(bias_tobe)

def reset_model(model):
    """reset model bias to 0"""
    for layer_idx, heads in SELECTED_HEADS_BY_LAYER.items():
        zero_bias = torch.zeros(model.config.hidden_size, dtype=model.dtype, device=model.device)
        model.model.layers[layer_idx].self_attn.o_proj.bias = torch.nn.parameter.Parameter(zero_bias)

def generate(model, question_tokens, qa_prefix_tokens, max_length=600, **kwargs):

    tokens_without_template = question_tokens
    tokens_with_template = qa_prefix_tokens
    answer_token_ids = []

    for _ in range(max_length):
        with torch.no_grad():
            # edit
            reset_model(model)
            cur_activations, _ = get_activations(model, tokens_without_template)
            edit_model_bias(model, cur_activations, **kwargs)

            # predict next token
            outputs = model(tokens_with_template)
            next_token_logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            token = torch.tensor([probs.argmax().item()]).unsqueeze(0).to(model.device)

            # update tokens
            tokens_without_template = torch.cat((tokens_without_template, token), dim=1)
            tokens_with_template = torch.cat((tokens_with_template, token), dim=1)

            # collect answer token ids
            answer_token_ids.append(token.cpu().numpy()[0][0])

            if token.cpu().numpy()[0][0] == 151643 or token.cpu().numpy()[0][0] == 151644 or token.cpu().numpy()[0][0] == 151645:
                break

    return answer_token_ids

def generate_fast(model, question_tokens, qa_prefix_tokens, max_length=600, **kwargs):
    """Use kv cache for base forward, but not for style forward. This implementation completely follows the
    implementation of the original paper, just using kv cache for acceleration."""
    tokens_without_template = question_tokens
    tokens_with_template = qa_prefix_tokens
    answer_token_ids = []

    past_kv_base = None

    for _ in range(max_length):
        with torch.no_grad():
            # edit
            reset_model(model)
            cur_activations, past_kv_base = get_activations(model,
                                                            tokens_without_template if past_kv_base is None else token,
                                                            use_cache=True,
                                                            past_key_values=past_kv_base)
            edit_model_bias(model, cur_activations, **kwargs)

            # predict next token, recalculate from the first token
            outputs = model(tokens_with_template)

            # determine next token
            next_token_logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            token = torch.tensor([probs.argmax().item()]).unsqueeze(0).to(model.device)

            # update tokens
            tokens_without_template = torch.cat((tokens_without_template, token), dim=1)
            tokens_with_template = torch.cat((tokens_with_template, token), dim=1)

            # collect answer token ids
            answer_token_ids.append(token.cpu().numpy()[0][0])

            if token.cpu().numpy()[0][0] == 151643 or token.cpu().numpy()[0][0] == 151644 or token.cpu().numpy()[0][0] == 151645:
                break

    return answer_token_ids

def generate_faster(model, question_tokens, qa_prefix_tokens, max_length=600, **kwargs):
    """Use kv cache for base forward and style forward. This implementation slightly relaxed the constraint of the original paper.
    But yields similar performance with better speed."""
    tokens_without_template = question_tokens
    tokens_with_template = qa_prefix_tokens
    answer_token_ids = []

    past_kv_base = None
    past_kv_style = None

    for _ in range(max_length):
        with torch.no_grad():
            # edit
            reset_model(model)
            cur_activations, past_kv_base = get_activations(model,
                                                            tokens_without_template if past_kv_base is None else token,
                                                            use_cache=True,
                                                            past_key_values=past_kv_base)
            edit_model_bias(model, cur_activations, **kwargs)

            # predict next token with kv cache
            # Note that past_kv_style here is computed by model variants
            # that have been modified in previous steps, meaning the kv cache
            # of each token is calculated with a different bias!
            outputs = model(tokens_with_template if past_kv_style is None else token,
                            use_cache=True,
                            past_key_values=past_kv_style)
            past_kv_style = outputs.past_key_values

            # determine next token
            next_token_logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            token = torch.tensor([probs.argmax().item()]).unsqueeze(0).to(model.device)

            # update tokens
            tokens_without_template = torch.cat((tokens_without_template, token), dim=1)
            tokens_with_template = torch.cat((tokens_with_template, token), dim=1)

            # collect answer token ids
            answer_token_ids.append(token.cpu().numpy()[0][0])

            if token.cpu().numpy()[0][0] == 151643 or token.cpu().numpy()[0][0] == 151644 or token.cpu().numpy()[0][0] == 151645:
                break

    return answer_token_ids

def main(args):
    # load model
    print("Loading model...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model,
                                                             low_cpu_mem_usage=True,
                                                             torch_dtype="auto",
                                                             device_map="auto")
    model.config.oproj_bias = True

    # load dataset
    print("Loading dataset...")
    if args.dataset == "DRC":
        with open("dataset/Valid_DRC.json", 'r', encoding='utf-8') as file:
            dataset = json.load(file)
            format_func = format_tqa_DRC
    elif args.dataset == "Shakespeare":
        with open("dataset/Valid_Shakespeare.json", 'r', encoding='utf-8') as file:
            dataset = json.load(file)
            format_func = format_tqa_Shakespeare

    # load activations
    global SRC_ACTIVATIONS, TGT_ACTIVATIONS
    print("Loading activations...")
    activations = torch.load(args.activations_path)
    source_activations = activations["source_activations"].to(model.device).to(model.dtype)
    SRC_ACTIVATIONS = rearrange(source_activations, 'b l (h d) -> b l h d', h=model.config.num_attention_heads)
    target_activations = activations["target_activations"].to(model.device).to(model.dtype)
    TGT_ACTIVATIONS = rearrange(target_activations, 'b l (h d) -> b l h d', h=model.config.num_attention_heads)

    # load selected heads
    global SELECTED_HEADS_BY_LAYER
    print("Loading selected heads...")
    with open(args.selected_heads_path, 'r', encoding='utf-8') as file:
        selected_heads = json.load(file)
    # constrain the number of editing heads
    selected_heads = selected_heads[:min(len(selected_heads), args.head_num)]
    # group by layer for convenience and efficiency
    for layer_idx, head_idx in selected_heads:
        if layer_idx not in SELECTED_HEADS_BY_LAYER:
            SELECTED_HEADS_BY_LAYER[layer_idx] = []
        SELECTED_HEADS_BY_LAYER[layer_idx].append(head_idx)

    # determine the style subspace
    print("Determining the style subspace...")
    svd_decomposition(rank=args.rank, adaRank=args.adaRank, var_threshold=args.var_threshold)

    # generate
    print("Start generating...")

    if args.generation_method == "baseline":
        generate_method = generate
    elif args.generation_method == "fast":
        generate_method = generate_fast
    elif args.generation_method == "faster":
        generate_method = generate_faster

    print(f"Generation method: {args.generation_method}")

    model.eval()

    cum_time = 0
    cum_token = 0
    answers = []
    for index, sample in enumerate(dataset):

        # question is the the questions itself, qa_prefix is the question with the qa template
        question = sample["question"]
        qa_prefix = format_func(question, "")
        if index == 0:
            print("sanity check:")
            print(f"___question___")
            print(question)
            print(f"___qa_prefix___")
            print(qa_prefix)
        question_tokens = tokenizer(question, return_tensors='pt').input_ids.to(model.device)
        qa_prefix_tokens = tokenizer(qa_prefix, return_tensors='pt').input_ids.to(model.device)

        tik = time.time()
        response = generate_method(model,
                                   question_tokens,
                                   qa_prefix_tokens,
                                   global_scaling_factor=args.global_scaling_factor,
                                   KNN_neighbor_num=args.KNN_neighbor_num)
        time_cost = time.time() - tik

        cum_time += time_cost
        cum_token += len(response)

        answer = tokenizer.decode(response, skip_special_tokens=True)
        print(index, answer)
        answers.append(answer)

    print(f"Question number: {len(dataset)}\n"
        f"Cumulative token number: {cum_token}\n"
        f"Time cost: {cum_time} s\n"
        f"Average generation speed: {cum_token/cum_time} token/s")

    # save results
    print("Saving results...")
    output_data = []
    for i in range(len(answers)):
        dict = {}
        dict["question"] = dataset[i]["question"]
        dict["answer"] = answers[i]
        output_data.append(dict)

    os.makedirs(args.save_dir, exist_ok=True)

    # save generation results
    save_name = args.save_dir.split("/")[-1]
    output_path = os.path.join(args.save_dir, save_name + "_results.json")
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, ensure_ascii=False, indent=4)

    # save speed
    output_path = os.path.join(args.save_dir, "speed.txt")
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(f"Question number: {len(dataset)}\n"
        f"Cumulative token number: {cum_token}\n"
        f"Time cost: {cum_time} s\n"
        f"Average generation speed: {cum_token/cum_time} token/s")

    # save rank
    output_path = os.path.join(args.save_dir, "rank.json")
    with open(output_path, 'w', encoding='utf-8') as file:
        ranks = {f"{layer_idx}_{head_idx}": rank 
                 for (layer_idx, head_idx), rank in SS_RANK.items()}
        json.dump(ranks, file, ensure_ascii=False, indent=4)

    print("Results saved to", args.save_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
