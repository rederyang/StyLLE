import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_heads", default=64, type=int)
    parser.add_argument("--val_ratio", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()

def get_top_heads(source_activations, target_activations, model_layers, model_heads, seed, num_to_select):

    idx2layer_head = lambda idx: (idx // model_heads, idx % model_heads)

    # convert everything to numpy
    source_activations = source_activations.numpy()
    target_activations = target_activations.numpy()

    # split train and val set
    train_idxs = np.arange(len(source_activations))
    train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
    val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

    X_train = np.stack([array for i in train_set_idxs for array in [source_activations[i], target_activations[i]]], axis=0)
    y_train = np.array([label for _ in train_set_idxs for label in [0, 1]])
    X_val = np.stack([array for i in val_set_idxs for array in [source_activations[i], target_activations[i]]], axis=0)
    y_val = np.array([label for _ in val_set_idxs for label in [0, 1]])

    # linear probing for each head
    print("Train linear probing for each head...")
    val_accs = []
    for layer in tqdm(range(model_layers)):
        for head in range(model_heads):
            X_train_head = X_train[:, layer, head, :]
            X_val_head = X_val[:, layer, head, :]

            clf = LogisticRegression(random_state=seed, max_iter=1000).fit(X_train_head, y_train)
            y_val_pred = clf.predict(X_val_head)
            val_accs.append(accuracy_score(y_val, y_val_pred))

    # get top heads
    top_head_idxs = np.argsort(val_accs)[::-1][:num_to_select]
    top_heads = [idx2layer_head(int(head_idx)) for head_idx in top_head_idxs]

    # print head performance
    print("Selected heads performance:")
    for head_idx in top_head_idxs:
        layer, head = idx2layer_head(head_idx)
        val_acc = val_accs[head_idx]
        print(f"Layer {layer}, Head {head}: Validation accuracy = {val_acc:.4f}")

    return top_heads

def main(args):
    # set seeds
    print("Setting seeds...")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"seed: {args.seed}")

    # load model config
    print("Loading model config...")
    with open(os.path.join(args.model_dir, "config.json"), "r") as f:
        config = json.load(f)
    MODEL_LAYERS = config["num_hidden_layers"]
    MODEL_HEADS = config["num_attention_heads"]
    print(f"model layers: {MODEL_LAYERS}")
    print(f"model heads: {MODEL_HEADS}")

    # load activations
    print("Loading activations...")
    activations = torch.load(args.activations_path)
    source_activations = rearrange(activations["source_activations"], "b l (h d) -> b l h d", h=MODEL_HEADS)
    target_activations = rearrange(activations["target_activations"], "b l (h d) -> b l h d", h=MODEL_HEADS)
    print(f"source activations shape: {source_activations.shape}")
    print(f"target activations shape: {target_activations.shape}")

    # select heads
    print("Selecting heads...")
    top_heads = get_top_heads(source_activations, target_activations, MODEL_LAYERS, MODEL_HEADS, args.seed, args.num_heads)

    # save top heads
    print("Saving top heads...")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "selected_heads.json"), "w") as f:
        json.dump(top_heads, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)