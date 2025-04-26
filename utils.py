import torch
from baukit import TraceDict


def format_tqa_DRC(question, choice):
    return f"### Instruction:\n请你对下面的语句作出回复：\n\n### Input:\n{question}\n\n### Response:\n以下是我对该语句的回复：\n{choice}"

def format_tqa_Shakespeare(question, choice):
    return f"Please respond to the following statement, and do not output any unnecessary content: \n{question}\nOkay, my answer is as follows:\n{choice}"

def prepare_tqa_dataset(dataset, tokenizer, format_func): 
    """tokenize truthfulqa dataset with format_func, return prefix tokens, source tokens, target tokens"""
    prefix_tokens = []  # including qa template and question
    source_tokens = []  # prefix tokens + source style tokens
    target_tokens = []  # prefix tokens + target style tokens
    for qa_pair in dataset: 
        question = qa_pair['question']
        prefix = format_func(question, "")
        prefix_tokens.append(prefix)
        for answer in qa_pair['correct_answers']: 
            complate_qa = format_func(question, answer)
            complate_qa_tokens = tokenizer(complate_qa, return_tensors = 'pt').input_ids
            target_tokens.append(complate_qa_tokens)
        for answer in qa_pair['incorrect_answers']: 
            complate_qa = format_func(question, answer)
            complate_qa_tokens = tokenizer(complate_qa, return_tensors = 'pt').input_ids
            source_tokens.append(complate_qa_tokens)
    return prefix_tokens, source_tokens, target_tokens

def get_activations(model, tokens, use_cache=False, past_key_values=None): 
    """get activations of all heads in all layers"""
    ckpts = [f"model.layers.{i}.self_attn.head_out"
             for i in range(model.config.num_hidden_layers)]
    if past_key_values is not None:
        assert use_cache, "if use past_key_values, use_cache should be True"
    with torch.no_grad():
        tokens = tokens.to(model.device)
        with TraceDict(model, ckpts) as ret:
            outputs = model(
                tokens,
                use_cache=use_cache,
                past_key_values=past_key_values
            )
        new_past_key_values = outputs.past_key_values if use_cache else None

        head_wise_hidden_states = [ret[ckpt].output.detach() for ckpt in ckpts]
        # squeeze to remove the batch dimension if batch size is 1
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0).squeeze(dim=1)
    # head_wise_hidden_states: [num_layers, seq_len, num_heads * head_dim]
    # new_past_key_values: kv cache for next forward
    return head_wise_hidden_states, new_past_key_values
