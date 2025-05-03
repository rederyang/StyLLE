import os
import json
import argparse
import time
import transformers

from utils import format_tqa_DRC, format_tqa_Shakespeare

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

    # generate
    print("Start generating...")

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
        qa_prefix_tokens = tokenizer(qa_prefix, return_tensors='pt').input_ids.to(model.device)

        tik = time.time()
        response = model.generate(qa_prefix_tokens, max_length=600)
        time_cost = time.time() - tik

        cum_time += time_cost
        cum_token += len(response[0])

        response = response[0][qa_prefix_tokens.shape[1]:]
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
        dict["daiyu_answer"] = answers[i]  # FIXME: to remove
        dict["model_path"] = args.model_dir  # FIXME: to remove
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

    print("Results saved to", args.save_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
