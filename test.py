from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from tqdm.auto import tqdm
import torch
from train_pco import PROMPT_DICT
import argparse
import json
import os
import pandas as pd
import copy
import torch 


IGNORE_INDEX = -100
pco_list = ["fatigue", "depression", "anxiety", "nausea", "lymphedema"]

prompt_input = {
    k:"Instruction: Based on the input text, does the patient have {pco}?".format(pco=k)+ "\nInput: {input}\nResponse:" for k in pco_list
}

prompt_input["lymphedema"].replace("lymphedema", "lymphedema in arms or legs")

def _tokenize_fn(strings, tokenizer):

        tokenized = tokenizer(
            strings,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        
        input_ids = labels = tokenized.input_ids
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

def preprocess(sources, targets, tokenizer) :
    """Preprocess the data by tokenizing."""
    examples = sources + targets
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def main():
    
    # Create the parser
    parser = argparse.ArgumentParser(description='This is an inference code for PCO extraction model.')

    # Add the arguments
    parser.add_argument('--model_name', required=True, type=str, default="",
                    help='the name of the evaluated model')
    parser.add_argument('--device',  type=str, default="cpu",
                    help='device id')
    parser.add_argument('--dataset_name', required=True, type=str, default="",
                    help='inference file')
    parser.add_argument('--write_data_path', required=True, type=str, default="",
                    help='the path to save prediction')
    parser.add_argument("--llama", action="store_true", help="load llama-2 model")
    parser.add_argument("--lora", action="store_true", help="load llama-2 model trained with lora")
    parser.add_argument("--input_name", default="Text_snippet", type = str)

    # Parse the arguments
    args = parser.parse_args()

    model_name = args.model_name
    input_name = args.input_name

    if args.llama:
        tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = transformers.LlamaForCausalLM.from_pretrained(model_name)
        
    elif args.lora:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,  
            load_in_8bit=True,
            device_map='auto',
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side='left'
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side='left'
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if args.device == "cpu":
            device = "cpu"
    else:
        device = "cuda:{}".format(args.device)
        model = model.to(device)
        

    input_p = args.dataset_name
    original_data = pd.read_csv(input_p)

    pco_answers = {pco:[] for pco in pco_list}
    count_long_input = []

    with torch.no_grad():
        for ind, row in tqdm(original_data.iterrows(), total = len(original_data)):

            for pco, model_input_temp in prompt_input.items():
                model_input = model_input_temp.format(input = row[input_name])
                model_input_yes = preprocess(model_input, "Yes", tokenizer)
                td = {k:v.to(device) for k, v in model_input_yes.items()}

                if len(td['input_ids'][0]) < 4096:
                    output = model(**td)
                    yes_loss = copy.deepcopy(output.loss.cpu().item())
                    model_input_no = preprocess(model_input, "No", tokenizer)
                    td = {k:v.to(device) for k, v in model_input_no.items()}
                    output = model(**td)
                    no_loss = output.loss.cpu().item()
                    
                    if yes_loss<no_loss:
                        pco_answers[pco].append("Yes")
                    else:
                        pco_answers[pco].append("No")
                    
                else:
                    count_long_input.append(ind)
                    pco_answers[pco].append("No")
                
    write_data_path = args.write_data_path
    dir_path = os.path.dirname(write_data_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    df = original_data.assign(**pco_answers)
    df.to_csv(write_data_path, index=False)
    if len(count_long_input)!=0:
        print("These are the examples where the snippets are too long and assign answer no to all pco:", count_long_input)
    

if __name__ == "__main__":
    main()

