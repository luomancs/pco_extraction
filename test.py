from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from tqdm.auto import tqdm
import torch
from train import PROMPT_DICT
from utils import read_json_file
import argparse
import json
import os


from train import SupervisedDataset,DataCollatorForSupervisedDataset,Dict,transformers
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def main():
    
    # Create the parser
    parser = argparse.ArgumentParser(description='This is a demo script by ChatGPT.')

    # Add the arguments
    parser.add_argument('--model_name', required=True, type=str, default="",
                    help='the name of the evaluated model')
    parser.add_argument('--device',  type=str, default="0",
                    help='device id')
    parser.add_argument('--dataset_name', required=True, type=str, default="",
                    help='inference file')
    parser.add_argument('--write_data_path', required=True, type=str, default="",
                    help='the path to save prediction')
    parser.add_argument("--llama", action="store_true", help="load llama-2 model")
    parser.add_argument("--lora", action="store_true", help="load llama-2 model trained with lora")

    # Parse the arguments
    args = parser.parse_args()

    model_name = args.model_name
    if args.llama:
        tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = transformers.LlamaForCausalLM.from_pretrained(model_name)
        device = "cuda:{}".format(args.device)
        model = model.to(device)
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
        device = "cuda:{}".format(args.device)
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side='left'
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(model_name)
        device = "cuda:{}".format(args.device)
        model = model.to(device)
        

    input_p = args.dataset_name
    train_data_module = make_supervised_data_module(tokenizer, input_p)
    original_data = read_json_file(input_p)

    write_to_dict = []

    import torch 
    count_long_input = 0
    with torch.no_grad():
        for ind,td in tqdm(enumerate(train_data_module['train_dataset']), total = len(train_data_module['train_dataset'])):
            
            td = train_data_module["data_collator"]([td])
            td = {k:v.to(device) for k, v in td.items()}
            if len(td['input_ids'][0]) < 4096:
                count_long_input+=1
                output = model(**td)
                original_data[ind]['loss'] = output.loss.cpu().item()
                write_to_dict.append(original_data[ind])
            else:
                count_long_input+=1
                original_data[ind]['loss'] = 0.01
                write_to_dict.append(original_data[ind])
    write_data_path = args.write_data_path
    dir_path = os.path.dirname(write_data_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print(count_long_input)
    with open(write_data_path, 'w') as file:
        json.dump(write_to_dict, file, indent=4)
    

if __name__ == "__main__":
    main()


