# Patient Center Outcomes (PCO) Extraction
This is the repo for the PCO extraction using large language models project. In the following, we will walk through the training, data preparation, and inference. 

## Install and Activate Environment
In the enviorment.yml, the name of the enviormnet is my_env, and you could change according to your preference, and activate the environment using the name ( in second command below). 

```
conda env create -f enviroment.yml
conda activate my_env
```

## Data Preparation
For the training, the script reads json files which are in the following format. Each row contains three keys. 'instruction' describes the task, for example, if the task is to extract whether there a patient has fatigue, then an instruction can be "Based on the input text, does the patient has fatigue?". The 'input' is the clinical note, and the 'output' is the label. 
```
[
    {
        "instruction": "",
        "input": "",
        "output": ""
    },
]
```

For the inference, the script read csv file with 'Text_snippet' field as input text to the model. 

## Training Models
Below is a command that train a gpt-2 model, and train_biogpt.sh shows how to train a biogpt model.
```
train_data_path=<your_path_to_trianing_file>
output_dir=<your_path_to_train_model>
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate  launch train_pco.py \
    --model_name_or_path "gpt2" \
    --data_path $train_data_path \
    --output_dir $output_dir \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 1024
```

## Inference
The inference read the 'input_name' field from the csv file (default is 'Text_snippet'), and save the prediction into the 'output_file' (<your_path_to_write_the_prediction_file>). In the prediction file, there will be five columns for: "fatigue", "depression", "anxiety", "nausea", "lymphedema".

```
model_name=<your_path_to_train_model>
read_data_path=<your_path_to_test_file>
output_file=<your_path_to_write_the_prediction_file> 
input_name='Text_snippet'
python test.py \
    --model_name=$model_name  \
    --read_data_path=$read_data_path \
    --write_data_path=$output_file \
    --input_name=$input_name \
    --device 3 
```
