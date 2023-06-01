#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tqdm import tqdm
from openprompt.data_utils import PROCESSORS
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np
import csv

# In[ ]:


from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, ManualTemplate
from openprompt.prompts import SoftTemplate
from openprompt import PromptForClassification
import time
import os
from transformers import AutoTokenizer, OPTModel

# In[ ]:


parser = argparse.ArgumentParser("")
parser.add_argument("--template_id", type=int)
parser.add_argument("--soft_token_num", type=int, default=20)
parser.add_argument("--init_from_vocab", action="store_false")
args = parser.parse_args()

scriptsbase = "SuperGLUE/BoolQ"
scriptformat = "txt"
max_seq_l = 480
classes = ["1", "0"]



from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm("opt","facebook/opt-1.3b")

def read_csv_generator(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            yield row
dataset = []
with open("predicates.csv", "r", encoding="utf-8") as pred_file:
    reader = csv.DictReader(pred_file)
    predicates = [row["Predicate"].strip() for row in reader]


text_a_generator = read_csv_generator("bbc-text.csv")

for row_a in text_a_generator:
    text_a = row_a["text"].strip()    

    text_b_generator = read_csv_generator("predicates.csv")   
    for row_b in text_b_generator:
        text_b = row_b["Predicate"].strip()

        example = InputExample(
            guid=len(dataset),
            text_a=text_a,
            text_b=text_b
         )
        dataset.append(example)



mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, num_tokens=args.soft_token_num, initialize_from_vocab=args.init_from_vocab).from_file(f"scripts/{scriptsbase}/soft_template.txt", choice=args.template_id)
myverbalizer = ManualVerbalizer(
    classes=classes,
    label_words={"1": ["Yes"], "0": ["No"]},
    tokenizer=tokenizer)

wrapped_example = mytemplate.wrap_one_example(dataset[0])
print(wrapped_example)



prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer)






prompt_model.eval()
predictions = []
with torch.no_grad():
    # For every 100000 examples in dataset, we create a new data loader
    for i in range(0, len(dataset), 10000):
        subdataset = dataset[i : i + 10000]
        data_loader = PromptDataLoader(
            dataset=subdataset,
            template=mytemplate,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=max_seq_l, 
            decoder_max_length=3,
            batch_size= 4,shuffle=True, 
            teacher_forcing=False, 
            predict_eos_token=False,
            truncate_method="tail")
        for batch in tqdm(data_loader, desc='Progress monitor'):
            logits = prompt_model(batch)
            preds = torch.argmax(logits, dim=-1)
            for pred in preds:
                print(classes[pred.item()])
                predictions.append(classes[pred.item()])


predictions_file = "predictions.csv"
with open(predictions_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(predictions)
