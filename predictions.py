#!/usr/bin/env python
# coding: utf-8

# In[50]:


from openprompt.data_utils import InputExample
import csv
from datasets import load_dataset
from tqdm import tqdm
from transformers import GenerationMixin
import pandas as pd
import torch
import json


# In[51]:


def read_csv_generator(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            yield row


# In[52]:


dataset = []
with open("predicate_small.csv", "r", encoding="utf-8") as pred_file:
    reader = csv.DictReader(pred_file)
    predicates = [row["predicate"].strip() for row in reader]


text_a_generator = read_csv_generator("train.csv")
text_b_generator = read_csv_generator("predicate_small.csv")


# In[53]:


from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm("gpt2", "gpt2-large")


# In[54]:


from openprompt.data_utils import InputExample

for row_a in text_a_generator:
    text_a = row_a["Description"].strip()
    label = row_a["Label"].strip()
    
    text_b_generator = read_csv_generator("predicate_small.csv")  
    for row_b in text_b_generator:
        text_b = row_b["predicate"].strip()

        example = InputExample(
            guid=len(dataset),
            text_a=text_a,
            text_b=text_b,
            label=int(label)
        )
        dataset.append(example)

classes = ["1", "0"]


# In[55]:


from openprompt.prompts import ManualTemplate
for data in tqdm(dataset, desc='Progress monitor'):
            promptTemplate = ManualTemplate(
            text = '{"placeholder":"text_a"} Question: Is the text talking about{"placeholder":"text_b"}? Answer: {"mask"}.',
            tokenizer = tokenizer,
)


# In[56]:


from openprompt.prompts import ManualVerbalizer

promptVerbalizer = ManualVerbalizer(
    classes=classes,
    label_words={"1": ["Yes"], "0": ["No"]},
    tokenizer=tokenizer,
)


# In[57]:


from openprompt import PromptForClassification

promptModel = PromptForClassification(
    template=promptTemplate,
    plm=plm,
    verbalizer=promptVerbalizer,
)


# In[58]:


from openprompt import PromptDataLoader

data_loader = PromptDataLoader(
    dataset=dataset,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
)


# In[59]:


promptModel.eval()
predictions = []
with torch.no_grad():
    for batch in tqdm(data_loader, desc='Progress monitor'):
        logits = promptModel(batch)
        preds = torch.argmax(logits, dim=-1)
        print(classes[preds])
        predictions.append(classes[preds])


# In[61]:


predictions[111]


# In[62]:


predictions_file = "predictions.csv"
with open(predictions_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(predictions)


# In[ ]:




