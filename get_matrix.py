#!/usr/bin/env python
# coding: utf-8

# In[6]:


import csv

predictions_file = "predictions.csv"
output_file = "output.csv"


with open(predictions_file, "r", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    predictions = list(reader)[0]
def read_csv_generator(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            yield row


result_matrix = []
text_a_generator = read_csv_generator("train.csv")
text_b_list = list(read_csv_generator("predicate.csv"))

index = 0
result_row = [""]  
for row_b in text_b_list:
    text_b = row_b["predicate"].strip()
    result_row.append(text_b)
result_matrix.append(result_row)

for i, example_a in enumerate(text_a_generator):
    text_a = example_a["Description"].strip()
    label = example_a["Label"].strip()
    result_row = [text_a]

    for j, row_b in enumerate(text_b_list):
        text_b = row_b["predicate"].strip()
        result_row.append(predictions[index])
        index += 1

    result_matrix.append(result_row)


with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(result_matrix)


# In[ ]:




