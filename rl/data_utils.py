import random
import torch
import json


def read_dataset(path):
    # works with jsonl ony, expect ChatML format inside
    data = []
    with open(path, 'r') as file:
        for line in file:
            loaded_data = json.loads(line)
            if isinstance(loaded_data, list):
                data.append(loaded_data)
            else:
                data.append(loaded_data['messages'])
    return data

def get_raw_dataset(files):
    dataset = []
    for file in files:
        dataset.extend(read_dataset(file))
    return dataset

