from torch.utils.data import Dataset, DataLoader
from utils import prepare_dataset
from transformers import BertTokenizer
import torch
import numpy as np

import warnings
warnings.filterwarnings("ignore")

class IntentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, item):
        row = self.dataframe.loc[item]
        intent_idx = row.intent_idx
        query = row.query

        ids = self.tokenizer.encode(query, 
                    max_len=self.max_len)

        ids = ids + [0] * (self.max_len - len(ids))
        attention_mask = [int(i > 0) for i in ids]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "intent_idx": intent_idx
        }

if __name__ == "__main__":
    path = "../oos-eval/data/data_full.json"
    data, intents_dict = prepare_dataset(path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = IntentData(data, tokenizer, max_len=30)

    for batch in DataLoader(train_dataset, batch_size=100):
        print(batch["ids"])
        print(batch["attention_mask"])
        break