from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch
import torch.nn as nn

class IntentClassificationModel(nn.Module):
    def __init__(self):
        super(IntentClassificationModel, self).__init__()
        print("Importing model...")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.linear1 = nn.Linear(768, 600)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(600, 150)
        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.normal_(self.linear2.weight, std=0.02)


    def forward(self, ids, attention_mask):
        _, out = self.bert(
                        ids, 
                        attention_mask=attention_mask,
                        )
        
        x = self.dropout(out)
        x = self.linear1(x)
        x = self.relu(x)
        logits = self.linear2(x)

        return logits

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    query = "What is my balance"
    max_len = 30
    ids = tokenizer.encode(query, max_len=max_len)
    ids = ids + [0] * (30 - len(ids))
    attention_mask = [int(i > 0) for i in ids]

    ids_tensor = torch.tensor(ids, dtype=torch.long).view(1, -1)
    attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long).view(1, -1)

    model = IntentClassificationModel()
    logits = model(ids_tensor, attention_mask_tensor)
    print(logits)
