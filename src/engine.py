from model import IntentClassificationModel
from transformers import BertModel, BertTokenizer
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
from torch.optim import lr_scheduler
from utils import prepare_dataset
from dataset import IntentData
from torch.utils.data import Dataset, DataLoader
import time
import copy
import torch
import torch.nn as nn
from transformers import AdamW
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def train_model(model, dataloader, model_params):
    model.train()

    loss_criterion = model_params.get("loss_criterion")
    tokenizer = model_params.get("tokenizer")
    modelsdir = model_params.get("modelsdir")
    optimizer = model_params.get("optimizer")
    scheduler = model_params.get("scheduler")
    writer = model_params.get("writer")
    device = model_params.get("device")
        
    running_loss = 0.0

    y = []
    y_hat = []

    for i, data in tqdm(enumerate(dataloader)):
        ids = data.get("ids")
        attention_mask = data.get("attention_mask")
        intent_idx = data.get("intent_idx")
        
        ids = ids.to(device)
        attention_mask = attention_mask.to(device)
        intent_idx = intent_idx.to(device)

        with torch.set_grad_enabled(True):
            logits = model(ids, attention_mask)
            loss = loss_criterion(logits, intent_idx)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # y_pred = torch.argmax(logits, dim=-1).cpu().detach().numpy()
        # y_true = intent_idx.cpu().detach().numpy()

        # y.extend(y_true)
        # y_hat.extend(y_pred)

        # accuracy = accuracy_score(y_true, y_pred)

        # print(f"Loss = {loss.item():7.4f}, Accuracy = {accuracy:7.4f}")
    
    # overall_accuracy = accuracy_score(y, y_hat)
    # print(f"Overall training accuracy: {overall_accuracy:7.4}")

def eval_model(model, dataloader, model_params):
    tokenizer = model_params.get("tokenizer")
    loss_criterion = model_params.get("loss_criterion")
    running_loss = 0.0
    
    y = []
    y_hat = []

    for _, data in tqdm(enumerate(dataloader)):
        model.eval()
        
        ids = data.get("ids")
        attention_mask = data.get("attention_mask")
        intent_idx = data.get("intent_idx")

        ids = ids.to(device)
        attention_mask = attention_mask.to(device)
        intent_idx = intent_idx.to(device)

        with torch.set_grad_enabled(False):
            logits = model(ids, attention_mask)
            loss = loss_criterion(logits, intent_idx)

        y_pred = torch.argmax(logits, dim=-1).cpu().detach().numpy()
        y_true = intent_idx.cpu().detach().numpy()

        y.extend(y_true)
        y_hat.extend(y_pred)

    overall_accuracy = accuracy_score(y, y_hat)
    print(f"Overall validation accuracy: {overall_accuracy:7.4}")

    # print(f"Loss = {loss_avg:7.4f}, Jaccard = {jaccard_avg:6.4f}, Positive: {jaccards['positive'] / counts['positive']:6.4f}, Negative: {jaccards['negative'] / counts['negative']:6.4f}, Neutral: {jaccards['neutral']/counts['neutral']:6.4f}")
    # print(f"Loss = {loss_avg:7.4f}, Jaccard = {jaccard_avg:6.4f}, Positive: {jaccards['positive'] / counts['positive']:6.4f}")

    # return (loss_avg, jaccard_avg)

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    max_len = 36
    epochs = 10
    batch_size = 128
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path = "../oos-eval/data/data_full.json"
    data, intents_dict = prepare_dataset(path)

    num_train_steps = int(len(data.query("datatype == 'train'").index) * epochs / batch_size) + 1

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = IntentClassificationModel()
    model.to(device)

    train_dataset = IntentData(data.query("datatype == 'train'").reset_index(drop=True), 
            tokenizer, 
            max_len=max_len)
    valid_dataset = IntentData(data.query("datatype == 'val'").reset_index(drop=True), 
            tokenizer, 
            max_len=max_len)
    test_dataset = IntentData(data.query("datatype == 'test'").reset_index(drop=True), 
            tokenizer, 
            max_len=max_len)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=100, 
        num_training_steps=num_train_steps
    )

    model_params = {
        "loss_criterion": nn.CrossEntropyLoss(),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "device": device,
        "tokenizer": tokenizer
    }

    for epoch in range(epochs):
        print(f"Epoch = {epoch+1}")
        train_model(model, train_dataloader, model_params)
        eval_model(model, train_dataloader, model_params)
        eval_model(model, valid_dataloader, model_params)
        eval_model(model, test_dataloader, model_params)