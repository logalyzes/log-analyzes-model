import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import get_scheduler
from tqdm import tqdm
from typing import List

class LogDataset(Dataset):
    def __init__(self, data):
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        log = item['message']
        inputs = self.tokenizer(log, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}
        
        labels = {
            'is_anomal': torch.tensor(item['isAnomal'], dtype=torch.long),
            'need_attention': torch.tensor(item['needAttention'], dtype=torch.long),
            'level': torch.tensor(item['Level'], dtype=torch.long)
        }
        return inputs, labels

class MultiTaskBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier_is_anomal = nn.Linear(self.bert.config.hidden_size, 2)
        self.classifier_need_attention = nn.Linear(self.bert.config.hidden_size, 2)
        self.classifier_level = nn.Linear(self.bert.config.hidden_size, 5)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        
        is_anomal_logits = self.classifier_is_anomal(pooled_output)
        need_attention_logits = self.classifier_need_attention(pooled_output)
        level_logits = self.classifier_level(pooled_output)
        
        return is_anomal_logits, need_attention_logits, level_logits

class Trainer:
    def __init__(self, model_path='/path/to/your/model.pt', batch_size=100):
        self.model_path = model_path
        self.batch_size = batch_size

    def train_model(self, messages: List[dict]):
        """Train the model with consumed messages from the Kafka topic."""
        if not messages:
            print("No messages to train on.")
            return

        dataset = LogDataset(messages)
        train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiTaskBERT().to(device)
        self.train_model_instance(model, train_loader, device)
        self.save_model(model)

    def train_model_instance(self, model, train_loader, device):
        model.train()
        optimizer = AdamW(model.parameters(), lr=5e-5)
        loss_fn = CrossEntropyLoss()
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 3)

        for epoch in range(3):
            with tqdm(train_loader, unit="batch") as tepoch:
                for inputs, labels in tepoch:
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    labels = {k: v.to(device) for k, v in labels.items()}

                    model.zero_grad()
                    outputs = model(input_ids, attention_mask)
                    loss = (loss_fn(outputs[0], labels['is_anomal']) +
                            loss_fn(outputs[1], labels['need_attention']) +
                            loss_fn(outputs[2], labels['level'])) / 3
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    tepoch.set_postfix(loss=loss.item())

    def save_model(self, model):
        torch.save(model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")
