import requests
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import BertTokenizer



class Predictor:
    MODEL_URL = "https://haymanastorage.blob.core.windows.net/logalyze/multi_task_bert_large.pt"
    MODEL_SAVE_NAME = "model.pt"


    def __init__(self) -> None:
        self._load_model(self.MODEL_SAVE_NAME, self.MODEL_URL)
        
        self.model = torch.jit.load(self.MODEL_SAVE_NAME)
        self.model.eval()  

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



    def _load_model(self,save_path: str, url:str) -> None:
        if not os.path.exists(save_path):
            print("Loading File")

            res = requests.get(url, stream=True)
            if res.status_code == 200:
                total_size = int(res.headers.get('content-length', 0))
                block_size = 1 << 10
                t = tqdm(total=total_size, unit='iB', unit_scale=True)

                with open(save_path, 'wb') as f:
                    for data in res.iter_content(block_size):
                        t.update(len(data))
                        f.write(data)
                
                t.close()

                if total_size != 0 and t.n != total_size:
                    print("ERROR: Something went wrong during the download.")
                else:
                    print(f"Downloaded {save_path} successfully.")
            else:
                print(f"Failed to download from {url}. Status code: {res.status_code}")
        else:
            print(f"{save_path} already exists. No download needed.")



    def predict(self, log: str) -> list[int]:
        inputs = self.tokenizer(log, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            is_anomal_logits = outputs[0]
            need_attention_logits = outputs[1]
            level_logits = outputs[2]

        print(is_anomal_logits, need_attention_logits, level_logits)

        is_anomal_logits = is_anomal_logits.argmax(dim=-1).item()
        need_attention_logits = need_attention_logits.argmax(dim=-1).item()
        level_logits = level_logits.argmax(dim=-1).item()

        return [is_anomal_logits, need_attention_logits, level_logits]






