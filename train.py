"""改进的对比学习训练脚本"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

class TripletDataset(Dataset):
    """支持困难样本挖掘的数据集"""
    def __init__(self, df, tokenizer, knn_k=5):
        self.df = df
        self.tokenizer = tokenizer
        
        # 预计算所有歌曲向量
        self.song_vecs = self._encode_songs()
        
        # 构建KNN模型用于困难样本挖掘
        self.knn = NearestNeighbors(n_neighbors=knn_k)
        self.knn.fit(self.song_vecs)

    def _encode_songs(self):
        """预编码所有歌曲描述"""
        texts = self.df.apply(
            lambda r: f"{r['title']} {r['artist']} {r['genre']} {r['mood']}", axis=1
        ).tolist()
        
        # 获取tokenizer输出
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        # 提取input_ids并转换为numpy数组
        input_ids = encoded['input_ids'].numpy()
        
        # 确保是2D数组 (n_samples, seq_len)
        return input_ids

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        anchor_text = row['user_input']
        pos_row = self.df.iloc[idx]
        pos_text = f"{pos_row['title']} {pos_row['artist']} {pos_row['genre']} {pos_row['mood']}"
        _, neg_indices = self.knn.kneighbors([self.song_vecs[idx]])
        neg_idx = np.random.choice(neg_indices[0])
        neg_row = self.df.iloc[neg_idx]
        neg_text = f"{neg_row['title']} {neg_row['artist']} {neg_row['genre']} {neg_row['mood']}"
        return {
            'anchor': anchor_text,
            'positive': pos_text,
            'negative': neg_text
        }
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.df)

class ContrastiveModel(nn.Module):
    """改进的对比学习模型"""
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(768, 256)  # 降维提升效率
        
    def forward(self, x):
        if isinstance(x, dict):
            # 处理tokenizer输出格式
            outputs = self.bert(**x)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        elif hasattr(x, 'dim'):
            # 处理预编码的input_ids张量
            if x.dim() == 2:
                x = x.unsqueeze(0)  # 确保有batch维度
            outputs = self.bert(input_ids=x)
            embeddings = outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError("Unsupported input type for model forward")
            
        return F.normalize(self.proj(embeddings), dim=-1)

def train_epoch(model, dataloader, optimizer, device, tokenizer):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        anchors = list(batch['anchor'])
        positives = list(batch['positive'])
        negatives = list(batch['negative'])

        # 正确地将 tokenizer 输出转到 device
        def batch_to_device(batch_encoding, device):
            return {k: v.to(device) for k, v in batch_encoding.items()}

        anchor_inputs = batch_to_device(tokenizer(anchors, padding=True, truncation=True, return_tensors="pt"), device)
        positive_inputs = batch_to_device(tokenizer(positives, padding=True, truncation=True, return_tensors="pt"), device)
        negative_inputs = batch_to_device(tokenizer(negatives, padding=True, truncation=True, return_tensors="pt"), device)

        anchor = model(anchor_inputs)
        positive = model(positive_inputs)
        negative = model(negative_inputs)

        loss = F.triplet_margin_loss(anchor, positive, negative)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    # 加载数据
    df = pd.read_csv("data/train/train_data.csv")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # 数据加载器
    dataset = TripletDataset(df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 训练循环
    for epoch in range(10):
        loss = train_epoch(model, dataloader, optimizer, device, tokenizer)
        print(f"Epoch {epoch+1} | Loss: {loss:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), "models/song_encoder.pt")

if __name__ == "__main__":
    main()