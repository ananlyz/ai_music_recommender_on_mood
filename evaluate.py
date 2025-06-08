"""模型评估脚本"""
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score
from transformers import AutoTokenizer
from train import ContrastiveModel, TripletDataset
import pandas as pd

def evaluate_model(model, test_df, tokenizer, device, top_k=5):
    """评估模型性能"""
    model.eval()
    
   # 1. 先获取所有歌曲的文本描述
    song_texts = test_df.apply(
        lambda r: f"{r['title']} {r['artist']} {r['genre']} {r['mood']}", axis=1
    ).tolist()

    # 2. 批量编码所有歌曲，得到 [num_songs, 256] 的向量
    with torch.no_grad():
        song_inputs = tokenizer(song_texts, padding=True, truncation=True, return_tensors="pt")
        song_inputs = {k: v.to(device) for k, v in song_inputs.items()}
        all_song_vecs = model(song_inputs)  # shape: [num_songs, 256]
    
    correct = 0
    all_labels = []
    all_preds = []
    
    for idx in range(len(test_df)):
        # 获取查询和真实歌曲
        query = test_df.iloc[idx]['user_input']
        true_song_idx = idx
        
        # 编码查询
        inputs = tokenizer(query, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            query_vec = model(inputs)
        
        # 计算相似度
        similarities = torch.matmul(query_vec, all_song_vecs.T)
        top_indices = torch.topk(similarities, k=top_k).indices.cpu().numpy()
        
        # 计算指标
        pred_labels = np.zeros(len(test_df))
        pred_labels[top_indices] = 1
        
        true_labels = np.zeros(len(test_df))
        true_labels[true_song_idx] = 1
        
        all_labels.append(true_labels)
        all_preds.append(pred_labels)
        
        if true_song_idx in top_indices:
            correct += 1
    
    # 汇总指标
    accuracy = correct / len(test_df)
    precision = precision_score(np.concatenate(all_labels), np.concatenate(all_preds))
    recall = recall_score(np.concatenate(all_labels), np.concatenate(all_preds))
    
    print(f"\n评估结果 (Top-{top_k}):")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")

def main():
    # 加载数据
    test_df = pd.read_csv("data/test/test_data.csv")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveModel().to(device)
    model.load_state_dict(torch.load("models/song_encoder.pt"))
    
    # 执行评估
    evaluate_model(model, test_df, tokenizer, device)

if __name__ == "__main__":
    main()