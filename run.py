"""改进的音乐推荐系统主程序"""
import torch
import numpy as np
from transformers import AutoTokenizer
from train import ContrastiveModel
import pandas as pd
from tqdm import tqdm

class MusicRecommender:
    def __init__(self, model_path, song_data_path):
        # 加载设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        self.model = ContrastiveModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # 加载歌曲数据
        self.song_df = pd.read_csv(song_data_path)
        self._precompute_song_vectors()
    
    def _precompute_song_vectors(self):
        """预计算所有歌曲向量"""
        print("正在预计算歌曲向量...")
        song_texts = self.song_df.apply(
            lambda r: f"{r['title']} {r['artist']} {r['genre']} {r['mood']}", 
            axis=1
        ).tolist()
        
        batch_size = 32
        self.song_vectors = []
        for i in tqdm(range(0, len(song_texts), batch_size)):
            batch = song_texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}  # 修正
            with torch.no_grad():
                vectors = self.model(inputs).cpu().numpy()
                self.song_vectors.append(vectors)
        
        self.song_vectors = np.concatenate(self.song_vectors)
    
    def recommend(self, query, top_k=5):
        """根据用户输入推荐歌曲"""
        inputs = self.tokenizer(query, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # 修正
        with torch.no_grad():
            query_vec = self.model(inputs).cpu().numpy()
        similarities = np.dot(query_vec, self.song_vectors.T)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            song = self.song_df.iloc[idx]
            results.append({
                'title': song['title'],
                'artist': song['artist'],
                'genre': song['genre'],
                'mood': song['mood'],
                'score': float(similarities[idx])
            })
        return results

def main():
    # 初始化推荐器
    recommender = MusicRecommender(
        model_path="models/song_encoder.pt",
        song_data_path="data/train/train_data.csv"
    )
    
    print("\n🎵 智能音乐推荐系统 (输入'q'退出)")
    while True:
        query = input("\n请描述您当前的心情或状态: ")
        if query.lower() == 'q':
            break
        
        # 获取推荐
        recommendations = recommender.recommend(query)
        
        # 显示结果
        print("\n为您推荐的歌曲:")
        for i, song in enumerate(recommendations, 1):
            print(f"{i}. {song['title']} - {song['artist']}")
            print(f"   风格: {song['genre']} | 情绪: {song['mood']} | 匹配度: {song['score']:.3f}")

if __name__ == "__main__":
    main()