"""数据预处理与训练样本生成"""
import os
import pandas as pd
import numpy as np
from collections import defaultdict

# 情绪关键词映射表
MOOD_KEYWORDS = {
    'happy': ['happy', 'joy', 'upbeat', 'cheerful'],
    'sad': ['sad', 'melancholy', 'depressing', 'gloomy'], 
    'energetic': ['energetic', 'intense', 'powerful', 'pumping'],
    'calm': ['calm', 'peaceful', 'relaxing', 'soothing']
}

def extract_mood(tags):
    """从标签中提取主要情绪"""
    if not isinstance(tags, str):
        return None
        
    tags = tags.lower().split(',')
    mood_counts = defaultdict(int)
    
    for tag in tags:
        for mood, keywords in MOOD_KEYWORDS.items():
            if any(kw in tag for kw in keywords):
                mood_counts[mood] += 1
                
    if not mood_counts:
        return None
    return max(mood_counts.items(), key=lambda x: x[1])[0]

def generate_state_description(row):
    """根据歌曲元数据生成状态描述"""
    mood = row['mood']
    genre = row['genre']
    
    # 基础模板
    templates = {
        'happy': [
            f"开心快乐的{genre}时光",
            f"心情愉悦地听{genre}音乐"
        ],
        'sad': [
            f"感到孤独时听的{genre}",
            f"情绪低落时的{genre}陪伴"
        ],
        'energetic': [
            f"充满能量时听的{genre}",
            f"运动健身时的{genre}动力"
        ],
        'calm': [
            f"需要放松时的{genre}音乐",
            f"平静安宁的{genre}时刻"
        ]
    }
    
    if mood not in templates:
        return f"听{genre}音乐"
    return np.random.choice(templates[mood])

def main():
    # 加载原始数据
    songs = pd.read_csv("data/processed/songs.csv")
    
    print("正在处理数据...")
    # 清洗无效数据
    songs = songs.dropna(subset=['title', 'artist', 'energy'])
    
    # 提取情绪标签
    songs['mood'] = songs['energy'].apply(extract_mood)
    songs = songs.dropna(subset=['mood'])
    
    # 生成训练样本
    samples = []
    for _, row in songs.iterrows():
        samples.append({
            'user_input': generate_state_description(row),
            'song_id': row['track_id'],
            'title': row['title'],
            'artist': row['artist'],
            'genre': row['genre'],
            'mood': row['mood']
        })
    
    # 保存训练数据
    train_df = pd.DataFrame(samples)
    # 确保训练目录存在
    os.makedirs("data/train", exist_ok=True)
    train_df.to_csv("data/train/train_data.csv", index=False)
    print(f"生成 {len(train_df)} 条训练样本")

if __name__ == "__main__":
    main()