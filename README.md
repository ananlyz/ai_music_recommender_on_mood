# 写在前面
这是吉林大学软件学院软件工程2023级人工智能引论的实践作业，系统目的，系统设计，系统实现，算法设计，系统测试均为学生一人完成，另外本作业是学生在人工智能实践过程中的一次探索与尝试，虽已尽力追求原创性与完成度，但深知受限于学识与经验，其中必然存在诸多不足与值得商榷之处。恳请老师不吝赐教，予以悉心指正。您的宝贵意见是学生精进学习的关键指引，学生定当认真领会、努力改进。衷心期盼在老师的指导与鼓励下，本次实践能得到一个较为理想的评价。


# SmartMusic 智能音乐推荐系统 v4.0

基于对比学习的智能音乐推荐系统，使用DistilBERT模型理解用户状态，并通过困难样本挖掘优化歌曲匹配。

##  功能特点

-  **智能理解**：使用轻量级DistilBERT模型理解自然语言状态描述
-  **精准匹配**：基于Triplet Loss的对比学习模型
-  **性能优化**：支持困难样本挖掘提升推荐质量
-  **完整流程**：提供训练-评估-应用全流程解决方案
-  **交互界面**：友好的命令行推荐界面
-  **详细评估**：提供准确率、召回率等多维度评估指标

## 安装指南

### 系统要求
- Python 3.11+
- PyTorch 2.2+
- CUDA 12+ (如需GPU加速)

### 安装步骤
1. 进入项目：
```bash
cd ai_music_recommender_on_mood
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

##  数据准备

### 数据集
数据集：Million Song Dataset + Spotify + Last.fm

数据集链接：https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm

### 数据处理流程
1. 获取原始数据：
```bash
python get_data.py
```

2. 预处理数据：
```bash
python preprocess.py
```

### 数据格式
1. 原始数据格式 get_data执行后第一次数据处理：
```csv
track_id,title,artist,genre,tags
TRUMISQ128F9340BEE,Somebody Told Me,The Killers,"rock, alternative, indie, pop, alternative_rock, indie_rock",pumping
TRIQRZJ128F14898B4,When You Were Young,The Killers,"rock, alternative, indie, alternative_rock, indie_rock",powerful
```

2. 训练数据格式 preprocess执行后第二次数据处理：
```csv
user_input,song_id,title,artist,genre,mood
"开心快乐的rock, alternative, alternative_rock, 90s, funk时光",TRIODZU128E078F3E2,Under the Bridge,Red Hot Chili Peppers,"rock, alternative, alternative_rock, 90s, funk",happy
"感到孤独时听的rock, classic_rock, hard_rock, 80s, british, 70s",TRRSVFJ128F426FB31,Bohemian Rhapsody,Queen,"rock, classic_rock, hard_rock, 80s, british, 70s",sad
```

##  快速开始


### 训练模型
```bash
python train.py 
```

### 评估模型
```bash
python evaluate.py 
```

### 运行推荐系统
```bash
python run.py
```

##  项目结构

```
smartmusic/
├── data/                # 数据目录
│   ├── raw/             # 原始数据
│   ├── processed/       # 处理后的数据
│   └── train/           # 训练数据
│   └── test/            # 测试数据
├── models/              # 训练好的模型
├── get_data.py          # 数据获取脚本
├── preprocess.py        # 数据预处理
├── train.py             # 模型训练(含困难样本挖掘)
├── evaluate.py          # 模型评估
├── run.py               # 主程序入口
└── README.md            # 说明文档
```

##  评估指标

系统提供以下评估指标：
- Top-K准确率 (K=1,3,5)
- 精确率(Precision@K)
- 召回率(Recall@K)
- F1分数

##  使用示例

```bash
请描述您当前的心情或状态: 今天工作很累，想听些放松的音乐

为您推荐的歌曲:
1. River Flows in You - Yiruma
    风格: 钢琴 |  情绪: calm |  匹配度: 0.872
2. Take Five - Dave Brubeck
    风格: 爵士 |  情绪: calm |  匹配度: 0.843
3. Moonlight Sonata - Beethoven
    风格: 古典 | 情绪: calm |  匹配度: 0.812
```

##  后续计划

- [ ] 增加用户历史偏好学习
- [ ] 实现多模态推荐(结合音频特征)
- [ ] 部署为Web服务
- [ ] 增加推荐解释功能
- [ ] 支持多语言输入

##  许可证

本项目采用 [MIT License](LICENSE)