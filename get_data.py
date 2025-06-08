"""音乐推荐数据获取与预处理脚本"""
import requests
import zipfile
import os
import pandas as pd
from tqdm import tqdm
import random

# FMA数据集配置
FMA_SMALL_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"

def download_file(url, save_path, max_retries=3):
    """下载文件并显示进度条，带重试机制"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(save_path, 'wb') as f, tqdm(
                desc=save_path,
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)
            
            # 如果是zip文件，验证完整性
            if save_path.endswith('.zip'):
                try:
                    with zipfile.ZipFile(save_path, 'r') as zip_ref:
                        if zip_ref.testzip() is None:
                            return True
                        else:
                            print("下载的zip文件损坏，将重试...")
                            os.remove(save_path)
                            continue
                except zipfile.BadZipFile as e:
                    print(f"Zip文件验证失败: {str(e)}")
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    continue
            
            return True
            
        except (requests.RequestException, IOError) as e:
            print(f"下载出错 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if os.path.exists(save_path):
                os.remove(save_path)
            if attempt == max_retries - 1:
                print(f"无法下载文件，请检查网络或手动下载: {url}")
                print(f"保存路径: {os.path.abspath(save_path)}")
                return False

def process_fma_data(raw_dir, output_dir):
    """处理FMA音乐数据"""
    os.makedirs(output_dir, exist_ok=True)

    MOOD_MAP = {
        'Pop': 'happy',
        'Rock': 'energetic',
        'Hip-Hop': 'energetic',
        'Folk': 'calm',
        'Electronic': 'energetic',
        'Experimental': 'sad',
        'Instrumental': 'calm',
        'International': 'happy',
        'Jazz': 'calm',
        'Classical': 'calm',
        'Old-Time / Historic': 'sad',
        'Spoken': 'sad'
    }

    
    # 读取FMA元数据
    tracks = pd.read_csv(f"{raw_dir}/fma_metadata/tracks.csv", index_col=0, header=[0, 1])
    genres = pd.read_csv(f"{raw_dir}/fma_metadata/genres.csv")
    
    # 提取必要字段
    processed = pd.DataFrame()
    processed['track_id'] = tracks.index
    processed['title'] = tracks[('track', 'title')]
    processed['artist'] = tracks[('artist', 'name')]
    processed['genre'] = tracks[('track', 'genre_top')]
    processed['tags'] = processed['genre'].map(MOOD_MAP)
    
    # 保存处理后的数据
    processed.to_csv(f"{output_dir}/songs.csv", index=False)
    print(f"处理完成，共 {len(processed)} 首歌曲")

def energy_to_label(e, v):
    happy = ['happy', 'joy', 'upbeat', 'cheerful']
    sad = ['sad', 'melancholy', 'depressing', 'gloomy']
    energetic = ['energetic', 'intense', 'powerful', 'pumping']
    calm = ['calm', 'peaceful', 'relaxing', 'soothing']

    if pd.isnull(e):
        return "unknown"
    try:
        e = float(e)
    except:
        return "unknown"
    if e < 0.4:
        return random.choice(calm)
    elif e < 0.7:
        # 用valence进一步细分happy和sad
        if pd.isnull(v):
            return random.choice(happy + sad)
        try:
            v = float(v)
        except:
            return random.choice(happy + sad)
        if v >= 0.5:
            return random.choice(happy)
        else:
            return random.choice(sad)
    else:
        return random.choice(energetic)

def process_msd_spotify_lastfm(raw_csv_path, output_csv_path):
    """
    处理 Million Song Dataset + Spotify + Last.fm 数据集，输出标准格式
    只保留 track_id,title,artist,genre,energy 字段
    energy 字段为情绪标签，genre 字段为原始 tags
    """
    import pandas as pd

    # 读取原始数据
    df = pd.read_csv(raw_csv_path)

    # 整理字段
    df_out = pd.DataFrame()
    df_out['track_id'] = df['track_id']
    df_out['title'] = df['name']
    df_out['artist'] = df['artist']
    df_out['genre'] = df['tags']
    df_out['energy'] = [energy_to_label(e, v) for e, v in zip(df['energy'], df['valence'])]

    # 去除缺失值
    df_out = df_out.dropna(subset=['track_id', 'title', 'artist', 'genre', 'energy'])

    # 保存为新的csv
    df_out.to_csv(output_csv_path, index=False)
    print(f"处理完成，共 {len(df_out)} 首歌曲，已保存到 {output_csv_path}")

def main():
    # 创建数据目录
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    """ print("正在下载FMA元数据...")
    if not download_file(FMA_METADATA_URL, "data/raw/fma_metadata.zip"):
        return  # 下载失败时退出 """
    
    """ print("解压数据...")
    try:
        with zipfile.ZipFile("data/raw/fma_metadata.zip", 'r') as zip_ref:
            zip_ref.extractall("data/raw")
    except zipfile.BadZipFile as e:
        print(f"解压失败: {str(e)}")
        print("请删除损坏的文件并重新运行脚本")
        return """
    
    """ print("处理数据...")
    process_fma_data("data/raw", "data/processed") """

    raw_csv = "data/raw/Music_Info.csv"
    output_csv = "data/processed/songs.csv"
    process_msd_spotify_lastfm(raw_csv, output_csv)

if __name__ == "__main__":
    main()