import pickle
import os
import random
import glob
import numpy as np

# ================= 配置区域 =================
# 散落 pkl 文件的文件夹路径
SOURCE_FOLDER = '../data/raw_pkls'  # 请把你所有的 pkl 文件都扔到这个新建的文件夹里
# 输出文件名
OUTPUT_TRAIN = 'hiphop_train.pkl'
OUTPUT_VAL = 'hiphop_val.pkl'
# 验证集比例 (0.2 = 20% 的数据用来考试)
VAL_RATIO = 0.2 

def merge_and_split():
    all_data = []
    
    # 1. 寻找所有 .pkl 文件
    pkl_files = glob.glob(os.path.join(SOURCE_FOLDER, '*.pkl'))
    print(f"🔍 发现了 {len(pkl_files)} 个 pkl 文件...")

    if len(pkl_files) == 0:
        print("❌ 错误：文件夹里没有 .pkl 文件！请检查路径。")
        return

    # 2. 循环读取并合并
    for pkl_path in pkl_files:
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                
            # 检查数据格式
            if isinstance(data, list):
                print(f"   - 读取 {os.path.basename(pkl_path)}: 包含 {len(data)} 条样本")
                all_data.extend(data)
            else:
                print(f"⚠️ 跳过 {os.path.basename(pkl_path)}: 格式不是 List")
        except Exception as e:
            print(f"❌ 读取错误 {pkl_path}: {e}")

    total_samples = len(all_data)
    print(f"\n📊 总共收集到 {total_samples} 条动作数据。")
    
    if total_samples == 0:
        return

    # 3. 打乱数据 (Shuffle)
    # 这一步极其重要！防止模型死记硬背顺序
    random.shuffle(all_data)

    # 4. 切分训练集和验证集
    split_idx = int(total_samples * (1 - VAL_RATIO))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    # 5. 保存文件
    with open(OUTPUT_TRAIN, 'wb') as f:
        pickle.dump(train_data, f)
    with open(OUTPUT_VAL, 'wb') as f:
        pickle.dump(val_data, f)

    print(f"\n✅ 处理完成！")
    print(f"   🏋️ 训练集 ({OUTPUT_TRAIN}): {len(train_data)} 条")
    print(f"   📝 验证集 ({OUTPUT_VAL}): {len(val_data)} 条")

if __name__ == '__main__':
    # 确保文件夹存在
    if not os.path.exists(SOURCE_FOLDER):
        os.makedirs(SOURCE_FOLDER)
        print(f"⚠️ 已创建文件夹 {SOURCE_FOLDER}，请把你的 pkl 文件放进去再运行！")
    else:
        merge_and_split()