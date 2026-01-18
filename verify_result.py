import torch
from mmengine.config import Config
from mmaction.apis import init_recognizer
from mmaction.registry import DATASETS
from mmengine.runner import Runner
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置 =================
CONFIG_FILE = 'hiphop_stgcn.py'
CHECKPOINT = 'work_dirs/hiphop_stgcn_result/epoch_50.pth' # 确保文件名对
VAL_PKL = 'hiphop_val.pkl'
# 你的 6 个类别名 (顺序必须和 generate_pkl.py 一致)
CLASSES = ['others','freeze']

def verify():
    # 1. 加载配置和模型
    print("🚀 正在加载模型...")
    cfg = Config.fromfile(CONFIG_FILE)
    model = init_recognizer(cfg, CHECKPOINT, device='cpu')
    
    # 2. 加载验证数据
    with open(VAL_PKL, 'rb') as f:
        val_data = pickle.load(f)
    
    y_true = []
    y_pred = []
    
    print(f"🔍 开始测试 {len(val_data)} 条验证集数据...")
    
    # 3. 逐条推理
    for i, item in enumerate(val_data):
        # 伪造一个数据结构喂给模型
        fake_anno = dict(
            frame_dir=item['frame_dir'],
            total_frames=item['total_frames'],
            img_shape=item['img_shape'],
            original_shape=item['original_shape'],
            start_index=0,
            label=-1,
            keypoint=item['keypoint']
        )
        
        # 使用 inference 接口 (需要稍微包装一下数据)
        # 这里为了简单，我们直接用 test_step 或者简易 pipeline
        # 但最稳妥的是直接用 demo 里的逻辑
        from mmaction.apis import inference_recognizer
        # 保存为临时文件让 inference_recognizer 读取不太方便，
        # 我们直接复用 demo_realtime.py 里的核心推理逻辑
        
        # --- 核心推理 Hack ---
        # 构造 batch
        from mmengine.dataset import Compose, pseudo_collate
        pipeline = Compose(cfg.test_dataloader.dataset.pipeline)
        data = pipeline(fake_anno)
        data = pseudo_collate([data])
        
        with torch.no_grad():
            result = model.test_step(data)[0]
            scores = result.pred_score.cpu().numpy()
            pred_label = np.argmax(scores)
        
        y_true.append(item['label'])
        y_pred.append(pred_label)
        
        status = "✅" if item['label'] == pred_label else "❌"
        print(f"[{i+1}/{len(val_data)}] 真实: {CLASSES[item['label']]} -> 预测: {CLASSES[pred_label]} {status}")

    # 4. 生成报告
    print("\n" + "="*40)
    print("📊 最终体检报告")
    print("="*40)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    print("\n混淆矩阵 (行=真实, 列=预测):")
    print(cm)
    
    # 详细指标
    print("\n详细指标:")
    print(classification_report(y_true, y_pred, target_names=CLASSES, labels=range(len(CLASSES)), zero_division=0))
    # 5. 简单的绘图建议
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Hiphop AI Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        print("\n🖼️ 混淆矩阵图已保存为: confusion_matrix.png")
    except:
        print("⚠️ 绘图库缺失，跳过绘图 (不影响结果)")

if __name__ == '__main__':
    verify()