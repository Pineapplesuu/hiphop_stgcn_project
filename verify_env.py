import torch
from mmaction.registry import MODELS
from mmaction.utils import register_all_modules
import sys

def check_environment():
    print("------- 开始环境体检 -------")
    
    # 1. 注册所有模块 (这一步不报错说明 mmaction 安装成功)
    try:
        register_all_modules()
        print("✅ MMAction2 库加载成功")
    except Exception as e:
        print(f"❌ MMAction2 加载失败: {e}")
        return

    # 2. 定义一个最小化的 ST-GCN 模型配置
    config = dict(
        type='STGCN',
        in_channels=3,
        graph_cfg=dict(layout='coco', mode='spatial') # COCO 17点格式
    )

    # 3. 尝试构建模型
    try:
        model = MODELS.build(config)
        print("✅ ST-GCN 模型构建成功")
    except Exception as e:
        print(f"❌ 模型构建失败: {e}")
        return

    # 4. 捏造假数据 (Batch=1, Channel=3, Frames=50, Nodes=17, Person=1)
    fake_input = torch.randn(1, 3, 50, 17, 1)
    
    # 5. 尝试前向推断
    try:
        # 只要这一步能跑通，说明 PyTorch 和各种算子都没问题
        output = model(fake_input)
        print(f"✅ 前向推断成功！输出特征形状: {output.shape}")
        print("\n🎉🎉🎉 结论：环境完全健康！ 🎉🎉🎉")
        print("你的‘炼丹炉’已经造好了，现在唯一缺的就是‘药材’(数据)。")
        print("请安心等待队友给你 my_dance_data.pkl，不用再折腾报错了。")
    except Exception as e:
        print(f"❌ 推断失败: {e}")

if __name__ == '__main__':
    check_environment()