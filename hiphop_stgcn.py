# -----------------------------------------------------------
# 最终修正版 hiphop_stgcn.py (针对 Skeleton 数据优化)
# -----------------------------------------------------------

# 1. 模型设置
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        graph_cfg=dict(layout='coco', mode='spatial'), # 17点骨骼布局
        in_channels=3), # x, y, score
    cls_head=dict(
        type='GCNHead',
        num_classes=2,       # <--- 你的分类数
        in_channels=256,
        loss_cls=dict(type='CrossEntropyLoss')),
    train_cfg=None,
    test_cfg=None)

# 2. 数据集设置
dataset_type = 'PoseDataset'
data_root = 'data/raw_video'
ann_file_train = 'hiphop_train.pkl'
ann_file_val = 'hiphop_val.pkl'

# --- 关键修改：精简后的 Pipeline (STGCN 专用) ---
train_pipeline = [
    dict(type='PoseDecode'),
    # STGCN 需要固定的时间长度，这里将所有视频统一采样为 100 帧
    dict(type='UniformSample', clip_len=100), 
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='PoseDecode'),
    dict(type='UniformSample', clip_len=100, num_clips=1, test_mode=True),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = val_dataloader

# 3. 评估与训练配置
val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=30, eta_min=0, by_epoch=True)
]

# 运行环境
default_scope = 'mmaction'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5, ignore_last=False), # 日志频率改快点，方便看进度
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, save_best='auto', max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook')
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ActionVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
work_dir = './work_dirs/hiphop_stgcn'