_base_ = [
    '../_base_/datasets/SLO.py',
    '../_base_/schedules/SLO.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        in_channels=3,
        style='pytorch',
        init_cfg=dict(type='Pretrained', 
                      checkpoint='torchvision://resnet50')
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiTaskHead',
        task_heads={
            'T0': dict(type='LinearClsHead', num_classes=2,loss=dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=[0,0])),#ME   [负样本，正样本]log输出顺序不改变这里的配置，如果是负样本比例大此处就要减少负样本权重
            'T1': dict(type='LinearClsHead', num_classes=2,loss=dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=[0.1,0.9])),#DR[0.1,0.9]
            'T2': dict(type='LinearClsHead', num_classes=3,loss=dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=[0,0,0])),#glaucoma[0.06,0.47,0.47]
            'T3': dict(type='LinearClsHead', num_classes=2,loss=dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=[0,0])),#cataract[0.14,0.86]
            'T4': dict(type='LinearClsHead', num_classes=2,loss=dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=[0,0])),#optic_atrophy[0.02,0.98]
        },
        in_channels=2048,
    ))
optim_wrapper = dict(optimizer=dict(type='SGD', lr=1e-3, momentum=0.9))
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
param_scheduler = dict(type='MultiStepLR', by_epoch=True, milestones=[80,], gamma=0.1)
