_base_ = [
    '../_base_/datasets/gamma_full.py',
    '../_base_/schedules/gamma.py',
    '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='FundusClassifier',
    fusion_mode='concat',
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(3,),
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet34')
    ),
    backbone_oct=dict(
        type='ResNet',
        depth=34,
        in_channels=256,
        num_stages=4,
        out_indices=(3,),
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet34')
    ),
    neck=dict(type='GlobalAveragePooling'),
    neck_oct=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=3,
        in_channels=512 * 2,  # 2 branch: fundus + oct
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    )
)
optim_wrapper = dict(optimizer=dict(_delete_=True, type='Adam', lr=1e-4))
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)
param_scheduler = dict(type='MultiStepLR', by_epoch=True, milestones=[40, ], gamma=0.1)
