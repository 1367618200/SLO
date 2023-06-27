# dataset settings
dataset_type = 'FundusDataset'
multi_dataset_type = 'ConcatEvalDataset'
data_root_path = '/home1/commonfile/CSURetina10K'
ann_file_path = '/home/chenqiongpu/SLO/anno_file/annotations'

# data_preprocessor = dict(
#     num_classes=5,
#     mean=[9.724, 18.767, 0.480],  # [R, G, B]
#     std=[9.775, 16.141, 0.699],  # [R, G, B]
#     to_rgb=True,  # convert image from BGR to RGB
# )

train_pipeline = [
    # load image
    dict(type='LoadImageFromFile'),

    dict(type='ResizeEdge', scale=512, edge='short'),
    dict(type='CenterCrop',crop_size=(640,512)),
    dict(type='CenterPad',size=(640,512),pad_val=0),
    dict(type='RandomRotate', prob=1.0, degree=30),#旋转

    dict(type='RandomFlip', prob=0.5, direction='horizontal'),#水平翻转
    dict(type='RandomFlip', prob=0.5, direction='vertical'),#垂直翻转
    dict(type='Brightness', magnitude_range=(0, 0.9)),
    # # dict(type='SaveDebugImage', prefix='unit_train_1_preprocessing', output_root='./UnitTestResults-SLO/UnitTest_train'),
    dict(type='CLAHE'),#均衡图像直方图。
    # dict(type='SaveDebugImage', prefix='unit_train_2_preprocessing', output_root='./UnitTestResults-SLO/UnitTest_train'),

    # dict(type='SaveDebugImage', prefix='unit_train_1_preprocessing', output_root='./UnitTestResults-SLO/UnitTest_train'),
    dict(type='PackMultiTaskInputs', multi_task_fields=('gt_label',)),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='SaveDebugImage', prefix='unit_val_0_preprocessing', output_root='./UnitTestResults-SLO/UnitTest_val'),
    dict(type='ResizeEdge', scale=512, edge='short'),
    dict(type='CenterCrop',crop_size=(640,512)),
    dict(type='CenterPad',size=(640,512),pad_val=0),
    # dict(type='SaveDebugImage', prefix='unit_val_1_preprocessing', output_root='./UnitTestResults-SLO/UnitTest_val'),
    dict(type='PackMultiTaskInputs', multi_task_fields=('gt_label',)),
]

test_pipeline = [
    # load image
    dict(type='LoadImageFromFile'),
    # dict(type='SaveDebugImage', prefix='unit_test_0_preprocessing', output_root='./UnitTestResults-SLO/UnitTest_test'),
    dict(type='ResizeEdge', scale=512, edge='short'),
    dict(type='CenterCrop',crop_size=(640,512)),
    dict(type='CenterPad',size=(640,512),pad_val=0),
    # dict(type='SaveDebugImage', prefix='unit_test_1_preprocessing', output_root='./UnitTestResults-SLO/UnitTest_test'),
    dict(type='PackMultiTaskInputs', multi_task_fields=('gt_label',)),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        data_root=data_root_path,
        ann_file=ann_file_path+'/train.json',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=16,
    dataset=dict(
        type=multi_dataset_type,
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root_path,
                ann_file=ann_file_path+'/train.json',
                pipeline=train_pipeline
            ),
            dict(
                type=dataset_type,
                data_root=data_root_path,
                ann_file=ann_file_path+'/val.json',
                pipeline=val_pipeline
            ),
            dict(
                type=dataset_type,
                data_root=data_root_path,
                ann_file=ann_file_path+'/test.json',
                pipeline=test_pipeline
            )
        ]
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(
    # type='MultiTasksMetric',
    type='FundusMultiDatasetsEvaluator',
    dataset_prefixes=['train', 'val', 'test'],  # 这里要和上面 val_dataloader 里对应
    # value_format='{:5.2f}'#增加参数

    metrics=dict(
        type='FundusMultiTasksMetric',
        task_metrics={
            'T0': [
                # dict(type='Accuracy_V1',topk=(1,)),
                dict(type='SingleLabelMetric_V1', average=None),  # , prefix='pr', num_classes=2
                dict(type='Kappa'),
            ],
            'T1': [
                # dict(type='Accuracy_V1',topk=(1,)),
                dict(type='SingleLabelMetric_V1', average=None),
                dict(type='Kappa'),
            ],
            'T2': [
                # dict(type='Accuracy_V1',topk=(1,)),
                dict(type='SingleLabelMetric_V1', average=None),
                dict(type='Kappa'),
            ],
            'T3': [
                # dict(type='Accuracy_V1',topk=(1,)),
                dict(type='SingleLabelMetric_V1', average=None),
                dict(type='Kappa'),
            ],
            'T4': [
                # dict(type='Accuracy_V1',topk=(1,)),
                dict(type='SingleLabelMetric_V1', average=None),
                dict(type='Kappa'),
            ],
        }
    )
)

test_dataloader = dict(
    batch_size=16,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        data_root=data_root_path,
        ann_file=ann_file_path+'/test.json',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = dict(
    type='FundusMultiTasksMetric',
    task_metrics={
        'T0': [
            dict(type='SingleLabelMetric_V1', average=None),
            dict(type='Kappa'),
        ],
        'T1': [
            dict(type='SingleLabelMetric_V1', average=None),
            dict(type='Kappa'),
        ],
        'T2': [
            dict(type='SingleLabelMetric_V1', average=None),
            dict(type='Kappa'),
        ],
        'T3': [
            dict(type='SingleLabelMetric_V1', average=None),
            dict(type='Kappa'),
        ],
        'T4': [
            dict(type='SingleLabelMetric_V1', average=None),
            dict(type='Kappa'),
        ],
    }
)
