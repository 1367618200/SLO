# dataset settings
dataset_type = 'FundusDataset'
data_root = '../data/Glaucoma_grading/'
classes = ["non", "early", "mid_advanced"]

data_preprocessor = dict(
    type='ListClsDataPreprocessor',
    data_samples_index=0,
    keys=['inputs', 'inputs_oct'],
    configs=[
        # for fundus image
        dict(
            num_classes=len(classes),
            # mean=[0., 0., 0.],  # [R, G, B]
            # std=[255., 255., 255.],  # [R, G, B]
            mean=[66.735, 33.085, 10.229],  # [R, G, B]
            std=[71.145, 37.092, 15.033],  # [R, G, B]
            to_rgb=True,  # convert image from BGR to RGB
        ),
        # for oct image
        dict(
            # mean=[0.],  # [all depths]
            # std=[255.],  # [all depths]
            mean=[71.982],  # [all depths]
            std=[21.776],  # [all depths]
        )
    ]
)

train_pipeline = [
    # load fundus and oct image
    dict(type='LoadFundusImageFromFile'),

    # transforms for fundus image
    dict(type='RandomResizedCrop', scale=256, crop_ratio_range=(0.90, 1.1), aspect_ratio_range=(0.90, 1.1)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', prob=1.0, degree=30),

    # transforms for oct image
    dict(type='OctTransform', module=dict(type='CenterCrop', crop_size=(512, 512))),
    dict(type='OctTransform', module=dict(type='RandomFlip', prob=0.5, direction='horizontal')),
    dict(type='OctTransform', module=dict(type='RandomFlip', prob=0.5, direction='vertical')),

    # pack fundus and oct inputs
    dict(type='PackFundusClsInputs'),
]

test_pipeline = [
    # load fundus and oct image
    dict(type='LoadFundusImageFromFile'),

    # transforms for fundus image
    dict(type='CenterCropSquare'),
    dict(type='ResizeEdge', scale=256, edge='short'),

    # transforms for oct image
    dict(type='OctTransform', module=dict(type='CenterCrop', crop_size=(512, 512))),

    # pack fundus and oct inputs
    dict(type='PackFundusClsInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        classes=classes,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img_path='training/multi-modality_images',
                         oct_img_path='training/multi-modality_images'),
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        classes=classes,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img_path='training/multi-modality_images',
                         oct_img_path='training/multi-modality_images'),
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = [
    dict(type='Kappa'),
    dict(type='Accuracy', topk=(1,)),
    dict(type='SingleLabelMetric', average='micro', num_classes=len(classes))
]

# for labeled dataset (same as val)
# test_dataloader = val_dataloader
# test_evaluator = val_evaluator

# for unlabeled dataset
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        lazy_init=True,
        classes=classes,
        data_root=data_root,
        ann_file='annotations/test.json',  # annotation file without ground truth
        data_prefix=dict(img_path='testing/multi-modality_images',
                         oct_img_path='testing/multi-modality_images'),
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_evaluator = [
    dict(
        type='DumpCSVResults',
        csv_title=['data'] + classes  # ['data', 'non', 'early', 'mid_advanced']
    )
]
