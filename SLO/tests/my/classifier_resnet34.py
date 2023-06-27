import matplotlib.pyplot as plt
from mmengine.dataset import DefaultSampler, default_collate
from torch.utils.data import DataLoader

from mmcls.datasets.fundus import FundusDataset
from mmcls.datasets.transforms.formatting_fundus import LoadFundusImageFromFile, PackFundusClsInputs
from mmcls.datasets.transforms.processing import RandomResizedCrop
from mmcls.datasets.transforms.processing_fundus import OctTransform, RandomRotate
from mmcls.models.classifiers.fundus_classifier import FundusClassifier
from mmcls.models.utils.data_preprocessor_fundus import ListClsDataPreprocessor


def load_dataset_sample(batch_size=1):
    dataset = FundusDataset(
        ann_file='sample_annotation.json',
        data_root='./',
        data_prefix=dict(
            img_path='D:/file/server/root/userfolder/data/Glaucoma_grading/training/multi-modality_images/',
            oct_img_path='D:/file/server/root/userfolder/data/Glaucoma_grading/training/multi-modality_images/'
        ),
        classes=["non", "early", "mid_advanced"],
        pipeline=[
            LoadFundusImageFromFile(),  # load fundus and oct file

            # for training
            # for fundus image
            RandomResizedCrop(scale=256, crop_ratio_range=(0.90, 1.1), aspect_ratio_range=(0.90, 1.1)),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', prob=0.5, direction='vertical'),
            RandomRotate(prob=1.0, degree=30),

            # for oct image
            OctTransform(module=dict(type='CenterCrop', crop_size=(512, 512))),
            OctTransform(module=dict(type='RandomFlip', prob=0.5, direction='horizontal')),
            OctTransform(module=dict(type='RandomFlip', prob=0.5, direction='vertical')),

            # # for testing
            # # for fundus image
            # CenterCropSquare(),
            # ResizeEdge(scale=256, edge='short'),
            # # for oct image
            # OctTransform(module=dict(type='CenterCrop', crop_size=(512, 512))),

            PackFundusClsInputs(),  # pack results of fundus and image
        ]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=DefaultSampler(dataset, shuffle=False),
        collate_fn=default_collate,
    )
    data = next(iter(dataloader))
    return data


def full_debug(model, data):
    print(data.keys())
    print(data['inputs'].shape)
    print(data['inputs_oct'].shape)

    for s in data['data_samples']:
        print(s.to_dict())
    print()

    # Train
    x = model.data_preprocessor(data, True)
    feat = model._run_forward(x, mode='tensor')
    print(feat)
    print()

    loss = model._run_forward(x, mode='loss')
    print(loss)
    print()

    # Test
    x = model.data_preprocessor(data, False)
    preds = model._run_forward(x, mode='predict')
    for pred in preds:
        pred = pred.to_dict()
        print(pred['pred_label'], pred['gt_label'])
    print()


def visual_tensor(tensor):
    img = tensor[0].numpy().transpose(1, 2, 0)

    if img.shape[-1] == 3:
        plt.imshow(img)
    else:
        plt.imshow(img[:, :, 100], cmap='gray')
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    model = FundusClassifier(
        backbone=dict(
            type='ResNet',
            depth=34,
            num_stages=4,
            out_indices=(3,),
            style='pytorch'
        ),
        backbone_oct=dict(
            type='ResNet',
            depth=34,
            in_channels=256,
            num_stages=4,
            out_indices=(3,),
            style='pytorch'
        ),
        neck=dict(type='GlobalAveragePooling'),
        neck_oct=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=3,
            in_channels=512 * 2,  # 2 branch
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
        ),
        data_preprocessor=ListClsDataPreprocessor(
            # type='ListClsDataPreprocessor',
            keys=['inputs', 'inputs_oct'],
            configs=[
                # for fundus image
                dict(
                    num_classes=3,
                    mean=[0., 0., 0.],  # [R, G, B]
                    std=[255., 255., 255.],  # [R, G, B]
                    # mean=[66.735, 33.085, 10.229],  # [R, G, B]
                    # std=[71.145, 37.092, 15.033],  # [R, G, B]
                    to_rgb=True,  # convert image from BGR to RGB
                ),
                # for oct image
                dict(
                    mean=[0.],  # [all depths]
                    std=[255.],  # [all depths]
                    # mean=[71.982],  # [all depths]
                    # std=[21.776],  # [all depths]
                )
            ]
        ),
    )
    batch_size = 2
    data = load_dataset_sample(batch_size)
    data = model.data_preprocessor(data, True)
    x1 = data['inputs']
    x2 = data['inputs_oct']

    print(x1.shape)
    print(x2.shape)
    visual_tensor(x1)
    visual_tensor(x2)

    print(model)
    full_debug(model, data)
