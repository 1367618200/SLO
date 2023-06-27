# dataset settings
_base_ = [
    './gamma.py'
]

train_dataloader = dict(
    dataset=dict(
        ann_file='annotations/train_full.json',
        data_prefix=dict(img_path='training/multi-modality_images',
                         oct_img_path='training/multi-modality_images')
    )
)

val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/train_full.json',
        data_prefix=dict(img_path='training/multi-modality_images',
                         oct_img_path='training/multi-modality_images')
    )
)

# test_dataloader = val_dataloader

test_dataloader = dict(
    dataset=dict(
        ann_file='annotations/test.json',  # annotation file without ground truth
        data_prefix=dict(img_path='testing/multi-modality_images',
                         oct_img_path='testing/multi-modality_images')
    )
)
