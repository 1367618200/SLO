from mmcls.datasets.fundus import FundusDataset
from mmcls.datasets.transforms.formatting_fundus import LoadFundusImageFromFile, PackFundusClsInputs
from mmcls.datasets.transforms.processing import RandomResizedCrop
from mmcls.datasets.transforms.processing_fundus import OctTransform

if __name__ == '__main__':
    dataset = FundusDataset(
        ann_file='sample_annotation.json',
        data_root='./',
        data_prefix=dict(
            img_path='D:/file/server/root/userfolder/data/Glaucoma_grading/training/multi-modality_images/',
            oct_img_path='D:/file/server/root/userfolder/data/Glaucoma_grading/training/multi-modality_images/'
        ),
        classes=["non", "early", "mid_advanced"],
        pipeline=[
            LoadFundusImageFromFile(),
            # for fundus image
            RandomResizedCrop(scale=256, crop_ratio_range=(0.90, 1.1), aspect_ratio_range=(0.90, 1.1)),

            # for oct image
            OctTransform(module=dict(type='CenterCrop', crop_size=(512, 512))),
            PackFundusClsInputs(),
        ]
    )
    idx = 1
    print(dataset.get_data_info(idx))
    print(dataset.get_cat_ids(idx))
    sample = dataset.prepare_data(idx)

    for k, v in sample.items():
        try:
            print(k, v.shape)
        except Exception:
            print(k, v)
            pass
