from mmcls.registry import DATASETS
from .base_dataset import BaseDataset

from mmengine.dataset import ConcatDataset as _ConcatDataset #NEW

# reserved for future purpose

# README: https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/advanced_tutorials/basedataset.md
@DATASETS.register_module()
class FundusDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

@DATASETS.register_module()
class ConcatEvalDataset(_ConcatDataset):
    def __init__(self, *args, **kwargs):
        super(ConcatEvalDataset, self).__init__(*args, **kwargs)

        # REF: https://github.com/open-mmlab/mmocr/blob/1.x/mmocr/datasets/dataset_wrapper.py#L77
        if not kwargs.get('lazy_init', False):
            self._metainfo.update(dict(cumulative_sizes=self.cumulative_sizes))