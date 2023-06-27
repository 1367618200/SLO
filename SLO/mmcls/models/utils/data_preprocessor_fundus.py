import torch.nn as nn

from mmcls.registry import MODELS
from .data_preprocessor import BaseDataPreprocessor, ClsDataPreprocessor


@MODELS.register_module()
class ListClsDataPreprocessor(BaseDataPreprocessor):
    def __init__(self,
                 keys=['inputs'],
                 configs=[],
                 data_samples_index=0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(keys) == len(configs)

        self.keys = keys
        self.data_samples_index = data_samples_index

        samplers = []
        for config in configs:
            samplers.append(ClsDataPreprocessor(**config))
        self.samplers = nn.ModuleList(samplers)

    def forward(self, data: dict, training: bool = False) -> dict:
        outputs = dict()

        for i, key in enumerate(self.keys):
            with_data_samples = i == self.data_samples_index

            input_data = dict()
            input_data['inputs'] = data[key]
            input_data['data_samples'] = data['data_samples'] if with_data_samples else None

            res = self.samplers[i].forward(input_data, training)
            outputs[key] = res['inputs']
            if with_data_samples:
                outputs['data_samples'] = res['data_samples']
        return outputs
