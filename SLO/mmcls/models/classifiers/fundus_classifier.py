from typing import List, Optional, Union

import torch
import torch.nn as nn

from mmcls.registry import MODELS
from mmcls.structures import ClsDataSample
from .base import BaseClassifier


# Based on: mmcls/models/classifiers/image.py  ImageClassifier
@MODELS.register_module()
class FundusClassifier(BaseClassifier):
    def __init__(
            self,
            backbone,
            backbone_oct: Optional[dict] = None,  # NEW

            neck: Optional[dict] = None,
            neck_oct: Optional[dict] = None,  # NEW

            head: Optional[dict] = None,
            pretrained: Optional[str] = None,
            train_cfg: Optional[dict] = None,
            data_preprocessor: Optional[Union[dict, nn.Module]] = None,  # NEW
            init_cfg: Optional[dict] = None,

            fusion_mode='concat',  # NEW
    ):

        if pretrained is not None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        if data_preprocessor is None:
            data_preprocessor = {}

        if isinstance(data_preprocessor, dict):  # NEW
            # The build process is in MMEngine, so we need to add scope here.
            data_preprocessor.setdefault('type', 'mmcls.ClsDataPreprocessor')

            if train_cfg is not None and 'augments' in train_cfg:
                # Set batch augmentations by `train_cfg`
                data_preprocessor['batch_augments'] = train_cfg

        super(FundusClassifier, self).__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        if not isinstance(backbone, nn.Module):
            backbone = MODELS.build(backbone)
        if neck is not None and not isinstance(neck, nn.Module):
            neck = MODELS.build(neck)
        if head is not None and not isinstance(head, nn.Module):
            head = MODELS.build(head)

        # NEW
        if not isinstance(backbone_oct, nn.Module):
            backbone_oct = MODELS.build(backbone_oct)
        if neck_oct is not None and not isinstance(neck_oct, nn.Module):
            neck_oct = MODELS.build(neck_oct)
        # END NEW

        self.backbone = backbone
        self.backbone_oct = backbone_oct  # NEW
        self.neck = neck
        self.neck_oct = neck_oct  # NEW
        self.head = head

        self.fusion_mode = fusion_mode  # NEW

    def forward(self,
                inputs: torch.Tensor,
                inputs_oct: Optional[torch.Tensor] = None,  # NEW
                data_samples: Optional[List[ClsDataSample]] = None,
                mode: str = 'tensor'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
          tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`ClsDataSample`.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            inputs_oct (torch.Tensor): (N, D, H, W)  # NEW
            data_samples (List[ClsDataSample], optional): The annotation
                data of every samples. It's required if ``mode="loss"``.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of
              :obj:`mmcls.structures.ClsDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'tensor':
            feats = self.extract_feat(inputs, inputs_oct=inputs_oct)  # NEW
            return self.head(feats) if self.with_head else feats
        elif mode == 'loss':
            return self.loss(inputs, inputs_oct=inputs_oct, data_samples=data_samples)  # NEW
        elif mode == 'predict':
            return self.predict(inputs, inputs_oct=inputs_oct, data_samples=data_samples)  # NEW
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    # NEW
    def _fundus_oct_fusion(self, x, x_oct):
        assert len(x) == len(x_oct)
        if self.fusion_mode == 'concat':
            out = []
            for fundus, oct in zip(x, x_oct):
                out.append(torch.cat((fundus, oct), dim=1))
            return tuple(out)
        else:
            raise NotImplementedError(self.fusion_mode)
    # END NEW

    def extract_feat(self, inputs, inputs_oct=None, stage='neck'):
        """Extract features from the input tensor with shape (N, C, ...).

        Args:
            inputs (Tensor): A batch of inputs. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
            stage (str): Which stage to output the feature. Choose from:

                - "backbone": The output of backbone network. Returns a tuple
                  including multiple stages features.
                - "neck": The output of neck module. Returns a tuple including
                  multiple stages features.
                - "pre_logits": The feature before the final classification
                  linear layer. Usually returns a tensor.

                Defaults to "neck".

        Returns:
            tuple | Tensor: The output of specified stage.
            The output depends on detailed implementation. In general, the
            output of backbone and neck is a tuple and the output of
            pre_logits is a tensor.
        """

        assert stage in ['backbone', 'neck', 'pre_logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')

        x = self.backbone(inputs)
        x_oct = self.backbone_oct(inputs_oct)  # NEW
        if stage == 'backbone':
            x = self._fundus_oct_fusion(x, x_oct)  # NEW
            return x

        if self.with_neck:
            x = self.neck(x)
            x_oct = self.neck(x_oct)  # NEW
        if stage == 'neck':
            x = self._fundus_oct_fusion(x, x_oct)  # NEW
            return x

        assert self.with_head and hasattr(self.head, 'pre_logits'), \
            "No head or the head doesn't implement `pre_logits` method."
        return self.head.pre_logits(x)

    def loss(self, inputs: torch.Tensor,
             inputs_oct: torch.Tensor = None,  # NEW
             data_samples: List[ClsDataSample] = None) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            inputs_oct (torch.Tensor): (N, D, H, W)  # NEW
            data_samples (List[ClsDataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feats = self.extract_feat(inputs, inputs_oct)
        return self.head.loss(feats, data_samples)

    def predict(self,
                inputs: torch.Tensor,
                inputs_oct: torch.Tensor,  # NEW
                data_samples: Optional[List[ClsDataSample]] = None,
                **kwargs) -> List[ClsDataSample]:
        """Predict results from a batch of inputs.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            inputs_oct (torch.Tensor): (N, D, H, W)  # NEW
            data_samples (List[ClsDataSample], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.
        """
        feats = self.extract_feat(inputs, inputs_oct)
        return self.head.predict(feats, data_samples, **kwargs)
