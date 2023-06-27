import warnings

import mmcv
import numpy as np
from mmcv.transforms import CenterCrop, Pad
from mmcv.transforms.base import BaseTransform

from mmcls.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
import cv2
from .auto_augment import BaseAugTransform

@TRANSFORMS.register_module()
class OctTransform(BaseTransform):
    def __init__(self, module,
                 key_prefix='oct_', key_fields=['img', 'img_shape', 'ori_shape']):
        super().__init__()
        self.key_prefix = key_prefix
        self.key_fields = key_fields

        if isinstance(module, dict):
            module = TRANSFORMS.build(module)
        self.module = module

    def transform(self, results, *args, **kwargs):
        inputs = dict()

        for key in results.keys():
            if key in self.key_fields:
                new_key = self.key_prefix + key
                inputs[key] = results[new_key]  # img <- oct_img

        outputs = self.module(inputs, *args, **kwargs)

        for key in outputs.keys():
            new_key = self.key_prefix + key
            results[new_key] = outputs[key]  # oct_img <- img
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += '(' + str(self.module) + ')'
        return repr_str


# Based on: https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/datasets/pipelines/transforms.py#L657
@TRANSFORMS.register_module()
class RandomRotate(BaseTransform):
    """Rotate the image.
    Args:
        prob (float): The rotation probability.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
        pad_val (float, optional): Padding value of image. Default: 0.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used. Default: None.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image. Default: False
    """

    def __init__(self,
                 prob,
                 degree,
                 pad_val=0,
                 center=None,
                 auto_bound=False):
        self.prob = prob
        assert prob >= 0 and prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'
        self.pal_val = pad_val
        self.center = center
        self.auto_bound = auto_bound

    def transform(self, results):
        rotate = True if np.random.rand() < self.prob else False
        degree = np.random.uniform(min(*self.degree), max(*self.degree))
        if rotate:
            # rotate image
            results['img'] = mmcv.imrotate(
                results['img'],
                angle=degree,
                border_value=self.pal_val,
                center=self.center,
                auto_bound=self.auto_bound)
            results['rotate_degree'] = degree # NEW
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, ' \
                    f'degree={self.degree}, ' \
                    f'pad_val={self.pal_val}, ' \
                    f'center={self.center}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str


# Based on mmcv.transforms.processing.CenterCrop
@TRANSFORMS.register_module()
class CenterCropSquare(CenterCrop):
    def __init__(self,
                 crop_size=None,
                 auto_pad: bool = False,
                 pad_cfg: dict = dict(type='Pad'),
                 clip_object_border: bool = True) -> None:
        super(CenterCrop, self).__init__()  # use init of BaseTransform

        self.auto_pad = auto_pad

        self.pad_cfg = pad_cfg.copy()
        # size will be overwritten
        if 'size' in self.pad_cfg and auto_pad:
            warnings.warn('``size`` is set in ``pad_cfg``,'
                          'however this argument will be overwritten'
                          ' according to crop size and image size')
        self.clip_object_border = clip_object_border

    def transform(self, results: dict) -> dict:
        """Apply center crop on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: Results with CenterCropped image and semantic segmentation
            map.
        """
        assert 'img' in results, '`img` is not found in results'
        img = results['img']
        # img.shape has length 2 for grayscale, length 3 for color
        img_height, img_width = img.shape[:2]

        crop_height = crop_width = min(img_height, img_width)  # NEW

        y1 = max(0, int(round((img_height - crop_height) / 2.)))
        x1 = max(0, int(round((img_width - crop_width) / 2.)))
        y2 = min(img_height, y1 + crop_height) - 1
        x2 = min(img_width, x1 + crop_width) - 1
        bboxes = np.array([x1, y1, x2, y2])

        # crop the image
        self._crop_img(results, bboxes)
        # crop the gt_seg_map
        self._crop_seg_map(results, bboxes)
        # crop the bounding box
        self._crop_bboxes(results, bboxes)
        # crop the keypoints
        self._crop_keypoints(results, bboxes)
        return results

@TRANSFORMS.register_module()
class CenterPad(Pad):
    def _pad_img(self, results: dict) -> None:
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get('img', 0)

        size = None
        if self.pad_to_square:
            max_size = max(results['img'].shape[:2])
            size = (max_size, max_size)
        if self.size_divisor is not None:
            if size is None:
                size = (results['img'].shape[0], results['img'].shape[1])
            pad_h = int(np.ceil(
                size[0] / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(
                size[1] / self.size_divisor)) * self.size_divisor
            size = (pad_h, pad_w)
        elif self.size is not None:
            size = self.size[::-1]
        if isinstance(pad_val, int) and results['img'].ndim == 3:
            pad_val = tuple(pad_val for _ in range(results['img'].shape[2]))

        # NEW
        h, w = results['img'].shape[:2]
        pad_h, pad_w = size
        left = max(0, (pad_w - w) // 2)
        right = max(0, pad_w - w - left)
        top = max(0, (pad_h - h) // 2)
        bottom = max(0, pad_h - h - top)

        padded_img = mmcv.impad(
            results['img'],
            # shape=size,
            padding=(left, top, right, bottom),
            pad_val=pad_val,
            padding_mode=self.padding_mode)

        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
        results['img_shape'] = padded_img.shape[:2]

@TRANSFORMS.register_module()
class CLAHE(BaseAugTransform):
    """  Contrast-limited adaptive histogram equalization

    """

    def __init__(self,  **kwargs):
        super().__init__( **kwargs)

    def transform(self, results):
        """Apply transform to results."""
        if self.random_disable():
            return results

        img = results['img']
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)# 将彩色图像转换为LAB颜色空间
        l_channel, a_channel, b_channel = cv2.split(lab_img)# 分割LAB图像的亮度和色度通道
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel_equalized = clahe.apply(l_channel)
        lab_img_equalized = cv2.merge((l_channel_equalized, a_channel, b_channel))# 合并亮度和色度通道
        equalized_img = cv2.cvtColor(lab_img_equalized, cv2.COLOR_LAB2BGR)# 将图像转换回BGR颜色空间
        results['img'] = equalized_img.astype(lab_img_equalized.dtype)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str



