import os
from collections.abc import Sequence
from typing import Optional
import cv2
from mmengine.logging import MMLogger
from mmengine.dist import get_rank


import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms.loading import LoadImageFromFile

from mmcls.registry import TRANSFORMS
from mmcls.structures import ClsDataSample
from .formatting import PackClsInputs, to_tensor


@TRANSFORMS.register_module()
class LoadFundusImageFromFile(LoadImageFromFile):
    def __init__(self,
                 oct_extensions=('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif'),
                 *args, **kwargs):
        super(LoadFundusImageFromFile, self).__init__(*args, **kwargs)
        self.oct_extensions = oct_extensions

    # Based on mmcv.transforms.loading.LoadImageFromFile.transform
    def _load_img_file(self, filename, color_type):
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        if self.to_float32:
            img = img.astype(np.float32)
        return img

    def transform(self, results: dict) -> Optional[dict]:
        # load fundus image
        results = super().transform(results)

        # load oct images
        oct_img_list = []
        oct_img_path = results['oct_img_path']
        for file in os.listdir(oct_img_path):
            if not file.endswith(self.oct_extensions):
                continue
            filename = os.path.join(oct_img_path, file)
            img = self._load_img_file(filename, color_type='grayscale')
            oct_img_list.append(img)

        oct_img = np.stack(oct_img_list, -1)
        results['oct_img'] = oct_img
        results['oct_img_shape'] = oct_img.shape[:2]
        results['oct_ori_shape'] = oct_img.shape[:2]
        return results


@TRANSFORMS.register_module()
class PackFundusClsInputs(PackClsInputs):
    """Pack the inputs data for the classification.

    **Required Keys:**

    - img
    - gt_label (optional)
    - ``*meta_keys`` (optional)

    **Deleted Keys:**

    All keys in the dict.

    **Added Keys:**

    - inputs (:obj:`torch.Tensor`): The forward data of models.
    - data_samples (:obj:`~mmcls.structures.ClsDataSample`): The annotation
      info of the sample.

    Args:
        meta_keys (Sequence[str]): The meta keys to be saved in the
            ``metainfo`` of the packed ``data_samples``.
            Defaults to a tuple includes keys:

            - ``sample_idx``: The id of the image sample.
            - ``img_path``: The path to the image file.
            - ``ori_shape``: The original shape of the image as a tuple (H, W).
            - ``img_shape``: The shape of the image after the pipeline as a
              tuple (H, W).
            - ``scale_factor``: The scale factor between the resized image and
              the original image.
            - ``flip``: A boolean indicating if image flip transform was used.
            - ``flip_direction``: The flipping direction.
    """

    def __init__(self, meta_keys=('sample_idx', 'img_path', 'ori_shape', 'img_shape',
                                  'scale_factor', 'flip', 'flip_direction'),
                 oct_meta_keys=('oct_img_path', 'oct_ori_shape', 'oct_img_shape',
                                'oct_scale_factor', 'oct_flip', 'oct_flip_direction')):
        super().__init__(meta_keys)
        if oct_meta_keys is not None:
            self.meta_keys += oct_meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()

        # prepare fundus tensor inputs
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            packed_results['inputs'] = to_tensor(img)

        # NEW
        # prepare oct tensor inputs
        if 'oct_img' in results:
            oct_img = results['oct_img']
            if len(oct_img.shape) < 3:
                oct_img = np.expand_dims(oct_img, -1)
            oct_img = np.ascontiguousarray(oct_img.transpose(2, 0, 1))
            packed_results['inputs_oct'] = to_tensor(oct_img)
        # END NEW

        data_sample = ClsDataSample()
        if 'gt_label' in results:
            gt_label = results['gt_label']
            data_sample.set_gt_label(gt_label)

        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        return packed_results


@TRANSFORMS.register_module()
class SaveDebugImage:
    def __init__(self, prefix='step0', output_root='./UnitTestResults-SLO'):
        """
        Args:
            prefix: 可以是任意数字或字符串，指定调用当前 SaveDebugImage 时保存的文件夹名称
        """
        self.prefix = prefix

        self.output_path = os.path.join(output_root, f'{prefix}')
        os.makedirs(self.output_path, exist_ok=True)

        self.debug_index = 0

    def __call__(self, results):
        if get_rank() != 0:
            # only save debug results on gpu 0
            return results

        if 'img' in results:
            # print keys of transform parameters
            logger = MMLogger.get_current_instance()
            print_results = dict()
            for k, v in results.items():
                if k == 'img' or 'oct_' in k:
                    continue
                print_results[k] = v
            logger.info('[{}]  prefix: {}  {}'.format(self.debug_index, self.prefix, print_results))

            # save results
            img = results['img']
            filename = os.path.basename(results['img_path'])

            output_path = os.path.join(self.output_path, f'{self.debug_index}_{filename}')
            cv2.imwrite(output_path, img)

        self.debug_index += 1
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class SaveDebugOctImage(SaveDebugImage):
    def __call__(self, results):
        if get_rank() != 0:
            # only save debug results on gpu 0
            return results

        if 'oct_img' in results:
            # print keys of transform parameters
            logger = MMLogger.get_current_instance()
            print_results = dict()
            for k, v in results.items():
                if k == 'oct_img' or 'oct_' not in k:
                    continue
                print_results[k] = v
            logger.info('[{}]  prefix: {}  {}'.format(self.debug_index, self.prefix, print_results))

            # save results
            oct_img = results['oct_img']
            filedir = os.path.basename(results['oct_img_path'])
            output_path = os.path.join(self.output_path, f'{self.debug_index}_{filedir}')
            os.makedirs(output_path, exist_ok=True)

            oct_img = oct_img.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
            for index, img in enumerate(oct_img):
                cv2.imwrite(os.path.join(output_path, f'{index}.png'), img)

            self.debug_index += 1
        return results