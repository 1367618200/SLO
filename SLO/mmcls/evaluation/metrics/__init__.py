# Copyright (c) OpenMMLab. All rights reserved.
from .multi_label import AveragePrecision, MultiLabelMetric
from .multi_task import MultiTasksMetric
from .single_label import Accuracy, SingleLabelMetric
from .voc_multi_label import VOCAveragePrecision, VOCMultiLabelMetric


from .fundus import FundusMultiTasksMetric, FundusMultiDatasetsEvaluator
from .fundus import Kappa, DumpCSVResults,Accuracy_V1, SingleLabelMetric_V1  # NEW

from .single_label2 import  SingleLabelMetric2,AccuracyV3
__all__ = [
    'Accuracy', 'SingleLabelMetric', 'MultiLabelMetric', 'AveragePrecision',
    'MultiTasksMetric', 'VOCAveragePrecision', 'VOCMultiLabelMetric','FundusMultiTasksMetric'
]
