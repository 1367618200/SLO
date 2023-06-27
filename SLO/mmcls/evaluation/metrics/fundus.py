import os
import warnings
from typing import Dict
from typing import List, Optional
from typing import Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.dist import (broadcast_object_list, collect_results, is_main_process)
from mmengine.evaluator import BaseMetric, Evaluator, DumpResults
from mmengine.evaluator.metric import _to_cpu
from mmengine.fileio import dump
from mmengine.logging import MMLogger, print_log
from sklearn.metrics import cohen_kappa_score

from mmcls.evaluation.metrics.multi_task import MultiTasksMetric
from mmcls.evaluation.metrics.single_label import Accuracy, SingleLabelMetric, to_tensor
from mmcls.registry import METRICS, EVALUATORS


# Based on: mmcls.evaluation.metrics.single_label  SingleLabelMetric()
@METRICS.register_module()
class Kappa(BaseMetric):
    r"""Kappa

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    default_prefix: Optional[str] = ''

    def __init__(self,
                 num_classes: Optional[int] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.num_classes = num_classes

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()
            pred_label = data_sample['pred_label']
            gt_label = data_sample['gt_label']
            if 'score' in pred_label:
                result['pred_score'] = pred_label['score'].cpu()
            else:
                num_classes = self.num_classes or data_sample.get(
                    'num_classes')
                # assert num_classes is not None, \
                #     'The `num_classes` must be specified if `pred_label` has ' \
                #     'only `label`.'
                result['pred_label'] = pred_label['label'].cpu()
                result['num_classes'] = num_classes
            result['gt_label'] = gt_label['label'].cpu()

            # NEW
            if result['gt_label'] == -100:
                continue
            # END NEW

            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method. `self.results`
        # are a list of results from multiple batch, while the input `results`
        # are the collected results.
        metrics = {}

        # concat
        target = torch.cat([res['gt_label'] for res in results])
        if 'pred_score' in results[0]:
            pred = torch.stack([res['pred_score'] for res in results])
        else:
            # If only label in the `pred_label`.
            pred = torch.cat([res['pred_label'] for res in results])

        kappa = self.calculate(pred, target)
        metrics['kappa'] = kappa
        return metrics

    @staticmethod
    def calculate(
            pred: Union[torch.Tensor, np.ndarray, Sequence],
            target: Union[torch.Tensor, np.ndarray, Sequence],
    ) -> float:
        """Calculate the precision, recall, f1-score and support.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. It can be labels (N, ), or scores of every
                class (N, C).
            target (torch.Tensor | np.ndarray | Sequence): The target of
                each prediction with shape (N, ).

        Returns:
            - float: 100. * kappa
        """

        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.int64)
        assert pred.size(0) == target.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match " \
            f'the target ({target.size(0)}).'

        if pred.ndim == 1:
            pred_label = pred.to(torch.int64).tolist()
            target = target.tolist()
        else:
            pred_score, pred_label = torch.topk(pred, k=1)
            pred_score = pred_score.flatten().tolist()
            pred_label = pred_label.flatten().tolist()
            target = target.flatten().tolist()

        kappa = cohen_kappa_score(target, pred_label, weights='quadratic')
        return 100. * kappa


# Based on: mmengine.evaluator.DumpResults
@METRICS.register_module()
class DumpCSVResults(DumpResults):
    default_prefix: Optional[str] = ''

    def __init__(self,
                 out_file_path,
                 csv_title=None,
                 is_dump_pkl=True,
                 collect_device='cpu') -> None:
        super(DumpResults, self).__init__(collect_device=collect_device)
        if not out_file_path.endswith(('.csv',)):
            raise ValueError('The output file must be a csv file.')
        self.csv_title = csv_title
        self.is_dump_pkl = is_dump_pkl
        self.out_file_path = out_file_path

    def compute_metrics(self, results: list) -> dict:
        """dump the prediction results to a pickle file."""

        if self.is_dump_pkl:
            out_pkl_path = self.out_file_path.replace('.csv', '.pkl')
            dump(results, out_pkl_path)
            print_log(f'Results has been saved to {out_pkl_path}.', logger='current')

        csv_data = []
        for x in results:
            name = os.path.basename(x['img_path']).split('.')[0]
            pred_label = x['pred_label']['label']  # tensor(int)  Shape: torch.Size([1])
            pred_score = x['pred_label']['score']  # tensor(float,)  Shape: torch.Size([num_classes])
            one_hot_label = F.one_hot(pred_label, num_classes=len(pred_score))
            one_hot_label = one_hot_label.flatten().tolist()
            csv_data.append([name] + one_hot_label)

        df = pd.DataFrame(csv_data, columns=self.csv_title)
        df.to_csv(self.out_file_path, index=False)
        print_log(f'Results has been saved to {self.out_file_path}.', logger='current')
        return {}


@METRICS.register_module()
class Accuracy_V1(Accuracy):
    def process(self, data_batch, data_samples: Sequence[dict]):
        for data_sample in data_samples:
            result = dict()
            pred_label = data_sample['pred_label']
            gt_label = data_sample['gt_label']

            if 'score' in pred_label:
                result['pred_score'] = pred_label['score'].cpu()
            else:
                result['pred_label'] = pred_label['label'].cpu()
            result['gt_label'] = gt_label['label'].cpu()

            # NEW
            if result['gt_label'] == -100:
                continue
            # END NEW

            # Save the result to `self.results`.
            self.results.append(result)


@METRICS.register_module()
class SingleLabelMetric_V1(SingleLabelMetric):
    def process(self, data_batch, data_samples: Sequence[dict]):
        for data_sample in data_samples:
            result = dict()
            pred_label = data_sample['pred_label']
            gt_label = data_sample['gt_label']

            if 'score' in pred_label:
                result['pred_score'] = pred_label['score'].cpu()
            else:
                num_classes = self.num_classes or data_sample.get(
                    'num_classes')
                assert num_classes is not None, \
                    'The `num_classes` must be specified if `pred_label` has ' \
                    'only `label`.'
                result['pred_label'] = pred_label['label'].cpu()
                result['num_classes'] = num_classes
            result['gt_label'] = gt_label['label'].cpu()

            # NEW
            if result['gt_label'] == -100:
                continue
            # END NEW

            # Save the result to `self.results`.
            self.results.append(result)


@METRICS.register_module()
class FundusMultiTasksMetric(MultiTasksMetric):
    def __init__(self,
                 value_format='{:6.2f}',
                 task_format='precision[{}] recall[{}] f-score[{}] kappa[{}]',
                 is_reverse_results=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.value_format = value_format
        self.task_format = task_format
        self.is_reverse_results = is_reverse_results

    def format_results(self, results, prefix='single-label/'):
        precision = results.get(f'{prefix}precision_classwise', None)
        recall = results.get(f'{prefix}recall_classwise', None)
        f1_score = results.get(f'{prefix}f1-score_classwise', None)
        kappa = results.get('kappa', None)

        precision = [self.value_format.format(i) for i in precision]
        recall = [self.value_format.format(i) for i in recall]
        f1_score = [self.value_format.format(i) for i in f1_score]
        kappa = self.value_format.format(kappa)

        if self.is_reverse_results:
            precision.reverse()
            recall.reverse()
            f1_score.reverse()

        precision = ', '.join(precision)
        recall = ', '.join(recall)
        f1_score = ', '.join(f1_score)

        metric_str = self.task_format.format(precision, recall, f1_score, kappa)
        return metric_str

    def compute_metrics(self, results):
        metrics = {}
        for task_name in self._metrics:
            all_results = {}
            for metric in self._metrics[task_name]:
                name = metric.__class__.__name__
                metric_results = metric.compute_metrics(results)
                all_results.update(metric_results)
            result_str = self.format_results(all_results)
            metrics[task_name] = result_str
        return metrics

    def evaluate(self, size):
        metrics = {}
        for task_name in self._metrics:
            all_results = {}
            for metric in self._metrics[task_name]:
                name = metric.__class__.__name__

                if metric.results is None:
                    metric.results = self.results

                if 'MultiTasksMetric' in name or metric.results:
                    results = metric.evaluate(size)
                else:
                    results = {metric.__class__.__name__: 0}
                all_results.update(results)

                for key in results:
                    name = f'{task_name}_{key}'
                    if name in results:
                        raise ValueError(f'There are multiple metric results with the same name {name}.')
                    metrics[name] = results[key]

            result_str = self.format_results(all_results)
            metrics[task_name] = result_str
        return metrics


# REF: https://github.com/open-mmlab/mmocr/blob/1.x/mmocr/evaluation/evaluator/multi_datasets_evaluator.py
@EVALUATORS.register_module()
class FundusMultiDatasetsEvaluator(Evaluator):
    """Wrapper class to compose class: `ConcatDataset` and multiple
    :class:`BaseMetric` instances.
    The metrics will be evaluated on each dataset slice separately. The name of
    the each metric is the concatenation of the dataset prefix, the metric
    prefix and the key of metric - e.g.
    `dataset_prefix/metric_prefix/accuracy`.

    Args:
        metrics (dict or BaseMetric or Sequence): The config of metrics.
        dataset_prefixes (Sequence[str]): The prefix of each dataset. The
            length of this sequence should be the same as the length of the
            datasets.
    """

    def __init__(self, metrics: Union[Union[ConfigDict, Dict], BaseMetric, Sequence],
                 dataset_prefixes: Sequence[str]) -> None:
        super().__init__(metrics)
        self.dataset_prefixes = dataset_prefixes

    def logging_results(self, final_metrics):
        if is_main_process():
            logger = MMLogger.get_current_instance()
            for dataset_prefix in self.dataset_prefixes:
                metrics = final_metrics[dataset_prefix]
                for task_name, results in metrics.items():
                    message = ' '.join(('{:6}'.format(dataset_prefix), task_name, results))
                    logger.info(message)

    def evaluate(self, size: int) -> dict:
        """Invoke ``evaluate`` method of each metric and collect the metrics
        dictionary.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation results of all metrics. The keys are the names
            of the metrics, and the values are corresponding results.
        """

        metric_results = dict()

        print_results = dict()
        for dataset_prefix in self.dataset_prefixes:
            print_results[dataset_prefix] = dict()

        dataset_slices = self.dataset_meta.get('cumulative_sizes', [size])

        assert len(dataset_slices) == len(self.dataset_prefixes)
        assert len(self.metrics) == 1 and \
               self.metrics[0].__class__.__name__ == 'FundusMultiTasksMetric', 'Not Implemented'

        multi_task_metric = self.metrics[0]  # the instance of FundusMultiTasksMetric

        for task_name in multi_task_metric._metrics:
            all_results = dict()
            for dataset_prefix in self.dataset_prefixes:
                all_results[dataset_prefix] = dict()

            for metric in multi_task_metric._metrics[task_name]:
                if len(metric.results) == 0:
                    warnings.warn(f'{task_name, metric.__class__.__name__} got empty `self.results`.')

                results = collect_results(metric.results, size, metric.collect_device)

                if is_main_process():
                    results = _to_cpu(results)
                    for start, end, dataset_prefix in zip([0] + dataset_slices[:-1], dataset_slices,
                                                          self.dataset_prefixes):
                        per_results = metric.compute_metrics(results[start:end])  # type: ignore
                        all_results[dataset_prefix].update(per_results)
                    metric.results.clear()

            if is_main_process():
                for dataset_prefix in self.dataset_prefixes:

                    result_str = multi_task_metric.format_results(all_results[dataset_prefix], prefix='')
                    print_results[dataset_prefix][task_name] = result_str

                    for key, value in all_results[dataset_prefix].items():
                        new_key = f'{dataset_prefix}/{task_name}/{key}'
                        metric_results[new_key] = value

        if is_main_process():
            metric_results = [metric_results]
        else:
            metric_results = [None]  # type: ignore
        broadcast_object_list(metric_results)

        self.logging_results(print_results)

        return metric_results[0]
