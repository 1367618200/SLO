import torch
from mmengine.evaluator import Evaluator

from mmcls.evaluation.metrics import Kappa, DumpCSVResults
from mmcls.evaluation.metrics.single_label import Accuracy, SingleLabelMetric
from mmcls.structures import ClsDataSample

if __name__ == '__main__':
    data_samples = []
    num_classes = 3
    for i in range(100):
        x = ClsDataSample()
        pred = torch.rand(num_classes)
        x.set_gt_label(torch.randint(0, num_classes, (1,)))
        x.set_pred_label(pred.argmax()).set_field(num_classes, 'num_classes')
        x.set_pred_score(pred)  # optional
        x.set_field(f'path/to/{i:04d}', 'img_path')
        data_samples.append(x)

    evaluator = Evaluator(
        metrics=[
            Kappa(),
            Accuracy(topk=(1,)),
            SingleLabelMetric(),
            DumpCSVResults('work_dirs/temp.csv', csv_title=['data', 'non', 'early', 'mid_advanced'])
        ])
    evaluator.process(data_samples)
    out = evaluator.evaluate(100)
    print(out)
