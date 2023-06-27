import pickle
import pandas as pd
import os
import torch.nn.functional as F

# also see: mmcls.evaluation.metrics.fundus DumpCSVResults
if __name__ == '__main__':
    with open('pred.pkl', 'rb') as f:
        data = pickle.load(f)

    csv_data = []
    # title = None
    title = ['data', 'non', 'early', 'mid_advanced']
    for x in data:
        name = os.path.basename(x['img_path']).split('.')[0]
        pred_label = x['pred_label']['label']
        pred_score = x['pred_label']['score']
        one_hot_label = F.one_hot(pred_label, num_classes=len(pred_score))
        one_hot_label = one_hot_label.flatten().tolist()
        csv_data.append([name] + one_hot_label)
    df = pd.DataFrame(csv_data, columns=title)
    df.to_csv('submission_sub1.csv', index=False)
