import copy
import json
import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split

ann_file = 'training/glaucoma_grading_training_GT.xlsx'
output_dir = 'annotations'
image_dirs = ['training/multi-modality_images', 'testing/multi-modality_images']


# README: https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/advanced_tutorials/basedataset.md
def init_json_dict():
    data = dict(
        metainfo=dict(
            # classes
        ),
        data_list=[
            # img_path, oct_img_path, gt_label
        ],
    )
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--root', default='../data/Glaucoma_grading')
    args = parser.parse_args()

    ann_root = os.path.join(args.root, output_dir)
    os.makedirs(ann_root, exist_ok=True)

    # train and val
    df = pd.read_excel(os.path.join(args.root, ann_file), converters={'data': str})

    raw_json_file = init_json_dict()
    raw_json_file['metainfo']['classes'] = list(df.head(0))[1:]

    data_list = []
    for i, row in df.iterrows():
        name = row['data']
        label = list(row)[1:].index(1)
        data_list.append(
            dict(img_path=f'{name}/{name}.jpg', oct_img_path=f'{name}/{name}', gt_label=label)
        )

    # train full
    json_file = copy.deepcopy(raw_json_file)
    json_file['data_list'] = data_list
    with open(os.path.join(ann_root, 'train_full.json'), 'w') as f:
        json.dump(json_file, f, indent=2)

    train_list, val_list = train_test_split(data_list, train_size=args.train_ratio, random_state=args.seed)
    train_list.sort(key=lambda x: os.path.basename(x['img_path']))
    val_list.sort(key=lambda x: os.path.basename(x['img_path']))

    # train
    json_file = copy.deepcopy(raw_json_file)
    json_file['data_list'] = train_list
    with open(os.path.join(ann_root, 'train.json'), 'w') as f:
        json.dump(json_file, f, indent=2)

    # val
    json_file = copy.deepcopy(raw_json_file)
    json_file['data_list'] = val_list
    with open(os.path.join(ann_root, 'val.json'), 'w') as f:
        json.dump(json_file, f, indent=2)

    # test
    json_file = copy.deepcopy(raw_json_file)
    for name in os.listdir(os.path.join(args.root, image_dirs[1])):
        json_file['data_list'].append(
            dict(img_path=f'{name}/{name}.jpg', oct_img_path=f'{name}/{name}'))
    with open(os.path.join(ann_root, 'test.json'), 'w') as f:
        json.dump(json_file, f, indent=2)

    print(f'Save jsons to {ann_root}.')
