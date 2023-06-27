import copy
import json
import pandas as pd
import argparse
import os
import random

ann_file1 = '/home/chenqiongpu/SLO/anno_file/SLO-v3.9.xlsx'
ann_file2 = '/home/chenqiongpu/SLO/anno_file/SLO+oct-disc-v3.9.xlsx'
ann_file3 = '/home/chenqiongpu/SLO/anno_file/SLO+oct-macular-v3.9.xlsx'
ann_root = '/home/chenqiongpu/SLO/anno_file/annotations'
json_name= 'test.json'


# README: https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/advanced_tutorials/basedataset.md
def init_json_dict():
    data = dict(
        metainfo=dict(
            # classes[ME,DR,glaucoma,cataract,optic_atrophy]
        ),
        data_list=[
            # img_path,gt_label
        ],
    )
    return data

def is_PNS(row):
    if row == "Negative":
        return 0
    elif row == "ME" or row == "DR" or row == "cataract" or row == "optic_atrophy":
        return 1
    else:
        return -100

def is_glaucoma(row):
    if row == "Negative":
        return 0
    elif row == "glaucoma"  or row == "others+glaucoma":
        return 1
    elif row == "suspicious":
        return 2
    else:
        return -100

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--root', default='/home1/commonfile/CSURetina10K')
    args = parser.parse_args()

    os.makedirs(ann_root, exist_ok=True)

    # train
    # df = pd.read_excel(ann_file1, sheet_name="train-1458",converters={'patient ID': str})#5-10
    # df2 = pd.read_excel(ann_file2, sheet_name="train-842",converters={'patient ID': str})#6-11
    # df3 = pd.read_excel(ann_file3, sheet_name="train-982",converters={'patient ID': str})#6-11
    
    # val
    # df = pd.read_excel(ann_file1, sheet_name="val-182",converters={'patient ID': str})#5-10
    # df2 = pd.read_excel(ann_file2, sheet_name="val-105",converters={'patient ID': str})#6-11
    # df3 = pd.read_excel(ann_file3, sheet_name="val-123",converters={'patient ID': str})#6-11
    
    # test
    df = pd.read_excel(ann_file1, sheet_name="test-182",converters={'patient ID': str})#5-10
    df2 = pd.read_excel(ann_file2, sheet_name="test-105",converters={'patient ID': str})#6-11
    df3 = pd.read_excel(ann_file3, sheet_name="test-123",converters={'patient ID': str})#6-11

    raw_json_file = init_json_dict()
    raw_json_file['metainfo']['classes'] = list(df.head(0))[5:10]
    

    data_list = []
    for i, row in df.iterrows():
        patient_ID = row['patient ID']
        laterality = row['laterality']
        image_column = row['Images']
        images_list = [image.strip() for image in image_column.split(',') if image.strip()]# 使用逗号分隔字符串，并去除空项
        random_image = random.choice(images_list)# 随机选择一个值

        source_label = list(row)[5:10]
        label={}
        label["T0"]=is_PNS(row[5])
        label["T1"]=is_PNS(row[6])
        label["T2"]=is_glaucoma(row[7])
        label["T3"]=is_PNS(row[8])
        label["T4"]=is_PNS(row[9])

        data_list.append(
            dict(img_path=f'SLOImages-v3.4/{patient_ID}/{laterality}/{random_image}', gt_label=label)
        )
    for i, row in df2.iterrows():
        patient_ID = row['patient ID']
        patient_name=row['patient name']
        laterality = row['laterality']
        image_column = row['Images']
        images_list = [image.strip() for image in image_column.split(',') if image.strip()]# 使用逗号分隔字符串，并去除空项
        images_list = [image for image in images_list if not image.startswith("OCT")]
        random_image = random.choice(images_list)# 随机选择一个值

        source_label = list(row)[6:11]
        label={}
        label["T0"]=is_PNS(row[6])
        label["T1"]=is_PNS(row[7])
        label["T2"]=is_glaucoma(row[8])
        label["T3"]=is_PNS(row[9])
        label["T4"]=is_PNS(row[10])

        data_list.append(
            dict(img_path=f'SLO+OCTImages-Disc-v3.4/{patient_ID}+{patient_name}/{laterality}/SLO/{random_image}', gt_label=label)
        )
    for i, row in df3.iterrows():
        patient_ID = row['patient ID']
        patient_name=row['patient name']
        laterality = row['laterality']
        image_column = row['Images']
        images_list = [image.strip() for image in image_column.split(',') if image.strip()]# 使用逗号分隔字符串，并去除空项
        images_list = [image for image in images_list if not image.startswith("OCT")]#只处理SLO图片
        random_image = random.choice(images_list)# 随机选择一个值

        source_label = list(row)[6:11]
        label={}
        label["T0"]=is_PNS(row[6])
        label["T1"]=is_PNS(row[7])
        label["T2"]=is_glaucoma(row[8])
        label["T3"]=is_PNS(row[9])
        label["T4"]=is_PNS(row[10])

        data_list.append(
            dict(img_path=f'SLO+OCTImages-Macular-v3.4/{patient_ID}+{patient_name}/{laterality}/SLO/{random_image}', gt_label=label)
        )

    json_file = copy.deepcopy(raw_json_file)
    json_file['data_list'] = data_list
    with open(os.path.join(ann_root, json_name), 'w') as f:
        json.dump(json_file, f, indent=2)


    print(f'Save jsons to {ann_root}.')
