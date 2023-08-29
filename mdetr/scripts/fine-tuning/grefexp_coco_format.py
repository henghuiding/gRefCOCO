###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################

"""
This script is used to convert the Generalized referring expression dataset annotations to COCO format as expected by MDETR.
data_path :  path to original grefexp annotations to be downloaded from https://github.com/henghuiding/gRefCOCO

"""
import argparse
import json
import os
import pickle
from pathlib import Path
import sys
PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from utils.spans import consolidate_spans
from utils.text import get_root_and_nouns
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("Conversion script")

    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="Path to the refexp data",
    )

    parser.add_argument(
        "--out_path",
        required=True,
        type=str,
        help="Path where to export the resulting dataset.",
    )

    parser.add_argument(
        "--coco_path",
        required=True,
        type=str,
        help="Path to coco 2014 dataset.",
    )

    return parser.parse_args()


def convert(dataset_path: Path, dataset_name: str, split: str, output_path, coco_path, next_img_id: int = 0, next_id: int = 0):
    """Do the heavy lifting on the given split (eg 'train')"""

    print(f"Exporting {split}...")

    with open(f"{coco_path}/annotations/instances_train2014.json", "r") as f:
        coco_annotations = json.load(f)
    coco_images = coco_annotations["images"]
    coco_anns = coco_annotations["annotations"]
    annid2cocoann = {item["id"]: item for item in coco_anns}
    imgid2cocoimgs = {item["id"]: item for item in coco_images}

    categories = coco_annotations["categories"]
    annotations = []
    images = []

    d_name = "grefcoco"

    with open(dataset_path / dataset_name, "rb") as f:
        data = json.load(f)

    for item in tqdm(data):

        if item["split"] != split:
            continue

        for s in item["sentences"]:
            refexp = s["sent"]
            _, _, root_spans, neg_spans = get_root_and_nouns(refexp)
            root_spans = consolidate_spans(root_spans, refexp)
            neg_spans = consolidate_spans(neg_spans, refexp)
            # filename = item["file_name"]
            if len(item["file_name"].split('_')) > 3:
                a, b, c, d = item["file_name"].split('_')
                filename = a + '_' + b + '_' +c + '.jpg'
            else:
                filename = item["file_name"]
            cur_img = {
                "file_name": filename,
                "height": imgid2cocoimgs[item["image_id"]]["height"],
                "width": imgid2cocoimgs[item["image_id"]]["width"],
                "id": next_img_id,
                "original_id": item["image_id"],
                "caption": refexp,
                "dataset_name": d_name,
                "tokens_negative": neg_spans,
            }
            if not isinstance(item['ann_id'], list):
                item['ann_id'] = [item['ann_id']]
            if not isinstance(item['category_id'], list):
                item['category_id'] = [item['category_id']]
            if item["ann_id"] == [-1]:
                cur_obj = {
                    "area": -1,
                    "iscrowd": -1,
                    "image_id": next_img_id,
                    "category_id": -1,
                    "id": next_id,
                    "bbox": [0, 0, 0, 0],
                    'empty': True,
                    # "segmentation": annid2cocoann[x]['segmentation'],
                    "original_id": item["ann_id"][0],
                    "tokens_positive": root_spans,
                }
                next_id += 1
                annotations.append(cur_obj)
            else:
                for x, y in zip(item["ann_id"], item["category_id"]):
                    cur_obj = {
                        "area": annid2cocoann[x]["area"],
                        "iscrowd": annid2cocoann[x]["iscrowd"],
                        "image_id": next_img_id,
                        "category_id": y,
                        "id": next_id,
                        'empty': False,
                        "bbox": annid2cocoann[x]["bbox"],
                        # "segmentation": annid2cocoann[x]['segmentation'],
                        "original_id": x,
                        "tokens_positive": root_spans,
                    }
                    next_id += 1
                    annotations.append(cur_obj)
            next_img_id += 1
            images.append(cur_img)

    ds = {
        "info": coco_annotations["info"],
        "licenses": coco_annotations["licenses"],
        "images": images,
        "annotations": annotations,
        "categories": coco_annotations["categories"],
    }
    print('done!')
    with open(output_path / f"finetune_{d_name}_{split}.json", "w") as j_file:
        json.dump(ds, j_file)

    return next_img_id, next_id


def main(args):
    data_path = Path(args.data_path)
    output_path = Path(args.out_path)

    os.makedirs(str(output_path), exist_ok=True)

    next_img_id, next_id = 0, 0
    for dataset_name in ["grefs(unc).json"]:
        for split in ["val", "train"]:
        # for split in ["testA", "testB", "val", "train"]:
            next_img_id, next_id = convert(
                data_path, dataset_name, split, output_path, args.coco_path, next_img_id=next_img_id, next_id=next_id,
            )


if __name__ == "__main__":
    main(parse_args())
