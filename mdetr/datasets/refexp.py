# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
import copy
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.utils.data
from transformers import RobertaTokenizerFast

import util.dist as dist
from util.box_ops import generalized_box_iou

from .coco import ModulatedDetection, make_coco_transforms


class RefExpDetection(ModulatedDetection):
    pass


class RefExpEvaluator(object):
    def __init__(self, refexp_gt, iou_types, k=(1, 5, 10), thresh_iou=0.5, thresh_score=0.7, thresh_F1=1.0):
        assert isinstance(k, (list, tuple))
        refexp_gt = copy.deepcopy(refexp_gt)
        self.refexp_gt = refexp_gt
        self.iou_types = iou_types
        self.img_ids = self.refexp_gt.imgs.keys()
        self.predictions = {}
        self.thresh_iou = thresh_iou
        self.thresh_score = thresh_score
        self.thresh_F1 = thresh_F1
        print(self.thresh_score)
    def accumulate(self):
        pass

    def update(self, predictions):
        self.predictions.update(predictions)

    def synchronize_between_processes(self):
        all_predictions = dist.all_gather(self.predictions)
        merged_predictions = {}
        for p in all_predictions:
            merged_predictions.update(p)
        self.predictions = merged_predictions

    def summarize(self):
        if dist.is_main_process():
            correct_image = 0
            num_image = 0
            nt = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
            for image_id in self.img_ids:
                TP = 0
                ann_ids = self.refexp_gt.getAnnIds(imgIds=image_id)
                # assert len(ann_ids) == 1
                img_info = self.refexp_gt.loadImgs(image_id)[0]

                target = self.refexp_gt.loadAnns(ann_ids)
                prediction = self.predictions[image_id]
                assert prediction is not None
                sorted_scores_boxes = sorted(
                    zip(prediction["scores"].tolist(), prediction["boxes"].tolist()), reverse=True
                )
                sorted_scores, sorted_boxes = zip(*sorted_scores_boxes)
                sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
                converted_bbox_all = []
                no_target_flag = False
                for one_target in target:
                    if one_target['category_id'] == -1:
                        no_target_flag = True
                    target_bbox = one_target["bbox"]
                    converted_bbox = [
                    target_bbox[0],
                    target_bbox[1],
                    target_bbox[2] + target_bbox[0],
                    target_bbox[3] + target_bbox[1],
                ]
                    converted_bbox_all.append(torch.tensor(converted_bbox))
                gt_bbox_all = torch.stack(converted_bbox_all, dim=0)

                sorted_scores_array = np.array(sorted_scores)
                idx = sorted_scores_array >= self.thresh_score
                filtered_boxes = sorted_boxes[idx]
                # filtered_boxes = sorted_boxes[0:1]
                giou = generalized_box_iou(filtered_boxes, gt_bbox_all.view(-1, 4))
                num_prediction = filtered_boxes.shape[0]
                num_gt = gt_bbox_all.shape[0]
                if no_target_flag:
                    if num_prediction >= 1:
                        nt["FN"] += 1
                    else:
                        nt["TP"] += 1
                    if num_prediction >= 1:
                        F_1 = 0.
                    else:
                        F_1 = 1.0
                else:
                    if num_prediction >= 1:
                        nt["TN"] += 1
                    else:
                        nt["FP"] += 1
                    for i in range(min(num_prediction, num_gt)):
                        top_value, top_index = torch.topk(giou.flatten(0, 1), 1)
                        if top_value < self.thresh_iou:
                            break
                        else:
                            top_index_x = top_index // num_gt
                            top_index_y = top_index % num_gt
                            TP += 1
                            giou[top_index_x[0], :] = 0.0
                            giou[:, top_index_y[0]] = 0.0
                    FP = num_prediction - TP
                    FN = num_gt - TP
                    F_1 = 2 * TP / (2 * TP + FP + FN)

                if F_1 >= self.thresh_F1:
                    correct_image += 1
                num_image += 1

            score = correct_image / num_image
            results = {}
            results['F1_score'] = score
            results['T_acc'] = nt['TN'] / (nt['TN'] + nt['FP'])
            results['N_acc'] = nt['TP'] / (nt['TP'] + nt['FN'])
            print(results, 'results')
            return results
        return None


def build(image_set, args):
    img_dir = Path(args.coco_path) / "train2014"

    refexp_dataset_name = args.refexp_dataset_name
    if refexp_dataset_name in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
        if args.test:
            test_set = args.test_type
            ann_file = Path(args.refexp_ann_path) / f"finetune_{refexp_dataset_name}_{test_set}.json"
        else:
            ann_file = Path(args.refexp_ann_path) / f"finetune_{refexp_dataset_name}_{image_set}.json"
    elif refexp_dataset_name in ["all"]:
        ann_file = Path(args.refexp_ann_path) / f"final_refexp_{image_set}.json"
    else:
        assert False, f"{refexp_dataset_name} not a valid datasset name for refexp"

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)
    dataset = RefExpDetection(
        img_dir,
        ann_file,
        transforms=make_coco_transforms(image_set, cautious=True),
        return_masks=args.masks,
        return_tokens=True,
        tokenizer=tokenizer,
    )
    return dataset