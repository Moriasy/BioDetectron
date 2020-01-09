import os
import sys
import copy
import torch
import logging
import numpy as np
from glob import glob
from skimage.io import imread
from itertools import combinations
from subprocess import Popen, PIPE
from collections import OrderedDict
from skimage.exposure import rescale_intensity

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import DatasetEvaluator
from detectron2.engine import HookBase, DefaultPredictor

from utils import box2csv
from metrics import BoundingBox, BoundingBoxes, BBFormat, BBType, Evaluator as MetricEvaluator


class GenericEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, cfg, output_dir, distributed=False):
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir
        self._dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)

    def reset(self):
        self._predictions = []
        self._results = {}

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("AP",)
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            tasks = tasks + ("keypoints",)
        return tasks

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {'groundtruth':input}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning("[GenericEvaluator] Did not receive valid predictions.")
            return {}

        self._results = OrderedDict()
        if "proposals" in self._predictions[0]:
            self._eval_box_proposals()
        if "instances" in self._predictions[0]:
            self._eval_predictions(set(self._tasks))
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)


    def _eval_box_proposals(self):
        self._logger.warning("[_eval_box_proposals] not implemented.")
        return

    def _eval_predictions(self, tasks):
        if "AP" in tasks:
            self._results["AP"] = {}
            ap_scores = []
            for n in range(len(self._predictions)):
                boxes = BoundingBoxes()

                for box_idx, box in enumerate(self._predictions[n]["groundtruth"]["instances"].get("gt_boxes")):
                    bbox = BoundingBox(
                        'eval_img',
                        self._predictions[n]["groundtruth"]["instances"].get("gt_classes")[box_idx],
                        box[0],
                        box[1],
                        box[2],
                        box[3],
                        format=BBFormat.XYX2Y2,
                        bbType=BBType.GroundTruth)

                    boxes.addBoundingBox(bbox)

                for box_idx, box in enumerate(self._predictions[n]["instances"].get("pred_boxes")):
                    bbox = BoundingBox(
                        "eval_img",
                        self._predictions[n]["instances"].get("pred_classes")[box_idx],
                        box[0],
                        box[1],
                        box[2],
                        box[3],
                        bbType=BBType.Detected,
                        format=BBFormat.XYX2Y2,
                        classConfidence=self._predictions[n]["instances"].get("scores")[box_idx],
                    )

                    boxes.addBoundingBox(bbox)

                metric_evaluator = MetricEvaluator()
                results = metric_evaluator.GetPascalVOCMetrics(boxes)

                for cls in results:
                    if n == 0:
                        ap_scores.append([])
                    ap_scores[n].append(cls["AP"])

            for cls_n in range(len(ap_scores)):
                self._results["AP"]["Class {} AP".format(cls_n)] = np.mean(ap_scores[cls_n])

        return


class BboxPredictor():
    def __init__(self, cfg, weights):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(cfg)
        self.cfg.MODEL.WEIGHTS = weights

        self.predictor = DefaultPredictor(self.cfg)

    def inference_on_folder(self, folder):
        imglist = glob(os.path.join(folder, '*.jpg')) + \
                  glob(os.path.join(folder, '*.tif')) + \
                  glob(os.path.join(folder, '*.png'))

        for path in imglist:
            image = imread(path)

            if len(image.shape) < 3:
                image = np.expand_dims(image, axis=-1)
                image = np.repeat(image, 3, axis=-1)
            elif image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)

            image = rescale_intensity(image, in_range='dtype', out_range=(0, 255))
            image = image.astype(np.uint8)

            boxes, classes, scores = self.detect_one_image(image)
            box2csv(boxes, classes, scores, os.path.splitext(path)[0] + '_predict.csv')


    def detect_one_image(self, image):
        instances = self.predictor(image)["instances"]

        boxes = list(instances.pred_boxes)
        boxes = [tuple(box.cpu().numpy()) for box in boxes]

        scores = list(instances.scores)
        scores = [score.cpu().numpy() for score in scores]

        classes = list(instances.pred_classes)
        classes = [cls.cpu().numpy() for cls in classes]

        boxes, classes, scores = self.check_iou(boxes, scores, classes)

        return boxes, classes, scores

    @staticmethod
    def bb_intersection_over_union(boxA, boxB):
        # from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def check_iou(self, boxes, scores, classes):
        if len(boxes) <= 1:
            return boxes

        while True:
            new_boxes = []
            new_scores = []
            new_classes = []
            overlap_boxes = []

            indices = list((i,j) for ((i,_),(j,_)) in combinations(enumerate(boxes), 2))

            for a,b in indices:
                iou = self.bb_intersection_over_union(boxes[a], boxes[b])

                if iou > 0.5:
                    if scores[a] > scores[b]:
                        overlap_boxes.append(b)
                    else:
                        overlap_boxes.append(a)
                    break


            for idx in range(len(boxes)):
                if idx not in overlap_boxes:
                    new_boxes.append(boxes[idx])
                    new_scores.append(scores[idx])
                    new_classes.append(classes[idx])

            if len(new_boxes) == len(boxes) or len(new_boxes) <= 1:
                break

            boxes = new_boxes
            scores = new_scores
            classes = new_classes

        return new_boxes, new_classes, new_scores

