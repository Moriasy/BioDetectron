import os
import sys
import copy
import torch
import logging
import visdom
import numpy as np
from subprocess import Popen, PIPE
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.engine import HookBase
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import DatasetEvaluator

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

                ap_scores.append(results[0]["AP"])

            mean_AP = np.mean(ap_scores)

            self._results["AP"] = mean_AP

        print(self._results["AP"])
        return
