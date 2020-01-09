import os
import numpy as np
from glob import glob
from skimage.io import imread
from datetime import datetime
from itertools import combinations
from skimage.exposure import rescale_intensity

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import DatasetEvaluators
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper
from detectron2.engine import default_argument_parser, DefaultTrainer, DefaultPredictor, launch, default_setup

from datasets import register_custom_datasets
from data import BoxDetectionLoader
from utils import copy_code, box2csv
from eval import GenericEvaluator


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        evaluators = [GenericEvaluator(dataset_name, cfg, cfg.OUTPUT_DIR)]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=BoxDetectionLoader(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=BoxDetectionLoader(cfg, True))


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


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    date_time = datetime.now().strftime("%m%d%y_%H%M%S")
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.TRAIN[0], date_time)

    if comm.get_rank() == 0:
        copy_code(cfg.OUTPUT_DIR)

    cfg.freeze()
    register_custom_datasets()
    default_setup(cfg, args)

    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="detectron")
    return cfg

def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        # if comm.is_main_process():
        #     verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load()

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(main,
           num_gpus_per_machine=args.num_gpus,
           num_machines=args.num_machines,
           machine_rank=args.machine_rank,
           dist_url=args.dist_url,
           args=(args, ),
           )