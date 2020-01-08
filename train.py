import os
from itertools import combinations
from datetime import datetime

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import DatasetEvaluators
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper
from detectron2.engine import default_argument_parser, DefaultTrainer, DefaultPredictor, launch, default_setup

from data import BoxDetectionLoader, DictGetter
from eval import GenericEvaluator
from utils import copy_code


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

        ### MUST ADD METADATA SOMEHOW!!!

    def detect_one_image(self, image):
        instances = self.predictor(image)["instances"]

        boxes = list(instances.pred_boxes)
        boxes = [tuple(box.cpu().numpy()) for box in boxes]

        scores = list(instances.scores)
        scores = [score.cpu().numpy() for score in scores]

        boxes = self.check_iou(boxes, scores)

        return boxes

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

    def check_iou(self, boxes, scores):
        if len(boxes) <= 1:
            return boxes

        while True:
            new_boxes = []
            new_scores = []
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

            if len(new_boxes) == len(boxes) or len(new_boxes) <= 1:
                break

            boxes = new_boxes
            scores = new_scores

        return new_boxes


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    date_time = datetime.now().strftime("%m%d%y_%H%M%S")
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.TRAIN[0], date_time)

    if comm.get_rank() == 0:
        copy_code(cfg.OUTPUT_DIR)

    ####### OSMAN DATA
    if "osman" in cfg.DATASETS.TRAIN:
        dict_getter = DictGetter("osman", train_path='/scratch/bunk/osman/mating_cells/COCO/DIR/train',
                                 val_path='/scratch/bunk/osman/mating_cells/COCO/DIR/val')

        DatasetCatalog.register("osman", dict_getter.get_train_dicts)
        MetadataCatalog.get("osman").thing_classes = ["good_mating", "bad_mating", "single_cell", "crowd"]
        MetadataCatalog.get("osman").thing_dataset_id_to_contiguous_id = {1:0, 2:1, 3:2, 4:3}

        DatasetCatalog.register("osman_val", dict_getter.get_val_dicts)
        MetadataCatalog.get("osman_val").thing_classes = ["good_mating", "bad_mating", "single_cell", "crowd"]
        MetadataCatalog.get("osman_val").thing_dataset_id_to_contiguous_id = {1:0, 2:1, 3:2,}

    ####### WEN DATA
    elif "wen" in cfg.DATASETS.TRAIN:
        dict_getter = DictGetter("wen", train_path='/scratch/bunk/wen/COCO/DIR/train2014',
                                 val_path='/scratch/bunk/wen/COCO/DIR/val2014')

        DatasetCatalog.register("wen", dict_getter.get_train_dicts)
        MetadataCatalog.get("wen").thing_classes = ["G1", "G2", "ms", "ears", "uncategorized", "ls", "multinuc", "mito"]
        MetadataCatalog.get("wen").thing_dataset_id_to_contiguous_id = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}

        DatasetCatalog.register("wen_val", dict_getter.get_val_dicts)
        MetadataCatalog.get("wen_val").thing_classes = ["G1", "G2", "ms", "ears", "uncategorized", "ls", "multinuc", "mito"]
        MetadataCatalog.get("wen_val").thing_dataset_id_to_contiguous_id = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}

    ####### WEN DATA
    elif "wings" in cfg.DATASETS.TRAIN:
        dict_getter = DictGetter("wings", train_path='/scratch/bunk/wings/images/COCO/DIR/train2014',
                                 val_path='/scratch/bunk/wings/images/COCO/DIR/val2014')

        DatasetCatalog.register("wings", dict_getter.get_train_dicts)
        MetadataCatalog.get("wings").thing_classes = ["wing"]
        MetadataCatalog.get("wings").thing_dataset_id_to_contiguous_id = {1:0, 2:0, 3:0}

        DatasetCatalog.register("wings_val", dict_getter.get_val_dicts)
        MetadataCatalog.get("wings_val").thing_classes = ["wing"]
        MetadataCatalog.get("wings_val").thing_dataset_id_to_contiguous_id = {1:0, 2:0, 3:0}

    cfg.freeze()
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