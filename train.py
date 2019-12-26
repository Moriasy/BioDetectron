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

from data import SKImageLoader, DictGetter
from eval import GenericEvaluator


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        evaluators = [GenericEvaluator(dataset_name, cfg, cfg.OUTPUT_DIR)]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=SKImageLoader(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=SKImageLoader(cfg, True))


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

        print(boxes)

        boxes = self.check_iou(boxes)

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

    def check_iou(self, results):
        if len(results) <= 1:
            return results

        while True:
            new_results = []
            overlap_results = []
            for a,b in combinations(results, 2):
                iou = self.bb_intersection_over_union(a, b)

                if iou > 0.33:
                    overlap_results.append(b)
                    break


            for result in results:
                if result not in overlap_results:
                    new_results.append(result)

            #print(len(new_results), len(overlap_results))
            if len(new_results) == len(results) or len(new_results) <= 1:
                break

            results = new_results

        return new_results


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file('./wings.yaml')

    cfg.OUTPUT_DIR = '/scratch/bunk/logs'

    date_time = datetime.now().strftime("%m%d%y_%H%M%S")
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.TRAIN[0], date_time)

    ####### OSMAN DATA
    if "osman" in cfg.DATASETS.TRAIN:
        dict_getter = DictGetter(train_path='/scratch/bunk/osman/mating_cells/COCO/DIR/train',
                                 val_path='/scratch/bunk/osman/mating_cells/COCO/DIR/val')

        DatasetCatalog.register("osman", dict_getter.get_train_dicts)
        MetadataCatalog.get("osman").thing_classes = ["good_mating", "bad_mating", "single_cell", "crowd"]

        DatasetCatalog.register("osman_val", dict_getter.get_val_dicts)
        MetadataCatalog.get("osman_val").thing_classes = ["good_mating", "bad_mating", "single_cell", "crowd"]

    ####### WEN DATA
    elif "wen" in cfg.DATASETS.TRAIN:
        dict_getter = DictGetter(train_path='/scratch/bunk/wen/COCO/DIR/train2014',
                                 val_path='/scratch/bunk/wen/COCO/DIR/val2014')

        DatasetCatalog.register("wen", dict_getter.get_train_dicts)
        MetadataCatalog.get("wen").thing_classes = ["G1", "G2", "ms", "ears", "uncategorized", "ls", "multinuc", "mito"]

        DatasetCatalog.register("wen_val", dict_getter.get_val_dicts)
        MetadataCatalog.get("wen_val").thing_classes = ["G1", "G2", "ms", "ears", "uncategorized", "ls", "multinuc", "mito"]

    ####### WEN DATA
    elif "wings" in cfg.DATASETS.TRAIN:
        dict_getter = DictGetter(train_path='/scratch/bunk/wings/images/COCO/DIR/train2014',
                                 val_path='/scratch/bunk/wings/images/COCO/DIR/val2014')

        DatasetCatalog.register("wings", dict_getter.get_train_dicts)
        MetadataCatalog.get("wings").thing_classes = ["good", "broken", "unusable"]

        DatasetCatalog.register("wings_val", dict_getter.get_val_dicts)
        MetadataCatalog.get("wings_val").thing_classes = ["good", "broken", "unusable"]

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