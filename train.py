import os
from datetime import datetime

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_argument_parser, DefaultTrainer
from detectron2.evaluation import DatasetEvaluators

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


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file('./res50.yaml')
    cfg.DATASETS.TRAIN = ("osman",)
    cfg.DATASETS.TEST = ("osman_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    #cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.RETINANET.NUM_CLASSES = 4
    cfg.TEST.EVAL_PERIOD = 100
    cfg.OUTPUT_DIR = '/scratch/bunk/osman/logs'
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.SOLVER.BASE_LR = 0.01
    #cfg.SOLVER.WARMUP_ITERS = 10

    # WEN
    cfg.MODEL.PIXEL_MEAN = [36.51, 36.51, 36.51]
    cfg.MODEL.PIXEL_STD = [30.87, 30.87, 30.87]

    # OSMAN
    cfg.MODEL.PIXEL_MEAN = [91.50, 91.50, 91.50]
    cfg.MODEL.PIXEL_STD = [14.19, 14.19, 14.19]

    date_time = datetime.now().strftime("%m%d%y_%H%M%S")
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, date_time)

    # cfg.freeze()
    # default_setup(cfg, args)

    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="detectron")
    return cfg


def main(args):
    ####### OSMAN DATA

    dict_getter = DictGetter(train_path='/scratch/bunk/osman/mating_cells/COCO/DIR/train',
                             val_path='/scratch/bunk/osman/mating_cells/COCO/DIR/val')

    DatasetCatalog.register("osman", dict_getter.get_train_dicts)
    MetadataCatalog.get("osman").thing_classes = ["good_mating", "bad_mating", "single_cell", "crowd"]

    DatasetCatalog.register("osman_val", dict_getter.get_val_dicts)
    MetadataCatalog.get("osman_val").thing_classes = ["good_mating", "bad_mating", "single_cell", "crowd"]

    ####### WEN DATA

    # dict_getter = DictGetter(train_path='/scratch/bunk/wen/COCO/DIR/train2014',
    #                          val_path='/scratch/bunk/wen/COCO/DIR/val2014')

    DatasetCatalog.register("wen", get_train_dicts)
    MetadataCatalog.get("wen").thing_classes = ["G1", "G2", "ms", "ears", "uncategorized", "ls", "multinuc", "mito"]

    DatasetCatalog.register("wen_val", get_val_dicts)
    MetadataCatalog.get("wen_val").thing_classes = ["G1", "G2", "ms", "ears", "uncategorized", "ls", "multinuc", "mito"]

    cfg = setup(args)

    # if args.eval_only:
    #     model = Trainer.build_model(cfg)
    #     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #         cfg.MODEL.WEIGHTS, resume=args.resume
    #     )
    #     res = Trainer.test(cfg, model)
    #     if comm.is_main_process():
    #         verify_results(cfg, res)
    #     return res

    trainer = Trainer(cfg)
    trainer.resume_or_load()

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)