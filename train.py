import os
from datetime import datetime

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_argument_parser, DefaultTrainer

from training import BioTrainer
from data import get_csv
from eval import VizHook


def get_train_dicts():
    global train_path
    return get_csv(train_path)

def get_val_dicts():
    global val_path
    return get_csv(val_path)


def setup(args):
    cfg = get_cfg()
    #cfg.merge_from_file("./faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    #cfg.merge_from_file("./res50.yaml")
    cfg.merge_from_file('./res50.yaml')
    cfg.DATASETS.TRAIN = ("wen_coco",)
    cfg.DATASETS.TEST = ("wen_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    #cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    #cfg.INPUT.FORMAT = "BGR"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    cfg.MODEL.RETINANET.NUM_CLASSES = 8
    cfg.MODEL.PIXEL_MEAN = [36.51, 36.51, 36.51]
    cfg.MODEL.PIXEL_STD = [30.87, 30.87, 30.87]
    cfg.TEST.EVAL_PERIOD = 500
    cfg.OUTPUT_DIR = '/scratch/bunk/osman/logs'
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    #cfg.INPUT.CROP.ENABLED = False
    cfg.SOLVER.BASE_LR = 0.01
    #cfg.SOLVER.WARMUP_ITERS = 10
    #cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (1,1,1,1)
    #cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True

    date_time = datetime.now().strftime("%m%d%y_%H%M%S")
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, date_time)

    # cfg.freeze()
    # default_setup(cfg, args)

    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="detectron")
    return cfg


def main(args):
    global train_path
    global val_path

    train_path = '/scratch/bunk/wen/COCO/DIR/train2014'
    val_path = '/scratch/bunk/wen/COCO/DIR/val2014'

    DatasetCatalog.register("osman", get_train_dicts)
    MetadataCatalog.get("osman").thing_classes = ["background", "good_mating", "bad_mating", "single_cell", "crowd"]

    DatasetCatalog.register("osman_val", get_val_dicts)
    MetadataCatalog.get("osman_val").thing_classes = ["background", "good_mating", "bad_mating", "single_cell", "crowd"]

    DatasetCatalog.register("wen", get_train_dicts)
    MetadataCatalog.get("wen").thing_classes = ["G1", "G2", "ms", "ears", "uncategorized", "ls", "multinuc", "mito"]

    DatasetCatalog.register("wen_val", get_val_dicts)
    MetadataCatalog.get("wen_val").thing_classes = ["G1", "G2", "ms", "ears", "uncategorized", "ls", "multinuc", "mito"]

    register_coco_instances('wen_coco', {}, '/scratch/bunk/wen/COCO/DIR/annotations/instances_train2014.json', '/scratch/bunk/wen/COCO/DIR/train2014/')

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

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load()

    # trainer.register_hooks([
    #     VizHook(cfg.TEST.EVAL_PERIOD, lambda: trainer.eval(cfg, trainer.model, 'osman_val'), cfg),
    #                         ])

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)