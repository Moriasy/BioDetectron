import os

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_argument_parser

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
    cfg.merge_from_file("./faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("osman",)
    cfg.DATASETS.TEST = ("osman_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.INPUT.MAX_SIZE_TRAIN = 1608
    cfg.SOLVER.WARMUP_ITERS = 250
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.CHECKPOINT_PERIOD = 250
    cfg.INPUT.FORMAT = "I"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.TEST.EXPECTED_RESULTS = ['bbox', 'AP', 38.5, 0.2]
    cfg.TEST.EVAL_PERIOD = 25
    # cfg.freeze()
    # default_setup(cfg, args)
    # Setup logger for "densepose" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="detectron")
    return cfg


def main(args):
    global train_path
    global val_path

    train_path = '/scratch/bunk/osman/mating_cells/COCO/DIR/train'
    val_path = '/scratch/bunk/osman/mating_cells/COCO/DIR/val'

    DatasetCatalog.register("osman", get_train_dicts)
    MetadataCatalog.get("osman").thing_classes = ["background", "good_mating", "bad_mating", "single_cell", "crowd"]

    DatasetCatalog.register("osman_val", get_val_dicts)
    MetadataCatalog.get("osman_val").thing_classes = ["background", "good_mating", "bad_mating", "single_cell", "crowd"]

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

    trainer = BioTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    trainer.register_hooks([
        VizHook(cfg.TEST.EVAL_PERIOD, lambda: trainer.eval(cfg, trainer.model, 'osman_val'), 'osman_val'),
                            ])

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)