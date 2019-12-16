import os
import torch
import logging
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.structures import ImageList
from detectron2.modeling import GeneralizedRCNN, RetinaNet
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper
from detectron2.evaluation import DatasetEvaluator, DatasetEvaluators, verify_results, inference_on_dataset
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, EvalHook, hooks

from data import SKImageLoader
from eval import GenericEvaluator


class RetNet(RetinaNet):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        # print(torch.min(images.tensor), torch.max(images.tensor))

        features = [features[f] for f in self.in_features]
        box_cls, box_delta = self.head(features)
        anchors = self.anchor_generator(features)

        if self.training:
            gt_classes, gt_anchors_reg_deltas = self.get_ground_truth(anchors, gt_instances)
            print('gt')
            print(torch.min(gt_classes[0]), torch.max(gt_classes[0]), torch.min(gt_anchors_reg_deltas[0]), torch.min(gt_anchors_reg_deltas[0]))
            print('pred')
            print(torch.min(box_cls[0]), torch.max(box_cls[0]), torch.min(box_delta[0]), torch.max(box_delta[0]))
            return self.losses(gt_classes, gt_anchors_reg_deltas, box_cls, box_delta)
        else:
            results = self.inference(box_cls, box_delta, anchors, images)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results


class BioTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = RetNet(cfg)

    # @classmethod
    # def build_optimizer(cls, cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    #     """
    #     Build an optimizer from config.
    #     """
    #     params: List[Dict[str, Any]] = []
    #     for key, value in model.named_parameters():
    #         if not value.requires_grad:
    #             continue
    #         lr = cfg.SOLVER.BASE_LR
    #         weight_decay = cfg.SOLVER.WEIGHT_DECAY
    #         if key.endswith("norm.weight") or key.endswith("norm.bias"):
    #             weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
    #         elif key.endswith(".bias"):
    #             # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
    #             # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
    #             # hyperparameters are by default exactly the same as for regular
    #             # weights.
    #             lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
    #             weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
    #         params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    #
    #     optimizer = torch.optim.Adam(params, lr)
    #     return optimizer

    # @classmethod
    # def build_evaluator(cls, cfg, dataset_name):
    #     output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    #     evaluators = [GenericEvaluator(dataset_name, cfg, output_folder)]
    #
    #     if len(evaluators) == 1:
    #         return evaluators[0]
    #
    #     return DatasetEvaluators(evaluators)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))

    @classmethod
    def eval(cls, cfg, model, name):
        evaluators = [cls.build_evaluator(cfg, name)]

        res = cls.test(cfg, model, evaluators)

        return res

    # def build_hooks(self):
    #     """
    #     Build a list of default hooks, including timing, evaluation,
    #     checkpointing, lr scheduling, precise BN, writing events.
    #     Returns:
    #         list[HookBase]:
    #     """
    #     cfg = self.cfg.clone()
    #     cfg.defrost()
    #     cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
    #
    #     ret = [
    #         hooks.IterationTimer(),
    #         hooks.LRScheduler(self.optimizer, self.scheduler),
    #         hooks.PreciseBN(
    #             # Run at the same freq as (but before) evaluation.
    #             cfg.TEST.EVAL_PERIOD,
    #             self.model,
    #             # Build a new data loader to not affect training
    #             self.build_train_loader(cfg),
    #             cfg.TEST.PRECISE_BN.NUM_ITER,
    #         )
    #         if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
    #         else None,
    #     ]
    #
    #     # Do PreciseBN before checkpointer, because it updates the model and need to
    #     # be saved by checkpointer.
    #     # This is not always the best: if checkpointing has a different frequency,
    #     # some checkpoints may have more precise statistics than others.
    #     if comm.is_main_process():
    #         ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))
    #
    #     def test_and_save_results():
    #         self._last_eval_results = self.test(self.cfg, self.model)
    #         return self._last_eval_results
    #
    #     # Do evaluation after checkpointer, because then if it fails,
    #     # we can use the saved checkpoint to debug.
    #     # ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
    #
    #     if comm.is_main_process():
    #         # run writers in the end, so that evaluation metrics are written
    #         ret.append(hooks.PeriodicWriter(self.build_writers()))
    #     return ret

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.
        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            # if comm.is_main_process():
            #     assert isinstance(
            #         results_i, dict
            #     ), "Evaluator must return a dict on the main process. Got {} instead.".format(
            #         results_i
            #     )
            #     logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            #     print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


# class RCNN(GeneralizedRCNN):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#
#     # def preprocess_image(self, batched_inputs):
#     #     """
#     #     Normalize, pad and batch the input images.
#     #     """
#     #     images = [x["image"].to(self.device) for x in batched_inputs]
#     #     images = ImageList.from_tensors(images, self.backbone.size_divisibility)
#     #     return images