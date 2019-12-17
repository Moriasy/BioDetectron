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

from data import SKImageLoader


class GenericEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, cfg, output_dir, distributed=False):
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir
        self._dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self.img_loader = SKImageLoader(cfg, is_train=False)

        self._metadata = MetadataCatalog.get(dataset_name)

    def reset(self):
        self._predictions = []
        self._results = {}

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            tasks = tasks + ("keypoints",)
        return tasks

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"], "ori_image": input["ori_image"], 'groundtruth':input}

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

        # if self._output_dir:
        #     PathManager.mkdirs(self._output_dir)
        #     file_path = os.path.join(self._output_dir, "instances_predictions.pth")
        #     with PathManager.open(file_path, "wb") as f:
        #         torch.save(self._predictions, f)

        self._results = OrderedDict()
        if "proposals" in self._predictions[0]:
            self._eval_box_proposals()
        if "instances" in self._predictions[0]:
            self._eval_predictions(set(self._tasks))
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)


    def _eval_box_proposals(self):
        print('Add proposal eval!!!')
        return

    def _eval_predictions(self, task):
        self._results['image_id'] = [self._predictions[0]['image_id']]
        self._results['ori_image'] = [self._predictions[0]['ori_image']]
        self._results['instances'] = [self._predictions[0]['instances']]
        self._results['groundtruth'] = [self._predictions[0]['groundtruth']]

        for n in range(min(len(self._results['groundtruth']), 10)):
            ori_img = self._results["ori_image"][n]
            img = self._results["image"][n]

            metadata = MetadataCatalog.get(self._dataset_name)

            viz = Visualizer(img, metadata)
            viz.draw_dataset_dict(self._results["groundtruth"][n]).save(
                os.path.join(self._output_dir, 'GT_{}.png'.format(n)))

            viz = Visualizer(ori_img, metadata)
            viz.draw_instance_predictions(self._results["instances"][n]).save(
                os.path.join(self._output_dir, 'pred_{}.png'.format(n)))

        self._results = {}


class VizdomVisualizer:
    """
    Based on the vizdom tools from Pix2Pix:
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    def __init__(self):
        self.ncols = 4
        self.vis = visdom.Visdom(server="http://localhost", port=1337, env="main")
        self.port = 1337

        if not self.vis.check_connection():
            self.create_visdom_connections()

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals):
        """Display current results on visdom; save current results to an HTML file.
        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        ncols = self.ncols
        if ncols > 0:        # show all the images in one visdom panel
            ncols = min(ncols, len(visuals))
            h, w = next(iter(visuals.values())).shape[:2]
            table_css = """<style>
                    table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                    table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                    </style>""" % (w, h)  # create a table css
            # create a table of images.
            title = self.name
            label_html = ''
            label_html_row = ''
            images = []
            idx = 0
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                label_html_row += '<td>%s</td>' % label
                images.append(image_numpy.transpose([2, 0, 1]))
                idx += 1
                if idx % ncols == 0:
                    label_html += '<tr>%s</tr>' % label_html_row
                    label_html_row = ''
            white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
            while idx % ncols != 0:
                images.append(white_image)
                label_html_row += '<td></td>'
                idx += 1
            if label_html_row != '':
                label_html += '<tr>%s</tr>' % label_html_row
            try:
                self.vis.images(images, nrow=ncols, win=1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win=2,
                              opts=dict(title=title + ' labels'))
            except VisdomExceptionBase:
                self.create_visdom_connections()

        else:     # show each image in a separate visdom panel;
            idx = 1
            try:
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1
            except VisdomExceptionBase:
                self.create_visdom_connections()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values
        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
