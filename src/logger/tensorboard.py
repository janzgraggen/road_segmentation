from datetime import datetime

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class TensorBoardWriter:
    """
    Class for experiment tracking via TensorBoard.
    """

    def __init__(
        self,
        logger,
        project_config,
        log_dir,
        run_name,
        loss_names,
        **kwargs,
    ):
        """
        API key is expected to be provided by the user in the terminal.

        Args:
            logger (Logger): logger instance.
            project_config (dict): config for the current experiment.
            log_dir (str): path to the directory where logs are saved.
        """
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{run_name}")
        self.loss_names = loss_names

        self.step = 0
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        """
        Define current step and mode for the tracker.

        Calculates the difference between method calls to monitor
        training/evaluation speed.

        Args:
            step (int): current step.
            mode (str): current mode (partition name).
        """
        self.mode = mode
        previous_step = self.step
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar(
                "steps_per_sec", (self.step - previous_step) / duration.total_seconds()
            )
            self.timer = datetime.now()

    def add_checkpoint(self, checkpoint_path, save_dir):
        raise NotImplementedError()

    def add_scalar(self, scalar_name, scalar):
        """
        Log a scalar to the experiment tracker.

        Args:
            scalar_name (str): name of the scalar to use in the tracker.
            scalar (float): value of the scalar.
        """
        self.writer.add_scalar(scalar_name, scalar, self.step)

    def add_scalars(self, scalars):
        """
        Log several scalars to the experiment tracker.

        Args:
            scalars (dict): dict, containing scalar name and value.
        """
        for name, value in scalars.items():
            self.writer.add_scalar(name, value)

    def add_image(self, image_name, image):
        """
        Log an image to the experiment tracker.

        Args:
            image_name (str): name of the image to use in the tracker.
            image (Path | Tensor | ndarray | list[tuple] | Image): image
                in the CometML-friendly format.
        """
        # self.writer.add_image(image_name, image, self.step)

    def add_audio(self, audio_name, audio, sample_rate=None):
        raise NotImplementedError()

    def add_text(self, text_name, text):
        """
        Log text to the experiment tracker.

        Args:
            text_name (str): name of the text to use in the tracker.
            text (str): text content.
        """
        self.writer.add_text(text_name, text, self.step)

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        raise NotImplementedError()

    def add_table(self, table_name, table: pd.DataFrame):
        raise NotImplementedError()

    def add_images(self, image_names, images):
        raise NotImplementedError()

    def add_pr_curve(self, curve_name, curve):
        raise NotImplementedError()

    def add_embedding(self, embedding_name, embedding):
        raise NotImplementedError()
