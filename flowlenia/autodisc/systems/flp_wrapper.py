import dataclasses
import io
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from adtool.utils.misc.torch_utils import (
    SphericPad,
    complex_mult_torch,
    roll_n,
    soft_clip,
)

from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator
from matplotlib.animation import FuncAnimation
from numpy import ndarray
from PIL import Image


from pydantic import BaseModel
from pydantic.fields import Field

from adtool.systems.System import System


from flowlenia.flowlenia_params import FlowLeniaParams
from flowlenia.environments import init_connected_cells


torch.set_default_dtype(torch.float32)


class FlowLeniaWrapper(System):

    def __init__(
        self,
        flowlenia_instance: FlowLeniaParams,
    ):
        super().__init__()
        self.flp = flowlenia_instance
        self.video_path = "flowlenia.mp4"

        self.locator = BlobLocator()

        self.nb_k = self.flp.cfg.k
        self.SX = self.flp.cfg.X
        self.SY = self.flp.cfg.Y

    def map(self, input: Dict) -> Dict:
        params = self._process_dict(input)

        init_state = self._init_state(params)

        final_state = self._rollout(init_state)

        output_dict = deepcopy(input)
        # must detach here as gradients are not used
        # and this also leads to a deepcopy error downstream
        # also, squeezing leading dimensions for convenience
        output_dict["output"] = self.orbit[-1].detach().clone()

        return output_dict

    def render(self, data_dict, mode: str = "PIL_image") -> Tuple[bytes, str]:
        # ignores data_dict, as the render is based on self.orbit
        # in which only the last state is stored in data_dict["output"]

        im_array = []
        for img in self.orbit:
            # need to squeeze leading dimensions
            parsed_img = img.cpu().detach().numpy()

            img = state2img(parsed_img)
            if img.dtype in [np.float32, np.float64]:
                img = np.uint8(img.clip(0, 1) * 255)
            if len(img.shape) == 2:
                img = np.repeat(img[..., None], 3, -1)
            im_array.append(img)

        if mode == "human":
            matplotlib.use("TkAgg")
            fig = plt.figure(figsize=(4, 4))
            animation = FuncAnimation(
                fig, lambda frame: plt.imshow(frame), frames=im_array
            )
            plt.axis("off")
            plt.tight_layout()
            return plt.show()
        elif mode == "PIL_image":
            byte_img = io.BytesIO()
            imageio.mimwrite(
                byte_img, im_array, "mp4", fps=10, output_params=["-f", "mp4"]
            )
            return byte_img.getvalue(), "mp4"
        else:
            raise NotImplementedError

    def _process_dict(self, input_dict: Dict) -> FlowLeniaParameters:
        """
        Converts data_dictionary and parses for the correct
        parameters for Lenia.
        """
        init_params = deepcopy(input_dict["params"])
        if not isinstance(init_params, FlowLeniaParameters):
            dyn_p = FlowLeniaDynamicalParameters(**init_params["dynamic_params"])
            init_state = init_params["init_state"]
            params = FlowLeniaParameters(dynamic_params=dyn_p, init_state=init_state)
        return params
