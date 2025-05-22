import typing
from copy import deepcopy

import torch
from addict import Dict
from adtool.wrappers.BoxProjector import BoxProjector

from adtool.utils.misc.torch_utils import roll_n
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator

from flowlenia.autodisc.systems.flp_wrapper import FlowLeniaWrapper

EPS = 0.0001
DISTANCE_WEIGHT = 2  # 1=linear, 2=quadratic, ...


def center_of_mass(input_array: torch.Tensor) -> torch.Tensor:

    normalizer = input_array.sum()

    grids = torch.meshgrid(*[torch.arange(0, i) for i in input_array.shape])

    center = torch.tensor(
        [
            (input_array * grids[dir].double()).sum() / normalizer
            for dir in range(input_array.ndim)
        ]
    )

    if torch.any(torch.isnan(center)):
        center = torch.tensor(
            [int((input_array.shape[0] - 1) / 2), int((input_array.shape[1] - 1) / 2)]
        )

    return center


class FlowLeniaStatistics(Leaf):
    """
    Outputs 17-dimensional embedding.
    """

    def __init__(
        self,
        system: FlowLeniaWrapper,
        premap_key: str = "output",
        postmap_key: str = "output",
    ):
        super().__init__()
        self.locator = BlobLocator()

        self.premap_key = premap_key
        self.postmap_key = postmap_key

        self.SX = system.SX
        self.SY = system.SY
        self.C = system.C

        # model
        self._statistic_names = [
            "evolutionary_activity",
            "size_mp4",
            "prediction_error",
        ]
        self._n_latents = len(self._statistic_names)

        # projector for behavior space
        self.projector = BoxProjector(premap_key=self.postmap_key)

    def map(self, input: typing.Dict) -> typing.Dict:
        """
        Compute statistics on Lenia's output
        Args:
            input: Lenia's output
            is_output_new_discovery: indicates if it is a new discovery
        Returns:
            Return a torch tensor in dict
        """

        intermed_dict = deepcopy(input)

        # store raw output
        tensor = intermed_dict[self.premap_key].detach().clone()
        raw_output_key = "raw_" + self.premap_key
        intermed_dict[raw_output_key] = tensor
        del intermed_dict[self.premap_key]

        embedding = self._calc_static_statistics(tensor)

        intermed_dict[self.postmap_key] = embedding
        intermed_dict = self.projector.map(intermed_dict)

        return intermed_dict

    def sample(self):
        return self.projector.sample()

    def _calc_distance(
        self, embedding_a: torch.Tensor, embedding_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the distance between 2 embeddings in the latent space
        /!\ batch mode embedding_a and embedding_b can be N*M or M
        """
        # l2 loss
        return (embedding_a - embedding_b).pow(2).sum(-1).sqrt()

    def _calc_static_statistics(self, final_obs: torch.Tensor) -> torch.Tensor:
        """Calculates the final statistics for lenia last observation"""

        # sum the last dimension
        final_obs = torch.sum(final_obs, dim=-1).squeeze()

        feature_vector = torch.zeros(self._n_latents)
        cur_idx = 0

        size_y = self.SY
        size_x = self.SX
        num_of_cells = size_y * size_x

        # calc initial center of mass and use it as a reference point to "center" the world around it
        mid_y = (size_y - 1) / 2
        mid_x = (size_x - 1) / 2
        mid = torch.tensor([mid_y, mid_x])

        activation_center_of_mass = torch.tensor(center_of_mass(final_obs))
        activation_shift_to_center = mid - activation_center_of_mass

        activation = final_obs
        centered_activation = roll_n(activation, 0, activation_shift_to_center[0].int())
        centered_activation = roll_n(
            centered_activation, 1, activation_shift_to_center[1].int()
        )

        # calculate the image moments
        activation_moments = calc_image_moments(centered_activation)

        # activation mass
        activation_mass = activation_moments.m00
        # activation is number of acitvated cells divided by the number of cells
        activation_mass_data = activation_mass / num_of_cells
        feature_vector[cur_idx] = activation_mass_data
        cur_idx += 1

        # activation volume
        activation_volume = torch.sum(activation > EPS)
        activation_volume_data = activation_volume / num_of_cells
        feature_vector[cur_idx] = activation_volume_data
        cur_idx += 1

        # activation density
        if activation_volume == 0:
            activation_density_data = 0
        else:
            activation_density_data = activation_mass / activation_volume
        feature_vector[cur_idx] = activation_density_data
        cur_idx += 1

        # mass distribution around the center
        distance_weight_matrix = calc_distance_matrix(self.SY, self.SX)
        if activation_mass <= EPS:
            activation_mass_distribution = 1.0
        else:
            activation_mass_distribution = torch.sum(
                distance_weight_matrix * centered_activation
            ) / torch.sum(centered_activation)

        activation_mass_distribution_data = activation_mass_distribution
        feature_vector[cur_idx] = activation_mass_distribution_data
        cur_idx += 1

        # activation moments
        activation_hu1_data = activation_moments.hu1
        feature_vector[cur_idx] = activation_hu1_data
        cur_idx += 1

        activation_hu2_data = activation_moments.hu2
        feature_vector[cur_idx] = activation_hu2_data
        cur_idx += 1

        activation_hu3_data = activation_moments.hu3
        feature_vector[cur_idx] = activation_hu3_data
        cur_idx += 1

        activation_hu4_data = activation_moments.hu4
        feature_vector[cur_idx] = activation_hu4_data
        cur_idx += 1

        activation_hu5_data = activation_moments.hu5
        feature_vector[cur_idx] = activation_hu5_data
        cur_idx += 1

        activation_hu6_data = activation_moments.hu6
        feature_vector[cur_idx] = activation_hu6_data
        cur_idx += 1

        activation_hu7_data = activation_moments.hu7
        feature_vector[cur_idx] = activation_hu7_data
        cur_idx += 1

        activation_hu8_data = activation_moments.hu8
        feature_vector[cur_idx] = activation_hu8_data
        cur_idx += 1

        activation_flusser9_data = activation_moments.flusser9
        feature_vector[cur_idx] = activation_flusser9_data
        cur_idx += 1

        activation_flusser10_data = activation_moments.flusser10
        feature_vector[cur_idx] = activation_flusser10_data
        cur_idx += 1

        activation_flusser11_data = activation_moments.flusser11
        feature_vector[cur_idx] = activation_flusser11_data
        cur_idx += 1

        activation_flusser12_data = activation_moments.flusser12
        feature_vector[cur_idx] = activation_flusser12_data
        cur_idx += 1

        activation_flusser13_data = activation_moments.flusser13
        feature_vector[cur_idx] = activation_flusser13_data
        cur_idx += 1

        return feature_vector
