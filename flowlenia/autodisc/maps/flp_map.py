import dataclasses
from copy import deepcopy
from dataclasses import asdict, dataclass

from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator
from adtool.
import sys

from flowlenia.flowlenia_params import Params, FlowLeniaParams, Rule_space
from flowlenia.autodisc.systems import FlowLeniaWrapper


class FlowLeniaParameterMap(Leaf):
    """
    Due to the complexities of initializing Lenia parameters,
    it's easier to make this custom parameter map.
    """

    def __init__(
        self,
        system: FlowLeniaWrapper,
        premap_key: str = "params",
        param_obj: Rule_space = None,
        **config_decorator_kwargs,
    ):
        super().__init__()

        if param_obj is None:
            param_obj = Rule_space(system.nb_k)

        self.param_obj = param_obj

        self.locator = BlobLocator()
        # if config options set by decorator, override config
        if len(config_decorator_kwargs) > 0:
            param_obj = dataclasses.replace(param_obj, **config_decorator_kwargs)

        print("premap_key", premap_key)
        self.premap_key = premap_key

        self.SX = system.SX
        self.SY = system.SY

    def map(self, input: Dict, override_existing: bool = True) -> Dict:
        """
        Takes input dictionary and overrides the
        premap_key with generated parameters for Lenia.
        """
        intermed_dict = deepcopy(input)

        # check if either "params" is not set or if we want to override
        if (override_existing and self.premap_key in intermed_dict) or (
            self.premap_key not in intermed_dict
        ):
            # overrides "params" with new sample
            intermed_dict[self.premap_key] = self.sample()
        else:
            # passes "params" through if it exists
            pass

        return intermed_dict

    def sample(self) -> Dict:
        """
        Samples from the parameter map to yield the parameters ready
        for processing by the model (LeniaCPPN), i.e., including the genome.
        """
        dp = self.param_obj.sample()
        p_dict = {
            "dynamic_params": asdict(dp),
        }
        return p_dict

    def mutate(self, parameter_dict: Dict) -> Dict:
        """
        Takes the dictionary of only parameters outputted by
        the explorer and mutates them.
        """
        intermed_dict = deepcopy(parameter_dict)

        # mutate dynamic parameters
        params = Params(**parameter_dict["dynamic_params"])
        mutated_dp = self.param_obj.mutate(params)

        intermed_dict["dynamic_params"] = asdict(mutated_dp)

        return intermed_dict
