from typing import Callable, Optional, Tuple, TypeAlias, Any
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.experimental.host_callback as hcb
import numpy as np
from jaxtyping import PyTree
import os
import gzip

from flowlenia.vizutils import VideoWriter, img_from_flp, loop_video
from flowlenia.utils import loop_tqdm
from tqdm import tqdm
from threading import Thread


cpus = jax.devices("cpu")

try:
    import _pickle as pickle  # type:ignore
except:
    import pickle

State: TypeAlias = PyTree

dflt_transform_fn = lambda data: {"t": data["t"], "s": data["s_"]}
id_host_transform_fn = lambda data: data["s"]

####### Host transform functions that are used to transform data on before logging


def host_transform_fn(data):
    s = data["s"]
    A, P = s.A, s.P
    fP = P.reshape((-1, P.shape[-1]))
    uP, iP, cP = jnp.unique(fP, axis=0, return_counts=True, return_inverse=True)
    return {"uP": uP, "cP": cP, "iP": iP, "A": A.sum(-1)}


def host_transform_fn_full(data):
    s = data["s"]
    A, P = s.A, s.P
    fP = P.reshape((-1, P.shape[-1]))
    uP, iP, cP = jnp.unique(fP, axis=0, return_counts=True, return_inverse=True)
    return {"uP": uP, "cP": cP, "iP": iP, "A": A.sum(-1), "s": s}


def host_transform_fn_light(data):
    s = data["s"]
    A, P = s.A, s.P
    fP = P.reshape((-1, P.shape[-1]))
    uP, iP, cP = jnp.unique(fP, axis=0, return_counts=True, return_inverse=True)
    aP = jax.ops.segment_sum(A.sum(-1).reshape((-1)), iP.reshape((-1,)))
    return {"uP": uP, "cP": cP, "iP": iP, "aP": aP}


class Simulator:
    """
    Simulator class to run simulations and log data
    """

    def __init__(
        self,
        model: PyTree,
        save_pth: Optional[str] = None,
        save_freq: int = 50,
        transform_fn: Callable[[dict], dict] = dflt_transform_fn,
        host_transform_fn: Callable[[dict], Any] = host_transform_fn,
        zip_files: bool = False,
        video_pth: Optional[str] = None,
        video_modes: list = ["both", "state", "param"],
        video_sample_rate: int = 5,
        video_frame_rate: int = 60,
        video_downscale_factor: int = 1,
    ):
        """
        Args:
            model (PyTree): callable and have initialize method implemented
            save_pth (Optional[str], optional): directory in which data will be saved. Defaults to None.
            save_freq (int, optional): frequency at which data is saved. Defaults to 50.
            transform_fn (Callable[[dict], dict], optional): function taking raw data dict and transforming it. Is called on xla side. Defaults to dflt_transform_fn.
            host_transform_fn (Callable[[dict], Any], optional): function taking output from transform_fn and outputting. Defaults to host_transform_fn.
            zip_files (bool, optional): whether to zip files or not. Defaults to False.
            video_pth (Optional[str], optional): path to save video. Defaults to None.
            video_modes (list, optional): modes to save video in. Defaults to ["both", "state", "param"].
            video_sample_rate (int, optional): rate at which video is sampled. Defaults to 5.
            video_frame_rate (int, optional): frame rate of the video. Defaults to 60.
            video_downscale_factor (int, optional): downscale factor of the video. Defaults to 1.
        """

        if save_pth is not None and not os.path.isdir(save_pth):
            os.makedirs(save_pth)
        elif save_pth is not None:
            print(f"path {save_pth} already exist, simulating can overwrite content")

        self.model = model
        self.save_pth = save_pth
        self.save_freq = save_freq
        self.transform_fn = transform_fn
        self.host_transform_fn = host_transform_fn
        self.zip_files = zip_files

        if video_pth is None:
            video_pth = save_pth + "/video/video.mp4"
        dir_pth = os.path.dirname(video_pth)
        if not os.path.isdir(dir_pth):
            os.makedirs(dir_pth)
        self.video_pth = video_pth
        self.video_modes = video_modes
        self.video_sample_rate = video_sample_rate
        self.video_frame_rate = video_frame_rate
        self.video_downscale_factor = video_downscale_factor

    # -------------------------------------------------------------------
    def _tap_func(self, data, *_):
        if self.save_pth is None:
            return
        t = data["t"]
        filepath = f"{self.save_pth}/{t}"
        data = self.host_transform_fn(data)
        if not self.zip_files:
            with open(filepath + ".pickle", "wb") as file:
                pickle.dump(data, file)
        else:
            with gzip.GzipFile(filepath + ".zip", "wb") as file:
                pickle.dump(data, file)

    def log_jax(self, data: dict):
        """
        save data to self.save_pth
        Args:
                data (dict): data
        """

        data = self.transform_fn(data)
        assert "t" in data.keys()
        hcb.id_tap(self._tap_func, data)
        return None

    def log(self, data: dict):
        """
        save data to self.save_pth
        Args:
                data (dict): data
        """

        data = self.transform_fn(data)
        assert "t" in data.keys()
        self._tap_func(data)
        return None

    # -------------------------------------------------------------------

    def simulate(self, s0: State, steps: int, key: jax.Array = jr.key(1)):
        """
        Simulate the model for steps steps
        Args:
            s0 (State): initial state
            steps (int): number of steps to simulate
            key (jax.Array, optional): random key. Defaults to jr.key(1).

        Returns:
            State: final state
        """

        @loop_tqdm(steps, desc="Simulating")
        @loop_video(
            steps,
            self.video_sample_rate,
            video_pth=self.video_pth,
            modes=self.video_modes,
            framerate=self.video_frame_rate,
            wall=self.model.cfg.wall_mask,
            downscale_factor=self.video_downscale_factor,
        )
        def _step(t: int, c: Tuple[State, jax.Array]):
            # --- sim step
            s, k = c
            k, _k = jr.split(k)
            s_ = self.model(s, _k)
            # --- log
            jax.lax.cond(
                t % self.save_freq == 0,
                lambda data: self.log_jax(data),
                lambda data: None,
                {"s": s, "s_": s_, "t": t},
            )
            return s_, k

        s = jax.lax.fori_loop(0, steps, _step, (s0, key))
        return s

    def flush_save_path(self):
        """Remove all zip files in save path"""
        if self.save_pth is not None:
            files = os.listdir(self.save_pth)
            for file in files:
                if file.endswith(".zip") or file.endswith(".pickle"):
                    os.remove(f"{self.save_pth}/{file}")


def load_single_file(pth: str, zip_files: bool = True):
    if zip_files:
        with gzip.GzipFile(pth, "rb") as file:
            o = pickle.load(file)
        return o
    else:
        with open(pth, "rb") as file:
            o = pickle.load(file)
        return o


def load_files(save_pth: str, zip_files: bool = True, limit=None, subsampling_rate=1):
    files = os.listdir(save_pth)
    ordered_files = sorted(files, key=lambda x: int(x.split(".")[0]))
    ordered_files = ordered_files[::subsampling_rate]
    if limit is not None:
        ordered_files = ordered_files[:limit]
    datas = [
        load_single_file(f"{save_pth}/{file}", zip_files) for file in ordered_files
    ]
    return datas


def load_and_make_video(
    save_pth: str, video_pth: str, zip_files: bool = True, mode="both", wall=None
):
    files = os.listdir(save_pth)
    ordered_files = sorted(files, key=lambda x: int(x.split(".")[0]))
    with VideoWriter(video_pth, 10) as vid:
        for file in tqdm(ordered_files):
            data = load_single_file(f"{save_pth}/{file}", zip_files)
            vid.add(img_from_flp(data["s"].A, data["s"].P, mode=mode, wall=wall))
