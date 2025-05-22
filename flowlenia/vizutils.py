import os

os.environ["FFMPEG_BINARY"] = "ffmpeg"
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
from jax.debug import callback
import typing
import shutil

cpus = jax.devices("cpu")


@jax.jit
def _rgb_from_h(h):
    i = (h * 6.0).astype(int)
    f = h * 6.0 - i

    q = 1.0 - f
    t = 1.0 + f

    return jax.lax.switch(
        i,
        [
            lambda _: jnp.array([1.0, t, 0.0]),
            lambda _: jnp.array([q, 1.0, 0.0]),
            lambda _: jnp.array([0.0, 1.0, t]),
            lambda _: jnp.array([0.0, q, 1.0]),
            lambda _: jnp.array([t, 0.0, 1.0]),
            lambda _: jnp.array([1.0, 0.0, q]),
        ],
        None,
    )


get_param_color = jax.vmap(jax.vmap(_rgb_from_h))


def get_activation_colormap(A, only_hue=False):
    if len(A.shape) == 2:
        return jnp.dstack([A, A, A])

    if only_hue:
        return get_param_color(A.sum(-1) % 1.0)

    C = A.shape[-1]
    if C == 1:
        return jnp.dstack([A, A, A])
    elif C == 2:
        return jnp.dstack([A[..., 0], A[..., 0], A[..., 1]])
    else:
        return A[..., :3]


def display_fl(states, n_step=100, save_path=None):
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(5, 5, forward=True)
    ims = []
    for i in range(n_step):
        A = states.A[i]
        img = get_activation_colormap(A)
        im = ax.imshow(img, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    if save_path:
        ani.save(save_path)
    else:
        ani.save("videos/flowlenia_temp.mp4")
    plt.show()


def img_from_flp(A, P, mode="both", wall=None):
    if mode == "state":
        img = get_activation_colormap(A)
    elif mode == "param":
        img = get_activation_colormap(P, only_hue=True)
    elif mode == "both":
        img = get_activation_colormap(P, only_hue=True) * A.sum(-1, keepdims=True)
    else:
        raise ValueError("mode should be 'state', 'param' or 'both'")
    if wall is not None:
        img = img + get_activation_colormap(wall != 0) * 0.5
    return jax.device_put(img, cpus[0])


def display_flp(states, n_step=100, save_path=None, mode="both", wall=None):
    # mode = "state" shows only the state, "param" shows only the parameters, "both" summarizes the state and colors it with the parameters
    is_list = isinstance(states, list)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(5, 5, forward=True)
    ims = []
    for i in range(n_step):
        if is_list:
            A, P = states[i].A, states[i].P
        else:
            A, P = states.A[i], states.P[i]
        img = img_from_flp(A, P, mode, wall)
        im = ax.imshow(img, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    if save_path:
        ani.save(save_path)
    else:
        ani.save(f"videos/flowlenia_parameters_{mode}_temp.mp4")
    plt.show()


def display_flpf(states, n_step=100, save_path=None):
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    fig.set_size_inches(10, 5, forward=True)
    ims = []
    for i in range(n_step):
        A, FM = states.A[i], states.FM[i]
        img_state = get_activation_colormap(A)
        img_food = get_activation_colormap(FM)
        padding = jnp.ones((A.shape[0], 1, 3))
        img = jnp.concatenate([img_state, padding, img_food], axis=1)
        im = ax.imshow(img, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    if save_path:
        ani.save(save_path)
    else:
        ani.save("videos/flowlenia_food_temp.mp4")
    plt.show()


class VideoWriter:
    def __init__(
        self,
        filename,
        fps=30.0,
        downscale_factor=1,
        temp_length=60,
        keep_temp=False,
        **kw,
    ):
        """
        Initialize the video writer.

        Parameters
        ----------
        filename : str
            The name of the video file.
        fps : float
            The number of frames per second.
        downscale_factor : float
            Downscale the video (upscale if < 1, downscale if > 1).
        temp_length : int
            The length of the temporary videos, in seconds.
        keep_temp : bool
            Keep the temporary files.
        **kw : dict
            Parameters for the video writer.
        """
        self.writer = None
        self.tmp_dir = filename.replace(".mp4", "_tmp")
        while os.path.exists(self.tmp_dir):
            self.tmp_dir = self.tmp_dir + "0"
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.fps = fps
        self.filename = filename
        self.params = dict(fps=fps, **kw)
        self.downscale_factor = downscale_factor
        self.frame_count = 0  # count the number of frames in the current video
        self.video_count = 0  # count the number of temporary videos
        self.keep_temp = keep_temp
        self.temp_length = temp_length  # in seconds

    def add(self, img):
        """
        Add a frame to the video.

        Parameters
        ----------
        img : np.ndarray
            The image to add to the video.
        """
        if self.downscale_factor != 1:
            img = jax.image.resize(
                img,
                (
                    int(img.shape[0] / self.downscale_factor),
                    int(img.shape[1] / self.downscale_factor),
                    3,
                ),
                method="lanczos5",
            )
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(
                size=(w, h),
                filename=f"{self.tmp_dir}/{self.video_count:>06}.mp4",
                **self.params,
            )
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1) * 255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)

        self.writer.write_frame(img)

        # self.writer.write_frame(img)
        self.frame_count += 1
        if self.frame_count > self.temp_length * self.fps:
            self.writer.close()
            self.video_count += 1
            self.frame_count = 0
            del self.writer
            self.writer = None
        del img

    def close(self):
        """
        Close the video writer and concatenate the temporary videos.
        """
        if self.writer:
            self.writer.close()
        # Concatenate the videos
        video_files = sorted(os.listdir(self.tmp_dir))
        clips = [mvp.VideoFileClip(f"{self.tmp_dir}/{f}") for f in video_files]
        final_clip = mvp.concatenate_videoclips(clips)
        final_clip.write_videofile(self.filename, verbose=False, logger=None)

        # remove the temporary directory
        if not self.keep_temp:
            shutil.rmtree(self.tmp_dir)
        else:
            print(f"Temporary files saved in {self.tmp_dir}")

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()


def loop_video(
    n: int,
    sample_rate: typing.Optional[int] = None,
    **kwargs,
) -> typing.Callable:
    _update_main, close_tqdm = build_vid(n, sample_rate, **kwargs)

    def _loop_vid(func):
        """
        Decorator that adds a tqdm progress bar to `body_fun`
        used in `jax.lax.fori_loop`.
        """

        def wrapper_video(i, val):
            state, _ = val
            _update_main(i, state)
            result = func(i, val)
            return close_tqdm(result, i)

        return wrapper_video

    return _loop_vid


def build_vid(
    n: int,
    sample_rate: int,
    **kwargs,
) -> typing.Tuple[typing.Callable, typing.Callable]:
    video_pth = kwargs.pop("video_pth", "videos/flowlenia_temp.mp4")
    modes = kwargs.pop("modes", ["both", "state", "param"])
    wall = kwargs.pop("wall", None)
    if wall is not None:
        wall = jax.device_put(wall, cpus[0])
    framerate = kwargs.pop("framerate", 30)
    downscale_factor = kwargs.pop("downscale_factor", 1)
    vid = {}

    def _define_vid(arg, transform):
        for mode in modes:
            pth = video_pth.replace(".mp4", f"_{mode}.mp4")
            vid[mode] = VideoWriter(pth, framerate, downscale_factor=downscale_factor)

    def _update_vid(state, transform):
        for mode in modes:
            vid[mode].add(img_from_flp(state.A, state.P, mode=mode, wall=wall))

    def _update_main(iter_num, state):
        _ = jax.lax.cond(
            iter_num == 0,
            lambda _: callback(_define_vid, None, None, ordered=True),
            lambda _: None,
            operand=None,
        )

        _ = jax.lax.cond(
            (iter_num % sample_rate == 0) | (iter_num == n - 1),
            lambda _: callback(_update_vid, state, None, ordered=True),
            lambda _: None,
            operand=None,
        )

    def _close_vid(arg, transform):
        for mode in modes:
            vid[mode].close()

    def close_vid(result, iter_num):
        _ = jax.lax.cond(
            iter_num == n - 1,
            lambda _: callback(_close_vid, None, None, ordered=True),
            lambda _: None,
            operand=None,
        )
        return result

    return _update_main, close_vid


# From https://github.com/kartikay-bagla/bump-plot-python
def bumpchart(
    df,
    show_rank_axis=True,
    rank_axis_distance=1.1,
    ax=None,
    scatter=False,
    holes=False,
    line_args={},
    scatter_args={},
    hole_args={},
):

    if ax is None:
        left_yaxis = plt.gca()
    else:
        left_yaxis = ax

    # Creating the right axis.
    right_yaxis = left_yaxis.twinx()

    axes = [left_yaxis, right_yaxis]

    # Creating the far right axis if show_rank_axis is True
    if show_rank_axis:
        far_right_yaxis = left_yaxis.twinx()
        axes.append(far_right_yaxis)

    for col in df.columns:
        y = df[col]
        x = df.index.values
        # Plotting blank points on the right axis/axes
        # so that they line up with the left axis.
        for axis in axes[1:]:
            axis.plot(x, y, alpha=0)

        left_yaxis.plot(x, y, **line_args, solid_capstyle="round")

        # Adding scatter plots
        if scatter:
            left_yaxis.scatter(x, y, **scatter_args)

            # Adding see-through holes
            if holes:
                bg_color = left_yaxis.get_facecolor()
                left_yaxis.scatter(x, y, color=bg_color, **hole_args)

    # Number of lines
    lines = len(df.columns)

    y_ticks = [*range(1, lines + 1)]

    # Configuring the axes so that they line up well.
    for axis in axes:
        axis.invert_yaxis()
        axis.set_yticks(y_ticks)
        axis.set_ylim((lines + 0.5, 0.5))

    # Sorting the labels to match the ranks.
    left_labels = df.iloc[0].sort_values().index
    right_labels = df.iloc[-1].sort_values().index

    left_yaxis.set_yticklabels(left_labels)
    right_yaxis.set_yticklabels(right_labels)

    # Setting the position of the far right axis so that it doesn't overlap with the right axis
    if show_rank_axis:
        far_right_yaxis.spines["right"].set_position(("axes", rank_axis_distance))

    return axes
