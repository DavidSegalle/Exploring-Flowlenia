import jax.numpy as jnp
import jax.random as jr
from typing import Callable
import numpy as np

from flowlenia.flowlenia_params import FlowLeniaParams, Config as ConfigParams


def init_activation_params_grid(flp, seed, init_mask, n_clusters=4, params=None):
    kinit, kstate, kparams = jr.split(jr.key(seed), 3)
    s = flp.initialize(params)
    A = (
        jr.uniform(kstate, (flp.cfg.X, flp.cfg.Y, flp.cfg.C))
        * (init_mask == 0)[..., None]
    )
    P = jnp.ones((flp.cfg.X, flp.cfg.Y, flp.cfg.k))
    for i in range(n_clusters * 2):
        for j in range(n_clusters * 2):
            kparams, k_ = jr.split(kparams, 2)
            P = P.at[
                i
                * (flp.cfg.X // (2 * n_clusters)) : (i + 1)
                * (flp.cfg.X // (2 * n_clusters)),
                j
                * (flp.cfg.Y // (2 * n_clusters)) : (j + 1)
                * (flp.cfg.Y // (2 * n_clusters)),
                :,
            ].set(jr.uniform(k_, (1, 1, flp.cfg.k)))
    Pmix = jnp.ones((flp.cfg.X, flp.cfg.Y, flp.cfg.k))
    for i in range(n_clusters * 2):
        for j in range(n_clusters * 2):
            kparams, k_ = jr.split(kparams, 2)
            Pmix = Pmix.at[
                i
                * (flp.cfg.X // (2 * n_clusters)) : (i + 1)
                * (flp.cfg.X // (2 * n_clusters)),
                j
                * (flp.cfg.Y // (2 * n_clusters)) : (j + 1)
                * (flp.cfg.Y // (2 * n_clusters)),
                :,
            ].set(jr.uniform(k_, (1, 1, flp.cfg.k)))
    s = s.replace(A=A, P=P, Pmix=Pmix)
    return s


def init_cross_map(
    cfg: ConfigParams,
    seed: int,
    callback: Callable,
    width: int = 8,
    value: float = -10.0,
    n=1,
    n_clusters=4,
    cluster_padding=4,
) -> FlowLeniaParams:
    """
    Initialize an environmet with walls that divide evenly the map in 2**(n-1) cells. The matter is divided between 2**(n_clusters-1) clusters.

    Args:
        cfg (ConfigParams): Configuration parameters for the environment.
        seed (int): Seed for random number generation.
        callback (Callable): Callback function for the environment.
        width (int, optional): Width of the walls. Defaults to 8.
        value (float, optional): Value of the walls. Defaults to -10.0.
        n (int, optional): Number of divisions of the map. Defaults to 1.
        n_clusters (int, optional): Number of divisions of the matter. Defaults to 4.
        cluster_padding (int, optional): number of pixels between the clusters and the walls. Defaults to 4.

    Returns:
        FlowLeniaParams: FlowLeniaParams object with the initialized environment.
        State: State object with the initialized activations.
    """

    wall_map = multi_cross_map(cfg.X, cfg.Y, width, value, n=n)
    init_mask = multi_cross_map(
        cfg.X, cfg.Y, width + cluster_padding, 1.0, n=n_clusters
    )
    cfg = cfg.replace(wall_mask=wall_map)
    flp = FlowLeniaParams(cfg, key=jr.key(seed), callback=callback)
    s = init_activation_params_grid(
        flp, seed, init_mask=init_mask, n_clusters=n_clusters
    )
    return flp, s


def init_connected_cells(
    cfg: ConfigParams,
    seed: int,
    callback: Callable,
    width: int = 8,
    value: float = -10.0,
    n=1,
    passage_width=4,
    n_clusters=4,
    cluster_padding=4,
) -> FlowLeniaParams:
    """
    Initialize an environment with walls that divide evenly the map in 2**(n-1) cells, with passage between the cells. The matter is divided between 2**(n_clusters-1) clusters.

    Args:
        cfg (ConfigParams): Configuration parameters for the environment.
        seed (int): Seed for random number generation.
        callback (Callable): Callback function for the environment.
        width (int, optional): Width of the walls. Defaults to 8.
        value (float, optional): Value of the walls. Defaults to -10.0.
        n (int, optional): Number of divisions of the map. Defaults to 1.
        passage_width (int, optional): Width of the passage between the cells. Defaults to 4.
        n_clusters (int, optional): Number of divisions of the matter. Defaults to 4.
        cluster_padding (int, optional): number of pixels between the clusters and the walls. Defaults to 4.

    Returns:
        FlowLeniaParams: FlowLeniaParams object with the initialized environment.
        State: State object with the initialized activations.
    """
    wall_map = multi_cells_connected(
        cfg.X, cfg.Y, width, value, n=n, passage_width=passage_width
    )
    init_mask = multi_cross_map(
        cfg.X, cfg.Y, width + cluster_padding, 1.0, n=n_clusters
    )
    cfg = cfg.replace(wall_mask=wall_map)
    flp = FlowLeniaParams(cfg, key=jr.key(seed), callback=callback)
    s = init_activation_params_grid(
        flp, seed, init_mask=init_mask, n_clusters=n_clusters
    )
    return flp, s


def wall_map_from_rectangles(X, Y, rectangle_list=[], value=-10.0):
    """
    Create a wall map from a list of rectangles.

    Args:
        X (int): Width of the map.
        Y (int): Height of the map.
        rectangle_list (list, optional): List of rectangles. Defaults to [].
        value (float, optional): Value of the walls. Defaults to -10.0.

    Returns:
        np.array: Wall map.
    """
    mask = np.zeros((X, Y))
    for r in rectangle_list:
        mask[r[0] : r[1], r[2] : r[3]] = value
    return mask


def create_walled_area(X, Y, width=10, value=-10.0):
    """
    Create a map with walls around the border.

    Args:
        X (int): Width of the map.
        Y (int): Height of the map.
        width (int, optional): Width of the walls. Defaults to 10.
        value (float, optional): Value of the walls. Defaults to -10.0.

    Returns:
        np.array: Wall map.
    """
    if width <= 0:
        return np.zeros((X, Y))
    top = (0, width, 0, Y)
    bottom = (X - width, X, 0, Y)
    left = (0, X, 0, width)
    right = (0, X, Y - width, Y)
    return wall_map_from_rectangles(
        X, Y, [top, bottom, left, right], value
    ) + create_walled_area(X, Y, width - 1, value)


def create_cornerless_walled_area(X, Y, width=10, value=-10.0, corner_padding=10):
    """
    Create a map with walls around the border, but without the corners.

    Args:
        X (int): Width of the map.
        Y (int): Height of the map.
        width (int, optional): Width of the walls. Defaults to 10.
        value (float, optional): Value of the walls. Defaults to -10.0.
        corner_padding (int, optional): Width of the cornerless area. Defaults to 10.

    Returns:
        np.array: Wall map.
    """
    m = create_walled_area(X, Y, width, value)
    m[0:corner_padding, 0:corner_padding] = 0
    m[0:corner_padding, Y - corner_padding : Y] = 0
    m[X - corner_padding : X, 0:corner_padding] = 0
    m[X - corner_padding : X, Y - corner_padding : Y] = 0
    for i in range(width):
        m = m + create_cornerless_walled_area(X, Y, i, value, corner_padding + 1)
    return m


def multi_cross_map(X, Y, width=10, value=-10.0, n=1):
    """
    Create a map with walls that divide evenly the map in 2**(n-1) cells.

    Args:
        X (int): Width of the map.
        Y (int): Height of the map.
        width (int, optional): Width of the walls. Defaults to 10.
        value (float, optional): Value of the walls. Defaults to -10.0.
        n (int, optional): Number of divisions of the map. Defaults to 1.

    Returns:
        np.array: Wall map.
    """
    assert (
        X % 2 == 0 and Y % 2 == 0 and width % 2 == 0
    ), "The map can not be created with these dimensions : Cells can not be divided evenly."
    if n <= 0:
        return np.zeros((X, Y))
    if n == 1:
        return create_walled_area(X, Y, width, value)
    else:
        small_cross = multi_cross_map(X // 2, Y // 2, width, value, n - 1)
        m = create_walled_area(X, Y, width // 2, value)
        m[0 : X // 2, 0 : Y // 2] = small_cross
        m[X // 2 : X, 0 : Y // 2] = small_cross
        m[0 : X // 2, Y // 2 : Y] = small_cross
        m[X // 2 : X, Y // 2 : Y] = small_cross
        return m


def multi_cells_connected(X, Y, width=10, value=-10.0, n=1, passage_width=8):
    """
    Create a map with walls that divide evenly the map in 2**(n-1) cells, with passage between the cells.

    Args:
        X (int): Width of the map.
        Y (int): Height of the map.
        width (int, optional): Width of the walls. Defaults to 10.
        value (float, optional): Value of the walls. Defaults to -10.0.
        n (int, optional): Number of divisions of the map. Defaults to 1.
        passage_width (int, optional): Width of the passage between the cells. Defaults to 8.

    Returns:
        np.array: Wall map.
    """

    def multi_cells_connected_rec(X, Y, width=10, value=-10.0, n=1, passage_width=8):
        assert (
            X % 2 == 0 and Y % 2 == 0 and width % 2 == 0
        ), "The map can not be created with these dimensions : Cells can not be divided evenly."
        if n <= 0:
            return np.zeros((X, Y))
        if n == 1:
            return create_cornerless_walled_area(
                X, Y, width, value, passage_width + width
            )
        else:
            small_cross = multi_cells_connected_rec(
                X // 2, Y // 2, width, value, n - 1, passage_width
            )
            m = create_cornerless_walled_area(X, Y, (width + passage_width) // 2, value)
            m[0 : X // 2, 0 : Y // 2] = small_cross
            m[X // 2 : X, 0 : Y // 2] = small_cross
            m[0 : X // 2, Y // 2 : Y] = small_cross
            m[X // 2 : X, Y // 2 : Y] = small_cross
            return m

    m = multi_cells_connected_rec(X, Y, width, value, n, passage_width)
    if n <= 0:
        return m
    return m + create_walled_area(X, Y, width, value)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    m = multi_cross_map(256, 256, 16, -10.0, 1)
    mp = multi_cells_connected(256, 256, 16, -10.0, 1)
    plt.imshow(mp != m)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.set_size_inches(5, 5, forward=True)
    plt.savefig("connected_cells.png")
