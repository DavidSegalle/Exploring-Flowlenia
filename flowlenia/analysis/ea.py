import numpy as np
from tqdm import tqdm
from typing import NamedTuple, Optional
import jax

from flowlenia.analysis.genomes import GenomeData
import os
from flowlenia.simutils import load_single_file


class EAData(NamedTuple):
    A_p: np.ndarray
    A_c: np.ndarray
    A_n: np.ndarray
    a_p: Optional[np.ndarray] = None
    a_c: Optional[np.ndarray] = None
    a_n: Optional[np.ndarray] = None
    delta_p: Optional[np.ndarray] = None
    delta_c: Optional[np.ndarray] = None
    delta_n: Optional[np.ndarray] = None


def compute_nnea_online(path: str, limit=None, subsampling_rate=1):
    """
    Compute NNEA (Non Neutral Evolutionary Activity) online based on the given path.

    Args:
        path (str): The input path.
        limit ([type], optional): The maximum number of files to process. Defaults to None.
        subsampling_rate (int, optional): Interval between files. Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: NNEA, delta_n, delta_p

    """

    files = os.listdir(path)
    ordered_files = sorted(files, key=lambda x: int(x.split(".")[0]))
    ordered_files = ordered_files[::subsampling_rate]
    if limit is not None:
        ordered_files = ordered_files[:limit]

    ts = np.arange(len(ordered_files))
    A_n = np.zeros((len(ts),))
    deltas_n = np.zeros((len(ts),))
    deltas_p = np.zeros((len(ts),))

    prev_uPt = None
    prev_aPt = None
    t = 0

    for file in tqdm(ordered_files):
        data = load_single_file(f"{path}/{file}")
        new_uPt = np.array(data["uP"])
        if "A" in data:
            new_iPt = np.array(data["iP"].reshape((-1,)))
            At = np.array(data["A"].reshape((-1,)))
            new_aPt = np.array(jax.ops.segment_sum(At, new_iPt))
        else:
            At = None
            new_aPt = np.array(data["aP"])

        if t > 0:
            C = new_aPt.sum()

            uP, Pi = np.unique(
                np.concatenate([prev_uPt, new_uPt]), axis=0, return_inverse=True
            )
            prev_pr = np.zeros((len(uP),))
            prev_pr[Pi[: len(prev_uPt)]] = prev_aPt
            prev_pr = prev_pr / prev_pr.sum()
            new_pr = np.zeros((len(uP),))
            new_pr[Pi[len(prev_uPt) :]] = new_aPt
            new_pr = new_pr / C
            delta_n = np.where(new_pr > prev_pr, C * (new_pr - prev_pr) ** 2, 0.0)

        else:
            delta_n = np.zeros_like(new_uPt)
        delta_p = len(new_uPt)

        deltas_n[t] = delta_n.sum()
        deltas_p[t] = delta_p
        prev_uPt = new_uPt
        prev_aPt = new_aPt
        t += 1

    A_n = np.cumsum(deltas_n)
    return A_n, deltas_n, deltas_p


def compute_nnea(data: GenomeData):
    """
    Compute NNEA (Non Neutral Evolutionary Activity) based on the given GenomeData.

    Args:
        data (GenomeData): The input GenomeData object.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: NNEA, delta_n, delta_p

    """
    n_species = data.n_uP
    ts = np.arange(len(data.uPt))
    A_n = np.zeros((len(ts),))
    deltas_n = np.zeros((len(ts), n_species))
    deltas_p = np.zeros((len(ts), n_species))

    props = np.zeros((n_species,))

    for t in tqdm(range(len(data.uPt))):
        ps = data.uPit[t]
        cs = data.aPt[t]
        C = cs.sum()
        delta_n = np.zeros(n_species)
        delta_p = np.zeros(n_species)
        delta_p[ps] = 1
        # --- Non neutral ---
        e = props[ps]
        pr = cs / C
        props[ps] = pr
        delta_n[ps] = np.where(pr > e, C * (pr - e) ** 2, 0.0)

        if t == 0:
            delta_n[ps] = 0.0

        deltas_n[t] = delta_n.sum()
        deltas_p[t] = delta_p.sum()
    A_n = np.cumsum(deltas_n)
    return A_n, deltas_n, deltas_p


def compute_ea_data(
    data: GenomeData, return_a: bool = False, return_deltas: bool = False
) -> EAData:
    """
    Compute EA (Evolutionary Activity) data based on the given GenomeData.

    Args:
        data (GenomeData): The input GenomeData object.
        return_a (bool, optional): Whether to return the activities. Defaults to False.
        return_deltas (bool, optional): Whether to return the deltas. Defaults to False.

    Returns:
        EAData: The computed EAData object containing the EA data.

    """

    n_species = data.n_uP
    ts = np.arange(len(data.uPt))

    A_p = np.zeros((len(ts),))
    A_c = np.zeros((len(ts),))
    A_n = np.zeros((len(ts),))

    if return_a:
        activities_p = np.zeros((len(ts), n_species))
        activities_c = np.zeros((len(ts), n_species))
        activities_n = np.zeros((len(ts), n_species))
    else:
        activities_p = None
        activities_c = None
        activities_n = None

    if return_deltas:
        deltas_p = np.zeros((len(ts), n_species))
        deltas_c = np.zeros((len(ts), n_species))
        deltas_n = np.zeros((len(ts), n_species))
    else:
        deltas_p = None
        deltas_c = None
        deltas_n = None

    a_p = np.zeros(n_species)
    a_c = np.zeros(n_species)
    a_n = np.zeros(n_species)

    props = np.zeros((n_species,))

    for t in tqdm(range(len(data.uPt))):
        ps = data.uPit[t]
        cs = data.aPt[t]
        C = cs.sum()
        delta_c = np.zeros_like(a_c)
        delta_p = np.zeros_like(a_p)
        delta_n = np.zeros_like(a_n)
        exist = np.zeros_like(a_p)

        exist[ps] = 1
        delta_c[ps] = cs
        # --- Existence based ---
        delta_p[ps] = 1
        # --- Non neutral ---
        e = props[ps]
        pr = cs / C
        props[ps] = pr
        delta_n[ps] = np.where(pr > e, C * (pr - e) ** 2, 0.0)
        if t == 0:
            delta_c[ps] = 0
            delta_p[ps] = 0
            delta_n[ps] = 0

        a_p = (a_p + delta_p) * exist
        a_c = (a_c + delta_c) * exist
        a_n = (a_n + delta_n) * exist

        A_p[t] = a_p.sum()
        A_c[t] = a_c.sum()
        A_n[t] = a_n.sum()

        if return_a:
            activities_p[t] = a_p  # type:ignore
            activities_c[t] = a_c  # type:ignore
            activities_n[t] = a_n  # type:ignore
        if return_deltas:
            deltas_p[t] = delta_p  # type:ignore
            deltas_c[t] = delta_c  # type:ignore
            deltas_n[t] = delta_n  # type:ignore

    return EAData(
        A_p=A_p,
        A_c=A_c,
        A_n=A_n,
        a_n=activities_n,
        a_p=activities_p,
        a_c=activities_c,
        delta_n=deltas_n,
        delta_p=deltas_p,
        delta_c=deltas_c,
    )
