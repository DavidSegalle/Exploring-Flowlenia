import jax
import numpy as np
from typing import NamedTuple
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from flowlenia.simutils import load_single_file
import os
from tqdm import tqdm


class GenomeData(NamedTuple):
    uPt: list  # unique genomes at each time
    fuPt: np.ndarray  # concatenated uniaue genomes
    fT: np.ndarray  # timetesps matching concatenated genomes
    uPit: list  # unique genome ids at each time
    cPt: list  # cell counts for each genome at each time
    aPt: list  # total matter for each genome at each time
    iPt: list  # id of genome for each cell at each time
    n_uP: int  # nb of different genomes
    At: list  # flattened map of matter at each step
    uP: np.ndarray  # all unique genomes
    T: np.ndarray
    # --- PCA ---
    uPt_pr: list  # same than uPt but projected through PCA
    pca: PCA


def compute_genome_data_online(path: list, limit=None, subsampling_rate=1):
    files = os.listdir(path)
    ordered_files = sorted(files, key=lambda x: int(x.split(".")[0]))
    ordered_files = ordered_files[::subsampling_rate]
    if limit is not None:
        ordered_files = ordered_files[:limit]

    N = len(ordered_files)  # number of data points

    uPt = []
    cPt = []
    iPt = []
    At = []
    aPt = []
    uPit = []
    n = []

    fuPt = None
    for file in tqdm(ordered_files):
        data = load_single_file(f"{path}/{file}")
        uPt.append(np.array(data["uP"]))
        cPt.append(np.array(data["cP"]))
        iPt.append(np.array(data["iP"].reshape((-1,))))
        if "A" in data:
            At.append(np.array(data["A"].reshape((-1,))))
            aPt.append(np.array(jax.ops.segment_sum(At[-1], iPt[-1])))
        else:
            At = None
            aPt.append(np.array(data["aP"]))

        if fuPt is None:
            fuPt = np.array(data["uP"])
        else:
            fuPt = np.concatenate([fuPt, np.array(data["uP"])])
        n.append(data["uP"].shape[0])

    uP, Pi = np.unique(fuPt, axis=0, return_inverse=True)
    i = 0
    for j in range(len(n)):
        uPit.append(Pi[i : i + n[j]])
        i += n[j]
    fuPt = np.concatenate(uPt)
    fT = np.concatenate([np.ones(u.shape[:1]) * i for i, u in enumerate(uPt)])

    n_uP = uP.shape[0]

    return GenomeData(
        uPt=uPt,
        cPt=cPt,
        iPt=iPt,
        aPt=aPt,
        At=At,
        n_uP=n_uP,
        uP=uP,
        uPit=uPit,
        fuPt=fuPt,
        fT=fT,
        pca=None,
        uPt_pr=None,
        T=len(ordered_files) * subsampling_rate,
    )


def compute_genome_data(data: list, verbose: bool = False):

    N = len(data)  # number of data points
    T = np.arange(len(data)) * 100

    uPt = [d["uP"] for d in data]
    cPt = [d["cP"] for d in data]
    iPt = [d["iP"].reshape((-1,)) for d in data]

    fuPt = np.concatenate(uPt)
    fT = np.concatenate([np.ones(u.shape[:1]) * i for i, u in enumerate(uPt)]) * 100

    if verbose:
        print("Computing unique genomes set")
    uP, Pi = np.unique(fuPt, axis=0, return_inverse=True)
    n_uP = uP.shape[0]
    if verbose:
        print(f"Found {n_uP} unique genomes")
    uPit = []
    i = 0
    for d in data:
        n = d["uP"].shape[0]
        uPit.append(Pi[i : i + n])
        i += n
    if "A" in data[0]:
        At = [d["A"].reshape((-1,)) for d in data]
        if verbose:
            print(jax.ops.segment_sum(At[0], iPt[0]))
        aPt = [jax.ops.segment_sum(a, ipt) for a, ipt in zip(At, iPt)]
    else:
        At = None
        aPt = [d["aP"] for d in data]

    if verbose:
        print("Computing pca projection")
    pca = PCA(n_components=5)
    pca.fit(uP)
    if verbose:
        print("		explained variance: ", pca.explained_variance_ratio_)
    uPt_pr = [pca.transform(upt) for upt in uPt]

    return GenomeData(
        uPt=uPt,
        cPt=cPt,
        iPt=iPt,
        aPt=aPt,
        At=At,
        n_uP=n_uP,
        uP=uP,
        uPit=uPit,
        fuPt=fuPt,
        fT=fT,
        pca=pca,
        uPt_pr=uPt_pr,
        T=T,
    )


def compute_coverage(data: GenomeData, bins: np.ndarray, use_projected: bool = False):
    """return coverage of discretized space by genomes at each time step and binned genomes"""
    pts = data.uPt_pr if use_projected else data.uPt
    d = pts[0].shape[-1]
    bin_uPt = [np.digitize(upt, bins) for upt in data.uPt]
    coverage = [np.unique(bupt).shape[0] / (len(bins) ** d) for bupt in bin_uPt]
    return coverage, bin_uPt


def plot_genomes_projection(data: GenomeData, d=2, cmap="rainbow"):

    fig = plt.figure(figsize=(16, 16))
    fuPt_pr = np.concatenate(data.uPt_pr)
    if d == 3:
        ax = fig.add_subplot(projection="3d")
        ax.scatter(*fuPt_pr[:, :3].T, c=data.fT, cmap=cmap)  # type:ignore
    else:
        ax = fig.add_subplot()
        ax.scatter(*fuPt_pr[:, :2].T, c=data.fT, cmap=cmap)  # type:ignore

    plt.show()


def plot_genomes_projection_in_time(data: GenomeData, d=2, cmap="rainbow"):
    fig = plt.figure(figsize=(16, 16))
    fuPt_pr = np.concatenate(data.uPt_pr)
    if d == 2:
        ax = fig.add_subplot()
        ax.scatter(fuPt_pr[:, 0], data.T, c=fuPt_pr[:, 1], cmap=cmap)  # type:ignore
    elif d == 3:
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            fuPt_pr[:, 0], fuPt_pr[:, 1], data.T, c=fuPt_pr[:, 2], cmap=cmap
        )  # type:ignore
    plt.show()
