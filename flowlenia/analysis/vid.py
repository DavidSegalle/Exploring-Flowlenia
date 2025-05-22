import moviepy.editor as mpy
import numpy as np
import os
import shutil
from tqdm import tqdm
from joblib import Parallel, delayed


# cut video in small segments and record their size
def analyse_video(path: str, clip_length: int):
    temp_folder = path.replace(".mp4", "_analysis_temp")
    os.makedirs(temp_folder, exist_ok=True)
    clip = mpy.VideoFileClip(path)
    sizes = []
    for i in tqdm(range(0, int(clip.duration), clip_length)):
        clip.subclip(i, i + clip_length).write_videofile(
            f"{temp_folder}/{i}.mp4", verbose=False, logger=None
        )
        sizes.append(os.path.getsize(f"{temp_folder}/{i}.mp4"))
        # delete temp file
        os.remove(f"{temp_folder}/{i}.mp4")
    # delete temp folder
    shutil.rmtree(temp_folder)
    return np.array(sizes[:-1])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd

    directory = "/home/thomiche/Documents/videos/neonego_0cells"
    videos = [
        f"{directory}/{video}"
        for video in os.listdir(directory)
        if video.endswith(".mp4")
    ]

    clip_length = 1

    sizes = Parallel(n_jobs=-1)(
        delayed(analyse_video)(video, clip_length) for video in videos
    )
    for i in range(len(videos)):
        pd.DataFrame(
            columns=["time", "size"],
            data={"time": np.arange(len(sizes[i])) * clip_length, "size": sizes[i]},
        ).to_csv(f"{videos[i]}_sizes.csv")
        plt.plot(sizes[i], label=f"Video {i}")
    plt.xlabel("Time (s)")
    plt.ylabel("Size (bytes)")
    plt.legend()
    plt.show()
