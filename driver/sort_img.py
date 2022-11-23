import os
from glob import glob

import numpy as np

rid = 40
for grid_id in [141]:

    files = glob(f"../output/integrins/grid_id{grid_id}/run_{rid}/visuals/*.png")
    ids = [int(f.split("/")[-1].split("_")[1].split(".")[0]) for f in files]

    sort_indx = np.argsort(ids)

    for i, indx in enumerate(sort_indx):
        curr_file = files[indx]
        new_file = f"../output/integrins/grid_id{grid_id}/run_{rid}/visuals/img_{i}.png"
        cmd = f"mv {curr_file} {new_file}"
        os.system(cmd)

    print("Making a movie...")
    cmd = f"ffmpeg -i ../output/integrins/grid_id{grid_id}/run_{rid}/visuals/img_%d.png -b:v 4M -s 500x500 -pix_fmt yuv420p mov_{grid_id}.mp4"
    os.system(cmd)
    print("Done!")
