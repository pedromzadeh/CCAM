import os
from glob import glob

import numpy as np

files = glob("../output/prw/grid_id0/run_0/visuals/*.png")
ids = [int(f.split("/")[-1].split("_")[1].split(".")[0]) for f in files]

sort_indx = np.argsort(ids)

for i, indx in enumerate(sort_indx):
    curr_file = files[indx]
    new_file = f"../output/prw/grid_id0/run_0/visuals/img_{i}.png"
    cmd = f"mv {curr_file} {new_file}"
    os.system(cmd)

print("Making a movie...")
cmd = "ffmpeg -i ../output/prw/grid_id0/run_0/visuals/img_%d.png \
    -b:v 4M -s 500x500 -pix_fmt yuv420p prw.mp4"
os.system(cmd)
print("Done!")
