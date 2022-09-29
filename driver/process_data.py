import os
from analysis import analysis


# data to process
for pol_type in ["sva", "ffcr"]:

    # path to collision results
    proj_root = os.path.dirname(__file__)
    path_to_data = os.path.join(proj_root, f"output/{pol_type}")

    # all .csv files to read -- 462 * 96
    files = analysis._all_files(path_to_data)

    # all collision outcomes
    binary_outcomes = analysis.outcomes(files)

    # store processed data
    path_to_res = os.path.join(proj_root, f"processed/{pol_type}")
    if not os.path.exists(path_to_res):
        os.makedirs(path_to_res, exist_ok=True)

    binary_outcomes.to_parquet(path_to_res)
