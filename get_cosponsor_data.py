import json
import argparse

import pandas as pd 
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_jsonlist_fpath")
    parser.add_argument("test_jsonlist_fpath")
    parser.add_argument("output_fpath")
    args = parser.parse_args()

    cosponsor_data = []
    cosponsor_keys = ['id', 'r_perc', 'd_perc', 'i_perc', 'total_sponsors']
    print("loading train data")
    with open(args.train_jsonlist_fpath, "r") as infile:
        for line in tqdm(infile):
            line_data = json.loads(line)
            cosponsor_data.append({k: line_data[k] for k in cosponsor_keys})

    print("loading test data")
    with open(args.test_jsonlist_fpath, "r") as infile:
        for line in tqdm(infile):
            line_data = json.loads(line)
            cosponsor_data.append({k: line_data[k] for k in cosponsor_keys})
    
    cosponsor_data = pd.DataFrame.from_records(cosponsor_data).set_index("id").fillna(0)
    cosponsor_data.to_csv(args.output_fpath)