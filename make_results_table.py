import os
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import pandas as pd
import tqdm
import torch

def read_result_from_file(fpath, fail_silently=True):
    """
    Reads a single floating-point result from a file
    """
    if not fpath.exists() and fail_silently:
        return None
    
    with open(fpath, 'r') as infile:
        return float(infile.read().strip())

def get_results_data(basedir='./', pattern='output*', ignore_cols_with_same_vals=True):
    """
    Get the results data in folders matching `pattern` in `basedir`
    """
    dirs = [(p.name, p) for p in Path(basedir).glob(pattern) if p.is_dir()]

    experiments = pd.DataFrame()
    column_names = []
    for run_name, run_dir in tqdm.tqdm(dirs):
        model_path = Path(run_dir, 'torch_model.pt')
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
        except FileNotFoundError:
            continue
        model_time = (
            datetime.fromtimestamp(model_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
        )
        run_data = {
            'run_name': run_name,
            'git_hash': checkpoint['git_hash'],
            'date': model_time,

            # hyperparameters
            **checkpoint['options'].__dict__, # works if we switch to argparse as well

            # results
            'saved_at_epoch': checkpoint['epoch'],

            'accuracy_train': read_result_from_file(Path(run_dir, 'accuracy.train.txt')),
            'accuracy_dev': read_result_from_file(Path(run_dir, 'accuracy.dev.txt')),
            'accuracy_dev_from_chkpt': checkpoint['dev_metrics']['accuracy'],
            'accuracy_test': read_result_from_file(Path(run_dir, 'accuracy.test.txt')),
            
            'perplexity_dev': read_result_from_file(Path(run_dir, 'perplexity.dev.txt')),
            'perplexity_test': read_result_from_file(Path(run_dir, 'perplexity.test.txt')),

            'maw': read_result_from_file(Path(run_dir, 'maw.txt'))
        }        
        
        # keep longest set of cols for data ordering (python>=3.6 keeps dict key order)
        if len(run_data.keys()) > len(column_names):
            column_names = list(run_data.keys())
        
        experiments = experiments.append(run_data, ignore_index=True)
    
    # reorder columns 
    experiments = experiments[column_names] 
    if ignore_cols_with_same_vals:
        # remove any columns where the values have not been altered run-to-run
        # see https://stackoverflow.com/a/39658662/5712749
        nunique_vals = experiments.apply(pd.Series.nunique)
        cols_to_drop = nunique_vals[nunique_vals <= 1].index
        experiments = experiments.drop(cols_to_drop, axis=1)

    return experiments.sort_values(by=['date'])

if __name__ == '__main__':
    results_data = get_results_data()
    results_data.to_csv('results.csv', index=False)

    # below code stores as google sheet
    import gspread_pandas

    GOOGLE_AUTH_KEY_DIRECTORY = "~/scholar-experiments-key/" # google_secret.json should be in this directory
    RESULTS_SHEET_NAME = "scholar-experiments"
    SHARE_WITH_USERS = ["hoyle@umd.edu", "pgoel1@cs.umd.edu"]

    config = gspread_pandas.conf.get_config(GOOGLE_AUTH_KEY_DIRECTORY)
    client = gspread_pandas.Client(config=config)
    sheet = gspread_pandas.Spread(
        spread=RESULTS_SHEET_NAME,
        client=client,
        create_spread=True,
    )
    prev_results = sheet.sheet_to_df(sheet='results')
    results_data = (
        prev_results.append(results_data).drop_duplicates(subset='run_name', keep='last')
    )
    sheet.df_to_sheet(results_data, index=False, replace=True, sheet='results')
    
    # sharing
    access_sheet = client.open(RESULTS_SHEET_NAME)
    authenticated_users = [p["emailAddress"] for p in access_sheet.list_permissions()]
    for user in SHARE_WITH_USERS:
        if user not in authenticated_users:
            access_sheet.share(user, perm_type="user", role="writer")

    print("Access results at:")
    print(sheet.url)

