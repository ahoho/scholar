import os
import argparse
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import pandas as pd
import tqdm
import torch

from compute_npmi import compute_npmi_at_n
import file_handling as fh

def read_result_from_file(fpath, fail_silently=True):
    """
    Reads a single floating-point result from a file
    """
    if not fpath.exists() and fail_silently:
        return None
    
    with open(fpath, 'r') as infile:
        return float(infile.read().strip())

def get_results_data(
    basedir,
    pattern,
    ignore_cols_with_same_vals=True,
    coherence_reference_dir="/fs/clip-political/scholar/congress_votes_dwnom"
):
    """
    Get the results data in folders matching `pattern` in `basedir`
    """
    dirs = [(p.name, p) for p in Path(basedir).glob(pattern) if p.is_dir()]

    ref_vocab = fh.read_json(Path(coherence_reference_dir, "train.vocab.json"))
    ref_counts = fh.load_sparse(Path(coherence_reference_dir, "test.npz")).tocsc()

    experiments = pd.DataFrame()
    column_names = []
    for run_name, run_dir in tqdm.tqdm(dirs):

        model_path = Path(run_dir, 'torch_model.pt')
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
        except FileNotFoundError:
            continue

        
        npmi_internal = None
        try:
            topics = fh.read_text(Path(run_dir, "topic.txt"))
        except FileNotFoundError:
            print(f"topics.txt not found for {run_name}. Will not calculate npmi")
            pass
        else:
            npmi_internal = compute_npmi_at_n(
                topics=topics,
                ref_vocab=ref_vocab,
                ref_counts=ref_counts,
                n=10, # could change?
                silent=True,
            )

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-directory', default='./')
    parser.add_argument('--file-pattern', default='output*')
    parser.add_argument('--ouput-fpath', '-o', default='results.csv')

    # google sheet information
    parser.add_argument('--do-not-save-to-sheets', action='store_false')
    parser.add_argument('--google-auth-key-dir', default="~/scholar-experiments-key/")
    parser.add_argument('--google-sheets-name', default='scholar-experiments')
    parser.add_argument('--share-with-users', nargs='+', default=[])

    args = parser.parse_args()

    results_data = get_results_data(basedir=args.base_dir, pattern=args.file_pattern)
    results_data.to_csv(args.output_fpath, index=False)

    # below code stores as google sheet
    if args.do_not_save_to_sheets:
        exit()
    
    import gspread_pandas

    config = gspread_pandas.conf.get_config(args.google_auth_key_dir)
    client = gspread_pandas.Client(config=config)
    sheet = gspread_pandas.Spread(
        spread=args.google_sheets_name,
        client=client,
        create_spread=True,
    )
    prev_results = sheet.sheet_to_df(sheet='results')
    results_data = (
        prev_results.append(results_data).drop_duplicates(subset='run_name', keep='last')
    )
    sheet.df_to_sheet(results_data, index=False, replace=True, sheet='results')
    
    # sharing
    access_sheet = client.open(args.google_sheets_name)
    authenticated_users = [p["emailAddress"] for p in access_sheet.list_permissions()]
    for user in args.share_with_users:
        if user not in authenticated_users:
            access_sheet.share(user, perm_type="user", role="writer")

    print("Access results at:")
    print(sheet.url)

