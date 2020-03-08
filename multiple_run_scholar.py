import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from run_scholar import main

if __name__ == "__main__":
    run_parser = argparse.ArgumentParser()
    run_parser.add_argument("--runs", default=1, type=int)
    run_parser.add_argument("--global-seed", type=int)
    run_parser.add_argument("--store-all", default=False, action='store_true')
    run_parser.add_argument("--dev-folds", type=int)
    run_args, additional_args = run_parser.parse_known_args()

    outdir_parser = argparse.ArgumentParser()
    outdir_parser.add_argument("-o")
    outdir_args, _ = outdir_parser.parse_known_args(additional_args)
    
    np.random.seed(run_args.global_seed)

    for run in range(run_args.runs):
        print(f"On run {run}")
        if run_args.dev_folds:
            fold = run % run_args.dev_folds
            if fold == 0:
                seed = np.random.randint(0, 1000000) # renew seed
            additional_args += ["--dev-fold", f"{fold}", "--dev-folds", f"{run_args.dev_folds}"] 
        else:
            fold = None
            seed = np.random.randint(0, 1000000)
        additional_args += ['--seed', f'{seed}']
        
        # run scholar
        main(additional_args)

        # load model and store metrics
        try:
            checkpoint = torch.load(Path(outdir_args.o, "torch_model.pt"))
        except EOFError:
            print("Got EOFError, restarting run")
            continue

        m = checkpoint['dev_metrics']
        ppl, npmi, acc = m['perplexity'], m['npmi'], m['accuracy']
        results = pd.DataFrame({
               'seed': seed,
               'fold': fold, 
               'perplexity_value': float(ppl['value']),
               'perplexity_epoch': int(ppl.get('epoch', 0)),

               'npmi_value': float(npmi['value']),
               'npmi_epoch': int(npmi.get('epoch', 0)),
               
               'accuracy_value': float(acc['value']),
               'accuracy_epoch': int(acc.get('epoch', 0)),
            },
            index=[run],
        )

        results.to_csv(
            Path(outdir_args.o, "dev_metrics.csv"),
            mode='a',
            header=run==0,
        )
        if run_args.store_all:
            seed_path = Path(outdir_args.o, str(seed))
            if not seed_path.exists():
                seed_path.mkdir()
            for fpath in Path(outdir_args.o).glob("*"):
                if fpath.name not in ['torch_model.pt', 'dev_metrics.csv'] and fpath.is_file():
                    shutil.copyfile(fpath, Path(seed_path, fpath.name))
