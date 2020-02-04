import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_scholar import main

if __name__ == "__main__":
    run_parser = argparse.ArgumentParser()
    run_parser.add_argument("--runs", default=1, type=int)
    run_args, additional_args = run_parser.parse_known_args()

    outdir_parser = argparse.ArgumentParser()
    outdir_parser.add_argument("-o")
    outdir_args, _ = outdir_parser.parse_known_args(additional_args)
    
    for run in range(run_args.runs):
        print(f"On run {run}")
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
