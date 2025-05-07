#!/usr/bin/env python

import os
import glob
import pandas as pd

def main():
    # Starting directory – adjust if needed
    base_dir = os.path.expanduser('~/data/CIVET/simulation')
    os.chdir(base_dir)

    # Find all scenario directories
    scenario_dirs = [
        d for d in os.listdir('.')
        if d.startswith('SCENARIO_') and os.path.isdir(d)
    ]

    results = []  # Will collect dictionaries for each (scenario, subfolder, test)

    for scenario_dir in scenario_dirs:
        # Find all civet_results.csv files under each scenario folder (recursive)
        civet_files = glob.glob(os.path.join(scenario_dir, '**', 'civet_results.csv'),
                                recursive=True)
        for fpath in civet_files:
            # Load the civet_results.csv
            df = pd.read_csv(fpath)
            # df has 3 columns: [ mutation_name, p_value, test_name ]
            # Adjust if your column positions/names differ
            mutation_col = df.columns[0]
            pval_col     = df.columns[1]
            test_col     = df.columns[2]

            # Group by test name
            for test_name, subdf in df.groupby(test_col):
                pvals = subdf[pval_col]
                n_p_lt_005 = (pvals < 0.05).sum()
                n_p_lt_1   = (pvals < 1.0).sum()
                n_p_lt_001 = (pvals < 0.01).sum()

                results.append({
                    'scenario':  scenario_dir,
                    'subfolder': os.path.dirname(fpath),
                    'test':      test_name,
                    'n_p_lt_0.05': n_p_lt_005,
                    'n_p_lt_1':   n_p_lt_1,
                    'n_p_lt_0.01': n_p_lt_001
                })

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv('summary_civet_results.csv', index=False)

    print("Done! Summary written to summary_civet_results.csv")

if __name__ == "__main__":
    main()
