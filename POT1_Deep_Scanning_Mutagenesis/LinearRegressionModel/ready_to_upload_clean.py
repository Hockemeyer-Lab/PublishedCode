import pandas as pd
import numpy as np
import re
import arviz as az
import matplotlib.pyplot as plt
import pymc as pm

# Define paths to your files relative to the script's directory
file1 = 'TIF_sg13VUS_2nd.csv'
file2 = 'TIF_sg13VUS.csv'
genotype_file = 'genotype_sg13.csv'
phenotype_output_file = '10(pheno_sg13).csv'
genotype_output_file = 'geno_sg13.csv'
filtered_phenotype_file = 'filtered_phenotype_sg13.csv'
filtered_genotype_file = 'filtered_genotype_sg13.csv'
output_summary_file = 'output_summary.csv'

def load_and_concatenate_files(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    merged_df = pd.concat([df1, df2], axis=0, join='outer', ignore_index=True, sort=False)
    merged_df.dropna(axis=1, how='all', inplace=True)
    merged_df = merged_df.loc[:, merged_df.notna().sum() >= 20]
    return merged_df

def sort_columns_alphanumerically(df):
    def alphanumeric_key(s):
        return [int(text) if text.isdigit() else text for text in re.split('([0-9]+)', s)]
    return df.reindex(sorted(df.columns, key=alphanumeric_key), axis=1)

def sort_numeric_columns_descending(df):
    numeric_columns = df.select_dtypes(include=['number']).columns
    for col in numeric_columns:
        df[col] = df[col].sort_values(ascending=False).values
    return df

def save_to_csv(df, filename):
    df.to_csv(filename, index=False)

def process_phenotype_data(input_file, output_file):
    df = pd.read_csv(input_file)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    non_nan_counts = df.count()
    filtered_df = df.loc[:, non_nan_counts > 20]
    thresholded_df = filtered_df.applymap(lambda x: np.nan if pd.isnull(x) else (1 if x >= 10 else 0))
    phenotype_frequency = thresholded_df.sum() / thresholded_df.notna().sum()
    phenotype_frequency_df = phenotype_frequency.reset_index()
    phenotype_frequency_df.columns = ['Well', 'Phenotype Frequency']
    phenotype_frequency_df.to_csv(output_file, index=False)

def process_genotype_data(input_file, output_file):
    df = pd.read_csv(input_file)
    def process_low_frequency_rows(df):
        low_freq_rows = df[(df.iloc[:, 1:] < 0.01).all(axis=1)]
        low_freq_sum = low_freq_rows.iloc[:, 1:].sum(axis=0)
        low_freq_row = pd.Series(['Low Frequency'] + low_freq_sum.tolist(), index=df.columns)
        low_freq_df = pd.DataFrame([low_freq_row])
        df = pd.concat([df, low_freq_df], ignore_index=True)
        df = df.drop(low_freq_rows.index)
        return df
    result_df = process_low_frequency_rows(df)
    result_df.to_csv(output_file, index=False)

def filter_and_reorder_files(phenotype_file, genotype_file, phenotype_output_file, genotype_output_file):
    phenotype_df = pd.read_csv(phenotype_file)
    genotype_df = pd.read_csv(genotype_file)
    phenotype_df = phenotype_df.set_index('Well').T.reset_index(drop=True)
    phenotype_wells = set(phenotype_df.columns)
    genotype_wells = set(genotype_df.columns[1:])
    common_wells = phenotype_wells.intersection(genotype_wells)
    common_wells_list = list(common_wells)
    filtered_phenotype_df = phenotype_df[common_wells_list]
    filtered_genotype_df = genotype_df[['mutation_name'] + common_wells_list]
    well_order = [f"{letter}{number:02d}" for letter in "ABCDEFGH" for number in range(1, 13)]
    final_well_order = [well for well in well_order if well in common_wells]
    filtered_phenotype_df = filtered_phenotype_df[final_well_order]
    filtered_genotype_df = filtered_genotype_df[['mutation_name'] + final_well_order]
    filtered_phenotype_df.to_csv(phenotype_output_file, index=False)
    filtered_genotype_df.to_csv(genotype_output_file, index=False)

def perform_bayesian_linear_regression(phenotype_file, genotype_file, output_summary_file):
    phenotype_df = pd.read_csv(phenotype_file, header=None).iloc[1:]
    genotype_df = pd.read_csv(genotype_file, header=None, index_col=0).iloc[1:]
    phenotype_df = phenotype_df.apply(pd.to_numeric, errors='coerce')
    genotype_df = genotype_df.apply(pd.to_numeric, errors='coerce')
    mutation_names = genotype_df.index
    y = phenotype_df.iloc[0].values
    X = genotype_df.transpose().values
    with pm.Model() as model:
        constant = pm.Normal('constant', mu=0, sigma=10)
        beta = pm.Beta('beta', alpha=1, beta=9, shape=(X.shape[1],))
        mu = constant + pm.math.dot(X, beta)
        sigma = pm.HalfNormal('sigma', sigma=1)
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)
        trace = pm.sample(500, return_inferencedata=True)
    summary_df = az.summary(trace)
    summary_df = summary_df.rename(index={'constant': 'constant'})
    for i, name in enumerate(mutation_names):
        summary_df = summary_df.rename(index={f'beta[{i}]': name})
    summary_df.to_csv(output_summary_file)

if __name__ == "__main__":
    # Step 1: Concatenate and clean data files
    merged_df = load_and_concatenate_files(file1, file2)
    merged_df = sort_columns_alphanumerically(merged_df)
    merged_df = sort_numeric_columns_descending(merged_df)
    save_to_csv(merged_df, 'merged_sg13_sorted.csv')
    
    # Step 2: Process phenotype data
    process_phenotype_data('merged_sg13_sorted.csv', phenotype_output_file)
    
    # Step 3: Process genotype data
    process_genotype_data(genotype_file, genotype_output_file)
    
    # Step 4: Filter and reorder data files
    filter_and_reorder_files(phenotype_output_file, genotype_output_file, filtered_phenotype_file, filtered_genotype_file)
    
    # Step 5: Perform Bayesian linear regression
    perform_bayesian_linear_regression(filtered_phenotype_file, filtered_genotype_file, output_summary_file)

    print("All steps completed successfully.")
