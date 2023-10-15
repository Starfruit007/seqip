import pandas as pd
import numpy as np


def duplicate_columns(input_file, output_file):
    df = pd.read_csv(input_file)
    columns = list(df.columns) + list(df.columns)[1:]
    duplicated_df_values = np.vstack((df.values[:,0:1].T, np.tile(df.values[:,1:],2).T)).T
    
    df_new = pd.DataFrame(duplicated_df_values,columns= columns)

    df_new.to_csv(output_file, index=False)

input_file = './datasets/ETT-small/ETTh1.csv'
output_file = './datasets/ETT-small/ETTh1_duplicated.csv'
duplicate_columns(input_file, output_file)
