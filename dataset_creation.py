import numpy as np
import pandas as pd 


def Datacret(frac_train,frac_valid):
    df = pd.read_csv('emails.csv')
    # print(df)
    # print(df.describe())

    print(df)
    df = df.drop("Email No.", axis='columns')
    cor_matrix = df.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool_))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.7)]
    print(to_drop)
    df = df.drop(to_drop, axis=1)

    df_train = df.sample(frac=frac_train, random_state=1)
    df = df.drop(df_train.index)
    df_valid = df.sample(frac=frac_valid/(1-frac_train), random_state=1)
    df_test = df.drop(df_valid.index)


    df_train.to_csv('train.csv')
    df_valid.to_csv('valid.csv')
    df_test.to_csv('test.csv')
    print(df_train,df_valid,df_test)