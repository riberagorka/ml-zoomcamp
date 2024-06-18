#!/usr/bin/env python
# coding: utf-8


import pickle
import pandas as pd
import argparse




def read_data(filename):
    df = pd.read_parquet("yellow_tripdata_2023-05.parquet")
    categorical = ['PULocationID', 'DOLocationID']

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def save_results():

    parser = argparse.ArgumentParser("Introduce the year and month you want:")
    parser.add_argument("--year", type=int, help="Year to be analized")
    parser.add_argument("--month", type=int, help="Month to be anaized")
    args = parser.parse_args()

    print(args.year, args.month)

    year = args.year
    month = args.month
    
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{str(year)}-0{str(month)}.parquet')
    df['ride_id'] = f'{int(year):04d}/{int(month):02d}_' + df.index.astype('str')


    with open("model.bin", "rb") as f_in:
        dv, model = pickle.load(f_in)
        
    categorical = ['PULocationID', 'DOLocationID']

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print(y_pred.mean())


    df_result = pd.DataFrame({
        "ride_id": df["ride_id"],
        "predictions": y_pred,
    })
    df_result


    #output_file = "result_file.parquet"
    #df_result.to_parquet(
    #    output_file,
    #    engine='pyarrow',
    #    compression=None,
    #    index=False
    #)

if __name__=="__main__":
    save_results()
