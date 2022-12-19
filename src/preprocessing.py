import pandas as pd
import numpy as np


def prepare_data_train(path):
    data = pd.read_csv('./instance/'+path, parse_dates=["date"])
    data["year"] = data["date"].apply(lambda x: x.year)
    data["month"] = data["date"].apply(lambda x: x.month)
    data["day"] = data["date"].apply(lambda x: x.day)
    data = data.drop(columns="date")
    data = data.drop(columns="id")
    target = data["price"]
    data = data.drop(columns="price")
    return data.to_numpy(), target.to_numpy()

def prepare_data_test(path):
    data = pd.read_csv('./instance/'+path, parse_dates=["date"])
    data["year"] = data["date"].apply(lambda x: x.year)
    data["month"] = data["date"].apply(lambda x: x.month)
    data["day"] = data["date"].apply(lambda x: x.day)
    data = data.drop(columns="date")
    data = data.drop(columns="id")
    data = data.drop(columns="price")
    return data.to_numpy()
