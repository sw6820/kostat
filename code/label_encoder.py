from sklearn.preprocessing import LabelEncoder
import pandas as pd

def get_label_encoder():
    ctd = pd.read_csv("data/class_table_chunk_kor.csv")
    ctd.loc[:,"code"] = ctd.loc[:,"code"].astype(str)

    label_encoder = LabelEncoder()
    label_encoder.fit(ctd["code"])
    return label_encoder