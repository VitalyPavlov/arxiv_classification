import random
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from bertopic import BERTopic
from collections import Counter
import matplotlib.pyplot as plt

from data.data_preprocessing import preprocess_text
from models.llm_model import LLM
from metrics.concentration import concentration_score


class Manager:
    def __init__(self, version: str):
        self.cfg = version
        self.bert_model = BERTopic.load(f"MaartenGr/{self.cfg.bert.model_name}")
        self.llm_model = LLM(self.cfg.llm)

    def data_preparation(self, df: pd.DataFrame) -> pd.DataFrame:
        df["abstract_post"] = df["abstract"].apply(lambda x: preprocess_text(x))
        return df

    def get_data(self, chunksize: int = 1_000) -> pd.DataFrame:
        data = pd.read_json(self.cfg.data.data_path, lines=True, chunksize=chunksize)
        df = next(data)
        df = self.data_preparation(df)
        return df

    def run_bertopic(self, df: pd.DataFrame) -> pd.DataFrame:
        assert "abstract_post" in df.columns

        topic, _ = self.bert_model.transform(df.abstract_post.values)

        df["pred_cluster_id"] = topic
        df["pred_cluster_name"] = df["pred_cluster_id"].apply(lambda x: self.bert_model.topic_labels_[x])

        return df

    def get_representative_texts(self, df: pd.DataFrame) -> dict:
        topic = df.pred_cluster_id.values
        topic_words = self.bert_model.get_topics()
        representative_texts = dict()

        for topic_id, _ in tqdm(topic_words.items()):
            if sum(topic == topic_id) > 0:
                representative_texts[topic_id] = random.choice(df.abstract.values[topic == topic_id])
            else:
                representative_texts[topic_id] = ""
        return representative_texts

    def run_llm(self, df: pd.DataFrame) -> None:
        topic_words = self.bert_model.get_topics()
        representative_texts = self.get_representative_texts(df)

        prediction = dict()
        for topic_id, text in tqdm(representative_texts.items()):
            top_words = [x for x in topic_words[topic_id]]

            result = self.llm_model.run_model(top_words, text)
            prediction[topic_id] = result

        with open(self.cfg.llm.output_file, "w") as f:
            json.dump(prediction, f)

    def get_label_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        assert "pred_cluster_id" in df.columns

        with open(self.cfg.llm.output_file, "r") as f:
            llm_prediction = json.load(f)

        df["pred_label"] = df["pred_cluster_id"].map(llm_prediction)
        return df

    def run_validation(self, df: pd.DataFrame, save=False) -> list:
        assert "categories" in df.columns

        df["high_level_categories"] = df.categories.apply(lambda x: set([xx.split(".")[0] for xx in x.split(" ")]))
        df["high_level_categories"] = df["high_level_categories"].apply(
            lambda x: [xx.split("-", 1)[1] if "-" in xx else xx for xx in x]
        )

        topic = df.pred_cluster_id.values

        metrics = []
        for label in tqdm(np.unique(topic)):
            mask = topic == label
            categories_count = Counter([xx for x in df.loc[mask, "high_level_categories"].values for xx in x])
            metrics.append(concentration_score(categories_count))

        if save:
            plt.hist(metrics)
            plt.xlabel("Metrics values per class")
            plt.ylabel("Count")
            plt.savefig(self.cfg.data.valid_path)

        return metrics
