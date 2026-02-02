import math
import random
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from bertopic import BERTopic
from tqdm import tqdm
import json

import nltk
import spacy
import re
import string
from typing import List, Iterable
from nltk.corpus import stopwords

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from jsonformer import Jsonformer


EMAIL_REGEX = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
HTML_TAG_REGEX = re.compile(r"<.*?>")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")


def remove_stopwords(text: str) -> str:
    """
    Remove stopwords from a text.
    """
    stop_words = set(stopwords.words("english"))
    tokens = text.split()
    filtered = [t for t in tokens if t.lower() not in stop_words]
    return " ".join(filtered)


def remove_emails_and_html(text: str) -> str:
    """
    Remove email addresses and HTML tags from a text.
    """
    no_emails = EMAIL_REGEX.sub(" ", text)
    no_html = HTML_TAG_REGEX.sub(" ", no_emails)
    no_html = re.sub(r"\s+", " ", no_html).strip()
    return no_html


def remove_punctuation(text: str) -> str:
    """
    Remove all punctuation characters from a text.
    """
    return text.translate(PUNCT_TABLE)


def filter_short_docs(docs: List[str], min_tokens: int = 5) -> List[str]:
    """
    Keep only documents that have at least `min_tokens` tokens.
    """
    idxs = []
    for i, doc in enumerate(docs):
        n_tokens = len(doc.split())
        if n_tokens >= min_tokens:
            idxs.append(i)
    return np.array(idxs)


def lemmatize_text(text: str) -> str:
    """
    Lemmatize text using spaCy.
    Returns a space-joined string of lemmas.
    """
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_space]
    return " ".join(lemmas)


def preprocess_text(text: str) -> str:
    text = remove_emails_and_html(text)
    text = remove_punctuation(text)
    text = text.lower()
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text
