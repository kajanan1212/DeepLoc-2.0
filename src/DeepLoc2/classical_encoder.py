import numpy as np

from typing import List
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def get_fixed_size_encoding(
    encoded_data: List[float], fixed_size: int = None
) -> List[float]:
    if fixed_size is None:
        return encoded_data
    if len(encoded_data) < fixed_size:
        encoded_data = np.concatenate(
            (encoded_data, np.zeros(fixed_size - len(encoded_data)))
        )
    elif len(encoded_data) > fixed_size:
        encoded_data = encoded_data[:fixed_size]
    return encoded_data


class ClassicEncoder:
    protein_alphabet = [
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
    ]

    def __init__(self):
        self.vocabulary_array_col = np.array(self.protein_alphabet).reshape(-1, 1)

        self.onehot_encoder = OneHotEncoder()
        self.onehot_encoder.fit(self.vocabulary_array_col)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.protein_alphabet)

        self.bow_encoder = CountVectorizer(analyzer="char")
        self.bow_encoder.fit(self.protein_alphabet)

        self.tf_idf_encoder = TfidfVectorizer(analyzer="char")
        self.tf_idf_encoder.fit(self.protein_alphabet)

        self.ngram_encoder = CountVectorizer(analyzer="char", ngram_range=(1, 2))
        self.ngram_encoder.fit(self.protein_alphabet)

    def get_one_hot_encoding(self, protein_sequence: str, length: int = 1024):
        data = np.array([list(protein_sequence)]).reshape(-1, 1)
        value = self.onehot_encoder.transform(data).toarray().reshape([1, -1])[0]
        return get_fixed_size_encoding(value, length)

    def get_label_encoding(self, protein_sequence: str, length: int = 1024):
        value = self.label_encoder.transform(list(protein_sequence))
        return get_fixed_size_encoding(value, length)

    def get_binary_encoding(self, protein_sequence: str, length: int = 1024):
        # Create a dictionary mapping each amino acid to its binary mask
        binary_masks = {
            aa: [0] * len(self.protein_alphabet) for aa in self.protein_alphabet
        }
        for i, aa in enumerate(self.protein_alphabet):
            binary_masks[aa][i] = 1

        value = []
        for aa in protein_sequence:
            value.extend(binary_masks[aa])

        return value

    def get_bow_encoding(self, protein_sequence: str):
        value = self.bow_encoder.transform([protein_sequence]).toarray().flatten()
        return get_fixed_size_encoding(value)

    def get_tfidf_encoding(self, protein_sequence: str, length: int = 1024):
        value = self.tf_idf_encoder.transform([protein_sequence]).toarray().flatten()
        return get_fixed_size_encoding(value)

    def get_ngram_encoding(self, protein_sequence: str, length: int = 1024):
        self.ngram_encoder.fit_transform(self.protein_alphabet)
        return self.ngram_encoder.transform([protein_sequence]).toarray()[0]