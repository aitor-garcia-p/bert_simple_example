"""
The logic to load a trained model and use it to make predictions
"""
import json
import os
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

from bert_examples.classification_dataset import ClassificationDataset


class SimpleBertClassificationInferencer:

    def __init__(self, model_path: str, max_len: int):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        labels_vocabulary_path = os.path.join(model_path, 'labels_vocab.json')
        with open(labels_vocabulary_path, 'r', encoding='utf-8') as f:
            self.labels_vocabulary = json.load(f)
            self.reverse_labels_vocabulary = {y: x for x, y in self.labels_vocabulary.items()}
        self.max_len = max_len

    @torch.no_grad()
    def predict(self, documents: List[str], batch_size: int = 8):
        dataset = ClassificationDataset.load_for_inference(documents, tokenizer=self.tokenizer, max_len=self.max_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_predicted_labels = []
        for batch in tqdm(dataloader):
            token_ids = batch[ClassificationDataset.TEXT_F]
            attention_mask = batch[ClassificationDataset.ATTENTION_MASK_F]

            output = self.model(input_ids=token_ids, attention_mask=attention_mask)
            logits = output.logits

            predicted_indices = torch.max(logits, dim=1)[1].tolist()
            predicted_labels = [self.reverse_labels_vocabulary[idx] for idx in predicted_indices]
            all_predicted_labels += predicted_labels

        return [{'label': all_predicted_labels[i], 'document': document} for i, document in enumerate(documents)]


if __name__ == '__main__':
    MODEL_PATH = 'D:/DATA/WORK/ibermatica_bert_formacion/checkpoints/test1_epoch1-microF1_0.823-loss_0.590'
    MAX_LEN = 50
    inferencer = SimpleBertClassificationInferencer(model_path=MODEL_PATH, max_len=MAX_LEN)

    documents = ['the team scored a goal in the first half', 'new microchips are being developed by IBM', 'the stock market is raising again',
                 'el nuevo lanzador ha anotado muchos puntos', 'las nuevas tarjetas gráficas son más rápidas', 'la Bolsa ha experimentado ganancias']

    results = inferencer.predict(documents=documents, batch_size=8)

    for res in results:
        print(res)
