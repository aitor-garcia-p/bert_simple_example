"""
The logic to load a trained model and use it to make predictions
"""
import json
import os
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

from bert_examples.classification_dataset import ClassificationDataset


class SimpleBertClassificationInferencer:
    """
    The class that encapsulates the loading of a trained model and using it to make predictions over new documents.
    """

    def __init__(self, model_path: str, max_len: int):
        """
        Initialize the inferencer loading a model
        :param model_path: the path to the model checkpoint to be loaded (as it was stored by the trainer)
        :param max_len: the max length to encode the input examples when going to make predictions
        """
        # Load the model from the pointed location
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        # Load the corresponding tokenizer, from the same location (HuggingFace helpers do most of the work for us)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        # Load the labels vocabulary from the same location (we need it to interpret the label indices outputted by the model)
        labels_vocabulary_path = os.path.join(model_path, 'labels_vocab.json')
        with open(labels_vocabulary_path, 'r', encoding='utf-8') as f:
            self.labels_vocabulary = json.load(f)
            self.reverse_labels_vocabulary = {y: x for x, y in self.labels_vocabulary.items()}
        self.max_len = max_len

    @torch.no_grad()
    def predict(self, documents: List[str], batch_size: int = 8) -> [List[Dict[str, str]]]:
        """
        The method that predicts the labels for new unseen documents using the loaded model
        :param documents: A list of strings (each one assumed to be a document for which we want to predict a label)
        :param batch_size: the size of batch when processing many documents
        :return: a list of dictionaries containing pairs of a document and its predicted label
        """
        # Load the documents into a dataset (so we can reuse the same logic to tokenize/encode the documents as during the model trainer)
        dataset = ClassificationDataset.load_for_inference(documents, tokenizer=self.tokenizer, max_len=self.max_len)
        # Instantiate a DataLoader using the dataset, so it serves the batches efficiently
        # (this is not mandatory, we could simply iterate over the dataset example by example, but it would be less efficient)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_predicted_labels = []
        for batch in tqdm(dataloader):
            # We iterate over the batches, get the content (like in the training) and run the model forward-pass
            # Note that we have not labels this time (we are trying to predict them)
            token_ids = batch[ClassificationDataset.TEXT_F]
            attention_mask = batch[ClassificationDataset.ATTENTION_MASK_F]

            output = self.model(input_ids=token_ids, attention_mask=attention_mask)
            # Since we are not passing any gold-labels to the model, there is no loss to retrieve, only the logits
            logits = output.logits  # a tensor of shape BxC (B=batch_size, C=num_labels)
            # Just pick the index of the most likely label (according to the model)
            predicted_indices = torch.max(logits, dim=1)[1].tolist()
            # We should get a list of label indices, one per document in the batch
            # We need to decode the label indices into their readable label name using the labels vocabulary
            predicted_labels: List[str] = [self.reverse_labels_vocabulary[idx] for idx in predicted_indices]
            all_predicted_labels += predicted_labels

        # return all the labels paired to their document (or any other kind of structure you may want)
        return [{'label': all_predicted_labels[i], 'document': document} for i, document in enumerate(documents)]


if __name__ == '__main__':
    MODEL_PATH = 'D:/DATA/WORK/ibermatica_bert_formacion/checkpoints/test1_epoch1-microF1_0.823-loss_0.590'
    MAX_LEN = 50
    inferencer = SimpleBertClassificationInferencer(model_path=MODEL_PATH, max_len=MAX_LEN)

    documents = ['the team scored a goal in the first half',
                 'new microchips are being developed by IBM',
                 'the stock market is raising again',
                 # Spanish examples (for zero-shot transfer learning demonstration)
                 'el nuevo lanzador ha anotado muchos puntos',
                 'las nuevas tarjetas gráficas son más rápidas',
                 'la Bolsa ha experimentado ganancias']

    results = inferencer.predict(documents=documents, batch_size=8)

    for res in results:
        print(res)
