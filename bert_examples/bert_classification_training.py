"""
A simple example of classification using BERT.
The code is just a toy to show the nuances of a training and the data preparation for BERT.
It is by no means optimal or something to be imitated.
Once the key ideas are understood, there are better ways (even already implemented in other software packages) to arrange and run such task.
"""
import json
import os
from typing import Dict, List

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, PreTrainedModel, BertConfig

from bert_examples.classification_dataset import ClassificationDataset


class SimpleBERTForClassificationTrainer:

    def __init__(self, checkpoints_folder: str, bert_model_name_or_path: str,
                 train_data: str, dev_data: str, labels_inventory: List[str], max_len=100, cache_dir: str = None):
        """ Instantiate the class to run a training with the given data files.
        The data must be a tab-separated text with LABEL <TAB> TEXT format per line.
        """
        self.checkpoints_folder = checkpoints_folder
        self.labels_vocabulary = {label: i for i, label in enumerate(sorted(set(labels_inventory)))}
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name_or_path, cache_dir=cache_dir)
        bert_config = BertConfig.from_pretrained(bert_model_name_or_path, cache_dir=cache_dir)
        bert_config.num_labels = len(self.labels_vocabulary)
        self.model: PreTrainedModel = BertForSequenceClassification.from_pretrained(bert_model_name_or_path, config=bert_config, cache_dir=cache_dir)

        # create a labels vocabulary out from the provided labels inventory (checking that there are no repeated values)
        self.train_set = ClassificationDataset.load_from_file(path=train_data,
                                                              tokenizer=self.tokenizer, labels_vocabulary=self.labels_vocabulary, max_len=max_len)
        self.dev_set = ClassificationDataset.load_from_file(path=dev_data,
                                                            tokenizer=self.tokenizer, labels_vocabulary=self.labels_vocabulary, max_len=max_len)

    def _instantiate_optimizer(self, lr: float):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)
        return optimizer

    def train(self, run_name: str, num_epochs: int, learning_rate: float, batch_size: int, cuda_device_num: int = -1, clip_grad_norm=1.0):
        cuda_device = f'cuda:{cuda_device_num}' if cuda_device_num >= 0 else 'cpu'
        self.model.to(cuda_device)

        optimizer = self._instantiate_optimizer(lr=learning_rate)
        train_dataloader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=True)
        dev_dataloader = DataLoader(dataset=self.dev_set, batch_size=batch_size, shuffle=False)

        best_dev_score = 0.0

        for epoch in range(num_epochs):
            self.model.train()
            epoch_progress_bar = tqdm(train_dataloader)
            for batch_number, batch in enumerate(epoch_progress_bar):
                token_ids = batch[ClassificationDataset.TEXT_F].to(cuda_device)
                attention_mask = batch[ClassificationDataset.ATTENTION_MASK_F].to(cuda_device)
                label_ids = batch[ClassificationDataset.LABEL_F].to(cuda_device)
                outputs = self.model(input_ids=token_ids, attention_mask=attention_mask, labels=label_ids)
                loss = outputs.loss
                logits = outputs.logits
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                train_micro_f1 = self._evaluate(logits, label_ids)
                epoch_progress_bar.postfix = f'Training MicroF1={train_micro_f1:.3f}; Loss:{loss:.3f}'
                epoch_progress_bar.desc = f'Epoch {epoch}/{num_epochs}'

            # Now, after each full epoch, evaluate the current state of the model using the dev data
            self.model.eval()
            with torch.no_grad():
                all_dev_scores = []
                all_dev_losses = []
                for batch_number, batch in enumerate(tqdm(dev_dataloader)):
                    token_ids = batch[ClassificationDataset.TEXT_F].to(cuda_device)
                    attention_mask = batch[ClassificationDataset.ATTENTION_MASK_F].to(cuda_device)
                    label_ids = batch[ClassificationDataset.LABEL_F].to(cuda_device)
                    outputs = self.model(input_ids=token_ids, attention_mask=attention_mask, labels=label_ids)
                    logits = outputs.logits
                    loss = outputs.loss
                    all_dev_losses.append(loss.item())
                    dev_micro_f1 = self._evaluate(logits, label_ids)
                    all_dev_scores.append(dev_micro_f1)
            # average them
            final_dev_micro_f1 = sum(all_dev_scores) / len(all_dev_scores)
            final_dev_loss = sum(all_dev_losses) / len(all_dev_losses)

            if final_dev_micro_f1 > best_dev_score:
                best_dev_score = final_dev_micro_f1
                # store the checkpoint, since it is better than the best one so far
                self._store_checkpoint(run_name=run_name, current_epoch=epoch, current_score=final_dev_micro_f1, current_dev_loss=final_dev_loss)

    @classmethod
    def _evaluate(cls, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Logits outputted by the model and the gold-labels to compare to.
        :param logits: BxC shaped tensor (B=batch_size, C=Num_classes)
        :param labels: B shaped tensor (B=batch_size)
        :return:
        """
        predictions = torch.max(logits, dim=1)[1].squeeze().cpu()
        labels = labels.squeeze().cpu()
        micro_f1 = f1_score(y_true=labels, y_pred=predictions, average='micro')
        return micro_f1

    def _store_checkpoint(self, run_name: str, current_epoch: int, current_score: float, current_dev_loss: float):
        checkpoint_name = f'{run_name}_epoch{current_epoch}-microF1_{current_score:0.3f}-loss_{current_dev_loss:0.3f}'
        checkpoint_output_path = os.path.join(self.checkpoints_folder, checkpoint_name)
        os.makedirs(os.path.dirname(checkpoint_output_path), exist_ok=True)
        self.model.save_pretrained(save_directory=checkpoint_output_path)
        self.tokenizer.save_pretrained(checkpoint_output_path)
        labels_vocabulary_path = os.path.join(checkpoint_output_path, 'labels_vocab.json')
        with open(labels_vocabulary_path, 'w', encoding='utf-8') as f:
            json.dump(self.labels_vocabulary, f, indent=4)


if __name__ == '__main__':
    # Let's try it
    CHECKPOINTS_FOLDER = 'D:/DATA/WORK/ibermatica_bert_formacion/checkpoints'
    TRAIN_DATA = '../example_data/classif_example_dataset_TRAIN.txt'
    DEV_DATA = '../example_data/classif_example_dataset_TEST.txt'
    BERT_MODEL_NAME_OR_PATH = 'bert-base-multilingual-cased'
    # LABEL_DICT = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
    LABELS_INVENTORY = ['World', 'Sports', 'Business', 'Sci/Tech']
    MAX_LEN = 50
    CUDA_DEVICE_NUM = 0

    # cache_dir stores pre-trained models and tokenizers when they are downloaded, to avoid downloading them everytime
    # Replace with whichever other suitable folder, or set to None to use a default location
    CACHE_DIR = 'D:/DATA/cache'
    trainer = SimpleBERTForClassificationTrainer(checkpoints_folder=CHECKPOINTS_FOLDER, bert_model_name_or_path=BERT_MODEL_NAME_OR_PATH,
                                                 train_data=TRAIN_DATA, dev_data=DEV_DATA, labels_inventory=LABELS_INVENTORY, max_len=MAX_LEN,
                                                 cache_dir=CACHE_DIR)
    trainer.train(run_name='test1', num_epochs=10, learning_rate=2E-5, batch_size=8, cuda_device_num=CUDA_DEVICE_NUM)
