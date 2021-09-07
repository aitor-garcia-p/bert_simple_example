"""
A simple example of classification using BERT.
The code is just a toy to show the nuances of a training and the data preparation for BERT.
It is by no means optimal or something to be blindly imitated.
Once the key ideas are understood, there are better ways (even already implemented in other software packages) to arrange and run such task.
In any case, it should work for the purpose of this example.
"""
import json
import os
import random
from typing import List

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, PreTrainedModel
from transformers.models.bert import BertForSequenceClassification, BertTokenizer, BertConfig

from bert_examples.classification_dataset import ClassificationDataset


class SimpleBERTForClassificationTrainer:
    """
    The class that encapsulates the logic for training
    """

    def __init__(self, checkpoints_folder: str, bert_model_name_or_path: str,
                 train_data: str, dev_data: str, labels_inventory: List[str], max_len=100, cache_dir: str = None, random_seed: int = 12345):
        """
        Instantiates the class to run a training with the given data files.
        The data loading is handled by the corresponding Dataset class.
        :param checkpoints_folder: path to folder to store the model checkpoints
        :param bert_model_name_or_path: pre-trained BERT model name (if a HF hosted model) or path (in your system) to load
        :param train_data: path to training data (in the specified LABEL<TAB>TEXT format)
        :param dev_data: path to development data for model validation (in the specified LABEL<TAB>TEXT format)
        :param labels_inventory: the list of valid labels
        :param max_len: the maximum length of each sequence, measured in BERT-tokens
        :param cache_dir: path to a folder to cache assets downloaded from HF, leave empty to use the default location
        :param random_seed: random seed to control randomness and allow reproducing the same results (set to -1 to avoid setting a fixed value)
        """
        # Fix all random sources to a certain configurable seed value, so the results can be reproduced
        if random_seed >= 0:
            set_all_random_seeds_to(seed=random_seed)
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
        """
        Instantiate the optimizer that will update the model parameters.
        :param lr: learning rate
        :return: the optimizer
        """
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

    def train(self, run_name: str, num_epochs: int, learning_rate: float, batch_size: int, cuda_device_num: int = -1):
        """
        Main train method that runs the training
        :param run_name: a name for the run, that will be used when storing the model checkpoints
        :param num_epochs: number of epochs to run the training
        :param learning_rate: the learning rate of the training
        :param batch_size: number of instances per batch
        :param cuda_device_num: number of cuda device to use (-1 means CPU)
        :return:
        """

        # Check the provided cuda device num
        cuda_device = f'cuda:{cuda_device_num}' if cuda_device_num >= 0 else 'cpu'
        # Move the model to the selected device
        self.model.to(cuda_device)

        # Instantiate optimizer
        optimizer = self._instantiate_optimizer(lr=learning_rate)
        clip_grad_norm = 1.0
        # Instantiate dataloaders that will serve the batches from the different datasets
        train_dataloader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=True)
        dev_dataloader = DataLoader(dataset=self.dev_set, batch_size=batch_size, shuffle=False)

        # Initialize the best score to zero
        # Each time we hit a better score we will store a new version (checkpoint) of the model
        best_dev_score = 0.0

        # Start the training for the indicated number of epochs
        for epoch in range(num_epochs):
            # Model needs to be put in "train" mode for the "training" part (it enables elements such as dropouts and batch-normalization, etc.)
            self.model.train()
            epoch_progress_bar = tqdm(train_dataloader)
            for batch_number, batch in enumerate(epoch_progress_bar):
                # A single training step
                # Get the elements from this batch (tokens, attention mask, and labels, as they were encoded in the Dataset class)
                token_ids = batch[ClassificationDataset.TEXT_F].to(cuda_device)
                attention_mask = batch[ClassificationDataset.ATTENTION_MASK_F].to(cuda_device)
                label_ids = batch[ClassificationDataset.LABEL_F].to(cuda_device)
                # Run the forward pass of the model, and get the loss and the logits
                outputs = self.model(input_ids=token_ids, attention_mask=attention_mask, labels=label_ids)
                loss = outputs.loss
                logits = outputs.logits
                # Run the backward on the loss, so the gradients get calculated (optionally clip the gradients)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
                # use the optimizer to update the model parameters (after that the gradients must be reset to zero)
                optimizer.step()
                optimizer.zero_grad()
                # Optional: evaluate the outputs to check how the train progresses, and report it to the training progress bar in console
                train_micro_f1 = self._evaluate(logits, label_ids)
                epoch_progress_bar.postfix = f'Training MicroF1={train_micro_f1:.3f}; Loss:{loss:.3f}'
                epoch_progress_bar.desc = f'Epoch {epoch}/{num_epochs}'

            # Now, after each full epoch, evaluate the current state of the model using the dev data
            # Model needs to be set to "eval" mode to disable dropouts and other elements that only apply for training
            self.model.eval()
            # ATTENTION: Autograd engine can be disabled during evaluation since we do not need gradients (this way we save memory and time)
            with torch.no_grad():
                all_dev_scores = []
                all_dev_losses = []
                # We go through the whole development dataset, accumulating the evaluation metrics
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
            # average the metrics from all the individual batches
            final_dev_micro_f1 = sum(all_dev_scores) / len(all_dev_scores)
            final_dev_loss = sum(all_dev_losses) / len(all_dev_losses)

            # Only if the resulting score is better that the best score so far, we store a new version of the model (a new checkpoint)
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
        """
        Uses HuggingFace utilities to store a model checkpoint, together with its tokenizer and the labels vocabulary
        :param run_name: a name to refer to this run when saving it
        :param current_epoch: the training epoch in which we are saving the checkpoint
        :param current_score: the validation (dev) score corresponding to this checkpoint
        :param current_dev_loss: the validation (dev) loss corresponding to this checkpoint
        :return:
        """
        checkpoint_name = f'{run_name}_epoch{current_epoch}-microF1_{current_score:0.3f}-loss_{current_dev_loss:0.3f}'
        checkpoint_output_path = os.path.join(self.checkpoints_folder, checkpoint_name)
        os.makedirs(os.path.dirname(checkpoint_output_path), exist_ok=True)
        self.model.save_pretrained(save_directory=checkpoint_output_path)
        self.tokenizer.save_pretrained(checkpoint_output_path)
        labels_vocabulary_path = os.path.join(checkpoint_output_path, 'labels_vocab.json')
        with open(labels_vocabulary_path, 'w', encoding='utf-8') as f:
            json.dump(self.labels_vocabulary, f, indent=4)


def set_all_random_seeds_to(seed: int = 12345):
    """
    An auxiliary method that sets all the random sources to a fixed seed, so the results become the same when the same seed is used.
    This help to reproduce the results for experiments.
    :param seed: the seed value to fix
    :return:
    """
    print(f'Setting all random seeds (Python/Numpy/Pytorch) to: {seed}')
    # random seeds for all the involved modules (pytorch, numpy, and general random)
    # if we were using GPU, additional stuff would be needed here
    # see: https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # also NOTE:
        # 'Deterministic mode can have a performance impact, depending on your model.
        # This means that due to the deterministic nature of the model, the processing speed
        # (i.e. processed batch items per second) can be lower than when the model is non-deterministic.'
    # ####################


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
                                                 cache_dir=CACHE_DIR, random_seed=12345)
    trainer.train(run_name='test8', num_epochs=10, learning_rate=2E-5, batch_size=8, cuda_device_num=CUDA_DEVICE_NUM)
