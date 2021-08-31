from typing import List, Dict, ClassVar

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import PreTrainedTokenizer


class ClassificationDataset(Dataset):
    """
    An implementation of Pytorch Dataset for document classification that accepts plain text data with each line formatted as:
     LABEL <TAB> TEXT
     For example:
     POSITIVE <TAB> This example is a positive one.
     NEGATIVE <TAB> This example is a negative one.
    """
    TEXT_F: ClassVar[str] = 'text'
    ATTENTION_MASK_F: ClassVar[str] = 'attention_mask'
    LABEL_F: ClassVar[str] = 'label'

    def __init__(self, instances: List[Dict[str, str]], labels_vocabulary: Dict[str, int], tokenizer: PreTrainedTokenizer, max_len: int):
        self.instances = instances
        self.labels_vocabulary = labels_vocabulary
        self.tokenizer = tokenizer
        self.max_len = max_len

    @classmethod
    def load_from_file(cls, path, tokenizer: PreTrainedTokenizer, labels_vocabulary: Dict[str, int], max_len: int):
        instances = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip() != '':  # avoid blank lines, if any
                    label, text = line.split('\t', maxsplit=1)
                    # filter the instances with labels that we do not have in our vocabulary (just for sanity check)
                    if label not in labels_vocabulary:
                        print(f'WARNING: label {label} found in the dataset, but it is not in the provided labels vocabulary')
                        continue
                    instances.append({cls.LABEL_F: label, cls.TEXT_F: text})

        return ClassificationDataset(instances=instances, labels_vocabulary=labels_vocabulary, tokenizer=tokenizer, max_len=max_len)

    @classmethod
    def load_for_inference(cls, documents: List[str], tokenizer: PreTrainedTokenizer, max_len: int):
        instances = [{cls.LABEL_F: 'N/A', cls.TEXT_F: text} for text in documents]
        dummy_labels_vocab = {'N/A': 0}  # we need some labels vocab to prevent the logic from breaking, but we are not going to use it
        return ClassificationDataset(instances=instances, labels_vocabulary=dummy_labels_vocab, tokenizer=tokenizer, max_len=max_len)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> T_co:
        # just pick the index-th instance and process it using the implemented logic
        return self._process_instance(self.instances[index])

    def _process_instance(self, instance: Dict[str, str]):
        """ This method process a raw instance to convert it into tensors suitable for the DeepLearning model"""
        # Extract the relevant fields from the instance, the text and the classification label
        text = instance[self.TEXT_F]
        label = instance[self.LABEL_F]

        # The text must be tokenized with the appropriate tokenizer (the one that belongs to the same Transformer model that we will use for training)
        tokens = self.tokenizer.tokenize(text)
        # To match BERT convention (at least for the pre-trained models we are going to use) we need the [CLS] and [SEP] special tokens
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        # Now we need to ensure that the number of tokens matches the max number we want
        if len(tokens) > self.max_len:
            # if too many tokens, crop the sequence to the max len (ensure that the very last token is still the [SEP] special token)
            tokens = tokens[self.max_len - 1] + self.tokenizer.sep_token
        elif len(tokens) < self.max_len:
            # if too few tokens, we add [PAD] tokens to pad the sequence to the desired length
            tokens += [self.tokenizer.pad_token] * (self.max_len - len(tokens))

        # two more things with the tokens,  encode them to tokenizer vocabulary indices and turn them into torch tensors,
        token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
        # and then calculate the attention mask,
        # attention mask is a mask that indicates with parts of the input contain information and which ones not (i.e. the padding positions)
        # What we are doing is to create a tensor mask with 0's in PAD positions, and 1's in non-PAD positions
        attention_mask = torch.ne(token_ids, self.tokenizer.pad_token_id).long()

        # Now encode the labels (this is easier)
        # Just obtain their corresponding index according to the vocabulary we have injected, and turn them into tensors
        # NOTE: this will break if we find an unknown label, we do not have any fallback or default label
        label_id = torch.tensor(self.labels_vocabulary[label])

        return {self.TEXT_F: token_ids,
                self.ATTENTION_MASK_F: attention_mask,
                self.LABEL_F: label_id}
