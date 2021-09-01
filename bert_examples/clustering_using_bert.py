"""
A simple example of using BERT encoding of documents to apply some clustering algorithm on top of it
"""
from collections import defaultdict
from typing import List, Tuple

from scipy import spatial
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, BertTokenizer
from transformers.models.bert.modeling_bert import BertModel

from bert_examples.classification_dataset import ClassificationDataset


class BertDocumentEncoder:
    """
    A class that encapsulates the logic to load and use a pre-trained BERT model to encode documents (using MEAN_POOLING)
    """

    def __init__(self, bert_model_name_or_path: str, cache_dir: str = None):
        self.model: PreTrainedModel = BertModel.from_pretrained(bert_model_name_or_path, cache_dir=cache_dir)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name_or_path, cache_dir=cache_dir)

    @torch.no_grad()
    def encode_documents(self, documents: List[str], max_len: int, batch_size: int = 8, cuda_device_num: int = -1):
        """
        Encode all the documents, generating their document-embeddings (vectors of size H, where H=768 for a BERT-base model)
        :param documents: the list of documents to encode (can be all your document base, to precompute it)
        :param max_len: max length of document to use when encoding
        :param batch_size: batch size
        :param cuda_device_num: number of cuda device to use (-1 means CPU)
        :return:
        """
        cuda_device = f'cuda:{cuda_device_num}' if cuda_device_num >= 0 else 'cpu'
        self.model.to(cuda_device)
        dataset = ClassificationDataset.load_for_inference(documents=documents, tokenizer=self.tokenizer, max_len=max_len)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        document_embeddings = []
        for batch in tqdm(dataloader, disable=len(documents) <= 1):
            token_ids = batch[ClassificationDataset.TEXT_F].to(cuda_device)
            attention_mask = batch[ClassificationDataset.ATTENTION_MASK_F].to(cuda_device)

            output = self.model(input_ids=token_ids, attention_mask=attention_mask)
            sequence_output = output.last_hidden_state  # this should be a BxSxH  (B=batch_size, S=seq_len, H=hidden_size)
            # We are going to implement an "MEAN_POOLING"
            # (i.e. pick the average of tokens contextual-word-embeddings to obtain a document-embedding)
            sequence_wise_sums = torch.sum(sequence_output, dim=1)  # this should lead to a BxH shaped tensor
            valid_positions = torch.sum(attention_mask, dim=1).unsqueeze(1)  # this should lead to a Bx1 shaped tensor
            averaged_embeddings = sequence_wise_sums / valid_positions  # this should also be a BxH shaped tensor
            document_embeddings += [averaged_embeddings]
        document_embeddings = torch.cat(document_embeddings, dim=0).cpu().numpy()
        return document_embeddings


class BERTSimilarityCalculator:
    """
    A class that encapsulates the calculation of similarity among documents (based on cosine-distance) using BERT document embeddings
    """

    def __init__(self, all_documents: List[str], documents_encoder: BertDocumentEncoder, max_len: int, batch_size: int = 8,
                 cuda_device_num: int = -1):
        self.documents_encoder = documents_encoder
        self.documents = all_documents
        self.document_embeddings = documents_encoder.encode_documents(documents=all_documents, max_len=max_len, batch_size=batch_size,
                                                                      cuda_device_num=cuda_device_num)

        self.max_len = max_len
        self.batch_size = batch_size

    def rank_similar_documents(self, single_document, num_top_docs: int = 5):
        """
        Compares the given document to all the other documents that were supplied in the creation of the class
        :param single_document: The single document for which obtain a rank of other similar documents
        :param num_top_docs: the number of most-similar candidates to return
        :return: a sorted list of tuples with the most-similar documents to the given one (and their similarity score)
        """
        encoded_question = self.documents_encoder.encode_documents([single_document], max_len=self.max_len, batch_size=self.batch_size)

        distances: List[float] = spatial.distance.cdist(encoded_question, self.document_embeddings, "cosine")[0]
        results: List[Tuple[int, float]] = list(zip(range(len(distances)), distances))
        results = sorted(results, key=lambda x: x[1])

        top_similar_documents: List[Tuple[str, float]] = []
        for i in range(num_top_docs):
            selected_top_document = self.documents[results[i][0]]
            top_similar_documents.append((selected_top_document, 1 - results[i][1]))

        return top_similar_documents


class BERTClusterer:
    """
    A class that encapsulates the logic to use BERT encodings over documents, and run a K-means clustering algorithm of top of them.
    """

    def __init__(self, all_documents: List[str], documents_encoder: BertDocumentEncoder, max_len: int, batch_size: int = 8,
                 cuda_device_num: int = -1):
        self.documents_encoder = documents_encoder
        self.documents = all_documents
        self.document_embeddings = documents_encoder.encode_documents(documents=all_documents, max_len=max_len, batch_size=batch_size,
                                                                      cuda_device_num=cuda_device_num)
        self.documents_by_cluster = defaultdict(list)
        self.kmeans = None
        self.max_len = max_len
        self.batch_size = batch_size
        self.cuda_device_num = cuda_device_num

    def compute_clusters(self, num_clusters: int, random_seed: int = 42):
        self.kmeans = KMeans(init='k-means++', n_clusters=num_clusters, random_state=random_seed).fit(self.document_embeddings)
        cluster_labels = self.kmeans.labels_
        self.documents_by_cluster = defaultdict(list)  # reset the clusters dict
        for doc_num, document in enumerate(self.documents):
            self.documents_by_cluster[cluster_labels[doc_num]].append(document)

    def print_cluster_members(self, max_members: int = 10):
        for cluster, members in sorted(self.documents_by_cluster.items(), key=lambda x: x[0]):
            print(f'Cluster {cluster} members (only showing top {max_members} from {len(members)}):')
            for member in members[:max_members]:
                print(f'\t - {member}')
            print('=============')

    def predict_unseen_documents(self, documents: List[str]):
        if self.kmeans is None:
            raise Exception('KMeans has not been computed yet, run compute_clusters method first')
        doc_embeddings = self.documents_encoder.encode_documents(documents=documents, max_len=self.max_len, batch_size=self.batch_size,
                                                                 cuda_device_num=self.cuda_device_num)
        predicted_clusters = self.kmeans.predict(documents=doc_embeddings)
        return [(doc, predicted_clusters[i]) for i, doc in enumerate(documents)]


if __name__ == '__main__':
    # Let's try all this
    BERT_MODEL_NAME_OR_PATH = 'bert-base-multilingual-cased'
    # Change this path to one that suits your system
    CACHE_DIR = 'D:/DATA/cache'

    BERT_DOCUMENTS_ENCODER = BertDocumentEncoder(bert_model_name_or_path=BERT_MODEL_NAME_OR_PATH, cache_dir=CACHE_DIR)
    MAX_LEN = 50
    BATCH_SIZE = 8
    CUDA_DEVICE_NUM = 0

    example_documents = ['the team has played a good match',
                         'the corporation is obtaining profits',
                         'cloud computing is gaining more relevance',
                         'una nueva remontada del equipo titular',
                         'las pérdidas amenazan con bancarrota',
                         'la computación cuántica revoluciona la informática']

    TEST_DATA = '../example_data/classif_example_dataset_TEST.txt'
    with open(TEST_DATA, 'r', encoding='utf-8') as f:
        ALL_DOCUMENTS = [line.split('\t', maxsplit=1)[1].strip() for line in f.readlines()]

    ################################
    # SIMILARITY CALCULATION EXAMPLE
    ################################
    sim_calculator = BERTSimilarityCalculator(all_documents=ALL_DOCUMENTS, documents_encoder=BERT_DOCUMENTS_ENCODER, max_len=MAX_LEN,
                                              batch_size=BATCH_SIZE, cuda_device_num=CUDA_DEVICE_NUM)

    NUM_TOP_DOCS = 10
    for doc in example_documents:
        print(f'Calculating similar docs to:\n >>> {doc}')
        top_docs = sim_calculator.rank_similar_documents(single_document=doc, num_top_docs=10)
        for i, (text, score) in enumerate(top_docs):
            print(f'\t\t{i + 1} - score: {score:.3f}, TEXT: {text}')
        print('================')

    #################################
    # CLUSTERING EXAMPLE
    #################################
    # clusterer = BERTClusterer(all_documents=ALL_DOCUMENTS, documents_encoder=BERT_DOCUMENTS_ENCODER, max_len=MAX_LEN, batch_size=BATCH_SIZE,
    #                           cuda_device_num=CUDA_DEVICE_NUM)
    #
    # clusterer.compute_clusters(num_clusters=8)
    # clusterer.print_cluster_members(max_members=10)
