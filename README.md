
## Example of BERT for Classification, Similarity and Clustering

This is a very basic (toy) code, only for educational purposes, that makes use of BERT for:

  - Training a basic document classifier
  - Using the trained model to classify new documents
  - Using BERT for document-similarity ranking (e.g. semantic search)
  - Using BERT to encode documents and group them using some clustering algorithm

The code is mostly complete and working. The training contains all the minimal parts that
are required (data-loading, model update, evaluation, model saving, etc.).
However, there are a lot of other details that are not covered here, because despite being
useful, they are not required for a simple model to be trained (learning-rate scheduling,
early-stopping, mixed-precision training, distributed training,
model logging/reporting, etc.).

### Used data

The data used for this little example has been borrowed from:
https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv

Only a small subset of label-title pairs has been used.
Even with this toy subset of data the resulting classifier works surprisingly well, showing
the power of BERT and Transformers in general.

### How to use this code

You need Python3.7+.
It is recommended to create a fresh Python virtual environment before installing the
dependencies.
You also need to install Pytorch 1.7+ (with CUDA support if you have a CUDA capable device
and you plan to use it). It is advisable to use a CUDA device for training, because the
process works 10-15x faster. However, since the data is small, you can also use CPU if you
don't mind waiting several minutes for completion.

Then install the dependencies listed in requirements.txt

```bash
pip install -r requirements.txt
```

You should be ready to go. Have a look at the "main" sections of each Python file, and if
necessary, adjust the paths to your system (there are a few hard-coded paths).