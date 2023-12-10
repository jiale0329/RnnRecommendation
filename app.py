import collections
import json
from lib2to3.pytree import convert
import os
import random
import re
import pprint
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

from ast import Num
from flask import Flask, jsonify, request
from typing import Dict, Text
from absl import app
from absl import flags
from absl import logging
from numpy import product
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

#Azure Storage account information
account_name = 'rnnrecommendation'
account_key = 'vmzPMn1mw4GYZb6r236X6YKXLt47USCrn09l80Fcpp1TnBf35W4CNVSeY3bu0jmtxW4qUbl0oyZS+ASt4QaVHg=='
container_name = 'twinklewebsystem'
connection_string = f"DefaultEndpointsProtocol=https;AccountName=rnnrecommendation;AccountKey=vmzPMn1mw4GYZb6r236X6YKXLt47USCrn09l80Fcpp1TnBf35W4CNVSeY3bu0jmtxW4qUbl0oyZS+ASt4QaVHg==;EndpointSuffix=core.windows.net"

PURCHASE_HISTORY_FILE_NAME = "purchaseHistory.dat"
PRODUCT_FILE_NAME = "product.dat"
PURCHASE_HISTORY_DATA_COLUMNS = ["UserID", "ProductID", "NumberOfPurchase", "Timestamp"]
PRODUCT_DATA_COLUMNS = ["ProductID", "Name", "ProductType"]
OUTPUT_TRAINING_DATA_FILENAME = "train_productlens_1m.tfrecord"
OUTPUT_TESTING_DATA_FILENAME = "test_productlens_1m.tfrecord"
OUTPUT_PRODUCT_VOCAB_FILENAME = "product_vocab.json"
OUTPUT_PRODUCT_TYPE_VOCAB_FILENAME = "product_type_vocab.txt"
OUTPUT_PRODUCT_NAME_UNIGRAM_VOCAB_FILENAME = "product_name_unigram_vocab.txt"
OUTPUT_PRODUCT_NAME_BIGRAM_VOCAB_FILENAME = "product_name_bigram_vocab.txt"
PAD_PRODUCT_ID = 0
PAD_NUMBEROFPURCHASE = 0.0
UNKNOWN_STR = "UNK"
VOCAB_PRODUCT_ID_INDEX = 0
VOCAB_COUNT_INDEX = 3

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

class ProductInfo(
    collections.namedtuple(
        "ProductInfo", ["product_id", "timestamp", "numberOfPurchase", "name", "productType"])):
  """Data holder of basic information of a product."""
  __slots__ = ()

  def __new__(cls,
              product_id=PAD_PRODUCT_ID,
              timestamp=0,
              numberOfPurchase=PAD_NUMBEROFPURCHASE,
              name="",
              productType=""):
    return super(ProductInfo, cls).__new__(cls, product_id, timestamp, numberOfPurchase,
                                         name, productType)

def read_data(data_directory, min_rating=None):
  ##Read purchasehistory.dat from cloud
  blob_name = PURCHASE_HISTORY_FILE_NAME
  blob_client = container_client.get_blob_client(blob_name)
  blob_data = blob_client.download_blob()
  file_contents = blob_data.readall().decode('utf-8')  
  
  local_file_path = os.path.join(data_directory, PURCHASE_HISTORY_FILE_NAME)
  with open(local_file_path, 'w', encoding='utf-8') as local_file:
      local_file.write(file_contents)
  purchasehistory_df = pd.read_csv(
      os.path.join(data_directory, PURCHASE_HISTORY_FILE_NAME),
      sep="::",
      names=PURCHASE_HISTORY_DATA_COLUMNS,
      encoding="unicode_escape")  # May contain unicode. Need to escape.
  purchasehistory_df["Timestamp"] = purchasehistory_df["Timestamp"].apply(int)
  if min_rating is not None:
    purchasehistory_df = purchasehistory_df[purchasehistory_df["NumberOfPurchase"] >= min_rating]
  ##Read product.dat from cloud
  blob_name = 'product.dat'
  blob_client = container_client.get_blob_client(blob_name)
  blob_data = blob_client.download_blob()
  file_contents = blob_data.readall().decode('utf-8')  
  
  local_file_path = os.path.join(data_directory, 'product.dat')
  with open(local_file_path, 'w', encoding='utf-8') as local_file:
      local_file.write(file_contents)
      
  product_df = pd.read_csv(
      os.path.join(data_directory, PRODUCT_FILE_NAME),
      sep="::",
      names=PRODUCT_DATA_COLUMNS,
      encoding="unicode_escape")  # May contain unicode. Need to escape.
  return purchasehistory_df, product_df  

def convert_to_timelines(purchasehistory_df):
  """Convert purchase history data to user."""
  timelines = collections.defaultdict(list)
  product_counts = collections.Counter()
  for user_id, product_id, numberOfPurchase, timestamp in purchasehistory_df.values:
    timelines[user_id].append(
        ProductInfo(product_id=product_id, timestamp=int(timestamp), numberOfPurchase=numberOfPurchase))
    product_counts[product_id] += 1
  # Sort per-user timeline by timestamp
  for (user_id, context) in timelines.items():
    context.sort(key=lambda x: x.timestamp)
    timelines[user_id] = context
  return timelines, product_counts

def generate_product_dict(product_df):
  """Generates product dictionary from products dataframe."""
  productdict = {
      product_id: ProductInfo(product_id=product_id, name=name, productType=productType)
      for product_id, name, productType in product_df.values
  }
  productdict[0] = ProductInfo()
  return productdict

def generate_product_types(productdict, products):
  product_types = []
  for product in products:
    if not productdict[product.product_id].productType:
      continue
    productTypes = [
        tf.compat.as_bytes(productType)
        for productType in productdict[product.product_id].productType.split("|")
    ]
    product_types.extend(productTypes)

  return product_types

def _pad_or_truncate_product_feature(feature, max_len, pad_value):
  feature.extend([pad_value for _ in range(max_len - len(feature))])
  return feature[:max_len]

def generate_examples_from_single_timeline(timeline,
                                           productdict,
                                           max_context_len=100,
                                           max_context_product_type_len=320):
  """Generate TF examples from a single user timeline.

  Generate TF examples from a single user timeline. Timeline with length less
  than minimum timeline length will be skipped. And if context user history
  length is shorter than max_context_len, features will be padded with default
  values.

  Args:
    timeline: The timeline to generate TF examples from.
    movies_dict: Dictionary of all MovieInfos.
    max_context_len: The maximum length of the context. If the context history
      length is less than max_context_length, features will be padded with
      default values.
    max_context_movie_genre_len: The length of movie genre feature.

  Returns:
    examples: Generated examples from this single timeline.
  """
  examples = []
  for label_idx in range(1, len(timeline)):
    start_idx = max(0, label_idx - max_context_len)
    context = timeline[start_idx:label_idx]
    # Pad context with out-of-vocab movie id 0.
    while len(context) < max_context_len:
      context.append(ProductInfo())
    label_product_id = int(timeline[label_idx].product_id)
    context_product_id = [int(product.product_id) for product in context]
    context_product_numberOfPurchase = [product.numberOfPurchase for product in context]
    context_product_type = generate_product_types(productdict, context)
    context_product_type = _pad_or_truncate_product_feature(
        context_product_type, max_context_product_type_len,
        tf.compat.as_bytes(UNKNOWN_STR))
    feature = {
        "context_product_id":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_product_id)),
        "context_product_numberOfPurchase":
            tf.train.Feature(
                float_list=tf.train.FloatList(value=context_product_numberOfPurchase)),
        "context_product_type":
            tf.train.Feature(
                bytes_list=tf.train.BytesList(value=context_product_type)),
        "label_product_id":
            tf.train.Feature(
                int64_list=tf.train.Int64List(value=[label_product_id]))
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    examples.append(tf_example)

  return examples

def generate_examples_from_timelines(timelines,
                                     productdf,
                                     min_timeline_len=3,
                                     max_context_len=100,
                                     max_context_product_type_len=320,
                                     train_data_fraction=0.9,
                                     random_seed=None,
                                     shuffle=True):
  """Convert user timelines to tf examples.

  Convert user timelines to tf examples by adding all possible context-label
  pairs in the examples pool.

  Args:
    timelines: The user timelines to process.
    movies_df: The dataframe of all movies.
    min_timeline_len: The minimum length of timeline. If the timeline length is
      less than min_timeline_len, empty examples list will be returned.
    max_context_len: The maximum length of the context. If the context history
      length is less than max_context_length, features will be padded with
      default values.
    max_context_movie_genre_len: The length of movie genre feature.
    train_data_fraction: Fraction of training data.
    random_seed: Seed for randomization.
    shuffle: Whether to shuffle the examples before splitting train and test
      data.

  Returns:
    train_examples: TF example list for training.
    test_examples: TF example list for testing.
  """
  examples = []
  productdict = generate_product_dict(productdf)
  progress_bar = tf.keras.utils.Progbar(len(timelines))
  for timeline in timelines.values():
    if len(timeline) < min_timeline_len:
      progress_bar.add(1)
      continue
    single_timeline_examples = generate_examples_from_single_timeline(
        timeline=timeline,
        productdict=productdict,
        max_context_len=max_context_len,
        max_context_product_type_len=max_context_product_type_len)
    examples.extend(single_timeline_examples)
    progress_bar.add(1)
  # Split the examples into train, test sets.
  if shuffle:
    random.seed(random_seed)
    random.shuffle(examples)
  last_train_index = round(len(examples) * train_data_fraction)

  train_examples = examples[:last_train_index]
  test_examples = examples[last_train_index:]
  return train_examples, test_examples

def generate_product_feature_vocabs(productdf, product_counts):
  """Generate vocabularies for movie features.

  Generate vocabularies for movie features (movie_id, genre, year), sorted by
  usage count. Vocab id 0 will be reserved for default padding value.

  Args:
    movies_df: Dataframe for movies.
    movie_counts: Counts that each movie is rated.

  Returns:
    movie_id_vocab: List of all movie ids paired with movie usage count, and
      sorted by counts.
    movie_genre_vocab: List of all movie genres, sorted by genre usage counts.
    movie_year_vocab: List of all movie years, sorted by year usage counts.
  """
  product_vocab = []
  product_type_counter = collections.Counter()
  for product_id, name, productTypes in productdf.values:
    count = product_counts.get(product_id) or 0
    product_vocab.append([product_id, name, productTypes, count])
    for productType in productTypes.split("|"):
      product_type_counter[productType] += 1

  product_vocab.sort(key=lambda x: x[VOCAB_COUNT_INDEX], reverse=True)  # by count
  product_type_vocab = [UNKNOWN_STR
                      ] + [x for x, _ in product_type_counter.most_common()]

  return (product_vocab, product_type_vocab)

def write_tfrecords(tf_examples, filename):
  """Writes tf examples to tfrecord file, and returns the count."""
  with tf.io.TFRecordWriter(filename) as file_writer:
    length = len(tf_examples)
    progress_bar = tf.keras.utils.Progbar(length)
    for example in tf_examples:
      file_writer.write(example.SerializeToString())
      progress_bar.add(1)
    return length

def write_vocab_json(vocab, filename):
  """Write generated movie vocabulary to specified file."""
  with open(filename, "w", encoding="utf-8") as jsonfile:
    json.dump(vocab, jsonfile, indent=2)


def write_vocab_txt(vocab, filename):
  with open(filename, "w", encoding="utf-8") as f:
    for item in vocab:
      f.write(str(item) + "\n")

def generate_datasets(extracted_data_dir,
                      output_dir,
                      min_timeline_length,
                      max_context_length,
                      max_context_product_type_length,
                      min_rating=None,
                      build_vocabs=True,
                      train_data_fraction=0.9,
                      train_filename=OUTPUT_TRAINING_DATA_FILENAME,
                      test_filename=OUTPUT_TESTING_DATA_FILENAME,
                      vocab_filename=OUTPUT_PRODUCT_VOCAB_FILENAME,
                      vocab_product_type_filename=OUTPUT_PRODUCT_TYPE_VOCAB_FILENAME):
  """Generates train and test datasets as TFRecord, and returns stats."""
  logging.info("Reading data to dataframes.")
  purchasehistory_df, product_df = read_data(extracted_data_dir, min_rating=min_rating)
  logging.info("Generating movie rating user timelines.")
  timelines, product_counts = convert_to_timelines(purchasehistory_df)
  logging.info("Generating train and test examples.")
  train_examples, test_examples = generate_examples_from_timelines(
      timelines=timelines,
      productdf=product_df,
      min_timeline_len=min_timeline_length,
      max_context_len=max_context_length,
      max_context_product_type_len=max_context_product_type_length,
      train_data_fraction=train_data_fraction)

  if not tf.io.gfile.exists(output_dir):
    tf.io.gfile.makedirs(output_dir)
  logging.info("Writing generated training examples.")
  train_file = os.path.join(output_dir, train_filename)
  train_size = write_tfrecords(tf_examples=train_examples, filename=train_file)
  logging.info("Writing generated testing examples.")
  test_file = os.path.join(output_dir, test_filename)
  test_size = write_tfrecords(tf_examples=test_examples, filename=test_file)
  stats = {
      "train_size": train_size,
      "test_size": test_size,
      "train_file": train_file,
      "test_file": test_file,
  }

  if build_vocabs:
    (product_vocab, product_type_vocab) = (
        generate_product_feature_vocabs(
            productdf=product_df, product_counts=product_counts))
    vocab_file = os.path.join(output_dir, vocab_filename)
    write_vocab_json(product_vocab, filename=vocab_file)
    stats.update({
        "vocab_size": len(product_vocab),
        "vocab_file": vocab_file,
        "vocab_max_id": max([arr[VOCAB_PRODUCT_ID_INDEX] for arr in product_vocab])
    })

    for vocab, filename, key in zip([product_type_vocab],
                                    [vocab_product_type_filename],
                                    ["product_type_vocab"]):
      vocab_file = os.path.join(output_dir, filename)
      write_vocab_txt(vocab, filename=vocab_file)
      stats.update({
          key + "_size": len(vocab),
          key + "_file": vocab_file,
      })

  return stats

#------------------------------------GENERATE FILES FROM DAT FILE(MAIN)---------------------------------------------
def generate_data():
    logging.info("Downloading and extracting data.")
    extracted_data_dir = "C:/Users/lohji/XinYingFYP/src/RnnRecommendation/Data"
    output_dir = "C:/Users/lohji/XinYingFYP/src/TwinkleWebApp/Recommendation/Data/processedData"
        
    stats = generate_datasets(
        extracted_data_dir=extracted_data_dir,
        output_dir=output_dir,
        min_timeline_length=3,
        max_context_length=10,
        max_context_product_type_length=10,
        min_rating=None,
        build_vocabs=True,
        train_data_fraction=0.9,
    )

    logging.info("Generated dataset: %s", stats)

#-------------------------------CONVERT DAT TO TF_RECORD FUNCTION----------------------------------------------------

def read_and_convert_product_file(product_file_path):
    product_data = []  # A list of dictionaries, each dictionary represents a product
    with open(product_file_path, 'r', encoding='unicode_escape') as product_file:
        for line in product_file:
            # Parse the product details from the line and create a dictionary
            product_info = line.strip().split('::')  # Adjust the data splitting based on your file format
            product_dict = {
                'product_id': int(product_info[0]),
                'product_name': product_info[1].encode('utf-8'),
                'product_type': product_info[2].encode('utf-8')
            }
            product_data.append(product_dict)

    return product_data

def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value]))

def _int_feature(value):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=[value]))

#-------------------------------CONVERT DAT TO TFRECORD FUNCTION(MAIN)----------------------------------------------
def convert_dat_to_tfrecord():
    # Define the path to the product detail file and the output TFRecord file
    product_file_path = 'C:/Users/lohji/XinYingFYP/src/TwinkleWebApp/Recommendation/Data/product.DAT'
    tfrecord_file_path = 'C:/Users/lohji/XinYingFYP/src/TwinkleWebApp/Recommendation/Data/processedData/100m-product.tfrecord'

    # Read and convert the product details
    product_data = read_and_convert_product_file(product_file_path)

    # Create a TFRecord dataset
    with tf.io.TFRecordWriter(tfrecord_file_path) as writer:
        for product in product_data:
            feature = {
                'product_id': _int_feature(product['product_id']),
                'product_name': _bytes_feature(product['product_name']),
                'product_type': _bytes_feature(product['product_type'])
                # Add more fields as needed
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    print(f'Product details from {product_file_path} have been written to {tfrecord_file_path}')

#------------------------------------CALL GENERATE DATA AND CONVERT DAT TO TFRECORD---------------------------------------        

generate_data()
convert_dat_to_tfrecord()

#------------------------------------TENSORFLOW RECOMMENDATION FUNCTION-----------------------------------------------------
#path_to_save_model = "C:/Users/lohji/XinYingFYP/src/TwinkleWebApp/TwinkleWebApp/TensorflowCode/model"

train_filename = "C:/Users/lohji/XinYingFYP/src/TwinkleWebApp/Recommendation/Data/processedData/train_productlens_1m.tfrecord"
train = tf.data.TFRecordDataset(train_filename)

test_filename = "C:/Users/lohji/XinYingFYP/src/TwinkleWebApp/Recommendation/Data/processedData/test_productlens_1m.tfrecord"
test = tf.data.TFRecordDataset(test_filename)
        
feature_description = {
    'context_product_id': tf.io.FixedLenFeature([10], tf.int64, default_value=np.repeat(0, 10)),
    'context_product_numberOfPurchase': tf.io.FixedLenFeature([10], tf.float32, default_value=np.repeat(0, 10)),
    'context_product_type': tf.io.FixedLenFeature([10], tf.string, default_value=np.repeat("Drama", 10)),
    'label_product_id': tf.io.FixedLenFeature([1], tf.int64, default_value=0),
}

feature_description_only_product_id = {
    'product_id': tf.io.FixedLenFeature([], tf.int64)
}

def _parse_function(example_proto):
  return tf.io.parse_single_example(example_proto, feature_description)

def parse_tfrecord_fn(example):
    # Parse the TFRecord example based on the feature description
    parsed_example = tf.io.parse_single_example(example, feature_description_only_product_id)
    return tf.strings.as_string(parsed_example['product_id'])

train_ds = train.map(_parse_function).map(lambda x: {
    "context_product_id": tf.strings.as_string(x["context_product_id"]),
    "label_product_id": tf.strings.as_string(x["label_product_id"])
})

test_ds = test.map(_parse_function).map(lambda x: {
    "context_product_id": tf.strings.as_string(x["context_product_id"]),
    "label_product_id": tf.strings.as_string(x["label_product_id"])
})
        
#read file and parse file and get unique product id
data = tf.data.TFRecordDataset('C:/Users/lohji/XinYingFYP/src/TwinkleWebApp/Recommendation/Data/processedData/100m-product.tfrecord')

parsed_product = data.map(parse_tfrecord_fn)
batched_data = parsed_product.batch(1_000)
unique_product_ids = np.unique(np.concatenate(list(batched_data)))
        
embedding_dimension = 32

query_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
        vocabulary=unique_product_ids, mask_token=None),
    tf.keras.layers.Embedding(len(unique_product_ids) + 1, embedding_dimension), 
    tf.keras.layers.GRU(embedding_dimension),
])

candidate_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
        vocabulary=unique_product_ids, mask_token=None),
    tf.keras.layers.Embedding(len(unique_product_ids) + 1, embedding_dimension)
])

metrics = tfrs.metrics.FactorizedTopK(
    candidates=parsed_product.batch(128).map(candidate_model)
)

task = tfrs.tasks.Retrieval(
    metrics=metrics
)
        
class Model(tfrs.Model):
    def __init__(self, query_model, candidate_model):
        super().__init__()
        self._query_model = query_model
        self._candidate_model = candidate_model
        self._task = task

    def compute_loss(self, features, training=False):
        watch_history = features["context_product_id"]
        watch_next_label = features["label_product_id"]

        query_embedding = self._query_model(watch_history)       
        candidate_embedding = self._candidate_model(watch_next_label)

        return self._task(query_embedding, candidate_embedding, compute_metrics=not training)

model = Model(query_model, candidate_model)

model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train_ds.shuffle(10_000).batch(12800).cache()
cached_test = test_ds.batch(2560).cache()  

model.fit(cached_train, epochs=3)
model.evaluate(cached_test, return_dict=True)

#------------------------------------HOSTING & SETUP REST API---------------------------------------------------------------
app = Flask(__name__)

@app.route('/recommend_product', methods=['POST'])
def recommend_product():
    try:       
        def make_recommendations(model, user_history, top_k=10):
            user_history = tf.convert_to_tensor(user_history, dtype=tf.string)
            user_history = tf.reshape(user_history, (1, -1))  # Reshape to (1, sequence_length)

            # Get the query model's embedding for the user's history
            query_embedding = model._query_model(user_history)

            # Get the candidate embeddings for all products
            candidate_embeddings = model._candidate_model(np.array(unique_product_ids))

            # Calculate the scores using the dot product between query and candidate embeddings
            scores = tf.matmul(query_embedding, candidate_embeddings, transpose_b=True)
            scores = tf.squeeze(scores, axis=0)  # Remove the batch dimension

            # Find the top-k product IDs with the highest scores
            top_k_indices = tf.argsort(scores, direction='DESCENDING')[:top_k]
            top_k_product_ids = tf.gather(unique_product_ids, top_k_indices)

            return top_k_product_ids.numpy()

        data = request.get_json()
        
        if 'strings' in data:
            user_history = data['strings']
            
        recommended_products = make_recommendations(model, user_history)

        # Print the top recommendations
        print("Top Recommendations:")
        print(recommended_products)

        recommended_products_strings = [product_id.decode('utf-8') for product_id in recommended_products]
        shuffle_list = sorted(recommended_products_strings, key=lambda x: random.random())

        return jsonify({"status": "success", "message": f"Script executed successfully.", "result": shuffle_list})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error executing script: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
