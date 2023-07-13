#pip install -qU \
#  tiktoken==0.4.0 \
#  openai==0.27.7 \
#  langchain==0.0.137 \
#  "pinecone-client[grpc]"==2.2.1

#pip install langchain

#wget -r -A.html -P rtdocs https://docs.opentrons.com/v2/

### this is where we vectorize the OpenTrons API

from langchain.document_loaders import ReadTheDocsLoader

loader = ReadTheDocsLoader('rtdocs')
docs = loader.load()

import tiktoken

tokenizer_name = tiktoken.encoding_for_model('gpt-3.5-turbo')
tokenizer_name.name

tokenizer = tiktoken.get_encoding(tokenizer_name.name)

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

from typing_extensions import Concatenate
from uuid import uuid4
from tqdm.auto import tqdm

chunks = []

for idx, page in enumerate(tqdm(docs)):
    content = page.page_content
    if len(content) > 100:
        url = page.metadata['source'].replace('rtdocs/', 'https://')
        texts = text_splitter.split_text(content)
        chunks.extend([{
            'id': str(uuid4()),
            'text': texts[i],
            'chunk': i,
            'url': url
        } for i in range(len(texts))])

import os
import openai

openai.api_key = "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"

embed_model = "text-embedding-ada-002"

res = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], engine=embed_model
)

import pinecone

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = "5027b332-54e4-4a94-82da-42c3c2a44dc8"
# find your environment next to the api key in pinecone console
env = "us-west1-gcp"

pinecone.init(api_key="5027b332-54e4-4a94-82da-42c3c2a44dc8", enviroment="us-west1-gcp")


index_name = 'opentronsapi-docs'

import time

# check if index already exists (it shouldn't if this is first time)
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=len(res['data'][0]['embedding']),
        metric='cosine'
    )
    # wait for index to be initialized
    time.sleep(1)

# connect to index
index = pinecone.GRPCIndex(index_name)
# view index stats
index.describe_index_stats()

from tqdm.auto import tqdm
from time import sleep

batch_size = 100  # how many embeddings we create and insert at once

for i in tqdm(range(0, len(chunks), batch_size)):
    # find end of batch
    i_end = min(len(chunks), i+batch_size)
    meta_batch = chunks[i:i_end]
    # get ids
    ids_batch = [x['id'] for x in meta_batch]
    # get texts to encode
    texts = [x['text'] for x in meta_batch]
    # create embeddings (try-except added to avoid RateLimitError)
    try:
        res = openai.Embedding.create(input=texts, engine=embed_model)
    except:
        done = False
        while not done:
            sleep(5)
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)
                done = True
            except:
                pass
    embeds = [record['embedding'] for record in res['data']]
    # cleanup metadata
    meta_batch = [{
        'text': x['text'],
        'chunk': x['chunk'],
        'url': x['url']
    } for x in meta_batch]
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    # upsert to Pinecone
    index.upsert(vectors=to_upsert)