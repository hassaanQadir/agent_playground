import sys
import os
import re
import json
import time
import openai
import pinecone
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
# Load environment variables
load_dotenv('.env')
# Use the environment variables for the API keys if available
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
# Load chains' data from JSON
with open('chains.json', 'r') as f:
    chainsData = json.load(f)
# Set the OpenAI and Pinecone API keys
openai.api_key = openai_api_key
pinecone.init(api_key=pinecone_api_key, enviroment="us-west1-gcp")
# Name of the index where we vectorized the OpenTrons API
index_name = 'opentronsapi-docs'
opentronsapi_docs = pinecone.Index(index_name)

def queryAugmenter(index, chain_id, query):
    """
    Query the vectorized database and return an augmented query
    :param index: The vectorized database to search through
    :param chain_id: The chain whose template will determine what is relevant context
    :param query: The question to ask
    :return: Relevant context for the query from the given vector databse
    """
    # Convert the chain_id to a string if it's an object
    if not isinstance(chain_id, (str, int, float)):
        chain_id = str(chain_id)

    # If chain_id is a string, extract the numbers at the end.
    # If it's a number, just use that number.
    if isinstance(chain_id, str):
        numbers = re.findall(r'\d+', chain_id)
        if numbers:  # If there are any numbers in the string
            chain_id = numbers[-1]  # Select the last group of numbers
        chain_id = int(chain_id)

    template = chainsData[chain_id]['chain{}_template'.format(chain_id)]

    embed_model = chainsData[0]['embed_model']
    res = openai.Embedding.create(
        input=[template, query],
        engine=embed_model
    )
    # retrieve from Pinecone
    xq = res['data'][0]['embedding']
    # get relevant contexts (including the questions)
    res = index.query(xq, top_k=5, include_metadata=True)
    # get list of retrieved text
    contexts = [item['metadata']['text'] for item in res['matches']]
    augmented_query = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query
    
    return augmented_query

def create_llmchain(chain_id):
    """
    Create a LLMChain for a specific chain by calling on prompts stored in chains.json

    :param chain_id: The ID of the chain
    :return: An instance of LLMChain
    """
    chat = ChatOpenAI(streaming=False, callbacks=[StreamingStdOutCallbackHandler()], temperature=0, openai_api_key=openai_api_key)
    template = chainsData[chain_id]['chain{}_template'.format(chain_id)]
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    example_human = HumanMessagePromptTemplate.from_template(chainsData[chain_id]['chain{}_example1_human'.format(chain_id)])
    example_ai = AIMessagePromptTemplate.from_template(chainsData[chain_id]['chain{}_example1_AI'.format(chain_id)])
    human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai, human_message_prompt])
    return LLMChain(llm=chat, prompt=chat_prompt)

chain_5 = create_llmchain(5)

def test(user_input):
    original_query = user_input
    augmented_query = queryAugmenter(opentronsapi_docs, chain_5, original_query)

    theQuery = augmented_query

    answer = chain_5.run(theQuery)
    answer += theQuery

    return answer

if __name__ == "__main__":
   answer = test("Inoculate a flask of Luria-Bertani (LB) broth with E.coli and grow the cells overnight at 37°C with shaking")
   with open('answer.txt', 'w') as f:
    f.write(answer)