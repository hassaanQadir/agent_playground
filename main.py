import sys
import os
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
# Load agents' data from JSON
with open('agents.json', 'r') as f:
    agentsData = json.load(f)
# Set the OpenAI and Pinecone API keys
openai.api_key = openai_api_key
pinecone.init(api_key=pinecone_api_key, enviroment="us-west1-gcp")
# Name of the index where we vectorized the OpenTrons API
index_name = 'opentronsapi-docs'
index = pinecone.Index(index_name)


def retry_on_error(func, arg, max_attempts=5):
    """
    Retry a function in case of an error. This guards against rate limit errors.
    The function will be retried max_attempts times with a delay of 4 seconds between attempts.
    :param func: The function to retry
    :param arg: The argument to pass to the function
    :param max_attempts: The maximum number of attempts
    :return: The result of the function or a string indicating an API error
    """
    for attempt in range(max_attempts):
        try:
            result = func(arg)
            return result
        except Exception as e:
            if attempt < max_attempts - 1:  # no need to sleep on the last attempt
                print(f"Attempt {attempt + 1} failed. Retrying in 4 seconds.")
                time.sleep(4)
            else:
                print(f"Attempt {attempt + 1} failed. No more attempts left.")
                API_error = "OpenTronsAPI Error"
                return API_error



def queryAugmenter(query):
    """
    Query the OpenTrons API index vectorized database and return an augmented query
    :param query: The question to ask
    :return: Relevant context for the query from the OpenTrons API vector databse
    """
    embed_model = agentsData[0]['embed_model']
    res = openai.Embedding.create(
        input=["Provide the exact code to perform this step:", query],
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

def askOpenTrons(augmented_query):
    # system message to 'prime' the model
    template = (agentsData[5]['agent5_template'])

    res = openai.ChatCompletion.create( 
        model=agentsData[0]['chat_model'],
        messages=[
            {"role": "system", "content": template},
            {"role": "user", "content": augmented_query}
        ]
    )
    return (res['choices'][0]['message']['content'])



def test(user_input):
    user_input += "Successfully accessed\n"
    user_input += "the molbio.ai\n"
    augmented_query = queryAugmenter("Put 1 ng of DNA stored 50ug/ml into the eppendorf with 100 ul of water")
    user_input += retry_on_error(askOpenTrons, augmented_query)
    return user_input

if __name__ == "__main__":
   #answer = main("Make glow in the dark e. coli")
   marker = sys.argv[-1]
   answer = test(marker + "\nbatman\n")
   with open('answer.txt', 'w') as f:
    f.write(answer)