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

def queryAugmenter(chain_id, query):
    """
    Query the OpenTrons API index vectorized database and return an augmented query
    :param query: The question to ask
    :return: Relevant context for the query from the OpenTrons API vector databse
    """
    template = agentsData[chain_id]['agent{}_template'.format(chain_id)]

    embed_model = agentsData[0]['embed_model']
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

def create_llmchain(agent_id):
    """
    Create a LLMChain for a specific agent by calling on prompts stored in agents.json

    :param agent_id: The ID of the agent
    :return: An instance of LLMChain
    """
    chat = ChatOpenAI(streaming=False, callbacks=[StreamingStdOutCallbackHandler()], temperature=0, openai_api_key=openai_api_key)
    template = agentsData[agent_id]['agent{}_template'.format(agent_id)]
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    example_human = HumanMessagePromptTemplate.from_template(agentsData[agent_id]['agent{}_example1_human'.format(agent_id)])
    example_ai = AIMessagePromptTemplate.from_template(agentsData[agent_id]['agent{}_example1_AI'.format(agent_id)])
    human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai, human_message_prompt])
    return LLMChain(llm=chat, prompt=chat_prompt)

chain_5 = create_llmchain(5)

def test(user_input):
    original_query = user_input
    augmented_query = queryAugmenter(5, original_query)

    theQuery = augmented_query

    answer = chain_5.run(theQuery)
    answer += theQuery

    return answer

if __name__ == "__main__":
   answer = test("Inoculate a flask of Luria-Bertani (LB) broth with E.coli and grow the cells overnight at 37°C with shaking")
   with open('answer.txt', 'w') as f:
    f.write(answer)