# main.py

from os import path
from typing import List, Dict
import logging
import traceback

import chromadb
import openai
from openai.types.chat import ChatCompletionMessageParam

from fastapi import FastAPI
from pydantic import BaseModel

from configs import GPT_API_KEY
from configs import GPT_MODEL_NAME
from configs import CHROMA_DOC_COUNT

logger = logging.getLogger('chromaDB_llm')
logger.setLevel(level=logging.DEBUG)
fh = logging.StreamHandler()
fh_formatter = logging.Formatter('%(asctime)s %(levelname)s ln:%(lineno)d %(funcName)s - %(message)s')
fh.setFormatter(fh_formatter)
logger.addHandler(fh)


def build_prompt(query: str, context: List[str]) -> List[ChatCompletionMessageParam]:
    '''
    Function build_prompt prepare prompt (query, context) for LLM. LLM is set genetare answer base only on context.

    Input: query and context
    Output: prompt for the LLM
    '''

    system: ChatCompletionMessageParam = {
        'role': 'system',
        'content': 'I am going to ask you a question, which I would like you to answer'
        'based only on the provided context, and not any other information.'
        'If there is not enough information in the context to answer the question,'
        'say "I am not sure", then try to make a guess.'
        'Break your answer up into nicely readable paragraphs.',
    }
    user: ChatCompletionMessageParam = {
        'role': 'user',
        'content': f'The question is {query}. Here is all the context you have:'
        f'{(" ").join(context)}',
    }

    return [system, user]


def get_chatGPT_response(query: str, context: List[str], model_name: str) -> str:
    '''
    Function get_chatGPT_response transmit the geberated prompt to LLM model and collect the response.

    Input: query, context, ChatGPT model name
    Output: response content from LLM
    '''

    response = openai.chat.completions.create(
        model=model_name,
        messages=build_prompt(query, context),
    )

    return response.choices[0].message.content


def handle_chromadb(path_to_user_file: str, user_chroma_query: str) -> Dict:
    '''
    Function handle_chromadb will read raw data from file, fulfill the chromadb collection and search for relevant documents

    Input: path to the file and user search query
    Output: process information and generated context for chatGPT
    '''

    logger.info('>>> handle_chromadb')
    process_info = {}

    # ChromaDB process
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("user_content")

    user_file_path = path.join(path.dirname(__file__), 'no_sale_countries.md')  # default value
    if path_to_user_file:
        user_file_path = path_to_user_file

    user_text_file = {
        'ids': [],
        'documents': [],
        'metadatas': []
    }

    with open(user_file_path, 'r') as file_read:
        for line_no, line in enumerate(file_read):
            striped_line = line.strip()
            if striped_line:
                user_text_file['ids'].append(str(line_no))
                user_text_file['documents'].append(striped_line)
                user_text_file['metadatas'].append({'source': 'line_{}'.format(line_no)})

    collection.add(
        ids=user_text_file['ids'],
        documents=user_text_file['documents'],
        metadatas=user_text_file['metadatas']
    )

    # Query the collection to get the n most relevant documents, n is defined in CHROMA_DOC_COUNT in configs.py
    chroma_query = ['What is the reason?']  # default value
    if user_chroma_query:
        chroma_query = [user_chroma_query]

    collected_documents = collection.query(
        query_texts=chroma_query, n_results=CHROMA_DOC_COUNT, include=['documents', 'metadatas']
    )

    process_info['processed_file'] = user_file_path
    process_info['processed_query'] = chroma_query
    # ChromaDB returns List of Lists and we Asked for one question
    process_info['gpt_context'] = collected_documents['documents'][0]

    logger.info('<<< handle_chromadb')

    return process_info


def handle_chatgpt(user_gpt_query: str, gpt_context: List[str]) -> Dict:
    '''
    Function handle_chatgpt will send content and related question to the chatGPT API, return chatGPT API response 

    Input: user question and related content
    Output: process information and resposne from chatGPT
    '''

    logger.info('>>> handle_chatgpt')
    process_info = {}

    # ChatGPT process
    openai.api_key = GPT_API_KEY
    model_name = GPT_MODEL_NAME

    gpt_query = 'Waht is the common reason?'  # default value
    if user_gpt_query:
        gpt_query = user_gpt_query

    try:
        # Get the response from GPT
        response_chatGPT = get_chatGPT_response(gpt_query, gpt_context, model_name)
    except Exception:
        logger.error('Exception: %s', traceback.format_exc())
        exceptiondata = traceback.format_exc().splitlines()
        response_chatGPT = '{} {}'.format(exceptiondata[-1], exceptiondata[1:-1])

    process_info['processed_query'] = gpt_query
    process_info['response_chatGPT'] = response_chatGPT

    logger.info('<<< handle_chatgpt')

    return process_info


class Configuration(BaseModel):
    path_to_user_file: str
    chroma_query: str
    gpt_query: str


app = FastAPI()


@app.post('/process_task/')
async def process_user_task(config: Configuration):
    item_dict = {}

    try:

        process_chroma = handle_chromadb(config.path_to_user_file, config.chroma_query)

        item_dict['chroma'] = process_chroma

        process_gpt = handle_chatgpt(config.gpt_query, process_chroma['gpt_context'])

        item_dict['chat_gpt'] = process_gpt

    except Exception:
        logger.error('Exception: %s', traceback.format_exc())
        exceptiondata = traceback.format_exc().splitlines()
        item_dict['Exception: '] = '{} {}'.format(exceptiondata[-1], exceptiondata[1:-1])

    return item_dict
