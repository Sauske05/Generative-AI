# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 08:06:32 2024

@author: Arun Joshi
"""

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('Arun Joshi CV.pdf')
pages = loader.load_and_split()
initial_page = pages[0]

#Check out the document loader documentation for more info.
#Shows the entire text content
print(initial_page.page_content)
'''metadata attribute can be used to check the metadata
of the page.. Use it as : initial_page.metadata'''

from langchain.text_splitter import RecursiveCharacterTextSplitter
recursive_splitter = RecursiveCharacterTextSplitter(
    separators = ['\n\n', '.', '\n'],
    chunk_size = 500,
    chunk_overlap=0
    )
#for page in pages:
chunks = recursive_splitter.split_documents([pages[i] for i in range(len(pages))])
#chunks[0]
#check_dat = [pages[i] for i in range(len(pages))]
#len(pages)
'''for i in chunks:
    print(len(i.page_content)) 
Useful for checking the len of the page content    
'''
document_texts = [doc.page_content for doc in chunks]
len(document_texts)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-mpnet-base-v2")
vectors = model.encode(document_texts)

import faiss
index= faiss.IndexFlatL2(768)
index.add(vectors)

search_query = "Where does Arun Joshi like doing?"
import numpy as np


def get_indexval_from_vectorstore(query, encoder):
    search_vector = encoder.encode(search_query)
    search_vector = np.array(search_vector).reshape(1,-1)
    similar_data_info = index.search(search_vector, k = 2)
    index_values = [val for val in similar_data_info[1][0]]
    print(index_values)
    return index_values

def get_similar_context(query, encoder):
    similar_data = ''
    index_values = get_indexval_from_vectorstore(query, encoder)
    for i in index_values:
        similar_data += chunks[i].page_content
        similar_data += '\n\n'
    return similar_data
    
similar_context = get_similar_context(search_query, model)
print(similar_context)

import os
from dotenv import load_dotenv 
load_dotenv() 
 
api_key = os.getenv("GEMINI_API_KEY")

import google.generativeai as genai
from langchain.prompts import PromptTemplate
genai.configure(api_key=api_key)
llm = genai.GenerativeModel("gemini-1.5-flash")
llm_prompt_template = """You are an assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know.
Use five sentences maximum and keep the answer concise.\n
Question: {question} \nContext: {context} \nAnswer:"""
llm_prompt = PromptTemplate.from_template(llm_prompt_template)

print(llm_prompt)

formatted_prompt = llm_prompt.format(
    question=search_query,
    context=get_similar_context(search_query, model)
)

answer = llm.generate_content(formatted_prompt)

print(answer.text)
