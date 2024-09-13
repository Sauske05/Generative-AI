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

search_query = "Where does Arun live?"
search_vector = model.encode(search_query)
import numpy as np
search_vector = np.array(search_vector).reshape(1,-1)
index.search(search_vector, k = 2)

