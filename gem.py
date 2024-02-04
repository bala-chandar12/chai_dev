import urllib
import warnings
from pathlib import Path as p
from pprint import pprint

#import pandas as pd
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
warnings.filterwarnings("ignore")
from querygrag import ret
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
import os
os.environ['GOOGLE_API_KEY'] = 'AIzaSyCvtMa0OoR0OZclO0uC87IV_TlxBkoSv6A'

prompt_template = """
  Please answer the question in as much detail as possible based on the provided context.
  Ensure to include all relevant details. If the answer is not available in the provided context,
  kindly respond with "The answer is not available in the context." Please avoid providing incorrect answers.
\n\n
  Context:\n {context}?\n
  Question: \n{question}\n

  Answer:
"""

prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
retriver=ret

model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)
chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
question = "what is security incident response?"
docs = ret.get_relevant_documents(question)



"""

GOOGLE_API_KEY='AIzaSyCvtMa0OoR0OZclO0uC87IV_TlxBkoSv6A'
model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=GOOGLE_API_KEY,
                             temperature=0.4,convert_system_message_to_human=True)
vector_index=ret()
qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True)
print("going to prediction")
k=qa_chain({"query": "what is security incident recident response"})
print(k["result"])
print("prediction made")


def load_model():
  #Function to load Model
  model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=GOOGLE_API_KEY,
                             temperature=0.4,convert_system_message_to_human=True)
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)
  return model,embeddings

persist_dir="db1"

def get_data():
  #Function to load data
  pdf_loader = PyPDFLoader("/content/medical_data_book.pdf")
  pages = pdf_loader.load_and_split()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
  context = "\n\n".join(str(p.page_content) for p in pages)
  texts = text_splitter.split_text(context)
  model,embeddings=load_model()
  vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})
  vector_index.persist()
  
  qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True

)
  return qa_chain


def get_query():
  #Function to get query from the user
  qa_chain=get_data()
  query = "Describe about Achondroplasia?"
  output = qa_chain({"query": query})
  print(output["result"])

pip freeze > requirements.txt

get_query()

def pred(query):
  

  model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=GOOGLE_API_KEY,
                             temperature=0.4,convert_system_message_to_human=True)
  vector_index=ret()
  qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True)
  print(qa_chain({"query": query})["result"])




"""
