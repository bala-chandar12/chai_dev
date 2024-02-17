from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings

from langchain.document_loaders import PyPDFLoader
GOOGLE_API_KEY='AIzaSyCvtMa0OoR0OZclO0uC87IV_TlxBkoSv6A'



# Load and process the text files
# loader = TextLoader('single_text_file.txt')
# loader = DirectoryLoader('./document/SIR/', glob="./*.txt", loader_cls=TextLoader)

# documents = loader.load()

# #splitting the text into
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# texts = text_splitter.split_documents(documents)
# print(texts[3])


# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk


# Embed and store the texts
#from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
# Supplying a persist_directory will store the embeddings on disk
#persist_directory = 'db3'

## here we are using OpenAI embeddings but in future we will swap out to local embeddings
#model_name = "intfloat/e5-large-v2"

#hf = HuggingFaceEmbeddings(model_name=model_name)
#embedding = hf
#embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)


#vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)

# persiste the db to disk
#vectordb.persist()
#vectordb = None

# Now we can load the persisted database from disk, and use it as normal. 
"""vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)"""

#retriever = vectordb.as_retriever()


#docs = retriever.get_relevant_documents("What is Security Incident Response")
#print()
#print("test1")
#print(docs)






#from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
#from langchain.vectorstores import Chroma
import os
import together
from langchain.chains import ConversationalRetrievalChain
from querygrag import ret
retriever=ret()


os.environ["TOGETHER_API_KEY"] = "bcb47299a331e5736edb40b846e0b6f9654842e1e64faeaacc624e97244f9a89"




# set your API key
together.api_key = os.environ["TOGETHER_API_KEY"]

# list available models and descriptons
models = together.Models.list()

#together.Models.start("togethercomputer/llama-2-7b")




import logging
from typing import Any, Dict, List, Mapping, Optional

from pydantic import Extra, Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env
from langchain.chains import RetrievalQA

class TogetherLLM(LLM):
    """Together large language models."""

    model: str = "togethercomputer/llama-2-70b-chat"
    """model endpoint to use"""

    together_api_key: str = os.environ["TOGETHER_API_KEY"]
    """Together API key"""

    temperature: float = 0.7
    """What sampling temperature to use."""

    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion."""

    class Config:
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the API key is set."""
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,
                                          model=self.model,
                                          max_tokens=self.max_tokens,
                                          temperature=self.temperature,
                                          stop=["<|im_end|>","Answer:" ],
                                          )
        text = output['output']['choices'][0]['text']
        return text


    

llm = TogetherLLM(
    model= "mistralai/Mistral-7B-Instruct-v0.2",
    temperature = 0.1,
    max_tokens = 1024
)
# create the chain to answer questions
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationKGMemory
from langchain import PromptTemplate
from langchain.retrievers import TFIDFRetriever




template = """
Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)
memory=ConversationKGMemory(llm=llm,
           memory_key="history",
          input_key="question")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    verbose=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": prompt,
        "memory": memory,
                   #ConversationBufferMemory( memory_key="history",input_key="question"),
        }
)
#memory.clear()
#k=qa.run({"query": "What is Security Incident Response"})

#print(k)
def wrap(retrieve):
    print("inside wrap function")
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retrieve,
    verbose=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": prompt,
        "memory": memory,
                   #ConversationBufferMemory( memory_key="history",input_key="question"),
        }
      )
    
def createrag(file_path):
    print("inside createrag")
    pdf_loader = PyPDFLoader(file_path)
    print("pdf loader is created")
    pages = pdf_loader.load_and_split()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    # context = "\n\n".join(str(p.page_content) for p in pages)
    # texts = text_splitter.split_text(context)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(pages)
    print(texts[1])

    #loader = DirectoryLoader('./document/SIR/', glob="./*.txt", loader_cls=TextLoader)
    #documents = loader.load()
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    #texts = text_splitter.split_documents(documents)

    persist_directory = 'db5'
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)
    vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding,
                                 persist_directory=persist_directory)
    vectordb.persist()
    retriever = vectordb.as_retriever()
    print("retriver is created")
    wrap(retriever)


#createrag("C:/revival/live-transcription-flask-main/document/sb-itsm.pdf")


async def predict(que):
    memory.clear()
    k=qa.run({"query":que})
    print(k)
predict("what is security incident response")


"""

model_name = "intfloat/e5-large-v2"

hf = HuggingFaceEmbeddings(model_name=model_name)

persist_directory = 'db'

## Here is the nmew embeddings being used
embedding = hf #instructor_embeddings

# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)

retriever = vectordb.as_retriever()
docs = retriever.get_relevant_documents("What is Security Incident Response")
print(docs)


"""