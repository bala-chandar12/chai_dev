from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

# Load and process the text files
# loader = TextLoader('single_text_file.txt')
#loader = DirectoryLoader('./document/', glob="./*.txt", loader_cls=TextLoader)

#documents = loader.load()

#splitting the text into
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#texts = text_splitter.split_documents(documents)
#print(texts[3])


# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk


# Embed and store the texts
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

## here we are using OpenAI embeddings but in future we will swap out to local embeddings
model_name = "intfloat/e5-large-v2"

hf = HuggingFaceEmbeddings(model_name=model_name)
embedding = hf

"""vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding,
                                 persist_directory=persist_directory)"""

# persiste the db to disk
#vectordb.persist()
#vectordb = None

# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)

retriever = vectordb.as_retriever()

docs = retriever.get_relevant_documents("What is Security Incident Response")
print(docs)





