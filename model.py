from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

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