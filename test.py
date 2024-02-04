from querygrag import ret

retriever=ret()
print("query rag testing")
print()
docs = retriever.get_relevant_documents("What is Security Incident Response")
print(docs)