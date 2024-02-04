from querygrag import ret
import os
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
os.environ['GOOGLE_API_KEY'] = 'AIzaSyCvtMa0OoR0OZclO0uC87IV_TlxBkoSv6A'

retriever=ret()


prompt_templat = """
  Please answer the question in as much detail as possible based on the provided context.
  Ensure to include all relevant details. If the answer is not available in the provided context,
  kindly respond with "The answer is not available in the context." Please avoid providing incorrect answers.
\n\n
  Context:\n {context}?\n
  Question: \n{question}\n

  Answer:
"""
prompt_template = """
  Always answer the question!!!
\n\n
  Context:\n {context}?\n
  Question: \n{question}\n
  Answer:
"""
prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)
chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
question = "what is security incident"
docs = retriever.get_relevant_documents(question)
print(docs)
response = chain(
    {"input_documents":docs, "question": question}
    , return_only_outputs=True)
print(response)

