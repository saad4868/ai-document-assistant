from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

load_dotenv()
presist_directory="db\chroma_db"
embedding_model=HuggingFaceEmbeddings(
  model_name="all-MiniLM-L6-v2"
)
db=Chroma(
    embedding_function=embedding_model,
    persist_directory=presist_directory,

)
pipe=pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=200
)
model=HuggingFacePipeline(pipeline=pipe)
query="what is rag ?"
chathistory=[]

ret=db.as_retriever(serach_kwargs={"k":3})
docs=ret.invoke(query)
 
context="\n\n".join([doc.page_content for doc in docs])

prompt = f"""
You are a helpful assistant.
Use ONLY the following context to answer:Context:
{context}Question:
{query}Answer clearly:
"""
result = model.invoke(prompt)
print(f"ANSWER:{result}")
chathistory.append(HumanMessage(content=query))
chathistory.append(SystemMessage(content=str(result)))
