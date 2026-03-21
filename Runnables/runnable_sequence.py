from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser   
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv(override=True)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")
model = ChatHuggingFace(llm=llm, temperature=0.5)

prompt = PromptTemplate(
    input_variables=["question"],
    template="You are a helpful assistant. Answer the following question: {question}"
)

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser)
result = chain.invoke({"question": "What is the capital of France?"})

print(f"Result: {result}")