from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser   
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv(override=True)

def word_count(text):
    return len(text.split())

word_count_lambda = RunnableLambda(word_count)

word_count_result = word_count_lambda.invoke("Hello world, this is a test.")
print(f"Word count: {word_count_result}")
