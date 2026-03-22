from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser   
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv
import os

load_dotenv(override=True)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")
model = ChatHuggingFace(llm=llm, temperature=0.5)

prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="Post for LinkedIn: {topic}"
)

prompt2 = PromptTemplate(
    input_variables=["topic"],
    template="Post for Twitter {topic}?"
)

parser = StrOutputParser()

chain = RunnableParallel({
                            'tweet':RunnableSequence(prompt1, model, parser),
                            'linkedin':RunnableSequence(prompt2, model, parser)
                        })


result = chain.invoke({"topic": "RAG and RATF in AI"})

print(f"Result: {result}")