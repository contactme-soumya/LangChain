from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv(override=True)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


try: 
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.5
    )
    model = ChatHuggingFace(llm=llm, temperature=0.5)
except Exception as e:
    print(f"HuggingFaceEndpoint call failed: {e}")


template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables = ['topic']
)

template2 = PromptTemplate(
    template='Write a five line summary on following text /n {text}',
    input_variables = ['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'science and weather impact behind cricket ball swing'})
   
print(f'Result: {result}')
