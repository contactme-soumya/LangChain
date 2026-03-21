from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

load_dotenv(override=True)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


try: 
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2"
    )
    model = ChatHuggingFace(llm=llm, temperature=0.5)
except Exception as e:
    print(f"HuggingFaceEndpoint call failed: {e}")

parser = JsonOutputParser()

template1 = PromptTemplate(
    template='Write a short 3 lines report on {topic} \n {format_instruction}',
    input_variables = ['topic'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

prompt = template1.format()

result = model.invoke(prompt)

print('raw result',result)

final_parsed_result = parser.parse(result.content)

print('final_parsed_result',result)

chain = template1 | model | parser

chained_result = chain.invoke({'topic':'science and weather impact behind cricket ball swing'})
   
print(f'Chained Result: {chained_result}')
