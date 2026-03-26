#Will create a joke and print joke + word count of joke using RunnableSequence and RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser   
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv(override=True)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")
model = ChatHuggingFace(llm=llm, temperature=0.5)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a joke on the topic: {topic}"
)
parser = StrOutputParser()
def count_words(text):
    return len(text.split())

joke_generate_chain = RunnableSequence(prompt, model, parser)

joke_with_word_count_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(count_words)
})

final_chain = RunnableSequence(joke_generate_chain, joke_with_word_count_chain)
result = final_chain.invoke({"topic": "programming"})

print(f"result: {result['joke']} \n word count of joke: {result['word_count']}")