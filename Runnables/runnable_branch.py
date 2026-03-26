#Will create a joke and print joke + word count of joke using RunnableSequence and RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser   
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from dotenv import load_dotenv
import os
import random
import re

load_dotenv(override=True)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")
model = ChatHuggingFace(llm=llm, temperature=0.5)

instructions = ["Get out from my house", "Can you please do it for me"]

# Function to extract text within quotes
def extract_mood_from_quotes(text):
    """Extract first quoted text from the output as mood"""
    match = re.search(r'"([^"]+)"', text)
    if match:
        return match.group(1)  # Returns text inside quotes without quotes
    return text.strip().lower()

# RunnableLambda to extract mood from quoted text
mood_extractor = RunnableLambda(extract_mood_from_quotes)

# RunnableLambda to randomly select an instruction
random_instruction_selector = RunnableLambda(lambda x: {"x": random.choice(instructions)})

prompt = PromptTemplate(
    input_variables=["x"],
    template="Categorize the mood from two categories enum i.e. angry and request: {x}, result should be either angry or request"
)

parser = StrOutputParser()

# Chain to only format prompt (without calling LLM)
prompt_chain = random_instruction_selector | prompt

# Invoke and print formatted prompt
# formatted_prompt = prompt_chain.invoke({})
# print(f"Formatted Prompt: {formatted_prompt}")

# Optional: Uncomment below to call full chain with LLM
sentiment_finding_chain = prompt_chain | model | parser

# For testing output
# result = sentiment_finding_chain.invoke({})
# print(f"LLM Result: {result}")

# Extract mood from quoted text
# extracted_mood = mood_extractor.invoke(result)
# print(f"Extracted Mood: {extracted_mood}")

# prompt2 - Use extracted mood
prompt2 = PromptTemplate(
    input_variables=["mood"],
    template="Explain {mood} from sentence construction and grammar perspective in 1 line"
)

# Chain with prompt2
# response_chain = RunnableLambda(lambda x: {"mood": extracted_mood}) | prompt2 | model | parser
# response = response_chain.invoke({})
# print(f"Response based on mood: {response}")

# runtime runner: from mood string to runnable input dict
mood_to_dict = RunnableLambda(lambda mood: {"mood": mood.strip().lower()})

# generic chain that can be applied once we know mood
mood_response_chain = mood_to_dict | prompt2 | model | parser

branch_chain = RunnableBranch(
                                (lambda mood: mood == "angry", mood_response_chain),
                                (lambda mood: mood == "request", mood_response_chain),
                                RunnablePassthrough())


final_chain_comb_seq_and_branch_chain = sentiment_finding_chain | mood_extractor | branch_chain

final_joined_result = final_chain_comb_seq_and_branch_chain.invoke({})

print(f"Final joined result: {final_joined_result}")