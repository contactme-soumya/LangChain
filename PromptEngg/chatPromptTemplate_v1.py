from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate, 
    AIMessagePromptTemplate)
    
# SystemMessage / HumanMessage are already “final” messages, so ChatPromptTemplate has nothing to format. 
#langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

chat_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are an helpful {domain} expert"),
    HumanMessagePromptTemplate.from_template("Tell me about {topic}")
])

prompt = chat_template.invoke({"domain":"astrophysics","topic":"types of satellites"})

print(prompt)

