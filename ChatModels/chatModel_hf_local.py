import os
# Set environment variables FIRST, before any HuggingFace imports
os.environ["HF_HUB_CACHE"] = 'D:/HF_AI_Models'
os.environ["TRANSFORMERS_CACHE"] = 'D:/HF_AI_Models'
os.environ["HF_HOME"] = 'D:/HF_AI_Models'

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv(override=True)

# try:
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task="text-generation",
    pipeline_kwargs={"temperature": 0.5,"max_new_tokens": 100}
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of India?")
print(result.content)   
# except Exception as e:
#     print(f"Error: {e}")
#     print("\nTrying alternative: Using a model that's more likely to work with Inference API...")
#     llm = HuggingFacePipeline.from_model_id(
#         model_id='meta-llama/Llama-3.2-1B-Instruct',
#         task="text-generation",
#         pipeline_kwargs={"temperature": 0.5,"max_new_tokens": 100}
#     )
#     model = ChatHuggingFace(llm=llm)
#     result = model.invoke("What is the capital of India?")
#     print(result.content)
    