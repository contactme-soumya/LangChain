from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv(override=True)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables = ['topic']
)

template2 = PromptTemplate(
    template='Write a five line summary on following text /n {text}',
    input_variables = ['text']
)

prompt1 = template1.invoke({'topic':'science and weather impact behind cricket ball swing'})


# Remove the 'task' parameter - it's not valid for HuggingFaceEndpoint
# The model must be available via Hugging Face Inference API for chat completion
# If TinyLlama doesn't work, try other models like:
# - "mistralai/Mistral-7B-Instruct-v0.2"
# - "meta-llama/Llama-2-7b-chat-hf" (requires access)
# - "HuggingFaceH4/zephyr-7b-beta"
try:
    llm = HuggingFaceEndpoint(
        repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.5
    )
    
    model = ChatHuggingFace(llm=llm, temperature=0.5)
    
    result1 = model.invoke(prompt1)
    print('result1.content : ',result1.content)
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying alternative: Using a model that's more likely to work with Inference API...")
    # Alternative: Use a model that's known to work with Inference API
    try:
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            temperature=0.5
        )
        model = ChatHuggingFace(llm=llm, temperature=0.5)
        result1 = model.invoke(prompt1)
        print('result1.content : ',result1.content)
    except Exception as e2:
        print(f"Alternative also failed: {e2}")
        print("\nPossible issues:")
        print("1. The model may not be available via Hugging Face Inference API")
        print("2. Your API token may not have access to the model")
        print("3. You may need to specify a provider parameter")


prompt2 = template2.invoke({'text':result1.content})      

try:
    llm = HuggingFaceEndpoint(
        repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.5
    )
    
    model = ChatHuggingFace(llm=llm, temperature=0.5)
    
    result2 = model.invoke(prompt2)
    print('result2.content : ',result2.content)
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying alternative: Using a model that's more likely to work with Inference API...")
    # Alternative: Use a model that's known to work with Inference API
    try:
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            temperature=0.5
        )
        model = ChatHuggingFace(llm=llm, temperature=0.5)
        result2 = model.invoke(prompt2)
        print('result2.content : ',result2.content)
    except Exception as e2:
        print(f"Alternative also failed: {e2}")
        print("\nPossible issues:")
        print("1. The model may not be available via Hugging Face Inference API")
        print("2. Your API token may not have access to the model")
        print("3. You may need to specify a provider parameter")