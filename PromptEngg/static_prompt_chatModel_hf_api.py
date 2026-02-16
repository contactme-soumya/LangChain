from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv(override=True)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Remove the 'task' parameter - it's not valid for HuggingFaceEndpoint
# The model must be available via Hugging Face Inference API for chat completion
# If TinyLlama doesn't work, try other models like:
# - "mistralai/Mistral-7B-Instruct-v0.2"
# - "meta-llama/Llama-2-7b-chat-hf" (requires access)
# - "HuggingFaceH4/zephyr-7b-beta"
try:
    llm = HuggingFaceEndpoint(
        repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=512,
        temperature=0
    )
    
    model = ChatHuggingFace(llm=llm, temperature=0)
    
    result = model.invoke("In 6 lines plan an itinerary in Australia with 12 years kid")
    print(result.content)
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying alternative: Using a model that's more likely to work with Inference API...")
    # Alternative: Use a model that's known to work with Inference API
    try:
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            max_new_tokens=512,
            temperature=0
        )
        model = ChatHuggingFace(llm=llm, temperature=0)
        result = model.invoke("In 6 lines plan an itinerary in Australia with 12 years kid")
        print(result.content)
    except Exception as e2:
        print(f"Alternative also failed: {e2}")
        print("\nPossible issues:")
        print("1. The model may not be available via Hugging Face Inference API")
        print("2. Your API token may not have access to the model")
        print("3. You may need to specify a provider parameter")