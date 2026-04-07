from openai_api_key_verifier import verify_api_key, check_model_access, list_models, get_account_usage
import os

# Get API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Verify if the API key is valid
is_valid = verify_api_key(api_key)

if is_valid:
    print("API key is valid!")
    
    # Check for GPT-4 access
    if check_model_access(api_key, "gpt-4"):
        print("This key has GPT-4 access!")
    
    # List all available models
    list_models(api_key)
    
    # Get usage statistics
    get_account_usage(api_key)
else:
    print("API key is invalid.")