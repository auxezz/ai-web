# IMPORTS 
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_cpp import Llama
import json
import os
import google.generativeai as genai

from google.api_core.exceptions import GoogleAPICallError

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# CONFIGURATION FILES (use absolute paths)
MODEL_PATH = os.path.join(SCRIPT_DIR, "gemma2-9b-neuro-sama-q4_k_m.gguf")
MEMORY_FILE = os.path.join(SCRIPT_DIR, "memory.json")
PROMPT_FILE = os.path.join(SCRIPT_DIR, "prompt.json")
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.json")

# FLASK APP SETUP
app = Flask(__name__, static_folder=SCRIPT_DIR, static_url_path='')
CORS(app)  # Allow requests from the frontend

# HELPER FUNCTIONS 

def load_json_file(path, default_value):
    """
    Load a JSON file from disk.
    If the file doesn't exist or is corrupted, return the default value.
    """
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Using default value.")
        return default_value
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except json.JSONDecodeError:
        print(f"Warning: {path} is empty or corrupted. Using default value.")
        return default_value
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}. Using default value.")
        return default_value

def save_memory(path, memory_obj):
    """Save chat memory as JSON."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(memory_obj, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error: failed to save memory to {path}: {e}")

# LOAD CONFIGURATION 

# Load prompt configuration from prompt.json
prompt_config = load_json_file(PROMPT_FILE, {"system_prompt": "", "memory_length": 8})

system_prompt = prompt_config.get("system_prompt", "")
memory_length = prompt_config.get("memory_length", 8)

print(f"System prompt: {system_prompt}")
print(f"Memory length: {memory_length} messages")

# Load chat memory
memory = load_json_file(MEMORY_FILE, [])
if not isinstance(memory, list):
    print("Warning: memory.json is not a list. Resetting to empty.")
    memory = []

# Load configuration (model mode and API key)
config = load_json_file(CONFIG_FILE, {"use_gemini": False, "gemini_api_key": ""})
use_gemini = config.get("use_gemini", False)
# NOTE: The API key is loaded from the config file, but will only be used if use_gemini is true.
gemini_api_key = config.get("gemini_api_key", "")

# Initialize models
llm = None
gemini_model = None
# FIX: The model name 'gemini-1.5-flash' is causing a 404 error in the v1beta API. 
# Switching to the latest recommended alias, 'gemini-2.5-flash'.
GEMINI_MODEL_NAME = 'gemini-2.5-flash' 

def initialize_gemini():
    """Configures and initializes the Gemini model if an API key is present."""
    global gemini_model
    if gemini_api_key:
        print(f"Configuring Google Gemini API for {GEMINI_MODEL_NAME}...")
        try:
            # Re-configure with the latest key
            genai.configure(api_key=gemini_api_key)
            gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            print("Gemini API configured successfully!")
            return True
        except Exception as e:
            print(f"Error configuring Gemini API: {e}")
            gemini_model = None
            return False
    else:
        print("Warning: Gemini mode selected but no API key provided.")
        gemini_model = None
        return False

def initialize_local_llm():
    """Loads the local Llama model."""
    global llm
    print("Loading local AI model...")
    try:
        llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=8)
        print("Local model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading local model from {MODEL_PATH}: {e}")
        llm = None
        return False

# Initial model loading based on config
if use_gemini:
    initialize_gemini()
else:
    initialize_local_llm()


# API ROUTES 

@app.route("/")
def index():
    """Serve the main HTML page."""
    # Assuming index.html is correctly served from the static folder
    return app.send_static_file('index.html')

@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint. Returns online status to confirm the server is running."""
    return jsonify({"status": "online", "model_mode": "Gemini" if use_gemini else "Local"})

@app.route("/memory", methods=["GET"])
def get_memory():
    """Get chat history. Returns all stored messages as JSON."""
    return jsonify(memory)

@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint. Receives a user message, generates an AI response, and updates memory.
    """
    global memory, llm, gemini_model, use_gemini
    
    # Get user message from request
    data = request.get_json()
    user_message = data.get("message", "")
    
    if not user_message:
        return jsonify({"response": "I didn't receive a message."})

    # Add user message to memory
    memory.append({"role": "user", "content": user_message})

    # Build conversation history (only use last N messages based on memory_length)
    recent_messages = memory[-memory_length:]
    
    response_text = ""
    
    if use_gemini and gemini_model:
        # Use Gemini API
        print(f"Generating response with Gemini API ({GEMINI_MODEL_NAME}, using last {len(recent_messages)} messages)...")
        
        # Format history into a single string for prompt-based chat model
        # The structure is designed to guide the model's output to follow "Neuro:"
        conversation_parts = [system_prompt]
        for m in recent_messages:
            role_prefix = "User" if m["role"] == "user" else "Neuro"
            conversation_parts.append(f"{role_prefix}: {m['content']}")
        conversation_parts.append("Neuro:")
        
        full_prompt = "\n".join(conversation_parts)
        
        try:
            # Use generate_content for a single turn with prompt history
            response = gemini_model.generate_content(full_prompt)
            
            # Check for response blocking or empty text
            if response.candidates and response.candidates[0].finish_reason.name == 'SAFETY':
                print(f"Gemini blocked the prompt.")
                response_text = "Filtered"
            elif not response.text:
                print("Gemini returned empty response or candidate.")
                response_text = "Somone tell Vedal there is a problem with my Internet"
            else:
                response_text = response.text.strip()
        
        # CATCH API errors using the reliable GoogleAPICallError
        except GoogleAPICallError as e:
            print(f"Somone tell Vedal there is a problem with my API: {type(e).__name__}: {e}")
            # Check for common error types
            error_message = str(e).lower()
            if "invalid api key" in error_message or "permission denied" in error_message:
                response_text = "Somone tell Vedal there is a problem with my API"
            elif "quota" in error_message or "limit" in error_message or "resource_exhausted" in error_message:
                response_text = "API quota exceeded. Please try again later."
            else:
                # This should now be less likely to be a 404 model name issue
                response_text = f"API error: {str(e)[:100]}"
        except Exception as e:
            print(f"Unexpected Error during Gemini call: {type(e).__name__}: {e}")
            response_text = f"Somone tell Vedal there is a problem with my AI: {str(e)[:100]}"
    
    elif llm:
        # Use local model (llama_cpp)
        history = "\n".join([
            f"User: {m['content']}" if m["role"] == "user" else f"Neuro: {m['content']}"
            for m in recent_messages
        ])
        
        # Build the full prompt with system instructions + history
        prompt = f"{system_prompt}\n{history}\nNeuro:"

        # Generate response
        print(f"Generating response with local model (using last {len(recent_messages)} messages)...")
        try:
            # Set max_tokens to a reasonable limit, and stop tokens to prevent turn-taking issues
            output = llm(prompt, max_tokens=150, stop=["User:", "Neuro:", "</s>"], echo=False, temperature=0.7)
            response_text = output["choices"][0]["text"].strip()
        except Exception as e:
            print(f"Error during local LLM generation: {e}")
            response_text = "I ran into an issue with the local model. Please check the console."
    
    else:
        # No model is available (either local model failed to load, or Gemini mode is on without a key)
        response_text = "No AI model is currently available or configured. Check your settings and console."

    # Add response to memory
    memory.append({"role": "assistant", "content": response_text})
    save_memory(MEMORY_FILE, memory)

    return jsonify({"response": response_text})

@app.route("/clear_memory", methods=["POST"])
def clear_memory():
    """
    Clear chat history.
    Deletes all stored messages and resets memory.
    """
    global memory
    memory = []
    save_memory(MEMORY_FILE, memory)
    print("Memory cleared.")
    return jsonify({"status": "cleared"})

@app.route("/config", methods=["GET"])
def get_config():
    """Get current configuration (model mode)."""
    # Expose only status, not the actual key
    return jsonify({
        "use_gemini": use_gemini,
        "has_api_key": bool(gemini_api_key),
        "model_available": (use_gemini and gemini_model is not None) or (not use_gemini and llm is not None)
    })

@app.route("/config", methods=["POST"])
def update_config():
    """Update configuration (switch model or set API key)."""
    global config, use_gemini, gemini_api_key, llm, gemini_model
    
    try:
        data = request.get_json()
        
        # Update use_gemini if provided
        if "use_gemini" in data:
            new_use_gemini = data["use_gemini"]
            if new_use_gemini != use_gemini:
                use_gemini = new_use_gemini
                config["use_gemini"] = use_gemini
                
                # De-initialize the unused model
                if use_gemini:
                    llm = None
                    initialize_gemini()
                else:
                    gemini_model = None
                    initialize_local_llm()

                print(f"Switched to {'Gemini API' if use_gemini else 'local model'}")
        
        # Update API key if provided
        if "gemini_api_key" in data and data["gemini_api_key"] != gemini_api_key:
            gemini_api_key = data["gemini_api_key"]
            config["gemini_api_key"] = gemini_api_key
            print("API key updated")
            
            # Re-initialize Gemini immediately if the key changed and we are in Gemini mode
            if use_gemini:
                initialize_gemini()
        
        # Save config to file
        # Helper function save_memory is generic enough to use for any JSON file
        save_memory(CONFIG_FILE, config)
        
        return jsonify({
            "status": "success",
            "use_gemini": use_gemini,
            "model_available": (use_gemini and gemini_model is not None) or (not use_gemini and llm is not None)
        })
    
    except Exception as e:
        print(f"Error in update_config: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# start server | Logs for backend status
if __name__ == "__main__":
    import webbrowser
    import threading
    
    print("\nNeuro Chatbot Server")
    print("Server running at http://127.0.0.1:5000")
    print("Opening browser...")
    print("Press Ctrl+C to stop\n")
    
    # Open browser after a short delay to ensure server is ready
    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open('http://127.0.0.1:5000')
    
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000)
