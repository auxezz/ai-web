#  IMPORTS 
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_cpp import Llama
import json
import os

# CONFIGURATION FILES 
MODEL_PATH = "gemma2-9b-neuro-sama-q4_k_m.gguf"
MEMORY_FILE = "memory.json"
PROMPT_FILE = "prompt.json"

# FLASK APP SETUP
app = Flask(__name__)
CORS(app)  # Allow requests from the frontend

#  HELPER FUNCTIONS 

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

#  LOAD CONFIGURATION 

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

# Load AI model
print("Loading AI model...")
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=8)
print("Model loaded successfully!")

#  API ROUTES 

@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint. Returns online status to confirm the server is running."""
    return jsonify({"status": "online"})

@app.route("/memory", methods=["GET"])
def get_memory():
    """Get chat history. Returns all stored messages as JSON."""
    return jsonify(memory)

@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint. Receives a user message, generates an AI response, and updates memory."""
    global memory
    
    # Get user message from request
    data = request.get_json()
    user_message = data.get("message", "")
    
    if not user_message:
        return jsonify({"response": "I didn't receive a message."})

    # Add user message to memory
    memory.append({"role": "user", "content": user_message})

    # Build conversation history (only use last N messages based on memory_length)
    recent_messages = memory[-memory_length:]
    history = "\n".join([
        f"User: {m['content']}" if m["role"] == "user" else f"Neuro: {m['content']}"
        for m in recent_messages
    ])
    
    # Build the full prompt with system instructions + history | Also acts as save points for refreshing the website
    prompt = f"{system_prompt}\n{history}\nNeuro:"

    # Generate response
    print(f"Generating response (using last {len(recent_messages)} messages)...")
    output = llm(prompt, max_tokens=150, stop=["User:", "Neuro:"], echo=False)
    response_text = output["choices"][0]["text"].strip()

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

# start server | Logs for backend status
if __name__ == "__main__":
    print("\nNeuro Chatbot Server")
    print("Server running at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop\n") 
    app.run(host="127.0.0.1", port=5000)
