from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_cpp import Llama
import json
import os
import google.generativeai as genai

from google.api_core.exceptions import GoogleAPICallError

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Serve the parent directory (NeuroAIPage) as the static folder so
# sibling folders like Background/ and NeuroSpin/ are reachable
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

MODEL_PATH = os.path.join(SCRIPT_DIR, "../LLM/gemma2-9b-neuro-sama-q4_k_m.gguf")
MEMORY_FILE = os.path.join(SCRIPT_DIR, "memory.json")
PROMPT_FILE = os.path.join(SCRIPT_DIR, "../Configs/prompt.json")
CONFIG_FILE = os.path.join(SCRIPT_DIR, "../Configs/config.json")

app = Flask(__name__, static_folder=ROOT_DIR, static_url_path='')
CORS(app)  


def load_json_file(path, default_value):
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
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(memory_obj, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error: failed to save memory to {path}: {e}")


prompt_config = load_json_file(PROMPT_FILE, {"system_prompt": "", "memory_length": 8})

system_prompt = prompt_config.get("system_prompt", "")
memory_length = prompt_config.get("memory_length", 8)

print(f"System prompt: {system_prompt}")
print(f"Memory length: {memory_length} messages")

memory = load_json_file(MEMORY_FILE, [])
if not isinstance(memory, list):
    print("Warning: memory.json is not a list. Resetting to empty.")
    memory = []

config = load_json_file(CONFIG_FILE, {"use_gemini": False, "gemini_api_key": ""})
use_gemini = config.get("use_gemini", False)
gemini_api_key = config.get("gemini_api_key", "")

llm = None
gemini_model = None
GEMINI_MODEL_NAME = 'gemini-2.5-flash' 

def initialize_gemini():
    global gemini_model
    if gemini_api_key:
        print(f"Configuring Google Gemini API for {GEMINI_MODEL_NAME}...")
        try:
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

if use_gemini:
    initialize_gemini()
else:
    initialize_local_llm()



@app.route("/")
def index():
    # Neuro.html is inside the Main/ subfolder under the static root
    return app.send_static_file('Main/Neuro.html')

@app.route("/ping", methods=["GET"])
def ping():

    return jsonify({"status": "online", "model_mode": "Gemini" if use_gemini else "Local"})

@app.route("/memory", methods=["GET"])
def get_memory():
    return jsonify(memory)

@app.route("/chat", methods=["POST"])
def chat():
    global memory, llm, gemini_model, use_gemini
    
    data = request.get_json()
    user_message = data.get("message", "")
    
    if not user_message:
        return jsonify({"response": "I didn't receive a message."})

    memory.append({"role": "user", "content": user_message})

    recent_messages = memory[-memory_length:]
    
    response_text = ""
    
    if use_gemini and gemini_model:
        print(f"Generating response with Gemini API ({GEMINI_MODEL_NAME}, using last {len(recent_messages)} messages)...")
        
        conversation_parts = [system_prompt]
        for m in recent_messages:
            role_prefix = "User" if m["role"] == "user" else "Neuro"
            conversation_parts.append(f"{role_prefix}: {m['content']}")
        conversation_parts.append("Neuro:")
        
        full_prompt = "\n".join(conversation_parts)
        
        try:
            response = gemini_model.generate_content(full_prompt)
            
            if response.candidates and response.candidates[0].finish_reason.name == 'SAFETY':
                print(f"Gemini blocked the prompt.")
                response_text = "Filtered"
            elif not response.text:
                print("Gemini returned empty response or candidate.")
                response_text = "Somone tell Vedal there is a problem with my Internet"
            else:
                response_text = response.text.strip()
        
        except GoogleAPICallError as e:
            print(f"Somone tell Vedal there is a problem with my API: {type(e).__name__}: {e}")
            error_message = str(e).lower()
            if "invalid api key" in error_message or "permission denied" in error_message:
                response_text = "Somone tell Vedal there is a problem with my API"
            elif "quota" in error_message or "limit" in error_message or "resource_exhausted" in error_message:
                response_text = "API quota exceeded. Please try again later."
            else:
                response_text = f"API error: {str(e)[:100]}"
        except Exception as e:
            print(f"Unexpected Error during Gemini call: {type(e).__name__}: {e}")
            response_text = f"Somone tell Vedal there is a problem with my AI: {str(e)[:100]}"
    
    elif llm:
        history = "\n".join([
            f"User: {m['content']}" if m["role"] == "user" else f"Neuro: {m['content']}"
            for m in recent_messages
        ])
        
        prompt = f"{system_prompt}\n{history}\nNeuro:"

        print(f"Generating response with local model (using last {len(recent_messages)} messages)...")
        try:
            output = llm(prompt, max_tokens=150, stop=["User:", "Neuro:", "</s>"], echo=False, temperature=0.7)
            response_text = output["choices"][0]["text"].strip()
        except Exception as e:
            print(f"Error during local LLM generation: {e}")
            response_text = "I ran into an issue with the local model. Please check the console."
    
    else:
        response_text = "No AI model is currently available or configured. Check your settings and console."

    memory.append({"role": "assistant", "content": response_text})
    save_memory(MEMORY_FILE, memory)

    return jsonify({"response": response_text})

@app.route("/clear_memory", methods=["POST"])
def clear_memory():
    global memory
    memory = []
    save_memory(MEMORY_FILE, memory)
    print("Memory cleared.")
    return jsonify({"status": "cleared"})

@app.route("/config", methods=["GET"])
def get_config():
    return jsonify({
        "use_gemini": use_gemini,
        "has_api_key": bool(gemini_api_key),
        "model_available": (use_gemini and gemini_model is not None) or (not use_gemini and llm is not None)
    })

@app.route("/config", methods=["POST"])
def update_config():
    global config, use_gemini, gemini_api_key, llm, gemini_model
    
    try:
        data = request.get_json()
        
        if "use_gemini" in data:
            new_use_gemini = data["use_gemini"]
            if new_use_gemini != use_gemini:
                use_gemini = new_use_gemini
                config["use_gemini"] = use_gemini
                
                if use_gemini:
                    llm = None
                    initialize_gemini()
                else:
                    gemini_model = None
                    initialize_local_llm()

                print(f"Switched to {'Gemini API' if use_gemini else 'local model'}")
        
        if "gemini_api_key" in data and data["gemini_api_key"] != gemini_api_key:
            gemini_api_key = data["gemini_api_key"]
            config["gemini_api_key"] = gemini_api_key
            print("API key updated")
            
            if use_gemini:
                initialize_gemini()

        save_memory(CONFIG_FILE, config)
        
        return jsonify({
            "status": "success",
            "use_gemini": use_gemini,
            "model_available": (use_gemini and gemini_model is not None) or (not use_gemini and llm is not None)
        })
    
    except Exception as e:
        print(f"Error in update_config: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    import webbrowser
    import threading
    
    print("\nNeuro Chatbot Server")
    print("Server running at http://127.0.0.1:5000")
    print("Opening browser...")
    print("Press Ctrl+C to stop\n")
    
    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open('http://127.0.0.1:5000')
    
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000)
