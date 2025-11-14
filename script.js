// script.js - Chat application with Neuro animation
document.addEventListener("DOMContentLoaded", () => {
  // retrieve HTML elements
  const chatBox = document.getElementById("chatBox");
  const userInput = document.getElementById("userInput");
  const sendButton = document.getElementById("sendButton");
  const serverStatus = document.getElementById("serverStatus");
  const neuroCorner = document.getElementById("neuro-corner");

  const BASE_URL = "http://127.0.0.1:5000";

  // --- Fallback Neuro animation/audio (used when NeuroSpin is not present) ---
  let _fallbackAudioCtx = null;
  function fallbackPlayBeep() {
    try {
      if (!_fallbackAudioCtx) _fallbackAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = _fallbackAudioCtx.createOscillator();
      const gain = _fallbackAudioCtx.createGain();
      osc.type = 'square';
      osc.frequency.value = 800;
      osc.connect(gain);
      gain.connect(_fallbackAudioCtx.destination);
      gain.gain.setValueAtTime(0.08, _fallbackAudioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, _fallbackAudioCtx.currentTime + 0.05);
      osc.start(_fallbackAudioCtx.currentTime);
      osc.stop(_fallbackAudioCtx.currentTime + 0.05);
    } catch (e) {
      // ignore audio errors
    }
  }

  function fallbackAnimateText(element, prefix, text, charDelay = 50) {
    return new Promise((resolve) => {
      if (!element) return resolve();
      let index = 0;
      element.textContent = prefix;

      if (neuroCorner) neuroCorner.classList.add('talking');

      const interval = setInterval(() => {
        if (index < text.length) {
          element.textContent = prefix + text.substring(0, index + 1);
          if (index % 2 === 0) fallbackPlayBeep();
          index++;
          chatBox.scrollTop = chatBox.scrollHeight;
        } else {
          clearInterval(interval);
          if (neuroCorner) neuroCorner.classList.remove('talking');
          resolve();
        }
      }, charDelay);
    });
  }

  // Initialize NeuroSpin if available (safe)
  if (window.NeuroSpin && typeof window.NeuroSpin.init === 'function') {
    try {
      window.NeuroSpin.init('neuro-corner');
    } catch (e) {
      console.warn('NeuroSpin.init failed', e);
    }
  }

  // Create Clear History button (ensure sendButton exists)
  if (sendButton && sendButton.parentNode) {
    const clearButton = document.createElement("button");
    clearButton.id = "clearButton";
    clearButton.textContent = "Clear History";
    sendButton.parentNode.insertBefore(clearButton, sendButton.nextSibling);

    clearButton.addEventListener("click", async (e) => {
      e.preventDefault();
      try {
        const r = await fetch(BASE_URL + "/clear_memory", { method: "POST" });
        if (r.ok) {
          chatBox.innerHTML = "";
          loadMemory();
        } else {
          append("info", "Failed to clear history.");
        }
      } catch (err) {
        console.error("clearMemory error:", err);
        append("info", "Error clearing history.");
      }
    });
  }

  // Optional UI elements (model toggle / API key) - guard for absence
  const modelToggle = document.getElementById("modelToggle");
  const apiKeyInput = document.getElementById("apiKeyInput");
  const saveApiKeyBtn = document.getElementById("saveApiKey");

  async function loadConfig() {
    if (!modelToggle) return;
    try {
      const r = await fetch(BASE_URL + "/config");
      if (r.ok) {
        const config = await r.json();
        modelToggle.checked = !!config.use_gemini;
        toggleApiKeyInput(!!config.use_gemini);
      }
    } catch (err) {
      console.error("loadConfig error:", err);
    }
  }

  function toggleApiKeyInput(showApiKey) {
    if (!apiKeyInput || !saveApiKeyBtn) return;
    apiKeyInput.style.display = showApiKey ? "block" : "none";
    saveApiKeyBtn.style.display = showApiKey ? "inline-block" : "none";
  }

  if (modelToggle) {
    modelToggle.addEventListener("change", async () => {
      const useGemini = modelToggle.checked;
      toggleApiKeyInput(useGemini);
      try {
        const r = await fetch(BASE_URL + "/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ use_gemini: useGemini })
        });
        if (r.ok) {
          const result = await r.json();
          append("info", `Switched to ${useGemini ? 'Gemini API' : 'Local Model'}`);
        } else {
          append("info", `Failed to switch model`);
          modelToggle.checked = !useGemini;
        }
      } catch (err) {
        console.error("Model switch error:", err);
        append("info", `Error switching model mode`);
        modelToggle.checked = !useGemini;
      }
    });
  }

  if (saveApiKeyBtn) {
    saveApiKeyBtn.addEventListener("click", async () => {
      const apiKey = apiKeyInput ? apiKeyInput.value.trim() : "";
      if (!apiKey) { append("info", "Please enter an API key."); return; }
      try {
        const r = await fetch(BASE_URL + "/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ gemini_api_key: apiKey })
        });
        if (r.ok) { append("info", "API key saved successfully!"); if (apiKeyInput) apiKeyInput.value = ""; }
        else { append("info", "Failed to save API key."); }
      } catch (err) { console.error("API key save error:", err); append("info", "Error saving API key."); }
    });
  }

  // --- Utilities ---
  function append(kind, text) {
    const el = document.createElement("div");
    el.className = "msg " + (kind === "user" ? "user" : kind === "ai" ? "ai" : "info");
    el.textContent = (kind === "user" ? "You: " : kind === "ai" ? "Neuro: " : "") + text;
    chatBox.appendChild(el);
    chatBox.scrollTop = chatBox.scrollHeight;
    return el;
  }

  async function checkServer() {
    try {
      const r = await fetch(BASE_URL + "/ping", { method: "GET" });
      if (r.ok) { serverStatus.textContent = "online"; serverStatus.style.color = "#8fffd4"; }
      else { serverStatus.textContent = "offline"; serverStatus.style.color = "#ff99aa"; }
    } catch (e) { serverStatus.textContent = "offline"; serverStatus.style.color = "#ff99aa"; }
  }

  async function loadMemory() {
    try {
      const r = await fetch(BASE_URL + "/memory");
      if (!r.ok) return;
      const mem = await r.json();
      if (Array.isArray(mem)) {
        chatBox.innerHTML = "";
        mem.forEach(item => append(item.role === "user" ? "user" : "ai", item.content));
      }
    } catch (err) { console.error("loadMemory error:", err); }
  }

  // --- Main chat flow ---
  async function sendMessage(text) {
    if (!text || !text.trim()) return;
    append("user", text);
    if (userInput) userInput.value = "";

    const aiMsg = document.createElement("div");
    aiMsg.className = "msg ai";
    aiMsg.textContent = "Neuro: …thinking…";
    chatBox.appendChild(aiMsg);
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
      const r = await fetch(BASE_URL + "/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text })
      });
      if (!r.ok) { aiMsg.textContent = "Neuro: [server error " + r.status + "]"; return; }
      const data = await r.json();
      const reply = data.response ?? data.reply ?? "[no response]";

      if (window.NeuroSpin && typeof window.NeuroSpin.animateText === 'function') {
        await window.NeuroSpin.animateText(aiMsg, "Neuro: ", reply);
      } else {
        await fallbackAnimateText(aiMsg, "Neuro: ", reply);
      }
    } catch (err) {
      console.error("sendMessage error:", err);
      aiMsg.textContent = "Neuro: [connection error]";
    } finally { chatBox.scrollTop = chatBox.scrollHeight; }
  }

  if (sendButton) sendButton.addEventListener("click", (e) => { e.preventDefault(); sendMessage(userInput ? userInput.value : ""); });
  if (userInput) userInput.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); sendMessage(userInput.value); } });

  // Init
  checkServer();
  loadConfig();
  loadMemory();
  setInterval(checkServer, 5000);
});