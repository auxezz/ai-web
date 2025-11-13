// script.js - Chat application with Neuro animation
document.addEventListener("DOMContentLoaded", () => {
  // retrieve HTML elements
  const chatBox = document.getElementById("chatBox");
  const userInput = document.getElementById("userInput");
  const sendButton = document.getElementById("sendButton");
  const serverStatus = document.getElementById("serverStatus");
  const neuroCorner = document.getElementById("neuro-corner");
  
  // Audio setup for typing sounds
  let audioContext = null;
  let typingSoundInterval = null;

  // Play a short beep sound when talking (Too weak computers for RVC integration(Retrieval based voice conversion))
  function playTypingBeep() {
    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.value = 800;
    oscillator.type = 'square';
    
    gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.05);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.05);
  }

  // Start Neuro talking animation and typing sound
  function startNeuroTalking() {
    neuroCorner.classList.add('talking');
    typingSoundInterval = setInterval(() => {
      playTypingBeep();
    }, 100);
  }
  
  // Stop Neuro talking animation and typing sound
  function stopNeuroTalking() {
    neuroCorner.classList.remove('talking');
    if (typingSoundInterval) {
      clearInterval(typingSoundInterval);
      typingSoundInterval = null;
    }
  }
  
  // Animate text appearing character-by-character with Neuro bobbing and typing sound
  async function animateText(element, prefix, text) {
    return new Promise((resolve) => {
      // Calculate how long the animation should last
      const charDelay = 50; // milliseconds per character
      const totalDuration = text.length * charDelay;
      
      console.log(`Animating ${text.length} characters over ${totalDuration}ms`);
      
      let index = 0;
      element.textContent = prefix;
      
      // Start Neuro bobbing
      neuroCorner.classList.add('talking');
      console.log("Neuro started bobbing"); // temp debugging
      console.log("neuroCorner classes:", neuroCorner.className);
      console.log("neuroCorner computed style:", window.getComputedStyle(neuroCorner).animation);
      
      // Stop bobbing after full message is displayed
      const stopTimer = setTimeout(() => {
        neuroCorner.classList.remove('talking');
        console.log("Neuro stopped bobbing");
        console.log("neuroCorner classes after stop:", neuroCorner.className);
      }, totalDuration);
      
      // Display text character by character
      const interval = setInterval(() => {
        if (index < text.length) {
          element.textContent = prefix + text.substring(0, index + 1);
          
          // Play sound every 2 characters
          if (index % 2 === 0) {
            playTypingBeep();
          }
          
          index++;
          chatBox.scrollTop = chatBox.scrollHeight;
        } else {
          clearInterval(interval);
          resolve();
        }
      }, charDelay);
    });
  }
  
  // Create Clear History button
  const clearButton = document.createElement("button");
  clearButton.id = "clearButton";
  clearButton.textContent = "Clear History";
  sendButton.parentNode.insertBefore(clearButton, sendButton.nextSibling);
  
  // Clear History button click handler
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

  const BASE_URL = "http://127.0.0.1:5000";

  // Add a message to the chat box
  function append(kind, text) {
    const el = document.createElement("div");
    el.className = "msg " + (kind === "user" ? "user" : kind === "ai" ? "ai" : "info");
    el.textContent = (kind === "user" ? "You: " : kind === "ai" ? "Neuro: " : "") + text;
    chatBox.appendChild(el);
    chatBox.scrollTop = chatBox.scrollHeight;
    return el;
  }

  // Check if the backend server is online
  async function checkServer() {
    try {
      const r = await fetch(BASE_URL + "/ping", { method: "GET" });
      if (r.ok) {
        serverStatus.textContent = "online";
        serverStatus.style.color = "#8fffd4";
      } else {
        serverStatus.textContent = "offline";
        serverStatus.style.color = "#ff99aa";
      }
    } catch (e) {
      serverStatus.textContent = "offline";
      serverStatus.style.color = "#ff99aa";
    }
  }

  // Load chat history from backend
  async function loadMemory() {
    try {
      const r = await fetch(BASE_URL + "/memory");
      if (!r.ok) return;
      const mem = await r.json();
      
      if (Array.isArray(mem)) {
        chatBox.innerHTML = "";
        mem.forEach(item => {
          append(item.role === "user" ? "user" : "ai", item.content);
        });
      }
    } catch (err) {
      console.error("loadMemory error:", err);
    }
  }

  // Send a message to the backend and display the response
  async function sendMessage(text) {
    if (!text || !text.trim()) return;
    append("user", text);
    userInput.value = "";
    
    // Create placeholder for AI response
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

      if (!r.ok) {
        aiMsg.textContent = "Neuro: [server error " + r.status + "]";
        return;
      }

      const data = await r.json();
      console.log("backend response:", data);

      const reply = data.response ?? data.reply ?? "[no response]";
      
      // Animate the response with Neuro bobbing and typing sound
      await animateText(aiMsg, "Neuro: ", reply);

    } catch (err) {
      console.error("sendMessage error:", err);
      aiMsg.textContent = "Neuro: [connection error]";
    } finally {
      chatBox.scrollTop = chatBox.scrollHeight;
    }
  }

  // Send button click handler
  sendButton.addEventListener("click", (e) => {
    e.preventDefault();
    sendMessage(userInput.value);
  });

  // Enter key handler for input field
  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendMessage(userInput.value);
    }
  });

  // Initialize: check server and load chat history
  checkServer();
  loadMemory();
  
  // Check server status every 5 seconds
  setInterval(checkServer, 5000); //5000 milliseconds
});
