document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chat-form");
    const messageInput = document.getElementById("message-input");
    const chatContainer = document.getElementById("chat-container");
    const sendBtn = document.getElementById("send-btn");
    const clearChatBtn = document.getElementById("clear-chat-btn");
    const themeBtn = document.getElementById("theme-btn");

    // Theme toggling
    if (themeBtn) {
        themeBtn.addEventListener("click", () => {
            if (document.documentElement.getAttribute("data-theme") === "dark" || document.body.getAttribute("data-theme") === "dark") {
                document.body.removeAttribute("data-theme");
                document.documentElement.removeAttribute("data-theme");
                themeBtn.innerHTML = '<i class="ph ph-moon"></i>';
            } else {
                document.body.setAttribute("data-theme", "dark");
                document.documentElement.setAttribute("data-theme", "dark");
                themeBtn.innerHTML = '<i class="ph ph-sun"></i>';
            }
        });
    }

    // Auto-resize textarea
    messageInput.addEventListener("input", function() {
        this.style.height = "auto";
        this.style.height = (this.scrollHeight) + "px";
        if (this.value.trim() === "") {
            sendBtn.disabled = true;
        } else {
            sendBtn.disabled = false;
        }
    });

    // Handle Enter key to send (Shift+Enter for new line)
    messageInput.addEventListener("keydown", function(e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (this.value.trim() !== "") {
                chatForm.dispatchEvent(new Event("submit"));
            }
        }
    });

    // Initial disable of send btn
    sendBtn.disabled = true;

    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const message = messageInput.value.trim();
        if (!message) return;

        // Reset input
        messageInput.value = "";
        messageInput.style.height = "auto";
        sendBtn.disabled = true;

        // Add user message to UI
        appendMessage("user", message);

        // Show typing indicator
        const typingId = showTypingIndicator();

        try {
            // Send request to Flask backend
            const response = await fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ message: message })
            });

            const data = await response.json();
            
            // Remove typing indicator
            const typingEl = document.getElementById(typingId);
            if(typingEl) typingEl.remove();

            if (data.error) {
                appendMessage("bot", `**Error:** ${data.error}`);
            } else {
                appendMessage("bot", data.response);
            }
        } catch (error) {
            const typingEl = document.getElementById(typingId);
            if(typingEl) typingEl.remove();
            appendMessage("bot", `**Network Error:** Could not connect to the server. ${error.message}`);
        }
    });

    clearChatBtn.addEventListener("click", () => {
        // Keep the first welcome message
        const welcomeMessage = chatContainer.firstElementChild;
        chatContainer.innerHTML = "";
        if (welcomeMessage) {
            chatContainer.appendChild(welcomeMessage);
        }
    });

    function appendMessage(sender, text) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${sender}-message`;
        
        const avatarHtml = sender === "user" 
            ? `<div class="message-avatar"><img src="https://ui-avatars.com/api/?name=User&background=3b82f6&color=fff" alt="User"></div>`
            : `<div class="message-avatar"><i class="ph-fill ph-scales"></i></div>`;
        
        // Use marked.js to parse markdown text securely
        let parsedText = text;
        try {
            parsedText = marked.parse(text);
        } catch(e) {
            parsedText = `<p>${text}</p>`; // Fallback
        }
        
        msgDiv.innerHTML = `
            ${avatarHtml}
            <div class="message-content">
                ${parsedText}
            </div>
        `;
        
        chatContainer.appendChild(msgDiv);
        scrollToBottom();
    }

    function showTypingIndicator() {
        const id = "typing-" + Date.now();
        const msgDiv = document.createElement("div");
        msgDiv.id = id;
        msgDiv.className = "message bot-message";
        
        msgDiv.innerHTML = `
            <div class="message-avatar"><i class="ph-fill ph-scales"></i></div>
            <div class="message-content" style="padding: 18px 20px;">
                <div class="typing-indicator">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
            </div>
        `;
        
        chatContainer.appendChild(msgDiv);
        scrollToBottom();
        return id;
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
});
