const API_BASE_URL = window.location.origin;

const elements = {
    chatMessages: document.getElementById('chatMessages'),
    messageInput: document.getElementById('messageInput'),
    sendButton: document.getElementById('sendButton'),
    useContext: document.getElementById('useContext'),
    temperature: document.getElementById('temperature'),
    tempValue: document.getElementById('tempValue'),
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    transcriptCount: document.getElementById('transcriptCount'),
    trainingCount: document.getElementById('trainingCount')
};

let isTyping = false;

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy' && data.model_loaded) {
            elements.statusDot.classList.add('online');
            elements.statusDot.classList.remove('offline');
            elements.statusText.textContent = 'Online';
        } else {
            elements.statusDot.classList.add('offline');
            elements.statusDot.classList.remove('online');
            elements.statusText.textContent = 'Model not loaded';
        }
    } catch (error) {
        elements.statusDot.classList.add('offline');
        elements.statusDot.classList.remove('online');
        elements.statusText.textContent = 'Offline';
    }
}

async function loadStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        const data = await response.json();
        
        elements.transcriptCount.textContent = data.transcripts || 0;
        elements.trainingCount.textContent = data.training_examples || 0;
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

function createMessageElement(content, isUser, sources = null, timestamp = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = isUser 
        ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>'
        : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"></path><path d="M2 17l10 5 10-5"></path><path d="M2 12l10 5 10-5"></path></svg>';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = content;
    
    contentDiv.appendChild(bubble);
    
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        
        const sourcesTitle = document.createElement('div');
        sourcesTitle.className = 'message-sources-title';
        sourcesTitle.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="16" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
            Sources from training videos:
        `;
        sourcesDiv.appendChild(sourcesTitle);
        
        sources.forEach(source => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            sourceItem.innerHTML = `
                <div class="source-video">${source.video}</div>
                <div>Timestamp: ${source.timestamp} | Similarity: ${(source.similarity * 100).toFixed(1)}%</div>
            `;
            sourcesDiv.appendChild(sourceItem);
        });
        
        contentDiv.appendChild(sourcesDiv);
    }
    
    if (timestamp) {
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = new Date(timestamp).toLocaleTimeString();
        contentDiv.appendChild(timeDiv);
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    return messageDiv;
}

function createTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = 'typing-indicator';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"></path><path d="M2 17l10 5 10-5"></path><path d="M2 12l10 5 10-5"></path></svg>';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
    
    bubble.appendChild(typingDiv);
    contentDiv.appendChild(bubble);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    return messageDiv;
}

function removeWelcomeMessage() {
    const welcomeMsg = elements.chatMessages.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
}

function scrollToBottom() {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

async function sendMessage() {
    const message = elements.messageInput.value.trim();
    
    if (!message || isTyping) return;
    
    removeWelcomeMessage();
    
    const userMessage = createMessageElement(message, true);
    elements.chatMessages.appendChild(userMessage);
    scrollToBottom();
    
    elements.messageInput.value = '';
    elements.messageInput.style.height = 'auto';
    elements.sendButton.disabled = true;
    
    isTyping = true;
    const typingIndicator = createTypingIndicator();
    elements.chatMessages.appendChild(typingIndicator);
    scrollToBottom();
    
    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                temperature: parseFloat(elements.temperature.value),
                use_context: elements.useContext.checked
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        typingIndicator.remove();
        
        const assistantMessage = createMessageElement(
            data.response,
            false,
            data.sources,
            data.timestamp
        );
        elements.chatMessages.appendChild(assistantMessage);
        scrollToBottom();
        
    } catch (error) {
        console.error('Error sending message:', error);
        typingIndicator.remove();
        
        const errorMessage = createMessageElement(
            'Sorry, I encountered an error. Please try again.',
            false
        );
        elements.chatMessages.appendChild(errorMessage);
        scrollToBottom();
    } finally {
        isTyping = false;
    }
}

elements.messageInput.addEventListener('input', (e) => {
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 150) + 'px';
    
    elements.sendButton.disabled = !e.target.value.trim() || isTyping;
});

elements.messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!elements.sendButton.disabled) {
            sendMessage();
        }
    }
});

elements.sendButton.addEventListener('click', sendMessage);

elements.temperature.addEventListener('input', (e) => {
    elements.tempValue.textContent = e.target.value;
});

checkHealth();
loadStats();

setInterval(checkHealth, 30000);
setInterval(loadStats, 60000);
