// DOM Elements
const textInput = document.getElementById('textInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const resultCard = document.getElementById('resultCard');
const sentimentEmoji = document.getElementById('sentimentEmoji');
const sentimentLabel = document.getElementById('sentimentLabel');
const analyzedText = document.getElementById('analyzedText');
const charCount = document.getElementById('charCount');
const wordCount = document.getElementById('wordCount');
const charCountResult = document.getElementById('charCountResult');
const analysisTime = document.getElementById('analysisTime');
const exampleCards = document.querySelectorAll('.example-card');
const shareBtn = document.getElementById('shareBtn');

// API Configuration
const API_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? '/predict' 
    : `${window.location.protocol}//${window.location.host}/predict`;

// State
let startTime = 0;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    updateCharCount();
});

// Event Listeners Setup
function setupEventListeners() {
    analyzeBtn.addEventListener('click', handleAnalyze);
    clearBtn.addEventListener('click', handleClear);
    textInput.addEventListener('input', updateCharCount);
    textInput.addEventListener('keydown', handleKeyPress);
    
    exampleCards.forEach(card => {
        card.addEventListener('click', () => {
            textInput.value = card.dataset.text;
            updateCharCount();
            textInput.focus();
            // Auto-scroll to input
            textInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });
    });
    
    if (shareBtn) {
        shareBtn.addEventListener('click', handleShare);
    }
}

// Handle keyboard shortcuts
function handleKeyPress(e) {
    if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        handleAnalyze();
    }
}

// Update character count
function updateCharCount() {
    const text = textInput.value;
    const count = text.length;
    charCount.textContent = count;
    
    // Visual feedback for character limit
    if (count > 4500) {
        charCount.style.color = 'var(--error)';
    } else if (count > 4000) {
        charCount.style.color = 'var(--neutral)';
    } else {
        charCount.style.color = 'var(--primary)';
    }
}

// Handle analyze button click
async function handleAnalyze() {
    const text = textInput.value.trim();
    
    // Validation
    if (!text) {
        showNotification('Please enter some text to analyze', 'error');
        textInput.focus();
        return;
    }
    
    if (text.length < 5) {
        showNotification('Please enter at least 5 characters', 'error');
        return;
    }
    
    // Start loading
    setLoading(true);
    startTime = Date.now();
    
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sentiment: text
            })
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const data = await response.json();
        const endTime = Date.now();
        const duration = ((endTime - startTime) / 1000).toFixed(2);
        
        displayResult(data['Predicted Sentiment'], text, duration);
        showNotification('Analysis complete!', 'success');
        
    } catch (error) {
        console.error('Error:', error);
        showNotification('Failed to analyze. Please ensure the server is running.', 'error');
    } finally {
        setLoading(false);
    }
}

// Display results
function displayResult(sentiment, text, duration) {
    const isPositive = sentiment.toLowerCase() === 'positive';
    const isNegative = sentiment.toLowerCase() === 'negative';
    
    // Update emoji
    sentimentEmoji.textContent = isPositive ? '😊' : isNegative ? '😞' : '😐';
    
    // Update label
    sentimentLabel.textContent = sentiment;
    sentimentLabel.className = 'sentiment-label ' + 
        (isPositive ? 'positive' : isNegative ? 'negative' : 'neutral');
    
    // Update stats
    const words = text.trim().split(/\s+/).length;
    wordCount.textContent = words;
    charCountResult.textContent = text.length;
    analysisTime.textContent = duration + 's';
    
    // Update analyzed text
    analyzedText.textContent = `"${text}"`;
    
    // Show result card with animation
    resultCard.classList.remove('hidden');
    
    // Smooth scroll to result
    setTimeout(() => {
        resultCard.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'nearest' 
        });
    }, 100);
}

// Handle clear
function handleClear() {
    textInput.value = '';
    updateCharCount();
    resultCard.classList.add('hidden');
    textInput.focus();
    showNotification('Cleared!', 'success');
}

// Set loading state
function setLoading(isLoading) {
    if (isLoading) {
        analyzeBtn.classList.add('loading');
        analyzeBtn.disabled = true;
    } else {
        analyzeBtn.classList.remove('loading');
        analyzeBtn.disabled = false;
    }
}

// Show notification
function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existing = document.querySelector('.notification');
    if (existing) {
        existing.remove();
    }
    
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Add styles
    Object.assign(notification.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '16px 24px',
        borderRadius: '12px',
        background: type === 'error' ? 'var(--negative)' : 
                    type === 'success' ? 'var(--positive)' : 'var(--primary)',
        color: 'white',
        fontWeight: '600',
        fontSize: '0.95rem',
        boxShadow: '0 8px 25px rgba(0, 0, 0, 0.3)',
        zIndex: '10000',
        animation: 'slideInRight 0.3s ease',
        maxWidth: '300px'
    });
    
    document.body.appendChild(notification);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add notification animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Handle share
function handleShare() {
    const text = analyzedText.textContent;
    const sentiment = sentimentLabel.textContent;
    
    const shareText = `Sentiment Analysis Result:\n${text}\nSentiment: ${sentiment}`;
    
    if (navigator.share) {
        navigator.share({
            title: 'Sentiment Analysis',
            text: shareText
        }).catch(() => {
            copyToClipboard(shareText);
        });
    } else {
        copyToClipboard(shareText);
    }
}

// Copy to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copied to clipboard!', 'success');
    }).catch(() => {
        showNotification('Failed to copy', 'error');
    });
}

// Auto-resize textarea
textInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 400) + 'px';
});
