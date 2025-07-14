// Custom JavaScript for deepEMIA FastAPI GUI

document.addEventListener('DOMContentLoaded', function() {
    console.log('deepEMIA FastAPI GUI loaded successfully');
    
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // HTMX event listeners for better UX
    document.body.addEventListener('htmx:configRequest', function(evt) {
        // Add CSRF token if needed in the future
        console.log('HTMX request configured:', evt.detail);
    });
    
    document.body.addEventListener('htmx:responseError', function(evt) {
        console.error('HTMX Response Error:', evt.detail);
        showAlert('An error occurred while processing your request. Please try again.', 'danger');
    });
    
    document.body.addEventListener('htmx:timeout', function(evt) {
        console.error('HTMX Timeout:', evt.detail);
        showAlert('Request timed out. Please check your connection and try again.', 'warning');
    });
    
    document.body.addEventListener('htmx:beforeRequest', function(evt) {
        console.log('HTMX request starting:', evt.detail);
    });
    
    document.body.addEventListener('htmx:afterRequest', function(evt) {
        console.log('HTMX request completed:', evt.detail);
    });
    
    // Display server info
    displayServerInfo();
});

// Utility functions
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        <i class="bi bi-${getIconForType(type)}"></i> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Insert at top of main content
    const main = document.querySelector('main');
    if (main) {
        main.insertBefore(alertDiv, main.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                const bsAlert = new bootstrap.Alert(alertDiv);
                bsAlert.close();
            }
        }, 5000);
    }
}

function getIconForType(type) {
    const iconMap = {
        'success': 'check-circle',
        'danger': 'exclamation-triangle',
        'warning': 'exclamation-circle',
        'info': 'info-circle'
    };
    return iconMap[type] || 'info-circle';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDuration(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    
    if (h > 0) {
        return `${h}h ${m}m ${s}s`;
    } else if (m > 0) {
        return `${m}m ${s}s`;
    } else {
        return `${s}s`;
    }
}

function displayServerInfo() {
    // Get current hostname for external access info
    const hostname = window.location.hostname;
    const port = '8080';
    
    console.log(`FastAPI GUI accessible at:`);
    console.log(`- Local: http://localhost:${port}`);
    console.log(`- Network: http://${hostname}:${port}`);
    console.log(`- External: http://[VM_EXTERNAL_IP]:${port}`);
}

// Health check function
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        console.log('Health check:', data);
        return data.status === 'healthy';
    } catch (error) {
        console.error('Health check failed:', error);
        return false;
    }
}

// Periodic health check (every 30 seconds)
setInterval(async () => {
    const isHealthy = await checkHealth();
    if (!isHealthy) {
        showAlert('Connection to server lost. Please refresh the page.', 'warning');
    }
}, 30000);

// Future: WebSocket connection for real-time updates
function initWebSocket() {
    // Will implement WebSocket connection for real-time progress updates
    console.log('WebSocket initialization placeholder - will be implemented in Phase 2');
}

// Future: Authentication handling
function handleAuth() {
    console.log('Authentication handling placeholder - will be implemented in Phase 2');
}

// Future: File upload handling
function handleFileUpload() {
    console.log('File upload handling placeholder - will be implemented in Phase 2');
}

// Future: Progress tracking
function updateProgress(taskId, progress) {
    console.log(`Progress update for task ${taskId}: ${progress}%`);
}

// Smooth scrolling for anchor links
document.addEventListener('click', function(e) {
    if (e.target.matches('a[href^="#"]')) {
        e.preventDefault();
        const target = document.querySelector(e.target.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    }
});

// Add loading state to buttons
function addLoadingState(button, text = 'Loading...') {
    button.disabled = true;
    const originalText = button.innerHTML;
    button.innerHTML = `
        <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
        ${text}
    `;
    
    return function removeLoadingState() {
        button.disabled = false;
        button.innerHTML = originalText;
    };
}

// Theme toggle (for future dark mode support)
function toggleTheme() {
    console.log('Theme toggle placeholder - dark mode will be implemented later');
}

// Error boundary for unhandled errors
window.addEventListener('error', function(e) {
    console.error('Unhandled error:', e.error);
    showAlert('An unexpected error occurred. Please refresh the page if problems persist.', 'danger');
});

// Log successful initialization
console.log('deepEMIA FastAPI GUI JavaScript initialized successfully');
