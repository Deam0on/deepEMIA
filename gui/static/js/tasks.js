// Task management JavaScript for deepEMIA FastAPI GUI

let currentTaskId = null;
let taskPollingInterval = null;

// Initialize task interface
document.addEventListener('DOMContentLoaded', function() {
    // Load datasets for task form
    loadDatasets();
    
    // Setup form handlers
    setupTaskForm();
    
    // Setup threshold slider
    setupThresholdSlider();
    
    // Load task list
    loadTasksList();
});

async function loadDatasets() {
    try {
        const response = await fetch('/api/datasets/');
        const datasets = await response.json();
        
        const select = document.getElementById('datasetSelect');
        select.innerHTML = '<option value="">Select a dataset</option>';
        
        for (const [name, info] of Object.entries(datasets)) {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = `${name} (${info[2] ? info[2].join(', ') : 'no classes'})`;
            select.appendChild(option);
        }
    } catch (error) {
        console.error('Failed to load datasets:', error);
        showAlert('Failed to load datasets', 'danger');
    }
}

function setupTaskForm() {
    const form = document.getElementById('taskForm');
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        await runTask();
    });
}

function setupThresholdSlider() {
    const slider = document.getElementById('thresholdRange');
    const valueDisplay = document.getElementById('thresholdValue');
    
    slider.addEventListener('input', function() {
        valueDisplay.textContent = this.value;
    });
}

async function runTask() {
    const formData = {
        task: document.getElementById('taskSelect').value,
        dataset_name: document.getElementById('datasetSelect').value,
        threshold: parseFloat(document.getElementById('thresholdRange').value),
        rcnn_model: document.getElementById('rcnnSelect').value,
        visualize: document.getElementById('visualizeCheck').checked,
        download: document.getElementById('downloadCheck').checked,
        upload: document.getElementById('uploadCheck').checked
    };
    
    if (!formData.dataset_name) {
        showAlert('Please select a dataset', 'warning');
        return;
    }
    
    try {
        const response = await fetch('/api/tasks/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            currentTaskId = result.task_id;
            showTaskProgress(result.task_id);
            startTaskPolling(result.task_id);
            showAlert(`Task started: ${result.message}`, 'success');
        } else {
            showAlert(`Error: ${result.detail}`, 'danger');
        }
    } catch (error) {
        console.error('Failed to start task:', error);
        showAlert('Failed to start task', 'danger');
    }
}

function showTaskProgress(taskId) {
    document.getElementById('taskStatus').style.display = 'none';
    document.getElementById('taskProgress').style.display = 'block';
    
    // Reset progress
    updateProgress(0, 'starting');
}

function startTaskPolling(taskId) {
    if (taskPollingInterval) {
        clearInterval(taskPollingInterval);
    }
    
    taskPollingInterval = setInterval(async () => {
        await pollTaskStatus(taskId);
    }, 2000); // Poll every 2 seconds
}

async function pollTaskStatus(taskId) {
    try {
        const response = await fetch(`/api/tasks/status/${taskId}`);
        const status = await response.json();
        
        if (response.ok) {
            updateTaskDisplay(status);
            
            // Stop polling if task is finished
            if (['completed', 'failed', 'error', 'cancelled'].includes(status.status)) {
                clearInterval(taskPollingInterval);
                taskPollingInterval = null;
                loadTasksList(); // Refresh task list
            }
        }
    } catch (error) {
        console.error('Failed to poll task status:', error);
    }
}

function updateTaskDisplay(status) {
    // Update progress
    const progress = Math.max(0, Math.min(100, status.progress));
    updateProgress(progress, status.status);
    
    // Update command
    document.getElementById('taskCommand').textContent = status.command;
    
    // Update output
    document.getElementById('taskStdout').textContent = status.stdout || 'No output yet';
    document.getElementById('taskStderr').textContent = status.stderr || 'No errors';
    
    // Update status badge
    const statusElement = document.getElementById('taskStatus');
    statusElement.className = `alert alert-${getStatusClass(status.status)}`;
    statusElement.innerHTML = `<i class="bi bi-${getStatusIcon(status.status)}"></i> ${status.status.toUpperCase()}`;
    
    // Show elapsed time
    const elapsed = Math.round(status.elapsed_time);
    document.getElementById('progressPercent').textContent = 
        `${progress}% (${formatDuration(elapsed)} elapsed)`;
}

function updateProgress(percent, status) {
    const progressBar = document.getElementById('progressBar');
    const progressPercent = document.getElementById('progressPercent');
    
    progressBar.style.width = `${percent}%`;
    progressBar.className = `progress-bar progress-bar-${getStatusClass(status)}`;
    
    if (!progressPercent.textContent.includes('elapsed')) {
        progressPercent.textContent = `${percent}%`;
    }
}

function getStatusClass(status) {
    const statusMap = {
        'starting': 'info',
        'running': 'info',
        'completed': 'success',
        'failed': 'danger',
        'error': 'danger',
        'cancelled': 'warning'
    };
    return statusMap[status] || 'secondary';
}

function getStatusIcon(status) {
    const iconMap = {
        'starting': 'hourglass-split',
        'running': 'arrow-clockwise',
        'completed': 'check-circle',
        'failed': 'x-circle',
        'error': 'exclamation-triangle',
        'cancelled': 'dash-circle'
    };
    return iconMap[status] || 'info-circle';
}

async function loadTasksList() {
    try {
        const response = await fetch('/api/tasks/');
        const tasks = await response.json();
        
        const container = document.getElementById('tasksList');
        if (Object.keys(tasks).length === 0) {
            container.innerHTML = '<p class="text-muted">No tasks yet</p>';
            return;
        }
        
        const taskItems = Object.entries(tasks).map(([taskId, task]) => {
            const elapsed = formatDuration(Math.round(task.elapsed_time));
            return `
                <div class="card mb-2">
                    <div class="card-body p-2">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <span class="badge bg-${getStatusClass(task.status)}">${task.status}</span>
                                <strong>${task.task_type}</strong>
                                <small class="text-muted">(${elapsed})</small>
                            </div>
                            <div>
                                <button class="btn btn-outline-info btn-sm" onclick="viewTask('${taskId}')">
                                    <i class="bi bi-eye"></i> View
                                </button>
                                ${task.status === 'running' ? 
                                    `<button class="btn btn-outline-danger btn-sm" onclick="stopTask('${taskId}')">
                                        <i class="bi bi-stop"></i> Stop
                                    </button>` : ''
                                }
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = taskItems.join('');
        
    } catch (error) {
        console.error('Failed to load tasks list:', error);
    }
}

async function viewTask(taskId) {
    try {
        const response = await fetch(`/api/tasks/status/${taskId}`);
        const status = await response.json();
        
        if (response.ok) {
            currentTaskId = taskId;
            showTaskProgress(taskId);
            updateTaskDisplay(status);
            
            // Start polling if task is still running
            if (['starting', 'running'].includes(status.status)) {
                startTaskPolling(taskId);
            }
        }
    } catch (error) {
        console.error('Failed to view task:', error);
        showAlert('Failed to load task details', 'danger');
    }
}

async function stopTask(taskId) {
    if (!confirm('Are you sure you want to stop this task?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/tasks/${taskId}`, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showAlert(result.message, 'warning');
            loadTasksList();
        } else {
            showAlert(`Error: ${result.detail}`, 'danger');
        }
    } catch (error) {
        console.error('Failed to stop task:', error);
        showAlert('Failed to stop task', 'danger');
    }
}

// Utility function for duration formatting (from main.js)
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
