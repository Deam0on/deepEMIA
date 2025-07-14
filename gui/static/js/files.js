// File operations JavaScript for deepEMIA FastAPI GUI

// Initialize file interface
document.addEventListener('DOMContentLoaded', function() {
    loadFilesList();
    setupFileUpload();
});

async function loadFilesList() {
    try {
        const response = await fetch('/api/files/list-local');
        const result = await response.json();
        
        const container = document.getElementById('filesList');
        
        if (result.files && result.files.length > 0) {
            const fileItems = result.files.map(file => `
                <div class="d-flex justify-content-between align-items-center border-bottom py-2">
                    <div>
                        <i class="bi bi-file-earmark"></i>
                        <strong>${file.name}</strong><br>
                        <small class="text-muted">${formatFileSize(file.size)} • ${file.path}</small>
                    </div>
                    <button class="btn btn-outline-primary btn-sm" onclick="downloadFile('${file.path}')">
                        <i class="bi bi-download"></i>
                    </button>
                </div>
            `).join('');
            
            container.innerHTML = `
                <div class="mb-2">
                    <strong>${result.count} files found</strong>
                </div>
                ${fileItems}
            `;
        } else {
            container.innerHTML = `
                <div class="alert alert-info">
                    <i class="bi bi-info-circle"></i>
                    No files found. Upload some files to get started.
                </div>
            `;
        }
    } catch (error) {
        console.error('Failed to load files:', error);
        const container = document.getElementById('filesList');
        container.innerHTML = `
            <div class="alert alert-warning">
                <i class="bi bi-exclamation-triangle"></i>
                Failed to load files. Using demo content.
            </div>
        `;
    }
}

function setupFileUpload() {
    const form = document.getElementById('fileUploadForm');
    const fileInput = document.getElementById('fileInput');
    const progressBar = document.getElementById('uploadProgress');
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const files = fileInput.files;
        const uploadPath = document.getElementById('uploadPath').value;
        
        if (files.length === 0) {
            showAlert('Please select files to upload', 'warning');
            return;
        }
        
        // Show progress
        progressBar.style.display = 'block';
        updateUploadProgress(0);
        
        try {
            const formData = new FormData();
            for (let i = 0; i < files.length; i++) {
                formData.append('files', files[i]);
            }
            formData.append('upload_path', uploadPath);
            
            const response = await fetch('/api/files/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            // Simulate progress completion
            updateUploadProgress(100);
            
            setTimeout(() => {
                progressBar.style.display = 'none';
                if (response.ok) {
                    showAlert(result.message, 'success');
                    form.reset();
                    loadFilesList(); // Refresh file list
                } else {
                    showAlert(`Error: ${result.detail}`, 'danger');
                }
            }, 500);
            
        } catch (error) {
            console.error('Upload failed:', error);
            progressBar.style.display = 'none';
            showAlert('Upload failed: ' + error.message, 'danger');
        }
    });
}

function updateUploadProgress(percent) {
    const progressBar = document.querySelector('#uploadProgress .progress-bar');
    const progressText = document.querySelector('#uploadProgress .progress-text');
    
    progressBar.style.width = `${percent}%`;
    progressText.textContent = `${percent}%`;
}

async function browseGCSFiles() {
    try {
        const response = await fetch('/api/files/browse');
        const result = await response.json();
        
        // For now, just show a summary
        const summary = `
            <div class="alert alert-info">
                <strong>GCS Browser Results:</strong><br>
                • Directories: ${result.directories?.length || 0}<br>
                • Images: ${result.images?.length || 0}<br>
                • Other files: ${result.other_files?.length || 0}<br>
                • Total: ${result.total_count || 0}
            </div>
        `;
        
        // Update the files list with GCS content
        document.getElementById('filesList').innerHTML = summary;
        showAlert('GCS files loaded', 'info');
        
    } catch (error) {
        console.error('Failed to browse GCS:', error);
        showAlert('Failed to browse GCS files', 'danger');
    }
}

async function downloadResults() {
    try {
        const response = await fetch('/api/files/create-zip', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prefix: 'results/' })
        });
        
        const result = await response.json();
        
        if (response.ok && result.download_path) {
            showAlert(`ZIP created: ${result.download_path}`, 'success');
        } else {
            showAlert(result.message, 'info');
        }
    } catch (error) {
        console.error('Failed to create ZIP:', error);
        showAlert('Failed to create results ZIP', 'danger');
    }
}

function refreshFiles() {
    loadFilesList();
    showAlert('Files list refreshed', 'info');
}

async function downloadFile(filePath) {
    try {
        const response = await fetch(`/api/files/download/${encodeURIComponent(filePath)}`);
        const result = await response.json();
        
        showAlert(result.message, 'info');
    } catch (error) {
        console.error('Download failed:', error);
        showAlert('Download failed', 'danger');
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
