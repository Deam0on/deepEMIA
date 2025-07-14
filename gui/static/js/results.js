// Results viewing JavaScript for deepEMIA FastAPI GUI

// Results tab functionality
const ResultsManager = {
    archiveFolders: [],
    currentArchiveContents: null,

    async initResults() {
        console.log("Initializing Results tab...");
        await this.loadArchiveFolders();
    },

    async loadArchiveFolders() {
        try {
            showSpinner('resultsList');
            
            const response = await fetch('/api/files/browse-archive', {
                headers: {
                    'X-Admin-Password': SHA256('admin'),
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            this.archiveFolders = data.archive_folders || [];
            
            if (data.error) {
                console.warn("Archive loading warning:", data.error);
                showAlert(`Warning: ${data.message || 'Could not load all archive folders'}`, 'warning');
            }
            
            this.renderArchiveFolders();
            
        } catch (error) {
            console.error("Error loading archive folders:", error);
            showAlert('Failed to load archive folders', 'error');
            document.getElementById('resultsList').innerHTML = 
                '<div class="alert alert-warning">Failed to load archive folders. Please check your connection.</div>';
        }
    },

    renderArchiveFolders() {
        const resultsList = document.getElementById('resultsList');
        
        if (this.archiveFolders.length === 0) {
            resultsList.innerHTML = `
                <div class="alert alert-info">
                    <i class="bi bi-info-circle me-2"></i>
                    No archive folders found. Results will appear here after running tasks.
                </div>`;
            return;
        }

        const foldersHtml = this.archiveFolders.map(folder => `
            <div class="col-md-6 col-lg-4 mb-3">
                <div class="card h-100">
                    <div class="card-body">
                        <h6 class="card-title">
                            <i class="bi bi-folder me-2"></i>
                            ${folder.name}
                        </h6>
                        <p class="card-text text-muted small">
                            Archive from: ${this.formatTimestamp(folder.timestamp)}
                        </p>
                        <button class="btn btn-primary btn-sm" 
                                onclick="ResultsManager.viewArchiveContents('${folder.name}')">
                            <i class="bi bi-eye me-1"></i>
                            View Contents
                        </button>
                    </div>
                </div>
            </div>
        `).join('');

        resultsList.innerHTML = `
            <div class="mb-3">
                <h5>Archive Folders (${this.archiveFolders.length})</h5>
                <p class="text-muted">Browse and download results from previous task runs</p>
            </div>
            <div class="row">
                ${foldersHtml}
            </div>
        `;
    },

    async viewArchiveContents(folderName) {
        try {
            showSpinner('resultsList');
            
            const response = await fetch(`/api/files/list-archive-contents/${folderName}`, {
                headers: {
                    'X-Admin-Password': SHA256('admin'),
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            this.currentArchiveContents = data;
            
            if (data.error) {
                console.warn("Archive contents loading warning:", data.error);
                showAlert(`Warning: Could not load all contents for ${folderName}`, 'warning');
            }
            
            this.renderArchiveContents(folderName);
            
        } catch (error) {
            console.error("Error loading archive contents:", error);
            showAlert('Failed to load archive contents', 'error');
        }
    },

    renderArchiveContents(folderName) {
        const resultsList = document.getElementById('resultsList');
        
        const { png_files = [], csv_files = [], other_files = [], total_files = 0 } = this.currentArchiveContents;
        
        const backButton = `
            <div class="mb-3">
                <button class="btn btn-outline-secondary" onclick="ResultsManager.loadArchiveFolders()">
                    <i class="bi bi-arrow-left me-1"></i>
                    Back to Archive Folders
                </button>
            </div>
        `;

        if (total_files === 0) {
            resultsList.innerHTML = backButton + `
                <div class="alert alert-info">
                    <i class="bi bi-info-circle me-2"></i>
                    No files found in archive folder "${folderName}".
                </div>`;
            return;
        }

        const renderFileSection = (title, files, icon, bgClass) => {
            if (files.length === 0) return '';
            
            const fileItems = files.map(file => `
                <div class="list-group-item d-flex justify-content-between align-items-center">
                    <div>
                        <i class="bi bi-${icon} me-2"></i>
                        ${file.name}
                    </div>
                    <button class="btn btn-sm btn-outline-primary" 
                            onclick="ResultsManager.downloadFile('${file.path}', '${file.name}')">
                        <i class="bi bi-download"></i>
                    </button>
                </div>
            `).join('');

            return `
                <div class="card mb-3">
                    <div class="card-header ${bgClass}">
                        <h6 class="mb-0">
                            <i class="bi bi-${icon} me-2"></i>
                            ${title} (${files.length})
                        </h6>
                    </div>
                    <div class="list-group list-group-flush">
                        ${fileItems}
                    </div>
                </div>
            `;
        };

        const contentHtml = `
            <div class="mb-3">
                <h5>Archive Contents: ${folderName}</h5>
                <p class="text-muted">Total files: ${total_files}</p>
            </div>
            
            ${renderFileSection('PNG Images', png_files, 'image', 'bg-light')}
            ${renderFileSection('CSV Data', csv_files, 'file-earmark-spreadsheet', 'bg-light')}
            ${renderFileSection('Other Files', other_files, 'file-earmark', 'bg-light')}
        `;

        resultsList.innerHTML = backButton + contentHtml;
    },

    async downloadFile(filePath, fileName) {
        try {
            showAlert(`Preparing download for ${fileName}...`, 'info');
            
            const response = await fetch(`/api/files/download-from-gcs?file_path=${encodeURIComponent(filePath)}`, {
                headers: {
                    'X-Admin-Password': SHA256('admin')
                }
            });
            
            if (!response.ok) {
                throw new Error('Download failed');
            }
            
            const data = await response.json();
            
            if (data.note) {
                // This is a placeholder response - show the information
                showAlert(`Download info: ${data.message}`, 'info');
                console.log("Download data:", data);
            } else {
                // This would handle actual file download
                showAlert(`Downloaded ${fileName}`, 'success');
            }
            
        } catch (error) {
            console.error("Error downloading file:", error);
            showAlert(`Failed to download ${fileName}`, 'error');
        }
    },

    formatTimestamp(timestamp) {
        // Try to parse timestamp and format it nicely
        try {
            // Assuming timestamp format like "2024-01-15_14-30-00"
            const dateStr = timestamp.replace(/_/g, ' ').replace(/-/g, ':');
            const date = new Date(dateStr);
            
            if (isNaN(date.getTime())) {
                return timestamp; // Return original if parsing fails
            }
            
            return date.toLocaleString();
        } catch (error) {
            return timestamp; // Return original if parsing fails
        }
    }
};

// Initialize results interface
document.addEventListener('DOMContentLoaded', function() {
    loadResultsList();
    
    // Initialize when Results tab becomes active
    const resultsTab = document.querySelector('button[data-bs-target="#results"]');
    if (resultsTab) {
        resultsTab.addEventListener('shown.bs.tab', function() {
            ResultsManager.initResults();
        });
    }
});

async function loadResultsList() {
    // This is called on page load - we'll let the tab handler manage results loading
    const container = document.getElementById('resultsList');
    container.innerHTML = `
        <div class="alert alert-info">
            <i class="bi bi-info-circle me-2"></i>
            Click the Results tab to browse and download archive files.
        </div>
    `;
}

function refreshResults() {
    ResultsManager.loadArchiveFolders();
    showAlert('Results refreshed', 'info');
}
