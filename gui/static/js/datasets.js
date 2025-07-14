// Dataset management JavaScript for deepEMIA FastAPI GUI

// Initialize dataset interface
document.addEventListener('DOMContentLoaded', function() {
    loadDatasetsList();
    setupDatasetForm();
});

async function loadDatasetsList() {
    try {
        const response = await fetch('/api/datasets/');
        const datasets = await response.json();
        
        const container = document.getElementById('datasetsList');
        if (Object.keys(datasets).length === 0) {
            container.innerHTML = '<p class="text-muted">No datasets found</p>';
            return;
        }
        
        const datasetItems = Object.entries(datasets).map(([name, info]) => {
            const [created, description, classes] = info;
            const classCount = classes ? classes.length : 0;
            const classText = classCount > 0 ? classes.join(', ') : 'No classes';
            
            return `
                <div class="card mb-3">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <h5 class="card-title">${name}</h5>
                                <p class="card-text">${description || 'No description'}</p>
                                <small class="text-muted">
                                    <i class="bi bi-calendar"></i> Created: ${created || 'Unknown'}<br>
                                    <i class="bi bi-tags"></i> Classes (${classCount}): ${classText}
                                </small>
                            </div>
                            <div class="btn-group-vertical">
                                <button class="btn btn-outline-primary btn-sm" onclick="editDataset('${name}')">
                                    <i class="bi bi-pencil"></i> Edit
                                </button>
                                <button class="btn btn-outline-danger btn-sm" onclick="deleteDataset('${name}')">
                                    <i class="bi bi-trash"></i> Delete
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = datasetItems.join('');
        
    } catch (error) {
        console.error('Failed to load datasets:', error);
        showAlert('Failed to load datasets', 'danger');
    }
}

function setupDatasetForm() {
    const form = document.getElementById('datasetForm');
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        await saveDataset();
    });
    
    // Setup class tags input
    setupClassesInput();
}

function setupClassesInput() {
    const input = document.getElementById('datasetClasses');
    const container = document.getElementById('classesContainer');
    
    input.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' || e.key === ',') {
            e.preventDefault();
            addClassTag(this.value.trim());
            this.value = '';
        }
    });
    
    input.addEventListener('blur', function() {
        if (this.value.trim()) {
            addClassTag(this.value.trim());
            this.value = '';
        }
    });
}

function addClassTag(className) {
    if (!className) return;
    
    const container = document.getElementById('classesContainer');
    
    // Check if class already exists
    const existing = container.querySelector(`[data-class="${className}"]`);
    if (existing) return;
    
    const tag = document.createElement('span');
    tag.className = 'badge bg-secondary me-1 mb-1';
    tag.setAttribute('data-class', className);
    tag.innerHTML = `
        ${className}
        <button type="button" class="btn-close btn-close-white ms-1" 
                onclick="removeClassTag('${className}')" style="font-size: 0.75em;"></button>
    `;
    
    container.appendChild(tag);
}

function removeClassTag(className) {
    const container = document.getElementById('classesContainer');
    const tag = container.querySelector(`[data-class="${className}"]`);
    if (tag) {
        tag.remove();
    }
}

function getClassesFromTags() {
    const container = document.getElementById('classesContainer');
    const tags = container.querySelectorAll('[data-class]');
    return Array.from(tags).map(tag => tag.getAttribute('data-class'));
}

async function saveDataset() {
    const name = document.getElementById('datasetName').value.trim();
    const description = document.getElementById('datasetDescription').value.trim();
    const classes = getClassesFromTags();
    const password = document.getElementById('adminPassword').value;
    
    if (!name) {
        showAlert('Please enter a dataset name', 'warning');
        return;
    }
    
    if (!password) {
        showAlert('Admin password required', 'warning');
        return;
    }
    
    const datasetData = {
        name: name,
        description: description,
        classes: classes
    };
    
    try {
        const response = await fetch('/api/datasets/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Admin-Password': password
            },
            body: JSON.stringify(datasetData)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showAlert(result.message, 'success');
            resetDatasetForm();
            loadDatasetsList();
        } else {
            showAlert(`Error: ${result.detail}`, 'danger');
        }
    } catch (error) {
        console.error('Failed to save dataset:', error);
        showAlert('Failed to save dataset', 'danger');
    }
}

function resetDatasetForm() {
    document.getElementById('datasetForm').reset();
    document.getElementById('classesContainer').innerHTML = '';
    document.getElementById('formTitle').textContent = 'Add New Dataset';
    document.getElementById('saveButton').innerHTML = '<i class="bi bi-plus"></i> Add Dataset';
}

async function editDataset(name) {
    try {
        const response = await fetch('/api/datasets/');
        const datasets = await response.json();
        
        console.log('Available datasets:', datasets); // Debug log
        console.log('Editing dataset:', name, datasets[name]); // Debug log
        
        if (datasets[name]) {
            const datasetInfo = datasets[name];
            let created, description, classes;
            
            // Handle different data structures
            if (Array.isArray(datasetInfo)) {
                // New structure: [created, description, classes]
                [created, description, classes] = datasetInfo;
            } else if (typeof datasetInfo === 'object') {
                // Object structure
                created = datasetInfo.created;
                description = datasetInfo.description;
                classes = datasetInfo.classes;
            } else {
                // Fallback
                created = 'Unknown';
                description = '';
                classes = [];
            }
            
            // Populate form
            document.getElementById('datasetName').value = name;
            document.getElementById('datasetDescription').value = description || '';
            
            // Clear and add class tags
            document.getElementById('classesContainer').innerHTML = '';
            if (classes && Array.isArray(classes)) {
                classes.forEach(className => addClassTag(className));
            }
            
            // Update form title and button
            document.getElementById('formTitle').textContent = 'Edit Dataset';
            document.getElementById('saveButton').innerHTML = '<i class="bi bi-save"></i> Update Dataset';
            
            // Scroll to form
            document.getElementById('datasetForm').scrollIntoView({ behavior: 'smooth' });
        } else {
            showAlert('Dataset not found', 'error');
        }
    } catch (error) {
        console.error('Failed to load dataset for editing:', error);
        showAlert('Failed to load dataset details: ' + error.message, 'danger');
    }
}

async function deleteDataset(name) {
    const password = prompt('Enter admin password to delete dataset:');
    if (!password) return;
    
    if (!confirm(`Are you sure you want to delete dataset "${name}"? This action cannot be undone.`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/datasets/${name}`, {
            method: 'DELETE',
            headers: {
                'X-Admin-Password': password
            }
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showAlert(result.message, 'success');
            loadDatasetsList();
        } else {
            showAlert(`Error: ${result.detail}`, 'danger');
        }
    } catch (error) {
        console.error('Failed to delete dataset:', error);
        showAlert('Failed to delete dataset', 'danger');
    }
}

// Admin functions
async function loadDatasetsFromGCS() {
    const password = prompt('Enter admin password to load datasets from GCS:');
    if (!password) return;
    
    try {
        const response = await fetch('/api/datasets/load-from-gcs', {
            method: 'POST',
            headers: {
                'X-Admin-Password': password
            }
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showAlert(`Loaded ${result.count} datasets from GCS`, 'success');
            loadDatasetsList();
        } else {
            showAlert(`Error: ${result.detail}`, 'danger');
        }
    } catch (error) {
        console.error('Failed to load datasets from GCS:', error);
        showAlert('Failed to load datasets from GCS', 'danger');
    }
}

async function saveDatasetsToGCS() {
    const password = prompt('Enter admin password to save datasets to GCS:');
    if (!password) return;
    
    try {
        const response = await fetch('/api/datasets/save-to-gcs', {
            method: 'POST',
            headers: {
                'X-Admin-Password': password
            }
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showAlert('Datasets saved to GCS successfully', 'success');
        } else {
            showAlert(`Error: ${result.detail}`, 'danger');
        }
    } catch (error) {
        console.error('Failed to save datasets to GCS:', error);
        showAlert('Failed to save datasets to GCS', 'danger');
    }
}
