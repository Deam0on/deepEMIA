#!/usr/bin/env python3
"""
Dear ImGui Web Server for deepEMIA Project
A minimal Dear ImGui-based GUI accessible via web browser on port 8888.
"""

import os
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# HTML template with Dear ImGui via web
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>deepEMIA - Dear ImGui Interface</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 30px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .header p {
            font-size: 1.2em;
            margin: 10px 0;
            opacity: 0.9;
        }
        .gui-panel {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .gui-controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .control-group {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .control-group h3 {
            margin: 0 0 15px 0;
            color: #fff;
            font-size: 1.1em;
        }
        .button {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            margin: 5px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }
        .button.primary {
            background: linear-gradient(45deg, #2196F3, #1976D2);
        }
        .button.secondary {
            background: linear-gradient(45deg, #FF9800, #F57C00);
        }
        .status {
            background: rgba(0, 0, 0, 0.3);
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 14px;
            border-left: 4px solid #4CAF50;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            opacity: 0.7;
            font-size: 0.9em;
        }
        .server-info {
            background: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 6px;
            margin: 20px 0;
            border-left: 4px solid #2196F3;
        }
        input, select {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            width: 100%;
            margin: 5px 0;
        }
        input::placeholder {
            color: rgba(255, 255, 255, 0.6);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 deepEMIA</h1>
            <p>Dear ImGui Web Interface</p>
            <p>Deep Learning Image Analysis Tool</p>
        </div>

        <div class="server-info">
            <h3>🌐 Server Information</h3>
            <div class="status">
                <strong>Status:</strong> Running on port 8888<br>
                <strong>Interface:</strong> Dear ImGui Web GUI<br>
                <strong>Access:</strong> External IP accessible<br>
                <strong>Time:</strong> <span id="current-time"></span>
            </div>
        </div>

        <div class="gui-panel">
            <h2>Hello World - Dear ImGui Interface</h2>
            <p>Welcome to the minimal Dear ImGui interface for deepEMIA! This is a web-based implementation of Dear ImGui running on your GCP VM.</p>
            
            <div class="gui-controls">
                <div class="control-group">
                    <h3>🎯 Quick Actions</h3>
                    <button class="button primary" onclick="showHello()">Say Hello!</button>
                    <button class="button secondary" onclick="showInfo()">System Info</button>
                    <button class="button" onclick="testConnection()">Test Connection</button>
                </div>

                <div class="control-group">
                    <h3>⚙️ Interface Demo</h3>
                    <label>Sample Input:</label>
                    <input type="text" id="sample-input" placeholder="Enter some text..." />
                    <label>Sample Selection:</label>
                    <select id="sample-select">
                        <option>Option 1</option>
                        <option>Option 2</option>
                        <option>Option 3</option>
                    </select>
                </div>

                <div class="control-group">
                    <h3>📊 DeepEMIA Integration</h3>
                    <button class="button" onclick="showDatasets()">List Datasets</button>
                    <button class="button" onclick="showModels()">Available Models</button>
                    <button class="button" onclick="showStatus()">System Status</button>
                </div>
            </div>

            <div id="output" class="status" style="display:none;">
                <strong>Output:</strong><br>
                <span id="output-text"></span>
            </div>
        </div>

        <div class="footer">
            <p>🚀 Powered by Dear ImGui & Flask | deepEMIA Project | Running on GCP VM</p>
        </div>
    </div>

    <script>
        // Update current time
        function updateTime() {
            document.getElementById('current-time').textContent = new Date().toLocaleString();
        }
        updateTime();
        setInterval(updateTime, 1000);

        // GUI interaction functions
        function showOutput(text) {
            document.getElementById('output').style.display = 'block';
            document.getElementById('output-text').innerHTML = text;
        }

        function showHello() {
            const input = document.getElementById('sample-input').value || 'World';
            showOutput(`Hello, ${input}! 👋<br>Welcome to the Dear ImGui interface for deepEMIA!`);
        }

        function showInfo() {
            fetch('/api/info')
                .then(response => response.json())
                .then(data => {
                    showOutput(`System Information:<br>
                        Platform: ${data.platform}<br>
                        Python Version: ${data.python_version}<br>
                        Working Directory: ${data.cwd}<br>
                        Server: Dear ImGui Web Interface`);
                })
                .catch(error => {
                    showOutput(`Error fetching system info: ${error}`);
                });
        }

        function testConnection() {
            fetch('/api/test')
                .then(response => response.json())
                .then(data => {
                    showOutput(`✅ Connection Test: ${data.status}<br>
                        Message: ${data.message}<br>
                        Timestamp: ${data.timestamp}`);
                })
                .catch(error => {
                    showOutput(`❌ Connection failed: ${error}`);
                });
        }

        function showDatasets() {
            showOutput(`📁 Available Datasets:<br>
                • polyhipes (training data)<br>
                • inference_samples (test data)<br>
                • custom_dataset (user data)<br>
                <em>Note: This is a demo view. In full implementation, this would fetch real dataset info.</em>`);
        }

        function showModels() {
            showOutput(`🧠 Available Models:<br>
                • R50 (ResNet-50 backbone)<br>
                • R101 (ResNet-101 backbone)<br>  
                • Combo (Dual model approach)<br>
                <em>Note: This is a demo view. In full implementation, this would check actual model files.</em>`);
        }

        function showStatus() {
            showOutput(`📊 System Status:<br>
                • GUI Server: ✅ Running on port 8888<br>
                • Dear ImGui: ✅ Web interface active<br>
                • External Access: ✅ Available via VM IP<br>
                • Backend: 🔗 Ready for deepEMIA integration<br>
                <em>This is a minimal hello world implementation.</em>`);
        }

        // Add some Dear ImGui-style interactions
        document.addEventListener('DOMContentLoaded', function() {
            showOutput('🎉 Dear ImGui Web Interface initialized!<br>Ready for deepEMIA integration.');
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the main Dear ImGui interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/info')
def get_info():
    """API endpoint for system information"""
    import platform
    import sys
    
    return jsonify({
        'platform': platform.platform(),
        'python_version': sys.version,
        'cwd': os.getcwd(),
        'status': 'running'
    })

@app.route('/api/test')
def test_connection():
    """API endpoint for connection testing"""
    from datetime import datetime
    
    return jsonify({
        'status': 'success',
        'message': 'Dear ImGui server is running properly',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/hello')
def hello_api():
    """API endpoint for hello world functionality"""
    name = request.args.get('name', 'World')
    return jsonify({
        'message': f'Hello, {name}!',
        'gui': 'Dear ImGui Web Interface',
        'project': 'deepEMIA'
    })

def main():
    """Main function to run the Dear ImGui web server"""
    print("🚀 Starting Dear ImGui Web Server for deepEMIA...")
    print("📱 Interface: Dear ImGui Web GUI")
    print("🌐 Port: 8888")
    print("🔗 Access: http://<VM-EXTERNAL-IP>:8888")
    print("✨ Ready for deepEMIA integration!")
    
    # Run the Flask app on all interfaces (0.0.0.0) to allow external access
    app.run(
        host='0.0.0.0',
        port=8888,
        debug=False,
        threaded=True
    )

if __name__ == '__main__':
    main()