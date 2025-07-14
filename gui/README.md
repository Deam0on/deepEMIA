# Dear ImGui Web Interface for deepEMIA

This directory contains a minimal Dear ImGui-based web interface for the deepEMIA project, accessible via TCP port 8888 on your GCP VM's external IP.

## Quick Start

### 1. Start the GUI Server
```bash
# From the project root
./start_gui.sh

# Or manually:
python3 gui/dear_imgui_server.py
```

### 2. Access the Interface
Open your web browser and navigate to:
```
http://<YOUR-VM-EXTERNAL-IP>:8888
```

### 3. Firewall Configuration
Make sure port 8888 is open in your GCP firewall rules:
```bash
# Create firewall rule to allow port 8888
gcloud compute firewall-rules create allow-deareimgui-port \
    --allow tcp:8888 \
    --description "Allow Dear ImGui interface on port 8888"
```

## Features

- **Hello World Interface**: Simple Dear ImGui-style web interface
- **External Access**: Accessible via VM's external IP address
- **API Endpoints**: RESTful API for system information and testing
- **Responsive Design**: Works on desktop and mobile browsers
- **Real-time Updates**: Live status information and interactions

## Files

- `dear_imgui_server.py` - Main server implementation using Flask
- `README.md` - This documentation file

## API Endpoints

- `GET /` - Main Dear ImGui web interface
- `GET /api/info` - System information (platform, Python version, etc.)
- `GET /api/test` - Connection test endpoint
- `GET /api/hello?name=<name>` - Hello world API with optional name parameter

## Technical Details

- **Framework**: Flask web server serving Dear ImGui-style HTML/CSS/JS
- **Port**: 8888 (configurable)
- **Host**: 0.0.0.0 (accepts connections from any IP)
- **Interface Style**: Mimics Dear ImGui's look and feel using modern web technologies

## Future Enhancements

This is a minimal "hello world" implementation. Future versions could include:

- Integration with existing deepEMIA functionality
- Real dataset and model management
- Live training progress monitoring  
- Interactive parameter tuning
- File upload/download capabilities
- WebSocket support for real-time updates

## Troubleshooting

### Port Already in Use
```bash
# Check what's using port 8888
sudo netstat -tlnp | grep 8888

# Kill the process if needed
sudo kill -9 <PID>
```

### Permission Issues
```bash
# Make sure scripts are executable
chmod +x start_gui.sh
chmod +x gui/dear_imgui_server.py
```

### Dependencies Missing
```bash
# Install Flask if not available
pip install flask
```

## Development

To modify the interface, edit `dear_imgui_server.py`. The HTML template is embedded in the Python file for simplicity in this minimal implementation.

For production deployment, consider using a proper WSGI server like Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8888 gui.dear_imgui_server:app
```