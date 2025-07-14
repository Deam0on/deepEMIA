#!/bin/bash
# Start Dear ImGui Web Server for deepEMIA
# Usage: ./start_gui.sh

cd "$(dirname "$0")"

echo "🚀 Starting deepEMIA Dear ImGui Web Interface..."
echo "📱 Interface will be available at: http://<VM-EXTERNAL-IP>:8888"
echo "⚠️  Make sure port 8888 is open in your GCP firewall rules"
echo ""

# Check if Flask is available
python3 -c "import flask" 2>/dev/null || {
    echo "❌ Flask not found. Installing Flask..."
    pip install flask
}

# Start the server
python3 gui/dear_imgui_server.py