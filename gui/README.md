# deepEMIA FastAPI GUI

This directory contains the new FastAPI-powered web interface for deepEMIA, alongside the legacy Streamlit GUI.

## Quick Start

### Start FastAPI GUI (Recommended)
```bash
python start_gui.py --interface fastapi
```
- Access via VM external IP: `http://[VM_EXTERNAL_IP]:8888`
- API Documentation: `http://[VM_EXTERNAL_IP]:8888/api/docs`

### Start Streamlit GUI (Legacy)
```bash
python start_gui.py --interface streamlit
```
- Access via VM external IP: `http://[VM_EXTERNAL_IP]:8501`

## Dependencies

Install the new FastAPI dependencies:
```bash
pip install fastapi uvicorn jinja2 python-multipart
```

## Architecture

### FastAPI GUI (New)
- **Backend**: FastAPI for modern API endpoints
- **Frontend**: HTMX + Bootstrap for interactive UI
- **Templates**: Jinja2 templating system
- **Port**: 8888
- **Features**: 
  - Automatic API documentation
  - Better performance
  - Modern UI components
  - External IP access configured

### File Structure
```
gui/
├── fastapi_app.py          # Main FastAPI application
├── routes/                 # API route modules
│   ├── __init__.py
│   └── demo.py            # Demo routes for testing
├── templates/             # Jinja2 HTML templates
│   └── index.html         # Main page template
├── static/                # Static assets
│   ├── css/
│   │   └── main.css       # Custom styles
│   └── js/
│       └── main.js        # Custom JavaScript
├── streamlit_gui.py       # Legacy Streamlit interface
└── streamlit_functions.py # Streamlit utilities
```

## External Access Configuration

The FastAPI server is configured to listen on `0.0.0.0:8888`, making it accessible from:
- **Local**: `http://localhost:8888`
- **Network**: `http://[INTERNAL_IP]:8888`
- **External**: `http://[VM_EXTERNAL_IP]:8888`

Make sure your VM firewall allows incoming connections on port 8888.

## Development Roadmap

### Phase 1 - Foundation ✅
- [x] FastAPI backend setup
- [x] Bootstrap UI framework
- [x] HTMX integration
- [x] External IP access
- [x] Basic "Hello World" interface

### Phase 2 - Core Features 🚧
- [ ] Dataset management interface
- [ ] Model training controls
- [ ] Real-time progress tracking
- [ ] Results visualization
- [ ] Google Cloud Storage integration
- [ ] Admin authentication

### Phase 3 - Advanced Features
- [ ] WebSocket real-time updates
- [ ] Advanced file upload
- [ ] Dashboard analytics
- [ ] Multi-user support
- [ ] Dark mode theme

## Testing

1. Start the FastAPI server:
   ```bash
   python gui/fastapi_app.py
   ```

2. Test HTMX functionality by clicking the "Test HTMX" button on the main page

3. Check API documentation at `/api/docs`

4. Verify external access using your VM's external IP

## Migration from Streamlit

The new FastAPI GUI will gradually replace the Streamlit interface. For now, both interfaces are available:

- **Use FastAPI** for new features and better performance
- **Use Streamlit** for existing workflows during transition period

The FastAPI interface will eventually include all functionality from the Streamlit version plus additional features.
