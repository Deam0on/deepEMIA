# deepEMIA Documentation

Welcome to the deepEMIA (Deep Learning Electron Microscopy Image Analysis) documentation. This tool provides a complete pipeline for scientific image analysis using deep learning.

## Quick Links

- [Getting Started](getting-started.md) - Installation and setup
- [User Guide](user-guide.md) - How to use deepEMIA
- [Configuration](configuration.md) - Configuration reference
- [API Reference](api/overview.md) - Technical API documentation
- [Architecture](architecture/pipeline-overview.md) - System design and architecture
- [Examples](examples/basic-workflow.md) - Practical usage examples
- [Scale Bar Troubleshooting](examples/scalebar-troubleshooting.md) - Debug scale bar detection issues
- [Development](development/contributing.md) - Contributing to deepEMIA

## What is deepEMIA?

deepEMIA is a modular, end-to-end computer vision pipeline designed for scientific image analysis, particularly electron microscopy images. It provides:

- **Dataset Management**: Prepare and organize datasets for training
- **Model Training**: Train instance segmentation models with Detectron2
- **Model Evaluation**: Comprehensive evaluation with COCO metrics
- **Inference**: Run predictions on new images with advanced post-processing
- **Measurement Analysis**: Automatic extraction of geometric and physical properties
- **Cloud Integration**: Seamless integration with Google Cloud Storage

## Key Features

- Interactive CLI wizard for easy operation
- Web-based Streamlit interface
- Hyperparameter optimization with Optuna
- Multi-scale inference for improved detection
- Scale bar detection and automatic unit conversion
- Spatial constraint filtering
- CPU and GPU support

## Documentation Structure

### For Users
- **Getting Started**: First-time setup and installation
- **User Guide**: Day-to-day usage, CLI commands, GUI interface
- **Configuration**: Complete configuration file reference
- **Examples**: Real-world usage scenarios and workflows

### For Developers
- **API Reference**: Detailed function and class documentation
- **Architecture**: System design, pipeline structure, and data flow
- **Development**: Contributing guidelines, code style, and testing

## Getting Help

- Check the [Troubleshooting](user-guide.md#troubleshooting) section
- Review [Examples](examples/basic-workflow.md) for common workflows
- Open an issue on [GitHub](https://github.com/Deam0on/deepEMIA/issues)

## License

This project is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) - see LICENSE.md for details.
