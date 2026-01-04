# Changelog

All notable changes to deepEMIA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-01-04

### Added

- **Tile-Based Inference**: Process large images efficiently with configurable tile size and overlap
- **Model Ensemble**: Automatic multi-model ensemble combining predictions for improved accuracy
- **GPU Check Module**: Automatic GPU detection with interactive prompts and L4 GPU optimizations
- **Spatial Constraints**: Configurable containment and overlap rules for class relationships
- **Scale Bar OCR**: Automatic scale bar detection and calibration using EasyOCR
- **Dataset-Specific Configurations**: YAML-based dataset configuration files in `config/datasets/`
- **Iterative Inference**: Automatic multi-pass detection until no new instances found
- **Measurement Extraction**: Comprehensive morphological measurements (area, perimeter, axes, circularity, wavelength)
- **CLI Wizard**: Interactive command-line interface for guided workflow
- **Direct CLI**: Non-interactive CLI with full argument support
- **Streamlit GUI**: Web-based graphical interface (legacy)
- **Configuration Validation**: Comprehensive config validation with detailed error messages
- **ETA Tracking**: Persistent ETA data across runs for better time estimates
- **Safe File Operations**: Atomic writes with backup and rollback support

### Changed

- Restructured project layout with `src/` package organization
- Improved memory efficiency for large image processing
- Enhanced error handling with custom exception classes
- Updated inference pipeline to use tile-based processing by default
- Moved dataset configurations to dedicated `config/datasets/` directory

### Fixed

- Memory issues when processing very large electron microscopy images
- GPU memory management for L4 and similar GPU architectures
- Path handling for cross-platform compatibility

### Security

- Environment-variable based password hashing
- Path validation to prevent directory traversal
- Service account authentication for GCS

## [1.0.0] - Initial Release

### Added

- Core deep learning pipeline for electron microscopy analysis
- Detectron2-based instance segmentation
- Basic training and evaluation workflows
- COCO-style dataset support
- Google Cloud Storage integration
- Basic CLI interface

---

For detailed documentation, see the [docs/](docs/) directory.
