# Pipeline Overview

High-level architecture of the deepEMIA pipeline.

## System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                      User Interfaces                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  CLI Wizard  │  │  Direct CLI  │  │  Web GUI     │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                    ┌────────▼─────────┐
                    │   main.py        │
                    │   (Orchestrator) │
                    └────────┬─────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼────┐         ┌─────▼─────┐       ┌─────▼─────┐
   │ Prepare │         │   Train   │       │ Inference │
   └────┬────┘         └─────┬─────┘       └─────┬─────┘
        │                    │                    │
        │              ┌─────▼─────┐              │
        │              │ Evaluate  │              │
        │              └─────┬─────┘              │
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼─────────┐
                    │   Data Layer     │
                    │ - Datasets       │
                    │ - Models         │
                    │ - Mappers        │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Utilities       │
                    │ - Config         │
                    │ - Logging        │
                    │ - GCS            │
                    │ - Measurements   │
                    └──────────────────┘
```

## Core Components

### 1. User Interfaces

Three entry points for different use cases:

- **CLI Wizard** (`cli_main.py`): Interactive guided interface with dataset config management
- **Direct CLI** (`main.py`): Command-line with explicit arguments
- **Web GUI** (`gui_legacy/streamlit_gui.py`): Browser-based visual interface

### 2. Orchestration Layer

**main.py** coordinates all operations:
- Parses arguments
- Loads configuration (with dataset-specific overrides)
- Performs GPU availability check
- Manages GCS sync
- Tracks ETA
- Handles errors

### 3. Pipeline Functions

Core operations in `src/functions/`:
- **Prepare**: Dataset splitting and registration
- **Train**: Model training with optimization
- **Evaluate**: Performance assessment
- **Inference**: Prediction and measurement

### 4. Data Layer

Dataset and model management in `src/data/`:
- Dataset registration with Detectron2
- Model loading and configuration
- Custom data augmentation

### 5. Utilities

Supporting functions in `src/utils/`:
- Configuration management
- Logging system
- GCS operations
- Mask processing
- Measurements
- Scale bar detection

## Data Flow

### Training Flow

```text
Dataset (GCS) → Download → Prepare → Register → Train → Evaluate → Upload
                   ↓          ↓         ↓         ↓        ↓         ↓
                Local FS   Split     COCO     Model   Metrics    GCS
```

### Inference Flow

```text
Images (GCS) → Download → Load Model → Predict → Post-process → Measure → Upload
                 ↓           ↓           ↓          ↓            ↓         ↓
              Local FS    Predictor   Masks    Filtered    CSV/Images   GCS
```

## Key Design Patterns

### 1. Configuration-Driven

All behavior controlled by `config.yaml`:
- Paths
- Thresholds
- Hyperparameters
- Optimization settings

### 2. Modular Architecture

Clear separation of concerns:
- Data handling separate from processing
- Utilities independent of pipeline logic
- Functions composable and reusable

### 3. Error Handling

Robust error management:
- Custom exception hierarchy
- Graceful degradation
- Detailed logging
- Retry logic for network operations

### 4. Cloud Integration

Seamless GCS integration:
- Automatic download before processing
- Automatic upload after completion
- Archival with timestamps

## Technology Stack

### Core Frameworks

- **PyTorch**: Deep learning framework
- **Detectron2**: Instance segmentation
- **Optuna**: Hyperparameter optimization

### Image Processing

- **OpenCV**: Image operations
- **scikit-image**: Advanced processing
- **EasyOCR**: Scale bar text extraction

### Infrastructure

- **Google Cloud Storage**: Data storage
- **Streamlit**: Web interface
- **YAML**: Configuration format

## Performance Considerations

### Memory Management

- Batch processing for large datasets
- Automatic cleanup of intermediate files
- Memory-efficient mask storage (RLE)

### GPU Optimization

- L4-specific tuning
- Automatic CPU fallback
- Model quantization for CPU inference

### Scalability

- Parallel image loading
- Parallel mask processing
- Streaming CSV writes

## Security Features

### Authentication

- Environment variable password hashes
- GCS service account credentials

### File Security

- Path validation
- Safe file operations
- Restricted access to allowed directories

## Extension Points

The architecture supports extension through:

1. **Custom Mappers**: Add new augmentation strategies
2. **Post-processing Hooks**: Custom mask filtering
3. **Measurement Plugins**: Additional particle properties
4. **Model Backends**: Support new architectures

## See Also

- [Training Pipeline](training-pipeline.md)
- [Inference Pipeline](inference-pipeline.md)
- [Cloud Integration](cloud-integration.md)
