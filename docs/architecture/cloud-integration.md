# Cloud Integration

Architecture of Google Cloud Storage integration.

## GCS Structure

```text
gs://bucket-name/
├── DATASET/
│   ├── dataset1/
│   │   ├── images and annotations
│   ├── dataset2/
│   └── INFERENCE/
│       └── images for prediction
├── Archive/
│   └── timestamped results
└── dataset_info.json
```

## Sync Operations

### Download

Before processing:
- Downloads dataset from GCS to local
- Validates data integrity

### Upload

After processing:
- Uploads results to Archive
- Timestamps for versioning
- Preserves local copy

## Authentication

Uses Google Cloud SDK authentication:
- Service account credentials
- Application default credentials
- Environment variables

## Retry Logic

Robust network operations:
- Exponential backoff
- Configurable retries
- Timeout handling

## See Also

- [Getting Started](../getting-started.md)
- [Pipeline Overview](pipeline-overview.md)
