# Monitoring Guide

## Overview

This project uses **Prometheus** for metrics collection and **Grafana** for visualization. The monitoring stack tracks model performance, API latency, error rates, and system health.

## Architecture
```
┌─────────────────────┐
│   ML API (FastAPI)  │
│   Port: 8000        │
│   - /metrics        │  ← Prometheus scrapes
│   - /predict/cnn    │
│   - /predict/rag    │
└──────────┬──────────┘
           │ HTTP GET /metrics (every 15s)
    ┌──────▼──────────┐
    │   Prometheus    │
    │   Port: 9090    │
    │   - Time-series │
    │   - Alerting    │
    └──────┬──────────┘
           │ PromQL queries
    ┌──────▼──────────┐
    │    Grafana      │
    │   Port: 3000    │
    │   - Dashboards  │
    │   - Alerts      │
    └─────────────────┘
```

## Quick Start

### 1. Start the monitoring stack
```bash
# Start all services
docker-compose up -d

# Or use Make
make monitoring-up
```

### 2. Access the services

- **ML API Documentation**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (login: admin/admin)

### 3. Generate test traffic
```bash
# Moderate traffic
python scripts/generate_traffic.py --rate 2 --duration 300

# Stress test
python scripts/generate_traffic.py --rate 10 --duration 60

# Custom rate
python scripts/generate_traffic.py --rate 5 --duration 120 --model cnn
```

### 4. View metrics

Open Grafana at http://localhost:3000:
1. Login with `admin/admin`
2. Navigate to "Dashboards"
3. Open "ML Models Monitoring"

## Metrics

### Application Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `ml_predictions_total` | Counter | Total number of predictions | `model_type`, `status` |
| `ml_prediction_latency_seconds` | Histogram | Prediction latency distribution | `model_type` |
| `ml_model_accuracy` | Gauge | Current model accuracy | `model_type` |
| `ml_active_requests` | Gauge | Number of active requests | `model_type` |
| `ml_errors_total` | Counter | Total errors | `model_type`, `error_type` |

### Example Queries

**Average prediction rate:**
```promql
rate(ml_predictions_total[5m])
```

**P95 latency:**
```promql
histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m]))
```

**Success rate:**
```promql
rate(ml_predictions_total{status="success"}[5m]) 
/ 
rate(ml_predictions_total[5m])
```

**Error rate:**
```promql
rate(ml_predictions_total{status="error"}[5m])
```

## Dashboards

### ML Models Overview

The main dashboard includes:

1. **Prediction Rate**: Requests per second by model type
2. **Active Requests**: Current concurrent requests
3. **Latency (P95/P99)**: Response time percentiles
4. **Model Accuracy**: Current accuracy gauge
5. **Error Rate**: Errors per second
6. **Total Predictions**: Cumulative prediction count

### Customizing Dashboards

1. Login to Grafana
2. Click on dashboard name → "Dashboard settings"
3. Modify panels or add new ones
4. Save changes

To export:
```bash
# Export dashboard JSON
curl -u admin:admin http://localhost:3000/api/dashboards/uid/ml-models \
  | jq '.dashboard' > monitoring/grafana/dashboards/custom-dashboard.json
```

## Alerting (Future Enhancement)

Configure alerts in `monitoring/prometheus/alerts.yml`:
```yaml
groups:
  - name: ml_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(ml_predictions_total{status="error"}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"
          
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m])) > 1.0
        for: 5m
        annotations:
          summary: "High prediction latency"
```

## Benchmarking

Run performance benchmarks:
```bash
# Basic benchmark
python scripts/benchmark.py

# Custom benchmark
python scripts/benchmark.py \
  --model both \
  --requests 1000 \
  --concurrency 10
```

Results include:
- Total requests and success rate
- Throughput (req/s)
- Latency percentiles (min, max, mean, P95, P99)
- Standard deviation

## Troubleshooting

### Services not starting
```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs ml-api
docker-compose logs prometheus
docker-compose logs grafana

# Restart services
docker-compose restart
```

### Prometheus not scraping

1. Check Prometheus targets: http://localhost:9090/targets
2. Verify ML API is running: `curl http://localhost:8000/health`
3. Check Prometheus logs: `docker-compose logs prometheus`

### Grafana can't connect to Prometheus

1. Check datasource configuration in Grafana
2. Verify network connectivity: `docker network inspect ml-cicd-showcase_ml-network`
3. Test Prometheus from Grafana container:
```bash
   docker exec -it grafana curl http://prometheus:9090/-/healthy
```

### No metrics showing up

1. Generate some traffic first: `python scripts/generate_traffic.py`
2. Wait 15-30 seconds for Prometheus to scrape
3. Check metrics endpoint: `curl http://localhost:8000/metrics`

## Production Considerations

For production deployment:

1. **Persistence**: Volumes are configured for Prometheus and Grafana data
2. **Security**: 
   - Change Grafana admin password
   - Use HTTPS for API
   - Restrict Prometheus/Grafana access
3. **Scaling**: Use Prometheus federation for multi-instance setups
4. **Alerting**: Configure Alertmanager for notifications
5. **Retention**: Adjust Prometheus retention period in config

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [FastAPI + Prometheus](https://github.com/prometheus/client_python)