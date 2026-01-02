#!/bin/bash
#
# ML CI/CD Showcase - Complete Demo Script
#
# This script demonstrates the full capabilities of the ML CI/CD showcase project:
# - Model training with MLflow tracking
# - API serving with FastAPI
# - Monitoring with Prometheus and Grafana
# - Traffic generation for metrics visualization
#
# Usage:
#   ./scripts/demo.sh [command]
#
# Commands:
#   all        - Run complete demo (default)
#   train      - Train models only
#   services   - Start services only
#   traffic    - Generate traffic only
#   cleanup    - Stop all services and cleanup
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TRAFFIC_DURATION=${TRAFFIC_DURATION:-60}
TRAFFIC_RATE=${TRAFFIC_RATE:-2}
MLFLOW_ENABLED=${MLFLOW_ENABLED:-true}

# Change to project root
cd "$PROJECT_ROOT"

# Helper functions
print_header() {
    echo ""
    echo -e "${CYAN}======================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}======================================${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=${3:-30}
    local attempt=1

    print_info "Waiting for $name to be ready..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            print_success "$name is ready!"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    echo ""
    print_error "$name failed to start after $max_attempts attempts"
    return 1
}

check_dependencies() {
    print_step "Checking dependencies..."

    local missing=()

    if ! command -v docker &> /dev/null; then
        missing+=("docker")
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        missing+=("docker-compose")
    fi

    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        missing+=("python")
    fi

    if [ ${#missing[@]} -ne 0 ]; then
        print_error "Missing required dependencies: ${missing[*]}"
        exit 1
    fi

    print_success "All dependencies found"
}

train_models() {
    print_header "Training Models"

    # Check if MLflow is enabled and running
    if [ "$MLFLOW_ENABLED" = "true" ]; then
        print_step "Starting MLflow server..."
        docker-compose up -d mlflow
        sleep 5

        if wait_for_service "http://localhost:5000" "MLflow" 15; then
            print_step "Training CNN with MLflow tracking..."
            python train.py --model cnn --epochs 3 --mlflow
        else
            print_warning "MLflow not available, training without tracking..."
            python train.py --model cnn --epochs 3
        fi
    else
        print_step "Training CNN without MLflow..."
        python train.py --model cnn --epochs 3
    fi

    print_success "Model training complete!"
}

start_services() {
    print_header "Starting Services"

    print_step "Starting all services (MLflow, API, Prometheus, Grafana)..."
    docker-compose up -d mlflow ml-api prometheus grafana

    echo ""
    print_info "Waiting for services to initialize..."
    sleep 10

    # Wait for each service
    wait_for_service "http://localhost:5000" "MLflow" 20 || true
    wait_for_service "http://localhost:8000/health" "ML API" 30
    wait_for_service "http://localhost:9090" "Prometheus" 20
    wait_for_service "http://localhost:3000" "Grafana" 20

    print_success "All services are running!"
    echo ""
    print_services_info
}

print_services_info() {
    echo -e "${CYAN}┌─────────────────────────────────────────────────────┐${NC}"
    echo -e "${CYAN}│              Services Available                      │${NC}"
    echo -e "${CYAN}├─────────────────────────────────────────────────────┤${NC}"
    echo -e "${CYAN}│${NC}  MLflow UI:      ${GREEN}http://localhost:5000${NC}             ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  API Docs:       ${GREEN}http://localhost:8000/docs${NC}        ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  API Metrics:    ${GREEN}http://localhost:8000/metrics${NC}     ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  Prometheus:     ${GREEN}http://localhost:9090${NC}             ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}  Grafana:        ${GREEN}http://localhost:3000${NC}             ${CYAN}│${NC}"
    echo -e "${CYAN}│${NC}                  ${YELLOW}(admin/admin)${NC}                     ${CYAN}│${NC}"
    echo -e "${CYAN}└─────────────────────────────────────────────────────┘${NC}"
}

generate_traffic() {
    print_header "Generating Traffic"

    print_info "Generating traffic for ${TRAFFIC_DURATION} seconds at ${TRAFFIC_RATE} req/s..."
    print_info "Watch the Grafana dashboard to see metrics!"
    echo ""

    python scripts/generate_traffic.py \
        --rate "$TRAFFIC_RATE" \
        --duration "$TRAFFIC_DURATION" \
        --model cnn \
        --verbose

    print_success "Traffic generation complete!"
}

test_api() {
    print_header "Testing API Endpoints"

    print_step "Testing health endpoint..."
    curl -s http://localhost:8000/health | python -m json.tool
    echo ""

    print_step "Testing CNN prediction..."
    # Generate a simple test image (all zeros)
    local test_data='{"data": [['$(python -c "print(','.join(['0.0']*784))")']]}'
    curl -s -X POST http://localhost:8000/predict/cnn \
        -H "Content-Type: application/json" \
        -d "$test_data" | python -m json.tool
    echo ""

    print_success "API tests complete!"
}

show_metrics() {
    print_header "Current Metrics"

    print_step "Fetching metrics from API..."
    curl -s http://localhost:8000/metrics | grep -E "^ml_" | head -20
    echo ""

    print_step "Sample Prometheus queries:"
    echo "  - Prediction rate: rate(ml_predictions_total[1m])"
    echo "  - P95 latency: histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m]))"
    echo "  - Error rate: rate(ml_errors_total[5m])"
}

cleanup() {
    print_header "Cleanup"

    print_step "Stopping all services..."
    docker-compose down

    print_success "Cleanup complete!"
}

run_full_demo() {
    print_header "ML CI/CD Showcase - Full Demo"

    echo -e "${YELLOW}"
    echo "This demo will:"
    echo "  1. Check dependencies"
    echo "  2. Train CNN model (with MLflow tracking)"
    echo "  3. Start all services (API, Prometheus, Grafana, MLflow)"
    echo "  4. Test API endpoints"
    echo "  5. Generate traffic for monitoring visualization"
    echo ""
    echo "Press Ctrl+C at any time to stop."
    echo -e "${NC}"

    read -p "Press Enter to continue..." || true
    echo ""

    check_dependencies
    train_models
    start_services
    test_api

    echo ""
    print_info "Opening dashboards in browser..."

    # Try to open browser (works on macOS and Linux)
    if command -v open &> /dev/null; then
        open "http://localhost:3000" 2>/dev/null || true
        open "http://localhost:5000" 2>/dev/null || true
    elif command -v xdg-open &> /dev/null; then
        xdg-open "http://localhost:3000" 2>/dev/null || true
        xdg-open "http://localhost:5000" 2>/dev/null || true
    fi

    generate_traffic
    show_metrics

    echo ""
    print_header "Demo Complete!"
    print_services_info
    echo ""
    print_info "Services are still running. Run './scripts/demo.sh cleanup' to stop them."
}

# Main
case "${1:-all}" in
    all)
        run_full_demo
        ;;
    train)
        check_dependencies
        train_models
        ;;
    services)
        check_dependencies
        start_services
        ;;
    traffic)
        generate_traffic
        ;;
    test)
        test_api
        ;;
    metrics)
        show_metrics
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  all        Run complete demo (default)"
        echo "  train      Train models only"
        echo "  services   Start services only"
        echo "  traffic    Generate traffic only"
        echo "  test       Test API endpoints"
        echo "  metrics    Show current metrics"
        echo "  cleanup    Stop all services"
        echo ""
        echo "Environment variables:"
        echo "  TRAFFIC_DURATION  Traffic generation duration in seconds (default: 60)"
        echo "  TRAFFIC_RATE      Requests per second (default: 2)"
        echo "  MLFLOW_ENABLED    Enable MLflow tracking (default: true)"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Run '$0 help' for usage information."
        exit 1
        ;;
esac
