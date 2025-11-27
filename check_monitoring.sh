#!/bin/bash

echo "🔍 Monitoring Stack Health Check"
echo "=================================="
echo ""

# ML API
echo -n "1. ML API (8000): "
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Running"
    curl -s http://localhost:8000/health | python3 -m json.tool
else
    echo "❌ Not responding"
fi

echo ""

# Prometheus
echo -n "2. Prometheus (9090): "
if curl -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
    echo "✅ Running"
else
    echo "❌ Not responding"
fi

echo ""

# Grafana
echo -n "3. Grafana (3000): "
if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
    echo "✅ Running"
    curl -s http://localhost:3000/api/health | python3 -m json.tool
else
    echo "❌ Not responding"
fi

echo ""
echo "=================================="
echo "📱 Access URLs:"
echo "  • API Docs:    http://localhost:8000/docs"
echo "  • Prometheus:  http://localhost:9090"
echo "  • Grafana:     http://localhost:3000"
echo "                 (login: admin/admin)"
