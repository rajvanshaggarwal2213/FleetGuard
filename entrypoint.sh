#!/bin/bash
# Entrypoint: initialize database then start Streamlit

set -e

echo "=== FleetGuard AI Startup ==="
echo "Waiting for PostgreSQL to be ready..."

# Wait for PostgreSQL (max 30 seconds)
for i in {1..30}; do
    if pg_isready -h $(echo $DATABASE_URL | sed -n 's/.*@\([^:]*\):.*/\1/p') -p 5432 -U $(echo $DATABASE_URL | sed -n 's/.*:\/\/\([^:]*\):.*/\1/p') 2>/dev/null; then
        echo "PostgreSQL is ready!"
        break
    fi
    echo "Waiting for PostgreSQL... ($i/30)"
    sleep 1
done

echo "Initializing database tables..."
python scripts/init_db.py

echo "Starting Streamlit..."
exec streamlit run app/main.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true