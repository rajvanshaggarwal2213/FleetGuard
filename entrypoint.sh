#!/bin/bash
# Entrypoint: initialize database then start Streamlit

set -e

echo "=== FleetGuard AI Startup ==="
echo "Waiting for PostgreSQL to be ready..."

# Extract host and port from DATABASE_URL
# Format: postgresql://user:pass@host:port/dbname
if [[ -n "$DATABASE_URL" ]]; then
    DB_HOST=$(echo $DATABASE_URL | sed -n 's/.*@\([^:]*\):.*/\1/p')
    DB_PORT=$(echo $DATABASE_URL | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')

    # Wait for PostgreSQL (max 30 seconds)
    for i in {1..30}; do
        if pg_isready -h "$DB_HOST" -p "$DB_PORT" 2>/dev/null; then
            echo "PostgreSQL is ready!"
            break
        fi
        echo "Waiting for PostgreSQL... ($i/30)"
        sleep 1
    done
else
    echo "WARNING: DATABASE_URL not set, skipping PostgreSQL wait"
fi

echo "Initializing database tables..."
# Set PYTHONPATH so Python can find the app module
export PYTHONPATH=/app:$PYTHONPATH
python /app/scripts/init_db.py

echo "Starting Streamlit..."
exec streamlit run /app/app/main.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true