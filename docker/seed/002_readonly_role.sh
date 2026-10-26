#!/bin/bash
# Creates a dedicated read-only login for Dremio's source connection.
set -euo pipefail

RO_USER="${DREMIO_RO_DB_USER:-dremio_ro}"
RO_PASS="${DREMIO_RO_DB_PASSWORD:-ro-password}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '${RO_USER}') THEN
    CREATE ROLE ${RO_USER} LOGIN PASSWORD '${RO_PASS}';
  END IF;
END
\$\$;
GRANT USAGE ON SCHEMA public TO ${RO_USER};
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ${RO_USER};
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO ${RO_USER};
SQL
