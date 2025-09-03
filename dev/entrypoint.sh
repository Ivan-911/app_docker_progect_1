#!/bin/sh
set -e

DB_HOST=${DB_HOST:-db}  # По умолчанию db, если не указано
DB_PORT=${DB_PORT:-3306}

echo "⏳ Жду, пока MySQL на $DB_HOST:$DB_PORT поднимется..."

while ! nc -z "$DB_HOST" "$DB_PORT"; do
  sleep 1
done

echo "✅ MySQL готов!"
exec "$@"
