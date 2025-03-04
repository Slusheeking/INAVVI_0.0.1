#!/bin/bash
# Database Backup Script for Autonomous Trading System
# This script creates backups of both TimescaleDB and Redis databases

# Configuration
TIMESCALEDB_HOST="timescaledb"
TIMESCALEDB_PORT="5432"
TIMESCALEDB_USER="ats_user"
TIMESCALEDB_DB="ats_db"
REDIS_HOST="redis"
REDIS_PORT="6379"

# Backup directories
TIMESCALEDB_BACKUP_DIR="/timescaledb_backups"
REDIS_BACKUP_DIR="/redis_backups"

# Create backup directories if they don't exist
mkdir -p $TIMESCALEDB_BACKUP_DIR
mkdir -p $REDIS_BACKUP_DIR

# Get current date for backup filenames
DATE=$(date +"%Y%m%d_%H%M%S")
TIMESCALEDB_BACKUP_FILE="$TIMESCALEDB_BACKUP_DIR/timescaledb_backup_$DATE.sql.gz"
REDIS_BACKUP_FILE="$REDIS_BACKUP_DIR/redis_backup_$DATE.rdb"

# Log function
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1"
}

# Backup TimescaleDB
backup_timescaledb() {
    log "Starting TimescaleDB backup..."
    
    # Wait for TimescaleDB to be ready
    until pg_isready -h $TIMESCALEDB_HOST -p $TIMESCALEDB_PORT -U $TIMESCALEDB_USER
    do
        log "Waiting for TimescaleDB to be ready..."
        sleep 5
    done
    
    # Create backup
    log "Creating TimescaleDB backup: $TIMESCALEDB_BACKUP_FILE"
    PGPASSWORD=$POSTGRES_PASSWORD pg_dump -h $TIMESCALEDB_HOST -p $TIMESCALEDB_PORT -U $TIMESCALEDB_USER -d $TIMESCALEDB_DB -F c | gzip > $TIMESCALEDB_BACKUP_FILE
    
    if [ $? -eq 0 ]; then
        log "TimescaleDB backup completed successfully"
        
        # Cleanup old backups (keep last 7 days)
        find $TIMESCALEDB_BACKUP_DIR -name "timescaledb_backup_*.sql.gz" -type f -mtime +7 -delete
        log "Cleaned up old TimescaleDB backups"
    else
        log "TimescaleDB backup failed"
    fi
}

# Backup Redis
backup_redis() {
    log "Starting Redis backup..."
    
    # Wait for Redis to be ready
    until redis-cli -h $REDIS_HOST -p $REDIS_PORT ping
    do
        log "Waiting for Redis to be ready..."
        sleep 5
    done
    
    # Create backup
    log "Creating Redis backup: $REDIS_BACKUP_FILE"
    redis-cli -h $REDIS_HOST -p $REDIS_PORT --rdb $REDIS_BACKUP_FILE
    
    if [ $? -eq 0 ]; then
        log "Redis backup completed successfully"
        
        # Cleanup old backups (keep last 7 days)
        find $REDIS_BACKUP_DIR -name "redis_backup_*.rdb" -type f -mtime +7 -delete
        log "Cleaned up old Redis backups"
    else
        log "Redis backup failed"
    fi
}

# Main backup function
main() {
    log "Starting database backup process"
    
    # Backup TimescaleDB
    backup_timescaledb
    
    # Backup Redis
    backup_redis
    
    log "Database backup process completed"
}

# Run backup once immediately
main

# Set up cron job to run backup daily at 2 AM
if [ ! -f /etc/cron.d/database-backup ]; then
    echo "0 2 * * * root /backup_scripts/backup.sh >> /backup_scripts/backup.log 2>&1" > /etc/cron.d/database-backup
    chmod 0644 /etc/cron.d/database-backup
    log "Set up cron job for daily backups at 2 AM"
fi

# Keep container running
tail -f /dev/null