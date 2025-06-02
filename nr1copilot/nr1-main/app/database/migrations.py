
"""
Netflix-Level Database Migration System
Version-controlled schema management with rollback support
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import uuid4

from app.database.connection import db_manager

logger = logging.getLogger(__name__)


class Migration:
    """Individual migration definition"""
    
    def __init__(self, version: str, name: str, up_sql: str, down_sql: str = ""):
        self.version = version
        self.name = name
        self.up_sql = up_sql
        self.down_sql = down_sql
        self.id = str(uuid4())
    
    def __str__(self):
        return f"Migration {self.version}: {self.name}"


class NetflixLevelMigrationManager:
    """Enterprise database migration manager with version control"""
    
    def __init__(self):
        self.migrations: List[Migration] = []
        self.migrations_table = "schema_migrations"
        self._initialize_migrations()
    
    def _initialize_migrations(self):
        """Initialize all database migrations"""
        
        # Migration 001: Create schema_migrations table
        self.migrations.append(Migration(
            version="001",
            name="create_schema_migrations_table",
            up_sql="""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                version VARCHAR(20) NOT NULL UNIQUE,
                name VARCHAR(255) NOT NULL,
                applied_at TIMESTAMP DEFAULT NOW(),
                execution_time_ms BIGINT
            );
            CREATE INDEX IF NOT EXISTS idx_schema_migrations_version ON schema_migrations(version);
            CREATE INDEX IF NOT EXISTS idx_schema_migrations_applied_at ON schema_migrations(applied_at);
            """,
            down_sql="DROP TABLE IF EXISTS schema_migrations;"
        ))
        
        # Migration 002: Create users table
        self.migrations.append(Migration(
            version="002", 
            name="create_users_table",
            up_sql="""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                email VARCHAR(255) NOT NULL UNIQUE,
                username VARCHAR(100) NOT NULL UNIQUE,
                full_name VARCHAR(255),
                avatar_url TEXT,
                subscription_tier VARCHAR(50) DEFAULT 'free',
                status VARCHAR(20) DEFAULT 'active',
                preferences JSONB DEFAULT '{}',
                storage_used BIGINT DEFAULT 0,
                storage_limit BIGINT DEFAULT 1073741824,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                last_login TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
            CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
            CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);
            CREATE INDEX IF NOT EXISTS idx_users_subscription_tier ON users(subscription_tier);
            """,
            down_sql="DROP TABLE IF EXISTS users;"
        ))
        
        # Migration 003: Create videos table
        self.migrations.append(Migration(
            version="003",
            name="create_videos_table", 
            up_sql="""
            CREATE TABLE IF NOT EXISTS videos (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title VARCHAR(255) NOT NULL,
                description TEXT,
                filename VARCHAR(255) NOT NULL,
                file_path TEXT NOT NULL,
                file_size BIGINT NOT NULL,
                duration FLOAT,
                resolution VARCHAR(20),
                format VARCHAR(20),
                status VARCHAR(20) DEFAULT 'uploading',
                upload_progress FLOAT DEFAULT 0.0,
                processing_progress FLOAT DEFAULT 0.0,
                ai_analysis JSONB DEFAULT '{}',
                viral_score FLOAT,
                engagement_prediction JSONB DEFAULT '{}',
                thumbnail_url TEXT,
                tags TEXT[] DEFAULT '{}',
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                processed_at TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_videos_user_id ON videos(user_id);
            CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
            CREATE INDEX IF NOT EXISTS idx_videos_created_at ON videos(created_at);
            CREATE INDEX IF NOT EXISTS idx_videos_viral_score ON videos(viral_score);
            CREATE INDEX IF NOT EXISTS idx_videos_title ON videos USING gin(to_tsvector('english', title));
            """,
            down_sql="DROP TABLE IF EXISTS videos;"
        ))
        
        # Migration 004: Create projects table
        self.migrations.append(Migration(
            version="004",
            name="create_projects_table",
            up_sql="""
            CREATE TABLE IF NOT EXISTS projects (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title VARCHAR(255) NOT NULL,
                description TEXT,
                template_id UUID,
                brand_kit_id UUID,
                timeline_data JSONB DEFAULT '{}',
                assets JSONB DEFAULT '[]',
                settings JSONB DEFAULT '{}',
                status VARCHAR(20) DEFAULT 'draft',
                progress FLOAT DEFAULT 0.0,
                collaborators UUID[] DEFAULT '{}',
                permissions JSONB DEFAULT '{}',
                output_video_id UUID,
                export_settings JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                last_edited TIMESTAMP DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id);
            CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status);
            CREATE INDEX IF NOT EXISTS idx_projects_created_at ON projects(created_at);
            CREATE INDEX IF NOT EXISTS idx_projects_updated_at ON projects(updated_at);
            CREATE INDEX IF NOT EXISTS idx_projects_title ON projects USING gin(to_tsvector('english', title));
            """,
            down_sql="DROP TABLE IF EXISTS projects;"
        ))
        
        # Migration 005: Create social_publications table
        self.migrations.append(Migration(
            version="005",
            name="create_social_publications_table",
            up_sql="""
            CREATE TABLE IF NOT EXISTS social_publications (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                project_id UUID REFERENCES projects(id) ON DELETE SET NULL,
                video_id UUID REFERENCES videos(id) ON DELETE SET NULL,
                platform VARCHAR(50) NOT NULL,
                platform_account_id VARCHAR(255) NOT NULL,
                title VARCHAR(255) NOT NULL,
                description TEXT,
                hashtags TEXT[] DEFAULT '{}',
                scheduled_for TIMESTAMP,
                status VARCHAR(20) DEFAULT 'pending',
                platform_post_id VARCHAR(255),
                published_url TEXT,
                engagement_stats JSONB DEFAULT '{}',
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                published_at TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_social_publications_user_id ON social_publications(user_id);
            CREATE INDEX IF NOT EXISTS idx_social_publications_platform ON social_publications(platform);
            CREATE INDEX IF NOT EXISTS idx_social_publications_status ON social_publications(status);
            CREATE INDEX IF NOT EXISTS idx_social_publications_scheduled_for ON social_publications(scheduled_for);
            CREATE INDEX IF NOT EXISTS idx_social_publications_created_at ON social_publications(created_at);
            """,
            down_sql="DROP TABLE IF EXISTS social_publications;"
        ))
        
        # Migration 006: Create templates table
        self.migrations.append(Migration(
            version="006",
            name="create_templates_table",
            up_sql="""
            CREATE TABLE IF NOT EXISTS templates (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                creator_id UUID REFERENCES users(id) ON DELETE SET NULL,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                category VARCHAR(100) NOT NULL,
                template_data JSONB DEFAULT '{}',
                preview_url TEXT,
                thumbnail_url TEXT,
                usage_count INTEGER DEFAULT 0,
                rating FLOAT DEFAULT 0.0,
                viral_performance JSONB DEFAULT '{}',
                is_public BOOLEAN DEFAULT true,
                is_premium BOOLEAN DEFAULT false,
                price DECIMAL(10,2),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS idx_templates_category ON templates(category);
            CREATE INDEX IF NOT EXISTS idx_templates_is_public ON templates(is_public);
            CREATE INDEX IF NOT EXISTS idx_templates_is_premium ON templates(is_premium);
            CREATE INDEX IF NOT EXISTS idx_templates_usage_count ON templates(usage_count);
            CREATE INDEX IF NOT EXISTS idx_templates_rating ON templates(rating);
            CREATE INDEX IF NOT EXISTS idx_templates_name ON templates USING gin(to_tsvector('english', name));
            """,
            down_sql="DROP TABLE IF EXISTS templates;"
        ))
        
        # Migration 007: Create analytics table  
        self.migrations.append(Migration(
            version="007",
            name="create_analytics_table",
            up_sql="""
            CREATE TABLE IF NOT EXISTS analytics (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID REFERENCES users(id) ON DELETE SET NULL,
                event_type VARCHAR(100) NOT NULL,
                event_data JSONB DEFAULT '{}',
                session_id VARCHAR(255),
                user_agent TEXT,
                ip_address INET,
                video_id UUID,
                project_id UUID,
                template_id UUID,
                created_at TIMESTAMP DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS idx_analytics_user_id ON analytics(user_id);
            CREATE INDEX IF NOT EXISTS idx_analytics_event_type ON analytics(event_type);
            CREATE INDEX IF NOT EXISTS idx_analytics_created_at ON analytics(created_at);
            CREATE INDEX IF NOT EXISTS idx_analytics_session_id ON analytics(session_id);
            
            -- Partitioning for analytics (monthly partitions)
            CREATE OR REPLACE FUNCTION create_monthly_partition()
            RETURNS void AS $$
            DECLARE
                start_date date;
                end_date date;
                partition_name text;
            BEGIN
                start_date := date_trunc('month', CURRENT_DATE);
                end_date := start_date + interval '1 month';
                partition_name := 'analytics_' || to_char(start_date, 'YYYY_MM');
                
                EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF analytics
                    FOR VALUES FROM (%L) TO (%L)',
                    partition_name, start_date, end_date);
            END;
            $$ LANGUAGE plpgsql;
            """,
            down_sql="DROP TABLE IF EXISTS analytics; DROP FUNCTION IF EXISTS create_monthly_partition();"
        ))
        
        # Migration 008: Create system_health table
        self.migrations.append(Migration(
            version="008",
            name="create_system_health_table",
            up_sql="""
            CREATE TABLE IF NOT EXISTS system_health (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                service_name VARCHAR(100) NOT NULL,
                status VARCHAR(50) NOT NULL,
                response_time_ms FLOAT NOT NULL,
                cpu_usage FLOAT,
                memory_usage FLOAT,
                disk_usage FLOAT,
                details JSONB DEFAULT '{}',
                error_message TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS idx_system_health_service_name ON system_health(service_name);
            CREATE INDEX IF NOT EXISTS idx_system_health_status ON system_health(status);
            CREATE INDEX IF NOT EXISTS idx_system_health_created_at ON system_health(created_at);
            
            -- Auto-cleanup old health records (keep last 7 days)
            CREATE OR REPLACE FUNCTION cleanup_old_health_records()
            RETURNS void AS $$
            BEGIN
                DELETE FROM system_health 
                WHERE created_at < NOW() - INTERVAL '7 days';
            END;
            $$ LANGUAGE plpgsql;
            """,
            down_sql="DROP TABLE IF EXISTS system_health; DROP FUNCTION IF EXISTS cleanup_old_health_records();"
        ))
        
        # Migration 009: Add performance indexes and triggers
        self.migrations.append(Migration(
            version="009",
            name="add_performance_optimizations",
            up_sql="""
            -- Add updated_at triggers
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            
            CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
            
            CREATE TRIGGER update_videos_updated_at BEFORE UPDATE ON videos
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
            
            CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
            
            CREATE TRIGGER update_social_publications_updated_at BEFORE UPDATE ON social_publications
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
            
            CREATE TRIGGER update_templates_updated_at BEFORE UPDATE ON templates
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
            
            -- Add composite indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_videos_user_status ON videos(user_id, status);
            CREATE INDEX IF NOT EXISTS idx_projects_user_status ON projects(user_id, status);
            CREATE INDEX IF NOT EXISTS idx_social_publications_user_platform ON social_publications(user_id, platform);
            
            -- Add gin indexes for JSONB fields
            CREATE INDEX IF NOT EXISTS idx_videos_ai_analysis_gin ON videos USING gin(ai_analysis);
            CREATE INDEX IF NOT EXISTS idx_projects_timeline_data_gin ON projects USING gin(timeline_data);
            CREATE INDEX IF NOT EXISTS idx_analytics_event_data_gin ON analytics USING gin(event_data);
            """,
            down_sql="""
            DROP TRIGGER IF EXISTS update_users_updated_at ON users;
            DROP TRIGGER IF EXISTS update_videos_updated_at ON videos;
            DROP TRIGGER IF EXISTS update_projects_updated_at ON projects;
            DROP TRIGGER IF EXISTS update_social_publications_updated_at ON social_publications;
            DROP TRIGGER IF EXISTS update_templates_updated_at ON templates;
            DROP FUNCTION IF EXISTS update_updated_at_column();
            """
        ))
    
    async def ensure_migrations_table(self):
        """Ensure migrations table exists"""
        try:
            async with db_manager.get_connection() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        version VARCHAR(20) NOT NULL UNIQUE,
                        name VARCHAR(255) NOT NULL,
                        applied_at TIMESTAMP DEFAULT NOW(),
                        execution_time_ms BIGINT
                    );
                """)
                logger.info("✅ Schema migrations table ready")
        except Exception as e:
            logger.error(f"Failed to create migrations table: {e}")
            raise
    
    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions"""
        try:
            async with db_manager.get_connection() as conn:
                rows = await conn.fetch(
                    "SELECT version FROM schema_migrations ORDER BY version"
                )
                return [row["version"] for row in rows]
        except Exception as e:
            logger.error(f"Failed to get applied migrations: {e}")
            return []
    
    async def apply_migration(self, migration: Migration) -> bool:
        """Apply a single migration"""
        start_time = time.time()
        
        try:
            logger.info(f"Applying {migration}")
            
            async with db_manager.get_connection() as conn:
                async with conn.transaction():
                    # Execute migration SQL
                    await conn.execute(migration.up_sql)
                    
                    # Record migration
                    execution_time = int((time.time() - start_time) * 1000)
                    await conn.execute("""
                        INSERT INTO schema_migrations (version, name, execution_time_ms)
                        VALUES ($1, $2, $3)
                    """, migration.version, migration.name, execution_time)
            
            logger.info(f"✅ Applied {migration} in {execution_time}ms")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to apply {migration}: {e}")
            return False
    
    async def rollback_migration(self, migration: Migration) -> bool:
        """Rollback a single migration"""
        if not migration.down_sql:
            logger.error(f"No rollback SQL available for {migration}")
            return False
        
        try:
            logger.info(f"Rolling back {migration}")
            
            async with db_manager.get_connection() as conn:
                async with conn.transaction():
                    # Execute rollback SQL
                    await conn.execute(migration.down_sql)
                    
                    # Remove migration record
                    await conn.execute(
                        "DELETE FROM schema_migrations WHERE version = $1",
                        migration.version
                    )
            
            logger.info(f"✅ Rolled back {migration}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to rollback {migration}: {e}")
            return False
    
    async def migrate_up(self, target_version: Optional[str] = None) -> Dict[str, Any]:
        """Apply pending migrations up to target version"""
        await self.ensure_migrations_table()
        
        applied_migrations = await self.get_applied_migrations()
        pending_migrations = []
        
        for migration in self.migrations:
            if migration.version not in applied_migrations:
                pending_migrations.append(migration)
                if target_version and migration.version == target_version:
                    break
        
        if not pending_migrations:
            logger.info("✅ No pending migrations")
            return {"status": "up_to_date", "applied": 0}
        
        successful = 0
        failed = 0
        
        for migration in pending_migrations:
            if await self.apply_migration(migration):
                successful += 1
            else:
                failed += 1
                break  # Stop on first failure
        
        result = {
            "status": "completed" if failed == 0 else "failed",
            "applied": successful,
            "failed": failed,
            "total_pending": len(pending_migrations)
        }
        
        logger.info(f"Migration result: {result}")
        return result
    
    async def migrate_down(self, target_version: str) -> Dict[str, Any]:
        """Rollback migrations down to target version"""
        applied_migrations = await self.get_applied_migrations()
        rollback_migrations = []
        
        # Find migrations to rollback (in reverse order)
        for migration in reversed(self.migrations):
            if migration.version in applied_migrations and migration.version > target_version:
                rollback_migrations.append(migration)
        
        if not rollback_migrations:
            logger.info("✅ No migrations to rollback")
            return {"status": "up_to_date", "rolled_back": 0}
        
        successful = 0
        failed = 0
        
        for migration in rollback_migrations:
            if await self.rollback_migration(migration):
                successful += 1
            else:
                failed += 1
                break  # Stop on first failure
        
        result = {
            "status": "completed" if failed == 0 else "failed",
            "rolled_back": successful,
            "failed": failed,
            "total_rollback": len(rollback_migrations)
        }
        
        logger.info(f"Rollback result: {result}")
        return result
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status"""
        await self.ensure_migrations_table()
        
        applied_migrations = await self.get_applied_migrations()
        total_migrations = len(self.migrations)
        applied_count = len(applied_migrations)
        
        pending_migrations = []
        for migration in self.migrations:
            if migration.version not in applied_migrations:
                pending_migrations.append({
                    "version": migration.version,
                    "name": migration.name
                })
        
        return {
            "total_migrations": total_migrations,
            "applied_count": applied_count,
            "pending_count": len(pending_migrations),
            "current_version": applied_migrations[-1] if applied_migrations else None,
            "latest_version": self.migrations[-1].version,
            "applied_migrations": applied_migrations,
            "pending_migrations": pending_migrations
        }


# Global migration manager
migration_manager = NetflixLevelMigrationManager()
