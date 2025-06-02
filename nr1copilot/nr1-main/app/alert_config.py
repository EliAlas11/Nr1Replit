
"""
Netflix-Grade Alert Configuration
Configure Slack/Email alerting for crash and failure notifications
"""

import logging
import os
from typing import Dict, Any, Optional
from .netflix_recovery_system import recovery_system
from .utils.health import health_monitor, AlertLevel

logger = logging.getLogger(__name__)


class AlertConfigurator:
    """Configure and setup alert channels for crash notifications"""
    
    def __init__(self):
        self.configured_channels = []
        
    async def setup_alerting(self):
        """Setup all configured alert channels"""
        logger.info("🔔 Setting up Netflix-grade alerting system...")
        
        # Setup email alerting if configured
        await self._setup_email_alerts()
        
        # Setup Slack alerting if configured
        await self._setup_slack_alerts()
        
        # Setup custom log alerting
        await self._setup_log_alerts()
        
        # Register alert callbacks with health monitor
        await self._register_health_monitor_alerts()
        
        # Test alerting system
        await self._test_alert_system()
        
        logger.info(f"✅ Alert system configured with {len(self.configured_channels)} channels")

    async def _setup_email_alerts(self):
        """Setup email alerting"""
        try:
            smtp_server = os.getenv('ALERT_SMTP_SERVER')
            email_user = os.getenv('ALERT_EMAIL_USER')
            email_pass = os.getenv('ALERT_EMAIL_PASS')
            recipients = os.getenv('ALERT_EMAIL_RECIPIENTS')
            
            if smtp_server and email_user and email_pass and recipients:
                logger.info("📧 Email alerting configured")
                self.configured_channels.append("email")
                
                # Test email configuration
                logger.info(f"Email recipients: {recipients}")
                logger.info(f"SMTP server: {smtp_server}")
            else:
                logger.info("📧 Email alerting not configured (missing environment variables)")
                logger.info("Set ALERT_SMTP_SERVER, ALERT_EMAIL_USER, ALERT_EMAIL_PASS, ALERT_EMAIL_RECIPIENTS to enable")
                
        except Exception as e:
            logger.error(f"Email alert setup failed: {e}")

    async def _setup_slack_alerts(self):
        """Setup Slack alerting"""
        try:
            webhook_url = os.getenv('SLACK_WEBHOOK_URL')
            
            if webhook_url:
                logger.info("💬 Slack alerting configured")
                self.configured_channels.append("slack")
                logger.info(f"Slack webhook configured: {webhook_url[:50]}...")
            else:
                logger.info("💬 Slack alerting not configured")
                logger.info("Set SLACK_WEBHOOK_URL to enable Slack notifications")
                
        except Exception as e:
            logger.error(f"Slack alert setup failed: {e}")

    async def _setup_log_alerts(self):
        """Setup enhanced log alerting"""
        try:
            # Configure structured logging for alerts
            alert_logger = logging.getLogger("netflix.alerts")
            
            # Create file handler for alerts if logs directory exists
            if os.path.exists("logs"):
                alert_handler = logging.FileHandler("logs/alerts.jsonl")
                alert_formatter = logging.Formatter(
                    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "alert": "%(message)s"}'
                )
                alert_handler.setFormatter(alert_formatter)
                alert_logger.addHandler(alert_handler)
                alert_logger.setLevel(logging.INFO)
            
            logger.info("📝 Enhanced log alerting configured")
            self.configured_channels.append("logs")
            
        except Exception as e:
            logger.error(f"Log alert setup failed: {e}")

    async def _register_health_monitor_alerts(self):
        """Register alert callbacks with health monitor"""
        try:
            # Register critical alert callback
            async def critical_alert_callback(level: AlertLevel, message: str):
                await self._send_critical_alert(level.value, message)
            
            # Register warning alert callback
            async def warning_alert_callback(level: AlertLevel, message: str):
                await self._send_warning_alert(level.value, message)
            
            # Register with health monitor if it supports alert callbacks
            if hasattr(health_monitor, 'register_alert_callback'):
                health_monitor.register_alert_callback(AlertLevel.CRITICAL, critical_alert_callback)
                health_monitor.register_alert_callback(AlertLevel.EMERGENCY, critical_alert_callback)
                health_monitor.register_alert_callback(AlertLevel.WARNING, warning_alert_callback)
                
                logger.info("🔗 Health monitor alert callbacks registered")
            
        except Exception as e:
            logger.error(f"Health monitor alert registration failed: {e}")

    async def _send_critical_alert(self, level: str, message: str):
        """Send critical alert through all channels"""
        try:
            alert_message = f"🚨 CRITICAL ALERT: {message}"
            
            # Log critical alert
            logger.critical(alert_message)
            
            # Send through recovery system (which handles all channels)
            await recovery_system._send_alert(level, message, {
                "application": "ViralClip Pro v10.0",
                "environment": os.getenv("ENVIRONMENT", "production"),
                "server": os.getenv("SERVER_NAME", "unknown")
            })
            
        except Exception as e:
            logger.error(f"Critical alert sending failed: {e}")

    async def _send_warning_alert(self, level: str, message: str):
        """Send warning alert through configured channels"""
        try:
            alert_message = f"⚠️ WARNING: {message}"
            
            # Log warning
            logger.warning(alert_message)
            
            # Send through recovery system
            await recovery_system._send_alert(level, message)
            
        except Exception as e:
            logger.error(f"Warning alert sending failed: {e}")

    async def _test_alert_system(self):
        """Test the alert system"""
        try:
            if self.configured_channels:
                logger.info("🧪 Testing alert system...")
                
                # Test recovery system alerting
                await recovery_system.test_alerting()
                
                logger.info("✅ Alert system test completed")
            else:
                logger.warning("⚠️ No alert channels configured - alerts will only appear in logs")
                
        except Exception as e:
            logger.error(f"Alert system test failed: {e}")

    def get_alert_configuration(self) -> Dict[str, Any]:
        """Get current alert configuration"""
        return {
            "configured_channels": self.configured_channels,
            "email_configured": "email" in self.configured_channels,
            "slack_configured": "slack" in self.configured_channels,
            "log_alerting": "logs" in self.configured_channels,
            "environment_variables": {
                "ALERT_SMTP_SERVER": bool(os.getenv('ALERT_SMTP_SERVER')),
                "ALERT_EMAIL_USER": bool(os.getenv('ALERT_EMAIL_USER')),
                "ALERT_EMAIL_PASS": bool(os.getenv('ALERT_EMAIL_PASS')),
                "ALERT_EMAIL_RECIPIENTS": bool(os.getenv('ALERT_EMAIL_RECIPIENTS')),
                "SLACK_WEBHOOK_URL": bool(os.getenv('SLACK_WEBHOOK_URL'))
            },
            "setup_instructions": {
                "email": "Set ALERT_SMTP_SERVER, ALERT_EMAIL_USER, ALERT_EMAIL_PASS, ALERT_EMAIL_RECIPIENTS",
                "slack": "Set SLACK_WEBHOOK_URL to your Slack webhook URL"
            }
        }


# Global alert configurator
alert_configurator = AlertConfigurator()

