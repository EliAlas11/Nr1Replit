"""
Netflix Alert Configuration System v2.0
Production-grade alerting with Slack, email, and webhook integration
"""

import os
import logging
from typing import Dict, Any, List, Optional
from enum import Enum
import asyncio
import aiohttp
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high" 
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels"""
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"
    LOG = "log"


class ProductionAlertManager:
    """Netflix-grade production alert management"""

    def __init__(self):
        self.channels = {
            AlertChannel.SLACK: {
                "enabled": bool(os.getenv("SLACK_WEBHOOK_URL")),
                "webhook_url": os.getenv("SLACK_WEBHOOK_URL"),
                "channel": os.getenv("SLACK_CHANNEL", "#alerts"),
                "username": "ViralClip-Monitor"
            },
            AlertChannel.EMAIL: {
                "enabled": bool(os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true"),
                "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
                "smtp_port": int(os.getenv("SMTP_PORT", 587)),
                "username": os.getenv("ALERT_EMAIL_USER"),
                "password": os.getenv("ALERT_EMAIL_PASS"),
                "from_email": os.getenv("ALERT_FROM_EMAIL"),
                "to_emails": [email.strip() for email in os.getenv("ALERT_TO_EMAILS", "").split(",") if email.strip()]
            },
            AlertChannel.WEBHOOK: {
                "enabled": bool(os.getenv("CUSTOM_WEBHOOK_URL")),
                "webhook_url": os.getenv("CUSTOM_WEBHOOK_URL"),
                "secret": os.getenv("WEBHOOK_SECRET")
            }
        }

        # Alert rate limiting
        self.alert_history = {}
        self.rate_limit_window = 300  # 5 minutes
        self.max_alerts_per_window = 10

        logger.info("🚨 Production Alert Manager initialized")

    async def send_critical_alert(self, title: str, message: str, metadata: Dict[str, Any] = None) -> None:
        """Send critical alert - bypasses rate limiting"""
        await self.send_alert(AlertSeverity.CRITICAL, title, message, metadata)

    async def send_alert(self, severity: AlertSeverity, title: str, message: str, 
                        metadata: Dict[str, Any] = None) -> None:
        """Send alert through configured channels with rate limiting"""

        # Rate limiting check (except for critical alerts)
        if severity != AlertSeverity.CRITICAL and self._is_rate_limited(title):
            logger.warning(f"Alert rate limited: {title}")
            return

        alert_data = {
            "severity": severity.value,
            "title": title,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
            "service": "ViralClip Pro v10.0",
            "environment": os.getenv("ENVIRONMENT", "production"),
            "host": os.getenv("HOSTNAME", "unknown")
        }

        # Send to all enabled channels
        tasks = []

        if self.channels[AlertChannel.SLACK]["enabled"]:
            tasks.append(self._send_slack_alert(alert_data))

        if self.channels[AlertChannel.EMAIL]["enabled"] and severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            tasks.append(self._send_email_alert(alert_data))

        if self.channels[AlertChannel.WEBHOOK]["enabled"]:
            tasks.append(self._send_webhook_alert(alert_data))

        # Always log the alert
        self._log_alert(alert_data)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            logger.info(f"Alert sent to {success_count}/{len(tasks)} channels")

    def _is_rate_limited(self, alert_key: str) -> bool:
        """Check if alert should be rate limited"""
        current_time = datetime.utcnow().timestamp()

        if alert_key not in self.alert_history:
            self.alert_history[alert_key] = []

        # Clean old entries
        self.alert_history[alert_key] = [
            timestamp for timestamp in self.alert_history[alert_key]
            if current_time - timestamp < self.rate_limit_window
        ]

        # Check rate limit
        if len(self.alert_history[alert_key]) >= self.max_alerts_per_window:
            return True

        # Add current alert
        self.alert_history[alert_key].append(current_time)
        return False

    async def _send_slack_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send enhanced Slack alert"""
        try:
            webhook_url = self.channels[AlertChannel.SLACK]["webhook_url"]

            severity_emojis = {
                "low": "🟢",
                "medium": "🟡", 
                "high": "🔴",
                "critical": "💥"
            }

            color_map = {
                "low": "#36a64f",      # Green
                "medium": "#ff9500",   # Orange  
                "high": "#ff4444",     # Red
                "critical": "#8B0000"  # Dark Red
            }

            emoji = severity_emojis.get(alert_data["severity"], "⚠️")

            slack_message = {
                "username": self.channels[AlertChannel.SLACK]["username"],
                "icon_emoji": ":rotating_light:",
                "attachments": [{
                    "color": color_map.get(alert_data["severity"], "#cccccc"),
                    "title": f"{emoji} {alert_data['title']}",
                    "text": alert_data["message"],
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert_data["severity"].upper(),
                            "short": True
                        },
                        {
                            "title": "Environment", 
                            "value": alert_data["environment"],
                            "short": True
                        },
                        {
                            "title": "Service",
                            "value": alert_data["service"],
                            "short": True
                        },
                        {
                            "title": "Host",
                            "value": alert_data["host"],
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": alert_data["timestamp"],
                            "short": False
                        }
                    ],
                    "footer": "ViralClip Pro Monitoring",
                    "ts": int(datetime.utcnow().timestamp())
                }]
            }

            # Add metadata if present
            if alert_data["metadata"]:
                metadata_text = "\n".join([f"• {k}: {v}" for k, v in alert_data["metadata"].items()])
                slack_message["attachments"][0]["fields"].append({
                    "title": "Details",
                    "value": f"```{metadata_text}```",
                    "short": False
                })

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(webhook_url, json=slack_message) as response:
                    if response.status != 200:
                        logger.error(f"Slack alert failed: {response.status}")
                        raise Exception(f"Slack webhook returned {response.status}")
                    else:
                        logger.info("Slack alert sent successfully")

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            raise

    async def _send_email_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send production email alert"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            from email.mime.application import MIMEApplication

            config = self.channels[AlertChannel.EMAIL]

            if not config["to_emails"]:
                logger.warning("No email recipients configured")
                return

            msg = MIMEMultipart('alternative')
            msg['From'] = config["from_email"]
            msg['To'] = ", ".join(config["to_emails"])
            msg['Subject'] = f"🚨 {alert_data['severity'].upper()}: {alert_data['title']} - {alert_data['service']}"

            # HTML email body
            html_body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .alert-header {{ background-color: #f44336; color: white; padding: 15px; border-radius: 5px; }}
                    .alert-content {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                    .metadata {{ background-color: #e8e8e8; padding: 10px; margin: 10px 0; border-radius: 3px; font-family: monospace; }}
                    .footer {{ color: #666; font-size: 12px; margin-top: 20px; }}
                </style>
            </head>
            <body>
                <div class="alert-header">
                    <h2>🚨 System Alert: {alert_data['title']}</h2>
                </div>

                <div class="alert-content">
                    <p><strong>Service:</strong> {alert_data['service']}</p>
                    <p><strong>Environment:</strong> {alert_data['environment']}</p>
                    <p><strong>Severity:</strong> {alert_data['severity'].upper()}</p>
                    <p><strong>Host:</strong> {alert_data['host']}</p>
                    <p><strong>Time:</strong> {alert_data['timestamp']}</p>

                    <h3>Message:</h3>
                    <p>{alert_data['message']}</p>

                    {f'<h3>Additional Details:</h3><div class="metadata">{json.dumps(alert_data["metadata"], indent=2)}</div>' if alert_data['metadata'] else ''}
                </div>

                <div class="footer">
                    <p>This is an automated alert from ViralClip Pro monitoring system.</p>
                    <p>For support, contact your system administrator.</p>
                </div>
            </body>
            </html>
            """

            # Plain text version
            text_body = f"""
            SYSTEM ALERT: {alert_data['title']}

            Service: {alert_data['service']}
            Environment: {alert_data['environment']}
            Severity: {alert_data['severity'].upper()}
            Host: {alert_data['host']}
            Time: {alert_data['timestamp']}

            Message:
            {alert_data['message']}

            Additional Details:
            {json.dumps(alert_data['metadata'], indent=2) if alert_data['metadata'] else 'None'}

            ---
            This is an automated alert from ViralClip Pro monitoring system.
            """

            msg.attach(MIMEText(text_body, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))

            # Send email
            server = smtplib.SMTP(config["smtp_server"], config["smtp_port"])
            server.starttls()
            server.login(config["username"], config["password"])
            server.send_message(msg)
            server.quit()

            logger.info(f"Email alert sent to {len(config['to_emails'])} recipients")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            raise

    async def _send_webhook_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send alert to custom webhook"""
        try:
            webhook_url = self.channels[AlertChannel.WEBHOOK]["webhook_url"]
            secret = self.channels[AlertChannel.WEBHOOK]["secret"]

            headers = {"Content-Type": "application/json"}
            if secret:
                headers["X-Webhook-Secret"] = secret

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(webhook_url, json=alert_data, headers=headers) as response:
                    if response.status not in [200, 201, 202]:
                        logger.error(f"Webhook alert failed: {response.status}")
                        raise Exception(f"Webhook returned {response.status}")
                    else:
                        logger.info("Webhook alert sent successfully")

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            raise

    def _log_alert(self, alert_data: Dict[str, Any]) -> None:
        """Log alert to application logs with structured format"""
        log_level = {
            "low": logging.INFO,
            "medium": logging.WARNING, 
            "high": logging.ERROR,
            "critical": logging.CRITICAL
        }.get(alert_data["severity"], logging.WARNING)

        # Structured log entry
        log_entry = {
            "event_type": "alert",
            "severity": alert_data["severity"],
            "title": alert_data["title"],
            "message": alert_data["message"],
            "service": alert_data["service"],
            "environment": alert_data["environment"],
            "host": alert_data["host"],
            "timestamp": alert_data["timestamp"],
            "metadata": alert_data["metadata"]
        }

        logger.log(log_level, f"ALERT: {alert_data['title']} - {alert_data['message']}", extra={"structured": log_entry})

    async def test_alerts(self) -> Dict[str, bool]:
        """Test all configured alert channels"""
        results = {}

        test_alert = {
            "severity": "low",
            "title": "Alert System Test",
            "message": "This is a test alert to verify the alert system is working correctly.",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {"test": True, "version": "v10.0"},
            "service": "ViralClip Pro v10.0",
            "environment": os.getenv("ENVIRONMENT", "test"),
            "host": os.getenv("HOSTNAME", "test-host")
        }

        # Test Slack
        if self.channels[AlertChannel.SLACK]["enabled"]:
            try:
                await self._send_slack_alert(test_alert)
                results["slack"] = True
            except Exception as e:
                logger.error(f"Slack test failed: {e}")
                results["slack"] = False

        # Test Email
        if self.channels[AlertChannel.EMAIL]["enabled"]:
            try:
                await self._send_email_alert(test_alert)
                results["email"] = True
            except Exception as e:
                logger.error(f"Email test failed: {e}")
                results["email"] = False

        # Test Webhook
        if self.channels[AlertChannel.WEBHOOK]["enabled"]:
            try:
                await self._send_webhook_alert(test_alert)
                results["webhook"] = True
            except Exception as e:
                logger.error(f"Webhook test failed: {e}")
                results["webhook"] = False

        return results


# Global production alert manager
alert_manager = ProductionAlertManager()