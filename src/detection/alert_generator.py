"""
Alert generation and notification system
"""
import json
import time
from datetime import datetime
from typing import List, Dict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AlertGenerator:
    def __init__(self, config=None):
        self.config = config or {}
        self.alert_history = []
        self.alert_thresholds = {
            'critical': 90,
            'high': 75,
            'medium': 60,
            'low': 40
        }
    
    def generate_alert(self, account_id, risk_score, pattern_type=None, details=None):
        """Generate alert for suspicious activity"""
        
        # Determine alert level
        if risk_score >= self.alert_thresholds['critical']:
            level = "CRITICAL"
            priority = 1
        elif risk_score >= self.alert_thresholds['high']:
            level = "HIGH"
            priority = 2
        elif risk_score >= self.alert_thresholds['medium']:
            level = "MEDIUM"
            priority = 3
        else:
            level = "LOW"
            priority = 4
        
        alert = {
            'alert_id': f"ALT_{int(time.time())}_{account_id}",
            'timestamp': datetime.now().isoformat(),
            'account_id': account_id,
            'risk_score': risk_score,
            'alert_level': level,
            'priority': priority,
            'pattern_type': pattern_type,
            'details': details or {},
            'status': 'NEW',
            'resolved': False
        }
        
        self.alert_history.append(alert)
        return alert
    
    def get_active_alerts(self, min_priority=3):
        """Get all unresolved alerts"""
        return [a for a in self.alert_history 
                if not a['resolved'] and a['priority'] <= min_priority]
    
    def resolve_alert(self, alert_id, resolution_note=None):
        """Mark alert as resolved"""
        for alert in self.alert_history:
            if alert['alert_id'] == alert_id:
                alert['resolved'] = True
                alert['resolution_time'] = datetime.now().isoformat()
                alert['resolution_note'] = resolution_note
                return True
        return False
    
    def generate_summary_report(self):
        """Generate alert summary report"""
        active = self.get_active_alerts()
        
        summary = {
            'total_alerts': len(self.alert_history),
            'active_alerts': len(active),
            'critical_count': len([a for a in active if a['alert_level'] == 'CRITICAL']),
            'high_count': len([a for a in active if a['alert_level'] == 'HIGH']),
            'medium_count': len([a for a in active if a['alert_level'] == 'MEDIUM']),
            'low_count': len([a for a in active if a['alert_level'] == 'LOW']),
            'avg_risk_score': sum(a['risk_score'] for a in active) / len(active) if active else 0,
            'top_patterns': self._get_top_patterns(active)
        }
        
        return summary
    
    def _get_top_patterns(self, alerts):
        """Get most common fraud patterns"""
        patterns = {}
        for alert in alerts:
            pattern = alert.get('pattern_type', 'unknown')
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        return sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def export_alerts_json(self, filepath="alerts_export.json"):
        """Export alerts to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.alert_history, f, indent=2)
        print(f"Alerts exported to {filepath}")
    
    def create_alert_dashboard_data(self):
        """Create data structure for dashboard"""
        active_alerts = self.get_active_alerts()
        
        return {
            'metrics': {
                'total_active': len(active_alerts),
                'critical': len([a for a in active_alerts if a['alert_level'] == 'CRITICAL']),
                'high_risk_avg': sum(a['risk_score'] for a in active_alerts) / len(active_alerts) if active_alerts else 0
            },
            'alerts': active_alerts[-20:],  # Last 20 alerts
            'trends': self._calculate_trends()
        }
    
    def _calculate_trends(self):
        """Calculate alert trends over time"""
        # Group by hour
        hourly_counts = {}
        for alert in self.alert_history[-100:]:  # Last 100 alerts
            hour = alert['timestamp'][:13]  # YYYY-MM-DDTHH
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
        
        return {
            'hourly_data': hourly_counts,
            'trend_direction': 'increasing' if len(hourly_counts) > 0 else 'stable'
        }

class EmailNotifier:
    """Send email notifications for critical alerts"""
    
    def __init__(self, smtp_server, smtp_port, sender_email, sender_password):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
    
    def send_alert(self, recipient_email, alert):
        """Send alert email"""
        subject = f"[AML ALERT] {alert['alert_level']} - Account {alert['account_id']}"
        
        body = f"""
        Alert ID: {alert['alert_id']}
        Time: {alert['timestamp']}
        Level: {alert['alert_level']}
        Risk Score: {alert['risk_score']:.1f}
        Pattern: {alert.get('pattern_type', 'Unknown')}
        
        Details:
        {json.dumps(alert.get('details', {}), indent=2)}
        
        Please investigate immediately.
        """
        
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            return True
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False

class WebhookNotifier:
    """Send alerts via webhook"""
    
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
    
    def send_alert(self, alert):
        """Send alert to webhook"""
        import requests
        
        payload = {
            'alert_type': 'aml_fraud_detection',
            'alert': alert
        }
        
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Webhook failed: {e}")
            return False

if __name__ == "__main__":
    generator = AlertGenerator()
    
    # Test alert generation
    alert = generator.generate_alert(
        account_id="ACC_12345",
        risk_score=95,
        pattern_type="smurfing",
        details={"num_splits": 25, "total_amount": 150000}
    )
    
    print(f"Alert generated: {alert['alert_id']}")
    print(f"Summary: {generator.generate_summary_report()}")