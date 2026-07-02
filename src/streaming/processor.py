"""
Real-time transaction processor
"""
import threading
import json
from collections import defaultdict
import time

class StreamingProcessor:
    def __init__(self, redis_client, model=None):
        self.redis_client = redis_client
        self.model = model
        self.recent_transactions = defaultdict(list)
        self.alert_count = 0
        self.processed_count = 0
        self.running = False
        
    def process_transaction(self, transaction):
        """Process a single transaction"""
        self.processed_count += 1
        
        # Add to recent history
        account = transaction.get('from_account')
        if account:
            self.recent_transactions[account].append(transaction)
            # Keep last 20 transactions per account
            if len(self.recent_transactions[account]) > 20:
                self.recent_transactions[account].pop(0)
        
        # Check for fraud patterns
        risk_score = self.calculate_risk(transaction)
        
        if risk_score > 70:
            alert = {
                'transaction': transaction,
                'risk_score': risk_score,
                'alert_level': 'HIGH' if risk_score > 85 else 'MEDIUM',
                'processed_at': time.time()
            }
            return alert
        
        return None
    
    def calculate_risk(self, transaction):
        """Calculate risk score for transaction"""
        risk = 0
        
        # Check amount
        amount = transaction.get('amount', 0)
        if amount > 50000:
            risk += 20
        if amount > 100000:
            risk += 15
        
        # Check round numbers
        if amount > 0 and amount % 1000 == 0:
            risk += 15
        
        # Check velocity
        account = transaction.get('from_account')
        if account:
            recent = self.recent_transactions.get(account, [])
            if len(recent) > 5:
                risk += 25
            if len(recent) > 10:
                risk += 30
        
        # Check pattern type
        if transaction.get('is_fraud', 0) == 1:
            risk += 40
        
        return min(risk, 100)
    
    def run_consumer(self):
        """Run the consumer in a loop"""
        self.running = True
        self.redis_client.start()
        print("🚀 Streaming processor started...")
        
        for transaction in self.redis_client.consume_transactions():
            if not self.running:
                break
                
            alert = self.process_transaction(transaction)
            
            if alert:
                self.alert_count += 1
                # Publish alert
                self.redis_client.publish_alert(alert)
                print(f"🔴 Alert #{self.alert_count}: Risk {alert['risk_score']:.0f}% - "
                      f"{transaction.get('pattern_type', 'unknown')}")
                
                if self.alert_count % 10 == 0:
                    print(f"📊 Processed {self.processed_count} transactions, "
                          f"{self.alert_count} alerts generated")
    
    def run_alert_consumer(self):
        """Consume and display alerts"""
        self.running = True
        
        for alert in self.redis_client.consume_alerts():
            if not self.running:
                break
            self._display_alert(alert)
    
    def _display_alert(self, alert):
        """Display alert in console"""
        tx = alert.get('transaction', {})
        print(f"""
        ⚠️ FRAUD ALERT ⚠️
        Risk Score: {alert.get('risk_score', 0):.0f}%
        Level: {alert.get('alert_level', 'UNKNOWN')}
        Pattern: {tx.get('pattern_type', 'unknown')}
        Amount: ₹{tx.get('amount', 0):.2f}
        From: {tx.get('from_account', 'unknown')}
        To: {tx.get('to_account', 'unknown')}
        {'='*40}
        """)
    
    def stop(self):
        self.running = False
        if self.redis_client:
            self.redis_client.stop()

    def get_stats(self):
        """Get processing statistics"""
        return {
            'processed_count': self.processed_count,
            'alert_count': self.alert_count,
            'stream_stats': self.redis_client.get_stats()
        }