"""
Redis stream for real-time transaction processing
"""
import json
import time
import random
from datetime import datetime
import redis
import threading
from collections import deque

class RedisStreamClient:
    def __init__(self, host='localhost', port=6379, db=0):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.stream_key = 'upi_transactions'
        self.alerts_stream = 'fraud_alerts'
        self.running = False
        
    def publish_transaction(self, transaction):
        """Publish a transaction to Redis stream"""
        try:
            transaction['timestamp'] = datetime.now().isoformat()
            self.client.xadd(
                self.stream_key,
                {'data': json.dumps(transaction)}
            )
            return True
        except Exception as e:
            print(f"Error publishing: {e}")
            return False
    
    def consume_transactions(self, batch_size=10, block_ms=1000):
        """Consume transactions from Redis stream"""
        last_id = '0'
        while self.running:
            try:
                # Read new messages
                messages = self.client.xread(
                    {self.stream_key: last_id},
                    count=batch_size,
                    block=block_ms
                )
                
                if messages:
                    for stream, msg_list in messages:
                        for msg_id, msg_data in msg_list:
                            last_id = msg_id
                            # Get data from message
                            data = msg_data.get('data', '{}')
                            try:
                                transaction = json.loads(data)
                                yield transaction
                            except:
                                continue
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Consumer error: {e}")
                time.sleep(1)
    
    def publish_alert(self, alert):
        """Publish fraud alert to alerts stream"""
        alert['timestamp'] = datetime.now().isoformat()
        self.client.xadd(
            self.alerts_stream,
            {'data': json.dumps(alert)}
        )
    
    def consume_alerts(self):
        """Consume alerts from alerts stream"""
        last_id = '0'
        while self.running:
            try:
                messages = self.client.xread(
                    {self.alerts_stream: last_id},
                    count=10,
                    block=1000
                )
                
                if messages:
                    for stream, msg_list in messages:
                        for msg_id, msg_data in msg_list:
                            last_id = msg_id
                            data = msg_data.get('data', '{}')
                            try:
                                alert = json.loads(data)
                                yield alert
                            except:
                                continue
            except Exception as e:
                print(f"Alert consumer error: {e}")
                time.sleep(1)
    
    def start(self):
        self.running = True
    
    def stop(self):
        self.running = False
    
    def get_stats(self):
        """Get stream statistics"""
        try:
            stream_length = self.client.xlen(self.stream_key)
            alerts_length = self.client.xlen(self.alerts_stream)
            return {
                'transactions_in_stream': stream_length,
                'alerts_in_stream': alerts_length
            }
        except:
            return {'transactions_in_stream': 0, 'alerts_in_stream': 0}