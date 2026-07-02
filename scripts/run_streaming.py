"""
Run real-time transaction streaming
"""
import sys
import threading
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.streaming.redis_stream import RedisStreamClient
from src.streaming.generator import TransactionGenerator
from src.streaming.processor import StreamingProcessor

def main():
    print("=" * 60)
    print("📡 REAL-TIME UPI TRANSACTION STREAMING")
    print("=" * 60)
    
    # Connect to Redis
    print("\n[1] Connecting to Redis...")
    redis_client = RedisStreamClient()
    
    try:
        redis_client.client.ping()
        print("✅ Redis connected!")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("\nMake sure Redis is running:")
        print("  docker run --name redis-aml -p 6379:6379 -d redis:alpine")
        return
    
    # Start streaming components
    print("\n[2] Starting streaming components...")
    
    # Start generator in background thread
    generator = TransactionGenerator(redis_client)
    gen_thread = threading.Thread(
        target=generator.run,
        kwargs={'interval': 0.3, 'batch_size': 3},
        daemon=True
    )
    gen_thread.start()
    print("✅ Transaction generator started")
    
    # Start processor in background thread
    processor = StreamingProcessor(redis_client)
    proc_thread = threading.Thread(
        target=processor.run_consumer,
        daemon=True
    )
    proc_thread.start()
    print("✅ Streaming processor started")
    
    # Start alert consumer
    alert_thread = threading.Thread(
        target=processor.run_alert_consumer,
        daemon=True
    )
    alert_thread.start()
    print("✅ Alert consumer started")
    
    print("\n" + "=" * 60)
    print("🚀 STREAMING IS LIVE!")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    # Show stats every 10 seconds
    try:
        while True:
            time.sleep(10)
            stats = processor.get_stats()
            print(f"\n📊 Stats: {stats['processed_count']} processed, "
                  f"{stats['alert_count']} alerts generated")
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping streaming...")
        generator.stop()
        processor.stop()
        redis_client.stop()
        print("✅ Streaming stopped!")

if __name__ == "__main__":
    main()