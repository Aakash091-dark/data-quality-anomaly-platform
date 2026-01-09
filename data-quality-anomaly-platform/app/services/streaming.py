import json
import time
import pandas as pd
from kafka import KafkaConsumer
from typing import List, Dict, Any
from app.services.data_quality import run_data_quality_checks
from app.services.feature_engineering import generate_features
from app.models.inference import load_model, score_anomaly
from app.services.alerting import check_and_alert

class StreamingService:
    def __init__(self, topic: str, bootstrap_servers: str = 'localhost:9092', group_id: str = 'anomaly-detector'):
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.model = None
        self.buffer = []
        self.buffer_size = 10 # Process every 10 records, or use time window
    
    def start(self):
        print(f"Connecting to Kafka topic {self.topic}...")
        try:
            consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                auto_offset_reset='latest',
                enable_auto_commit=True,
                group_id=self.group_id,
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            print("Connected. Listening for messages...")
            
            # Load model once
            self.model = load_model()
            
            for message in consumer:
                self.process_message(message.value)
                
        except Exception as e:
            print(f"Error connecting to Kafka: {e}")

    def process_message(self, data: Dict[str, Any]):
        self.buffer.append(data)
        
        if len(self.buffer) >= self.buffer_size:
            self.process_batch()

    def process_batch(self):
        print(f"Processing batch of {len(self.buffer)} records...")
        df = pd.DataFrame(self.buffer)
        
        # 1. Data Quality
        quality_result = run_data_quality_checks(df)
        print(f"Quality Check: {quality_result['data_type']}")
        
        # 2. Feature Engineering
        features = generate_features(df, quality_result["data_type"])
        
        # 3. Anomaly Detection
        if self.model:
            anomaly_result = score_anomaly(self.model, features)
            print(f"Anomaly Result: {anomaly_result}")
            
            # Check for anomaly and alert
            check_and_alert(anomaly_result, context=f"Stream Batch ({len(df)} records)")
        else:
            print("Model not loaded, skipping inference.")
            
        # Clear buffer
        self.buffer = []

if __name__ == "__main__":
    # Example usage
    service = StreamingService(topic="sensor_data")
    service.start()
