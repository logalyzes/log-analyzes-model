from confluent_kafka import Producer, Consumer, KafkaError, KafkaException, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic
import json
import sys
from tqdm import tqdm
from .trainer import Trainer

class KafkaManager:
    def __init__(self, bootstrap_servers: str, trainer: Trainer):
        self.bootstrap_servers = bootstrap_servers
        self.admin_client = AdminClient({'bootstrap.servers': self.bootstrap_servers})
        self.producer = Producer({'bootstrap.servers': self.bootstrap_servers})
        self.max_message_size = 1000000  # Set a default max message size of 1MB
        self.trainer = trainer

    def produce_message(self, topic_name: str, message: dict):
        """Produce a message to the Kafka topic."""
        message_str = json.dumps(message)
        message_size = sys.getsizeof(message_str.encode('utf-8'))
        if message_size > self.max_message_size:
            raise ValueError(f"Message size {message_size} exceeds the maximum allowed size {self.max_message_size}")
        self.producer.produce(topic_name, message_str.encode('utf-8'))
        self.producer.flush()

    def produce_messages(self, topic_name: str, messages: list):
        """Produce multiple messages to the Kafka topic."""
        for message in messages:
            self.produce_message(topic_name, message)
        self.producer.flush()

    def get_topic_size(self, topic_name: str) -> int:
        """Get the size of waiting consumable data on a topic."""
        consumer = Consumer({'bootstrap.servers': self.bootstrap_servers, 'group.id': 'size-checker'})
        partitions = consumer.list_topics(topic=topic_name).topics[topic_name].partitions
        total_size = 0

        for partition in partitions.values():
            tp = TopicPartition(topic_name, partition.id)
            low, high = consumer.get_watermark_offsets(tp)
            total_size += (high - low)

        consumer.close()
        return total_size

    def consume_messages(self, topic_name: str, batch_size: int = 100):
        """Consume messages from the Kafka topic."""
        consumer = Consumer({
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': 'trainer-group',
            'auto.offset.reset': 'earliest'
        })
        consumer.subscribe([topic_name])
        
        messages = []
        try:
            while True:
                batch = consumer.consume(batch_size, timeout=10.0)
                if not batch:
                    break
                for msg in batch:
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            continue
                        else:
                            raise KafkaException(msg.error())
                    message = json.loads(msg.value().decode('utf-8'))
                    messages.append(message)
                tqdm.write(f"Consumed {len(messages)} messages so far...")
        except Exception as e:
            print(f"Error while consuming messages: {e}")
        finally:
            consumer.close()
        return messages

    def train_model_if_needed(self, topic_name: str):
        """Check the topic size and train the model if necessary."""
        size = self.get_topic_size(topic_name)
        if size >= 5000:
            print("Starting training process...")
            messages = self.consume_messages(topic_name)
            self.trainer.train_model(messages)
            
    def create_consumer(self, group_id: str, topic_name: str):
        """Create a Kafka consumer."""
        consumer = Consumer({
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest'
        })
        consumer.subscribe([topic_name])
        return consumer
