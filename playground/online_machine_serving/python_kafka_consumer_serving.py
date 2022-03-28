
if __name__ == '__main__':
    from kafka import KafkaConsumer

    consumer = KafkaConsumer('testTopic', bootstrap_servers='localhost:9092')

    while True:
        for msg in consumer:
            print(msg)