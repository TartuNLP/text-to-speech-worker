import json
import logging
import hashlib
from sys import getsizeof
from time import time, sleep

from pydantic import ValidationError

import pika
import pika.exceptions
from pika import credentials, BlockingConnection, ConnectionParameters

from tts_worker.schemas import Response, Request
from tts_worker.synthesizer import Synthesizer
from tts_worker.config import mq_config

logger = logging.getLogger(__name__)


class MQConsumer:
    def __init__(self, tts_worker: Synthesizer):
        """
        Initializes a RabbitMQ consumer class that listens for requests for a specific worker and responds to
        them.
        """
        self.tts_worker = tts_worker
        self.routing_keys = []
        self.queue_name = None
        self.channel = None

        self._generate_queue_config()

    def _generate_queue_config(self):
        """
        Produce routing keys with the following format: exchange_name.speaker_name
        """
        routing_keys = []
        for speaker in self.tts_worker.speakers:
            key = f'{mq_config.exchange}.{speaker}'
            routing_keys.append(key)
        self.routing_keys = sorted(routing_keys)
        hashed = hashlib.sha256(str(self.routing_keys).encode('utf-8')).hexdigest()[:8]
        self.queue_name = \
            f'{mq_config.exchange}.{self.tts_worker.model_name}_{hashed}'

    def start(self):
        """
        Connect to RabbitMQ and start listening for requests. Automatically tries to reconnect if the connection
        is lost.
        """
        while True:
            try:
                self._connect()
                logger.info('Ready to process requests.')
                self.channel.start_consuming()
            except pika.exceptions.AMQPConnectionError as e:
                logger.error(e)
                logger.info('Trying to reconnect in 5 seconds.')
                sleep(5)
            except KeyboardInterrupt:
                logger.info('Interrupted by user. Exiting...')
                self.channel.close()
                break

    def _connect(self):
        """
        Connects to RabbitMQ, (re)declares the exchange for the service and a queue for the worker binding
        any alternative routing keys as needed.
        """
        logger.info(f'Connecting to RabbitMQ server: {{host: {mq_config.host}, port: {mq_config.port}}}')
        connection = BlockingConnection(ConnectionParameters(
            host=mq_config.host,
            port=mq_config.port,
            credentials=credentials.PlainCredentials(
                username=mq_config.username,
                password=mq_config.password
            ),
            heartbeat=mq_config.heartbeat,
            client_properties={
                'connection_name': mq_config.connection_name
            }
        ))
        self.channel = connection.channel()
        self.channel.queue_declare(queue=self.queue_name, arguments={
            'x-expires': mq_config.x_expires,
        })
        self.channel.exchange_declare(exchange=mq_config.exchange, exchange_type='direct')

        for route in self.routing_keys:
            self.channel.queue_bind(exchange=mq_config.exchange, queue=self.queue_name,
                                    routing_key=route)

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=self.queue_name, on_message_callback=self._on_request, arguments={
            'x-priority': mq_config.x_priority
        })

    @staticmethod
    def _respond(channel: pika.adapters.blocking_connection.BlockingChannel, method: pika.spec.Basic.Deliver,
                 properties: pika.BasicProperties, response: Response):
        """
        Publish the response to the callback queue and acknowledge the original queue item.
        """

        channel.basic_publish(exchange='',
                              routing_key=properties.reply_to,
                              properties=pika.BasicProperties(
                                  correlation_id=properties.correlation_id,
                                  content_type='application/json'),
                              body=response.encode())

        channel.basic_ack(delivery_tag=method.delivery_tag)

    def _on_request(self, channel: pika.adapters.blocking_connection.BlockingChannel, method: pika.spec.Basic.Deliver,
                    properties: pika.BasicProperties, body: bytes):
        """
        Pass the request to the worker and return its response.
        """
        t1 = time()
        logger.info(f"Received request: {{id: {properties.correlation_id}, size: {getsizeof(body)} bytes}}")
        try:
            request = json.loads(body)
            request = Request(**request)
            response = self.tts_worker.process_request(request)
        except ValidationError as error:
            response = Response(status=f'Error parsing input: {str(error)}', status_code=422)
        except Exception as e:
            logger.exception(f'Unexpected error: {e}')
            response = Response(status_code=500, status="Unknown internal error.")

        response = response
        response_size = getsizeof(response)

        self._respond(channel, method, properties, response)
        t2 = time()

        logger.info(f"Request processed: {{id: {properties.correlation_id}, duration: {round(t2 - t1, 3)} s, "
                    f"size: {response_size} bytes}}")
