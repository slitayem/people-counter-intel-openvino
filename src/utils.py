"""
Helper functions module
@author: Saloua Litayem
"""

import logging
import json

import paho.mqtt.client as mqtt

import logger

def connect_mqtt(host, port, keepalive_interval=60):
    """ Connect to the MQTT client
    """
    client = None
    try:
        client = mqtt.Client()
        client.connect(host, port, keepalive_interval)
    except Exception as exp:
        logging.error("Error while connecting to the MQTT server")
        logger.log_exception(exp)

    return client


def human_readable_size(size, decimal_places=2):
    """Convert size in bytes to a human readable value"""
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if size < 1024.0 or unit == 'PiB':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


def publish_messages(mqtt_client, topics_messages):
    """Publish messages to the provided topics
    :param mqtt_client: MQTT client object
    :param topics_messages: Dictionary of topics messages
    mappping
    """
    for topic, message in topics_messages.items():
        mqtt_client.publish(topic, json.dumps(message))
