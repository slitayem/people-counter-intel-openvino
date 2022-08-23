#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Logging module
@author: Saloua Litayem
"""
import os
import yaml
import traceback
import logging
import logging.config


class SetupLogging:
    """
    Set up logging
    """
    def __init__(self):
        self.default_config = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "logging.yml")
        getattr(logging, "DEBUG")

    def setup_logging(self, default_level=logging.INFO):
        """
        setup logging
        """
        path = self.default_config
        try:
            with open(path, 'r') as config_file:
                config_ = yaml.load(config_file, Loader=yaml.FullLoader)
                logging.config.dictConfig(config_)
        except OSError:
            logging.exception(
                "Error while loading logging configuration. Using default configs")
            logging.basicConfig(level=default_level)

def log_exception(exp):
    """
    Log exceptions
    :param exp: Exception object
    """
    logging.error(
        f"Exception: {exp.__class__} ({exp.__doc__}): {traceback.format_exc()}")
