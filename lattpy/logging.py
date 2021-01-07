# coding: utf-8
#
# This code is part of lattpy.
#
# Copyright (c) 2020, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import logging

logger = logging.getLogger("lattpy")

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
frmt = "[%(asctime)s] %(levelname)-8s - %(name)-15s - %(funcName)-25s -  %(message)s"
formatter = logging.Formatter(frmt, datefmt='%H:%M:%S')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
logger.setLevel(logging.WARNING)
