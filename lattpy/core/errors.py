# coding: utf-8
"""
Created on 13 May 2020
author: Dylan Jones
"""


class ConfigurationError(Exception):

    def __init__(self, msg='', hint=''):
        if hint:
            msg += f'({hint})'
        super().__init__(msg)

