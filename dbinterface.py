#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 20:36:28 2019

@author: felix
"""

import requests

url = 'https://biomec.uqam.ca/ktk/ktkwrapper.php'

requests.post(url, data={
        'command': 'fetchall',
        'projectlabel': 'FC_JL16E',
        'username': 'felix',
        'password': 'poutpout'},
        verify=False)
