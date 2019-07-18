#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 20:36:28 2019

@author: felix
"""

import requests

url = 'http://localhost/~felix/biomec.uqam.ca/py/pywrapper.php'

result = requests.post(url, data={
                                'command': 'fetchall',
                                'projectlabel': 'dummyProject',
                                'username': 'dummyUser',
                                'password': 'dummyPassword'},
                                verify=False)

content = result.content.decode("utf-8")

exec(content)
