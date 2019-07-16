#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reformat the console output of dict in ipython.

This module reformats the console output of dict in ipython, so that instead
of just using __repr__, it displays a nicer list of keys. This is very useful
for nested dicts, since their __repr__ representation is recursive and becomes
unmanagable when the dict becomes larger.

It the module is not imported in the ipython interpreter, it is simply
ignored.

Author: Felix Chenier
Date: July 16th, 2019
felixchenier.com
"""

def _ktk_format_dict(value, p, cycle):
    """Format a dict nicely on screen in ipython."""
    if cycle:
        p.pretty("...")
    else:
        p.text('dict with keys:\n')
        if len(value.keys()) > 0:
            for the_key in value.keys():
                p.text('    ')
                p.pretty(str(the_key))
                p.text('\n')
        else:
            p.text('<empty dict>')

# Format output for dict types in IPython
try:
    formatter = get_ipython().display_formatter.formatters['text/plain']
    formatter.for_type(dict, lambda n, p, cycle: _ktk_format_dict(n, p, cycle))
except:
    pass
