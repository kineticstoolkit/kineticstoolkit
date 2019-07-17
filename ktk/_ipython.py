#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reformat the console output of dict in ipython.

This module reformats the console output of dict in ipython, so that instead
of just using __repr__, it displays a nicer list of keys with abbreviated
values if required, so that there is a maximum of one key per line. This is
very useful for nested dicts, since their __repr__ representation is
recursive and becomes unmanagable when the dict becomes larger.

It the module is not imported in the ipython interpreter, it is simply
ignored.

Author: Felix Chenier
Date: July 16th, 2019
felixchenier.com
"""
try:
    
    import IPython.lib.pretty as pretty

    def _ktk_format_dict(value, p, cycle):
        """Format a dict nicely on screen in ipython."""
        
        if cycle:
            p.pretty("...")
        else:
            
            # Find the widest field name
            the_keys = list(value.keys())
            the_lengths = []
            for the_key in the_keys:
                the_lengths.append(len(the_key))
            the_max_length = max(the_lengths)

            max_length_to_show = 77 - the_max_length
            
            # Start printing
            p.text(str(type(value)) + ':\n')
            if len(value.keys()) > 0:
                
                for the_key in the_keys:
                    
                    # Print the starting { if needed
                    if the_key == the_keys[0]:
                        p.text('{')
                    else:
                        p.text(' ')
                    
                    # Print the key
                    to_show = pretty.pretty(str(the_key))
                    p.text(to_show.rjust(the_max_length+2, ' '))
                    p.text(': ')
                    
                    # Print the value
                    to_show = pretty.pretty(value[the_key])
                    if len(to_show) <= max_length_to_show:
                        p.text(to_show)
                    else:
                        p.text(to_show[0:max_length_to_show-2] + '...')
    
                    # Print the ending } if needed
                    if the_key is the_keys[-1]:
                        p.text('}')
                    else:
                        p.text(',\n')
            else:
                p.text('{}')

    
    formatter = get_ipython().display_formatter.formatters['text/plain']
    formatter.for_type(dict, lambda n, p, cycle: _ktk_format_dict(n, p, cycle))

except:
    pass
