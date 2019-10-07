"""
Test the ktk_gui module.
"""

import ktk.gui as gui


def test_message():
    gui.message('Test message')
    gui.message('')
