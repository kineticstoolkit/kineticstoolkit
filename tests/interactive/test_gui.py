"""
Test the ktk_gui module.
"""

import ktk.gui as gui
import matplotlib.pyplot as plt


def test_message():
    gui.message('Test message')
    gui.message('')

    plt.plot([0, 1])
    gui.message('Test message')
    gui.message('')
    plt.close()

    plt.plot([0, 1])
    gui.message('Test message')
    plt.close()

    gui.message('Test message')
    gui.message('')


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
