#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:12:17 2024

@author: felix
"""

import tkinter as tk
import matplotlib
import multiprocessing as mp
import os


from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Tkinter Matplotlib Demo")

        # prepare data
        data = {
            "Python": 11.27,
            "C": 11.16,
            "Java": 10.46,
            "C++": 7.5,
            "C#": 5.26,
        }
        languages = data.keys()
        popularity = data.values()

        # create a figure
        figure = Figure(figsize=(6, 4), dpi=100)

        # create FigureCanvasTkAgg object
        figure_canvas = FigureCanvasTkAgg(figure, self)

        # create the toolbar
        NavigationToolbar2Tk(figure_canvas, self)

        # create axes
        axes = figure.add_subplot()

        # create the barchart
        axes.bar(languages, popularity)
        axes.set_title("Top 5 Programming Languages")
        axes.set_ylabel("Popularity")

        figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def update(self):
        print("Allo")
        self.after(100, self.update)


def _gui_app(conn, cwd: str):
    matplotlib.use("TkAgg")
    app = App()
    app.after(100, app.update)
    app.mainloop()


if __name__ == "__main__":
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=_gui_app, args=(child_conn, os.getcwd()))
    p.start()
    # p.join()
