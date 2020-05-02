#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"

import nbformat
import subprocess
import os


class CellReader():

    def __init__(self, fid):
        self.fid = fid
        self.next_cell = ''
        self.line = ''
        self.exclude = False
        # Read up to next cell separator
        self.read_cell()

    def read_cell(self):
        """
        Read text up to triple-" or '# %%' and returns content as nb cell.
        """

        def end_formatting_temporary_cell():
            if self.next_cell == 'code':
                self.temporary_cell.source = cell_text
                self.temporary_cell.outputs = []
                self.temporary_cell.execution_count = None
            elif self.next_cell == 'markdown':
                self.temporary_cell.source = cell_text

        # Create the temporary cell
        self.temporary_cell = nbformat.v4.new_markdown_cell()
        self.temporary_cell.cell_type = self.next_cell

        end_reached = False
        cell_text = ''

        while end_reached is False:
            self.line = self.fid.readline()

            if len(self.line) == 0:
                if cell_text == '':
                    return None
                else:
                    end_formatting_temporary_cell()
                    return self.temporary_cell

            elif self.next_cell == '':
                # Not in any cell yet
                if '# %%' in self.line:
                    self.next_cell = 'code'
                    if 'exclude' in self.line:
                        self.exclude = True
                    else:
                        self.exclude = False
                    return self.read_cell()
                elif '"""' in self.line:
                    self.next_cell = 'markdown'
                    return self.read_cell()

            elif self.next_cell == 'code':
                if '# %%' in self.line:
                    end_formatting_temporary_cell()
                    self.next_cell = 'code'
                    if 'exclude' in self.line:
                        self.exclude = True
                    else:
                        self.exclude = False
                    return self.temporary_cell
                elif '"""' in self.line:
                    end_formatting_temporary_cell()
                    self.next_cell = 'markdown'
                    return self.temporary_cell
                else:
                    if self.exclude is False:
                        cell_text += self.line

            elif self.next_cell == 'markdown':
                if '"""' in self.line:
                    end_formatting_temporary_cell()
                    self.next_cell = 'code'
                    self.exclude = False
                    return self.temporary_cell
                else:
                    cell_text += self.line


def compile(input_filename, output_filename, header='', threads_running=[0]):
    temp_filename = output_filename + '.ipynb'

    fid = open(input_filename)
    cell_reader = CellReader(fid)

    nb = nbformat.v4.new_notebook()

    # Header
    if header != '':
        cell = nbformat.v4.new_markdown_cell()
        cell.source = header
        nb.cells.append(cell)

    # Contents
    cell = cell_reader.read_cell()
    while cell is not None:
        if len(cell.source) > 0 and cell.source != '\n':
            nb.cells.append(cell)
        cell = cell_reader.read_cell()

    fid.close()

    fid = open(temp_filename, 'w')
    nbformat.write(nb, fid)
    fid.close()

    subprocess.call(['jupyter-nbconvert', '--execute', temp_filename,
                     '--output', output_filename])

    os.remove(temp_filename)

    # Tell the caller that there's a thread less running.
    threads_running[0] -= 1
