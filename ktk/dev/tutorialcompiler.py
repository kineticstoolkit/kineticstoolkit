import nbformat
import subprocess
import os



class CellReader():

    def __init__(self, fid):
        self.fid = fid
        self.next_cell = 'markdown'
        self.line = ' '
        # Read up to next cell separator
        self.read_cell()

    def read_cell(self):
        self.temporary_cell = nbformat.v4.new_markdown_cell()
        self.temporary_cell.cell_type = self.next_cell

        if len(self.line) == 0:
            return None

        self.line = self.fid.readline()
        cell_text = ''
        while '# %%' not in self.line and len(self.line) > 0:

            # Add this line to the cell
            if self.temporary_cell.cell_type == 'code':
                if ('# %%' not in self.line):
                    cell_text += self.line

            elif self.temporary_cell.cell_type == 'markdown':
                if (('# %%' not in self.line) and
                         ('"""' not in self.line) and
                         ("'''" not in self.line)):
                    cell_text += self.line

            self.line = self.fid.readline()

        if len(self.line) == 0:
            self.next_cell = None
        elif 'markdown' in self.line:
            self.next_cell = 'markdown'
        elif 'code' in self.line:
            self.next_cell = 'code'
        elif 'exclude' in self.line:
            self.next_cell = 'exclude'
        else:
            self.next_cell = 'code'

        # Remove leading and trailing new lines
        if len(cell_text) > 0:
            while cell_text[0] == '\n':
                cell_text = cell_text[1:]
            while cell_text[-1] == '\n':
                cell_text = cell_text[:-1]
            self.temporary_cell.source = cell_text

        if self.temporary_cell.cell_type == 'code':
            self.temporary_cell.outputs = []
            self.temporary_cell.execution_count = None
        elif self.temporary_cell.cell_type == 'exclude':
            self.temporary_cell = self.read_cell()

        return self.temporary_cell


def compile(input_filename, output_filename):

    temp_filename = output_filename + '.ipynb'

    fid = open(input_filename)
    cell_reader = CellReader(fid)

    nb = nbformat.v4.new_notebook()

    cell = cell_reader.read_cell()
    while cell is not None:
        nb.cells.append(cell)
        cell = cell_reader.read_cell()

    fid.close()

    fid = open(temp_filename, 'w')
    nbformat.write(nb, fid)
    fid.close()

    subprocess.call(['jupyter-nbconvert', '--execute', temp_filename,
                     '--output', output_filename])

    os.remove(temp_filename)
