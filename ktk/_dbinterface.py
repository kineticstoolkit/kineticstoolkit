"""
Module that manages the DBInterface class.

Author: Felix Chenier
Date: April 2020
"""

import ktk
import requests
import os
from io import StringIO
import pandas as pd
import warnings


class DBInterface():
    """Interface for Felix Chenier's BIOMEC database."""

    @property
    def participants(self):
        table = self._tables['Participants']['ParticipantLabel']
        return table.drop_duplicates().to_list()

    @property
    def sessions(self):
        table = self._tables['Sessions']['SessionLabel']
        return table.drop_duplicates().to_list()

    @property
    def trials(self):
        table = self._tables['Trials']['TrialLabel']
        return table.drop_duplicates().to_list()

    @property
    def files(self):
        table = self._tables['Files']['FileLabel']
        return table.drop_duplicates().to_list()


    def __init__(self, project_label, user='', password='', root_folder='',
                 url='https://felixchenier.uqam.ca/biomec',
                 filters=dict()):

        # Simple assignations
        self.project_label = project_label
        self.url = url

        # Get username and password if not supplied
        if user == '':
            self.user, self._password = ktk.gui.get_credentials()
        else:
            self.user = user
            self._password = password

        # Assign root folder
        if root_folder == '':
            ktk.gui.message('Please select the folder that contains the '
                            'project data.')
            self.root_folder = ktk.gui.get_folder()
            ktk.gui.message('')
        else:
            self.root_folder = root_folder

        # Assign tables
        self.refresh()


    def __repr__(self):
        s =  f'--------------------------------------------------\n'
        s += 'DBInterface\n'
        s += f'--------------------------------------------------\n'
        s += f'          url: {self.url}\n'
        s += f'         user: {self.user}\n'
        s += f'project_label: {self.project_label}\n'
        s += f'--------------------------------------------------\n'
        s += f'participants:\n'
        s += str(self.participants) + '\n'
        s += f'--------------------------------------------------\n'
        s += f'sessions:\n'
        s += str(self.sessions) + '\n'
        s += f'--------------------------------------------------\n'
        s += f'trials:\n'
        s += str(self.trials) + '\n'
        s += f'--------------------------------------------------\n'
        s += f'files:\n'
        s += str(self.files) + '\n'
        s += f'--------------------------------------------------\n'
        return s


    def _fetch_table(self, table_name):
        global _module_user, _module_password
        url = self.url + '/kineticstoolkit/dbinterface.php'
        result = requests.post(url, data={
            'username': self.user,
            'password': self._password,
            'projectlabel': self.project_label,
            'action': 'fetchtable',
            'table': table_name})

        csv_text = result.content.decode("iso8859_15")
        if '# INVALID USER/PASSWORD COMBINATION' in csv_text:
            raise ValueError('Invalid user/password combination')

        try:
            return pd.read_csv(StringIO(csv_text),
                               comment='#',
                               delimiter='\t')
        except Exception:
            print(csv_text)
            raise ValueError('Unknown exception, see above.')


    def _scan_files(self):

        # Scan all files in root folder
        dict_files = {'FileID': [], 'FileName': []}

        warned_once = False
        for folder, _, files in os.walk(self.root_folder):
            if len(files) > 0:
                for file in files:
                    if 'dbfid' in file:
                        dbfid = int(file.split('dbfid')[1].split('n')[0])
                        if dbfid in dict_files['FileID']:
                            # Duplicate file

                            if warned_once is False:
                                warnings.warn('Duplicate file(s) found.')
                                warned_once = True

                            dup_index = dict_files['FileID'].index(dbfid)
                            dup_file = dict_files['FileName'][dup_index]

                            print('----------------------------------')
                            print(folder + '/' + file)
                            print('has the same dbfid as')
                            print(dup_file)

                        else:
                            dict_files['FileID'].append(dbfid)
                            dict_files['FileName'].append(
                                folder + '/' + file)

        # Convert to a Pandas DataFrame
        return pd.DataFrame(dict_files)

    def _filter_table(self, table, filters):
        """Apply filters on a DataFrame, return the filtered DataFrame."""
        # Remove all columns which name contains 'ID'
        columns = table.columns
        new_columns = []
        for column in columns:
            if not 'ID' in column:
                new_columns.append(column)
        table = table[new_columns]

        # Filter
        for column in filters:
            if column in table.columns:
                table = table[table[column] == filters[column]]
        return table

    def get(self, participant='', session='', trial='', file=''):
        """
        Extract information from a project.

        Parameters
        ----------
        participant : str, optional
            Participant label (for example, 'P01'). The default is ''.
        session : str, optional
            Session label (for example, 'SB4320'). The default is ''.
        trial : str, optional
            Trial label (for example, 'StaticAnatomic'). The default is ''.
        file : str, optional
            File label (for example, 'Kinematics'). The default is ''.

        Returns
        -------
        dict
            A record of the specified information.

        """
        # Assign the tables
        if participant == '' and session == '' and trial == '' and file == '':
            table_name = 'Projects'
            contents_name = 'Participants'
            contents_field = 'ProjectLabel'
            contents_filter = self.project_label
        elif participant != '' and session == '' and trial == '' and file == '':
            table_name = 'Participants'
            contents_name = 'Sessions'
            contents_field = 'ParticipantLabel'
            contents_filter = participant
        elif participant != '' and session != '' and trial == '' and file == '':
            table_name = 'Sessions'
            contents_name = 'Trials'
            contents_field = 'SessionLabel'
            contents_filter = session
        elif participant != '' and session != '' and trial != '' and file == '':
            table_name = 'Trials'
            contents_name = 'Files'
            contents_field = 'TrialLabel'
            contents_filter = trial
        elif participant != '' and session != '' and trial != '' and file != '':
            table_name = 'Files'
            contents_name = ''
            contents_field = 'FileLabel'
            contents_filter = file
        else:
            raise ValueError('Bad combination of arguments')

        table = self._tables[table_name]
        if contents_name != '':
            contents = self._tables[contents_name]
        else:
            contents = None

        # Filter
        filters = dict()
        if participant != '':
            filters['ParticipantLabel'] = participant
        if session != '':
            filters['SessionLabel'] = session
        if trial != '':
            filters['TrialLabel'] = trial
        if file != '':
            filters['FileLabel'] = file

        table = self._filter_table(table, filters)
        if contents is not None:
            contents = self._filter_table(contents, filters)

        # Change nans to None
        table = table.where(pd.notnull(table), None)

        # Convert to dict
        dict_out = table.to_dict('record')

        if len(dict_out) < 1:
            raise ValueError('No item found.')
        elif len(dict_out) > 1:
            raise ValueError('More than one item found.')
        else:
            dict_out = dict_out[0]

        # Add the contents
        if contents is not None:
            contents = contents[
                contents[contents_field] == contents_filter]
            contents = contents[contents_name[0:-1] + 'Label']
            dict_out[contents_name] = contents.to_list()

        return dict_out

    def refresh(self):
        """Update from database and reindex files."""
        def repetition_to_str(repetition):
            try:
                repetition = str(int(repetition))
            except Exception:
                repetition = ''
            return repetition

        self._tables = dict()

        self._tables['Projects'] = self._fetch_table('Projects')

        # Participants
        self._tables['Participants'] = self._fetch_table('Participants')
        self._tables['Participants']['ProjectLabel'] = self.project_label

        # Sessions
        self._tables['Sessions'] = self._fetch_table('Sessions')
        self._tables['Sessions'] = self._tables['Sessions'].merge(
            self._tables['Participants'][
                ['ParticipantID', 'ParticipantLabel']], how='inner')
        self._tables['Sessions']['SessionLabel'] = (
            self._tables['Sessions']['PlaceLabel'] +
            self._tables['Sessions']['SessionRepetition'].apply(repetition_to_str))

        # Trials
        self._tables['Trials'] = self._fetch_table('Trials')
        self._tables['TrialTypes'] = self._fetch_table('TrialTypes')
        self._tables['Trials'] = self._tables['Trials'].merge(
            self._tables['TrialTypes'], how='inner')
        self._tables['Trials'] = self._tables['Trials'].merge(
            self._tables['Sessions'][[
                'SessionID', 'SessionLabel', 'ParticipantLabel']], how='inner')
        self._tables['Trials']['TrialLabel'] = (
            self._tables['Trials']['TrialTypeLabel'] +
            self._tables['Trials']['TrialRepetition'].apply(repetition_to_str))

        # Files
        self._tables['Files'] = self._fetch_table('Files')
        self._tables['FileTypes'] = self._fetch_table('FileTypes')
        self._tables['FileAssociations'] = self._scan_files()

        self._tables['Files'] = self._tables['Files'].merge(
            self._tables['FileTypes'], how='inner')
        self._tables['Files'] = self._tables['Files'].merge(
            self._tables['FileAssociations'], how='inner')
        self._tables['Files'] = self._tables['Files'].merge(
            self._tables['Trials'][[
                'TrialID', 'TrialLabel', 'SessionLabel',
                'ParticipantLabel']], how='inner')
        self._tables['Files']['dbfid'] = ('dbfid' +
            self._tables['Files']['FileID'].apply(str) + 'n')
        self._tables['Files']['FileLabel'] = \
            self._tables['Files']['FileTypeLabel']
