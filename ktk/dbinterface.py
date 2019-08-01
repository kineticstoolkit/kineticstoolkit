#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions that manage projects as hosted on Felix Chenier's BIOMEC database.

A tutorial is also available.
"""

from ktk import gui

import requests
import os


def __dir__():
    """Create dir for tab completion."""
    return ['fetch_project']


def fetch_project(project_label, user='', password='', root_folder='',
                  url='https://biomec.uqam.ca'):
    """
    Fetch a project information from Felix Chenier's BIOMEC database.

    Parameters
    ----------
    project_label : str
        Project label, for example 'FC_XX18E'.
    user : str, optional
        User name on BIOMEC. The default is ''. If left this way, a dialog
        box will prompt the user to enter his credentials.
    password : str, optional
        Password on BIOMEC. The default is ''.
    root_folder : str, optional
        Location of the project's data files. The default is ''. If left this
        way, a dialog box will prompt the user to select the root folder.
    url : str, optional
        URL for the BIOMEC database. The default is 'https://biomec.uqam.ca'.

    Returns
    -------
    project : Project
        Project information.

    """
    # Get username and password of not supplied
    if user == '':
        user, password = gui.get_credentials()

    # Append the relative url to the base url
    url += '/py/pywrapper.php'

    # Do the request and executes it
    result = requests.post(url, data={
                                    'command': 'fetchall',
                                    'projectlabel': project_label,
                                    'username': user,
                                    'password': password}, verify=False)

    content = result.content.decode("iso8859_15")
    global project
    exec(content)

    # Add root folder
    if root_folder == '':
        project['root_folder'] = gui.get_folder()
    else:
        project['root_folder'] = root_folder

    # Scan all files in root folder
    folder_list = []
    file_list = []
    for folder, _, files in os.walk(project['root_folder']):
        if len(files) > 0:
            for file in files:
                folder_list.append(folder)
                file_list.append(file)

    # Assign files to File instances
    project['files'] = []
    project['missing_files'] = []
    project['duplicate_files'] = []

    for participant_id in project['participants'].keys():
        participant = project['participants'][participant_id]
        for session_id in participant['sessions'].keys():
            session = participant['sessions'][session_id]
            for trial_id in session['trials'].keys():
                trial = session['trials'][trial_id]
                for file_id in trial['files'].keys():
                    file = trial['files'][file_id]
                    dbfid = file['dbfid']

                    # Now find this file
                    file_found = False
                    file_duplicate = False

                    for i in range(0, len(file_list)):
                        if dbfid in file_list[i]:

                            if file_found is False:
                                file_found = True
                                file['filename'] = os.path.join(
                                            folder_list[i],
                                            file_list[i])
                            else:
                                file_duplicate = True

                    file_tuple = (dbfid, participant_id, session_id,
                                  trial_id, file_id)

                    if file_duplicate:
                        project['duplicate_files'].append(file_tuple)
                    elif file_found:
                        project['files'].append(file_tuple)
                    else:
                        project['missing_files'].append(file_tuple)

    return project
