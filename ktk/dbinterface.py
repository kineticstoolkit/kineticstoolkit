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
                  url='https://felixchenier.uqam.ca/biomec'):
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
        URL for the BIOMEC database. The default is
        'https://felixchenier.uqam.ca/biomec'.

    Returns
    -------
    project : Project
        Project information.

    """
    # Get username and password of not supplied
    if user == '':
        user, password = gui.get_credentials()

    # Do the request and executes it
    print("Fetching project on %s" % url)

    # Append the relative url to the base url
    url += '/py/pywrapper.php'
    result = requests.post(url, data={
                                    'command': 'fetchall',
                                    'projectlabel': project_label,
                                    'username': user,
                                    'password': password})

    content = result.content.decode("iso8859_15")

    try:
        global project
        exec(content)
        project
    except:
        raise(Exception(content))

    print("Assigning root folder")
    # Add root folder
    if root_folder == '':
        project['RootFolder'] = gui.get_folder()
    else:
        project['RootFolder'] = root_folder

    # Scan all files in root folder
    print("Building file associations")
    folder_list = []
    file_list = []
    for folder, _, files in os.walk(project['RootFolder']):
        if len(files) > 0:
            for file in files:
                folder_list.append(folder)
                file_list.append(file)

    # Assign files to File instances
    project['Files'] = []
    project['MissingFiles'] = []
    project['DuplicateFiles'] = []

    for participant_id in project['Participants'].keys():
        participant = project['Participants'][participant_id]
        for session_id in participant['Sessions'].keys():
            session = participant['Sessions'][session_id]
            for trial_id in session['Trials'].keys():
                trial = session['Trials'][trial_id]
                for file_id in trial['Files'].keys():
                    file = trial['Files'][file_id]
                    dbfid = file['dbfid']

                    # Now find this file
                    file_found = False
                    file_duplicate = False

                    for i in range(0, len(file_list)):
                        if dbfid in file_list[i]:

                            if file_found is False:
                                file_found = True
                                file['Filename'] = os.path.join(
                                            folder_list[i],
                                            file_list[i])
                            else:
                                file_duplicate = True

                    file_tuple = (dbfid, participant_id, session_id,
                                  trial_id, file_id)

                    if file_duplicate:
                        project['DuplicateFiles'].append(file_tuple)
                    elif file_found:
                        project['Files'].append(file_tuple)
                    else:
                        project['MissingFiles'].append(file_tuple)
    print("Done")

    return project
