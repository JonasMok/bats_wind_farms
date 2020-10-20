import pandas as pd
import numpy as np

import shutil, os


def get_audio_files(ip_dir):
    matches = []
    for root, dirnames, filenames in os.walk(ip_dir):
        for filename in filenames:
            if filename.lower().endswith('.wav'):
                #matches.append(os.path.join(root, filename))
                matches.append(os.path.join(filename))
    return matches

def pasta(x,y,z):
    '''x = name of the new folder to save the copied files
    #y = name of the folder where the files came from
    #z = the list of matched files '''

    path = os.path.abspath(y)
    if not os.path.isdir(x):
        os.makedirs(x)

    for root1, dirs1, files1 in os.walk(y):
        for file1 in files1:
            fullpath = os.path.join(path, file1)
            #print(fullpath)
            #print(z)
            if file1 in z:
                fullpath = os.path.join(path, file1)
                shutil.copy(fullpath,x)
            else:
                pass

def match(files, control):
    '''files = list of the file's names that would be copied
    control = the files inside of the folder that you want to copy'''
    files_2=[]
    for i in files:
        w = ''.join((i,'.wav'))
        files_2.append(w)

    match_2 =  list(set(files_2) & set(control))

    return match_2


#importing csv. file with the labels from with audio file have PIPI and PIPY calls
pipi = pd.read_csv('PIPI_PIPY_folder.csv')

#list of name files from the column
files = pipi['OUT FILE'].values.tolist()

#data_dir = 'Control_site_b'
#data_dir_2 = 'test'
#names of the directories to extract the files
data_dir_control = 'Control 0006246'
data_dir_ground_1 = 'T1 Ground 0006199'
data_dir_nacelle_1 = 'T1 Nacelle 0006325'
data_dir_ground_5 = 'T5 Ground 0006323'
data_dir_nacelle_5 = 'T5 Nacelle take 2'
data_dir_ground_9 = 'T9 Ground 0006331'
data_dir_nacelle_9 = 'T9 Nacelle 0006364'

#control
x0 = 'PIPI_control' #name of the new folder to save the copied files
y0 = data_dir_control #the folder where the files will be copied


control = get_audio_files(y0)
match_control = match(files, control)

z0= match_control #the list where

pasta(x0,y0,z0)

#ground 1

x1 = 'PIPI_ground_1'
y1 = data_dir_ground_1


ground_1 = get_audio_files(y1)
match_ground_1 = match(files,ground_1)

z1= match_ground_1

pasta(x1,y1,z1)

#nacelle 1
x_n1 = 'PIPI_nacelle_1'
y_n1 = data_dir_nacelle_1


nacelle_1 = get_audio_files(y_n1)
match_nacelle_1 = match(files,nacelle_1)

z_n1= match_nacelle_1

pasta(x_n1,y_n1,z_n1)

#ground 5
x5 = 'PIPI_ground_5'
y5 = data_dir_ground_5


ground_5 = get_audio_files(y5)
match_ground_5 = match(files, ground_5)

z5= match_ground_5

pasta(x5,y5,z5)

#nacelle 5
x_n5 = 'PIPI_nacelle_5'
y_n5 = data_dir_nacelle_5


nacelle_5 = get_audio_files(y_n5)
match_nacelle_5 = match(files,nacelle_5)

z_n5= match_nacelle_5

pasta(x_n5,y_n5,z_n5)


#ground 9
x9 = 'PIPI_ground_9'
y9 = data_dir_ground_9


ground_9 = get_audio_files(y9)
match_ground_9 = match(files, ground_9)

z9= match_ground_9

pasta(x9,y9,z9)

#nacelle 9
x_n9 = 'PIPI_nacelle_9'
y_n9 = data_dir_nacelle_9


nacelle_9 = get_audio_files(y_n9)
match_nacelle_9 = match(files,nacelle_9)

z_n9= match_nacelle_9

pasta(x_n9,y_n9,z_n9)
