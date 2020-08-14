import os

for cl in os.listdir('datasets/'):
    if cl != '.DS_Store':
        print('datasets/' + cl)
        for filename in os.listdir('datasets/' + cl):
            if filename >= cl + '_00600.jpg':
                os.remove('datasets/' + cl + '/' + filename)
