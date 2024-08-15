import os

total = 0
outliars = 0
all_files = []
for file in os.listdir('CTSpine/train/crops'):
    if 'crop' in file and 'distance_field' not in file:
        if file[-11:-7] != 'crop':
            outliars += 1
        total += 1
        all_files.append(file)
with open('CTSpine/train_files.txt', 'w') as f:
    for file in all_files:
        f.write(file + '\n')
with open('CTSpine/test_files.txt', 'w') as f:
    for file in all_files:
        f.write(file + '\n')
with open('CTSpine/test_files_200.txt', 'w') as f:
    for file in all_files:
        f.write(file + '\n')
