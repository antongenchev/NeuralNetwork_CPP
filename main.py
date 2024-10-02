import csv

with open('data/auto+mpg/auto-mpg.data', 'r') as f:
    lines = f.readlines()
data = []
for line in lines:
    parts = [x for x in line.strip().split(' ') if x != ''][:8]
    parts[-1] = parts[-1].split('\t')[0]
    parts = [x.strip('.') for x in parts]
    if '?' in parts:
        continue
    data.append(parts)

with open('data/data.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(data)
