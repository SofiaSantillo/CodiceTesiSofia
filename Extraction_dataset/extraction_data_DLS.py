import csv

# Path
input_path = 'Extraction_dataset/data_DLS.txt'
output_path = 'Data_DLS/data_DLS.csv'

delimiter = ' '  

with open(input_path, 'r') as infile, open(output_path, 'w', newline='') as outfile:
    reader = csv.reader(infile, delimiter=delimiter)
    writer = csv.writer(outfile, delimiter=',')  

    for row in reader:
        row_cleaned = [elem for elem in row if elem.strip() != '']
        writer.writerow(row_cleaned)


