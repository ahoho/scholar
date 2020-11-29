# Converts CSV with column headers to json lines, reading from stdin
# https://stackoverflow.com/questions/19697846/how-to-convert-csv-file-to-multiline-json
import csv
import json
import sys

for row in csv.DictReader(sys.stdin):
    json.dump(row, sys.stdout)
    sys.stdout.write('\n')
