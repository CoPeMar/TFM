import os
from pathlib import Path

# Specify your source and destination folders
File_1 = "C:/TFM_Data/ERA5_Profiles/ERA5izo4icaro_20240409_20250430/3cols/pres/combined.csv"
File_2 = "C:/TFM_Data/ERA5_Profiles/ERA5izo4icaro_20240409_20250430/3cols/r/combined.csv"
File_3 = "C:/TFM_Data/ERA5_Profiles/ERA5izo4icaro_20240409_20250430/3cols/t/combined.csv"
output_file = "C:/TFM_Data/ERA5_Profiles/ERA5izo4icaro_20240409_20250430/3cols/combined.csv"

# Get all files in the folder, sorted alphabetically
files = [File_1, File_2, File_3]

# Read and append each file to the output file
with open(output_file, 'w') as outf:
    # Read all lines from each file
    file_lines = []
    for file in files:
        with open(file, 'r') as inf:
            file_lines.append(inf.readlines())
    
    # Assume all files have the same number of lines
    num_lines = len(file_lines[0])
    
    for i in range(num_lines):
        row_parts = []
        for j, lines in enumerate(file_lines):
            line = lines[i].strip()
            if line:  # skip empty lines
                parts = line.split(',')
                if j == 0:
                    # First file: take all columns
                    row_parts.extend(parts)
                else:
                    # Other files: take only the last column
                    if parts:
                        row_parts.append(parts[-1])
        # Write the combined row
        outf.write(','.join(row_parts) + '\n')