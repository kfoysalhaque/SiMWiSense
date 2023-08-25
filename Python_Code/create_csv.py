"""
    Copyright (C) 2023 Khandaker Foysal Haque
    contact: haque.k@northeastern.edu
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import os
import csv
import numpy as np

def generate_train_val_csv(data_path, name):
    train_csv = os.path.join(data_path, 'train_set.csv')
    val_csv = os.path.join(data_path, 'val_set.csv')

    train_csv_file = open(train_csv, 'w', newline='')
    val_csv_file = open(val_csv, 'w', newline='')

    fieldnames = ['filename', 'label']
    writer_train = csv.DictWriter(train_csv_file, fieldnames=fieldnames)
    writer_train.writeheader()
    writer_val = csv.DictWriter(val_csv_file, fieldnames=fieldnames)
    writer_val.writeheader()

    for root, dirs, files in os.walk(data_path):
        if root[-5:] == 'batch':
            for file in files:
                filename = os.path.join(root[-7:], file)
                label = root[-7]
                rand = np.random.rand(1)
                if rand < 0.80:
                    writer_train.writerow({'filename': filename, 'label': label})
                else:
                    writer_val.writerow({'filename': filename, 'label': label}) 
    train_csv_file.close()
    val_csv_file.close()

def generate_test_csv(data_path):
    test_csv = os.path.join(data_path, 'test_set.csv')

    test_csv_file = open(test_csv, 'w', newline='')

    fieldnames = ['filename', 'label']
    writer_test = csv.DictWriter(test_csv_file, fieldnames=fieldnames)
    writer_test.writeheader()

    for root, dirs, files in os.walk(data_path):
        if root[-5:] == 'batch':
            for file in files:
                filename = os.path.join(root[-7:], file)
                label = root[-7]
                rand = np.random.rand(1)
                if rand < 1:
                    writer_test.writerow({'filename': filename, 'label': label})

    test_csv_file.close()

def process_data(Test, env, Bw, num_mon, stations, train_test_dir_name):
    data_pa = f"../Data/{Test}"

    for env in env:
        for name in train_test_dir_name:
            for station in stations:
                station_data_path = os.path.join(data_pa, env, Bw, num_mon, station)

                for station in stations:

                    if Test == "proximity":
                        data_path = os.path.join(station_data_path, "Slots", f"{name}_{station}")
                    else:
                        data_path = os.path.join(station_data_path, "Slots", name)

                    if name == "Train":
                        generate_train_val_csv(data_path, name)
                    else:
                        generate_test_csv(data_path)
