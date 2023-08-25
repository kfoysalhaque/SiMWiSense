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


import argparse
from create_csv import process_data

Env = ["Classroom", "Office"]
Bw = "80MHz"
num_mon = "3mo"
stations = ["m1", "m2", "m3"]
train_test_dir_name = ["Train", "Test"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('Test', help='Testing Scenario; options are: proximity / coarse / fine_grained')
    args = parser.parse_args()

    Test = args.Test

    process_data(Test, Env, Bw, num_mon, stations, train_test_dir_name)
