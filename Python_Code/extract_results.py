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
window_size=60

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('Test', help='Testing Scenario')
    parser.add_argument('station', help='name of the station')
    parser.add_argument('model_save', help='Name of the model')

    args = parser.parse_args()

    Test = args.Test
    station = args.station
    Bw = "80MHz"
    num_mon = "3mo"
    model_save= args.model_save



    # %%


    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    data_path = "../Data"
    data_dir = os.path.join(data_path, Test, Bw, num_mon, station)
    print (data_dir)
    model_dir = os.path.join(data_path, Test, Bw, num_mon, station, model_save)


    # %%
    
    from dataGenerator import DataGenerator
    test_csv = os.path.join(data_dir, 'test_set.csv')
    test_gen = DataGenerator(data_dir,test_csv,batchsize=64, shuffle=False)


    # %%
    from tensorflow.keras.models import load_model
    model = load_model(model_dir)

    final_loss, final_accuracy = model.evaluate(test_gen)
    print('Final Loss: {}, Final Accuracy: {}'.format(final_loss, final_accuracy))




