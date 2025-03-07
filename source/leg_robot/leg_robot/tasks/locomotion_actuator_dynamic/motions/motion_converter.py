import numpy as np
import pandas as pd
import os

class MotionConverter:
    def __init__(self, motion_file):
        current_path = os.path.dirname(os.path.abspath(__file__))
        self.motion_file = os.path.join(current_path, motion_file)
        self.load_motion()

    def load_motion(self):
        # check file extension
        if self.motion_file.endswith(".csv"):
            # load csv file
            self.motion_data: pd.DataFrame = pd.read_csv(self.motion_file)
        elif self.motion_file.endswith(".xlsx"):
            # load xlsx file
            self.motion_data: pd.DataFrame = pd.read_excel(self.motion_file)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

    def convert(self, output_file):
        if output_file.endswith(".npz"):
            self._convert_to_npz(output_file)
        elif output_file.endswith(".csv"):
            raise ValueError("Output file format .csv is not supported. Please use .npz.")

    def _convert_to_npz(self, output_file):
        # list all headers
        headers = self.motion_data.columns.tolist()

        # convert all columns to numpy arrays
        data_dict = {}
        for header in headers:
            data_dict[header] = self.motion_data[header].to_numpy()

        # save to npz file
        current_path = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(current_path, output_file)
        np.savez_compressed(output_file, **data_dict)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Csv motion file")
    parser.add_argument("--output_file", type=str, required=True, help="Output npz name")
    args, _ = parser.parse_known_args()

    motion_converter = MotionConverter(args.input_file)
    motion_converter.convert(args.output_file)