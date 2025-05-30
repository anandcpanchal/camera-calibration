from AutoCalib import calibrate
import argparse

def main(args):
    import pdb
    pdb.set_trace()
    calibrate(args.input_path, args.output_path, args.width_square_count-1, args.height_square_count-1, args.square_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A Python Implementation of Zhang-Camera-Calibration Algorithm.")

    # Add arguments here
    parser.add_argument("-i", "--input_path", type=str, help="A path to read input images")
    parser.add_argument("-o", "--output_path", type=str, help="A path to read output results")
    parser.add_argument("-W", "--width_square_count", type=int, help="Count of squares in the pattern along the width")
    parser.add_argument("-H", "--height_square_count", type=int, help="Count of squares in the pattern along the height")
    parser.add_argument("-s", "--square_size", type=int, help="Size of a square")

    args = parser.parse_args()
    main(args)
