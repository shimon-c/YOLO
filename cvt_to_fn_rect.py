"""
input from:
https://www.makesense.ai/
"""
import pandas as pd
import os
import argparse

def parse_arges():
    ap = argparse.ArgumentParser("cvt_to_fn_rect Convert to: fn,x1,y1,x2,y2")
    ap.add_argument('--input_csv', type=str,required=True, help="input_csv full path")
    ap.add_argument('--output_file_name', type=str, required=True, help="out file directorty taken from input")
    args = ap.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arges()
    df = pd.read_csv(args.input_csv)
    rows = [0,]*df.shape[0]
    dct = {'image': rows,
           'xmin': rows, 'ymin': rows,
                  'xmax': rows,'ymax': rows}
    odf = pd.DataFrame(data=dct)
    odf['image'] = df['image_name']
    odf['xmin'] = df['bbox_x']
    odf['ymin'] = df['bbox_y']
    odf['xmax'] = df['bbox_x'] + df['bbox_width']
    odf['ymax'] = df['bbox_y'] + df['bbox_height']
    dirn = os.path.dirname(args.input_csv)
    out_name = os.path.join(dirn, args.output_file_name)
    odf.to_csv(out_name, index=False)
    print(f'out file:{out_name}')

