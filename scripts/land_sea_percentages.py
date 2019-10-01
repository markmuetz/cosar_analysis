# coding: utf-8
import iris

if __name__ == '__main__':
    basedir = '/home/markmuetz/mirrors/rdf/um10.9_runs/archive/u-au197/land_sea_mask'

    land_mask_cube = iris.load_cube(f'{basedir}/qrparm.mask')
    land_mask = land_mask_cube.data[53:92, :]

    # Same as in COSAR
    num_cells = land_mask.shape[0] * land_mask.shape[1]

    sea_frac = (land_mask == 0).sum() / num_cells
    land_frac = (land_mask == 1).sum() / num_cells
    print(f'Sea percentage: {sea_frac * 100:.2f} %')
    print(f'Land percentage: {land_frac * 100:.2f} %')
