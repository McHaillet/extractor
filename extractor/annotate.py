#!/usr/bin/env python
import argparse
import pathlib
import mrcfile
import starfile
import numpy as np
from tqdm import tqdm


def spherical_mask(box_size, radius):
    center = (box_size - 1) / 2
    x, y, z = (np.arange(box_size) - center,
               np.arange(box_size) - center,
               np.arange(box_size) - center)
    # use broadcasting
    r = np.sqrt(((x / radius) ** 2)[:, np.newaxis, np.newaxis] +
                ((y / radius) ** 2)[:, np.newaxis] +
                (z / radius) ** 2).astype(np.float32)
    mask = np.zeros_like(r)
    mask[r < 1] = 1.
    return mask


def parse_ground_truth_shrec(ground_truth_path, particle_id):
    with open(ground_truth_path, 'r') as fstream:
        locations = [list(
            map(int, line.strip().split()[1:4])
        ) for line in fstream.readlines() if particle_id in line]
    return np.flip(np.array(locations), 1)  # flip to z, y, x


def parse_ground_truth_star(ground_truth_path, pixel_size_star, pixel_size_score_map):
    try:
        star_data = starfile.read(ground_truth_path)[0]
    except KeyError:
        star_data = starfile.read(ground_truth_path)

    x = star_data.filter(like='CoordinateX').to_numpy()
    y = star_data.filter(like='CoordinateY').to_numpy()
    z = star_data.filter(like='CoordinateZ').to_numpy()

    return (np.concatenate([z, y, x], axis=1) * pixel_size_star / pixel_size_score_map).round().astype(np.int32)


def annotate(ground_truth_positions, score_map, hit_tolerance_px=2, particle_radius_px=4):
    # mask edges of score volume
    score_map[0: particle_radius_px, :, :] = 0
    score_map[:, 0: particle_radius_px, :] = 0
    score_map[:, :, 0: particle_radius_px] = 0
    score_map[-particle_radius_px:, :, :] = 0
    score_map[:, -particle_radius_px:, :] = 0
    score_map[:, :, -particle_radius_px:] = 0

    # mask for gt
    tol_box = int(hit_tolerance_px) * 2 + 1
    gt_mask = spherical_mask(tol_box, hit_tolerance_px)

    ground_truth_mask = np.zeros_like(score_map)
    for i in tqdm(range(ground_truth_positions.shape[0])):
        zyx = ground_truth_positions[i]
        if score_map[zyx[0], zyx[1], zyx[2]] == 0:
            continue
        else:
            start = [x - hit_tolerance_px for x in zyx]
            ground_truth_mask[
            start[0]: start[0] + tol_box,
            start[1]: start[1] + tol_box,
            start[2]: start[2] + tol_box
            ] += gt_mask
    ground_truth_mask[ground_truth_mask > 1] = 1

    score_map *= ground_truth_mask  # remove everything outside of ground truth tolerance
    del ground_truth_mask

    ground_truth = np.zeros_like(score_map)

    # mask for iteratively selecting peaks
    cut_box = int(particle_radius_px) * 2 + 1
    cut_mask = (spherical_mask(cut_box, particle_radius_px) == 0) * 1
    n = 0

    pbar = tqdm(total=ground_truth_positions.shape[0])

    while score_map.max() > 0:

        ind = np.unravel_index(score_map.argmax(), score_map.shape)
        ground_truth[ind] = 1.

        start = [i - particle_radius_px for i in ind]
        score_map[
        start[0]: start[0] + cut_box,
        start[1]: start[1] + cut_box,
        start[2]: start[2] + cut_box
        ] *= cut_mask

        n += 1
        if n % 10 == 0:
            pbar.update(10)
    pbar.close()

    return ground_truth


def entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground-truth', type=pathlib.Path, required=True)
    parser.add_argument('--particle', type=str, required=False)
    parser.add_argument('--particle-radius', type=int, required=True,
                        help='Minimal radius of the particle, expressed in number of pixels in the score volume.')
    parser.add_argument('--tolerance', type=int, required=True,
                        help='Tolerated distance between ground truth and score peak, expressed in number of pixels.')
    parser.add_argument('--score-map', type=pathlib.Path, required=True)
    parser.add_argument('--ground-truth-pixel-size', type=float, required=False)
    parser.add_argument('--score-map-pixel-size', type=float, required=False)
    args = parser.parse_args()

    if args.ground_truth.suffix == '.txt' and args.particle is not None:
        ground_truth = parse_ground_truth_shrec(
            args.ground_truth,
            args.particle
        )
    elif args.ground_truth.suffix == '.star':
        ground_truth = parse_ground_truth_star(
            args.ground_truth,
            args.ground_truth_pixel_size,
            args.score_map_pixel_size
        )
    else:
        raise ValueError('Invalid ground truth file type either .txt or .star')

    ground_truth_volume = annotate(
        ground_truth,
        mrcfile.read(args.score_map),
        hit_tolerance_px=args.tolerance,
        particle_radius_px=args.particle_radius
    )
    mrcfile.write(
        str(args.score_map).replace('_scores', '_gt'),
        ground_truth_volume.astype(np.float32),
        voxel_size=args.score_map_pixel_size,
        overwrite=True
    )
