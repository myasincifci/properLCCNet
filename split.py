import os
import shutil

def main():
    dataset_path = './data/dataset_medium' # path to directory containing 'sequences' dir.

    sequence_00 = os.path.join(dataset_path, 'sequences', '00')
    total_len = len(os.listdir(os.path.join(sequence_00, 'image_2')))
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len

    os.mkdir(os.path.join(dataset_path, 'sequences', './01'))
    os.mkdir(os.path.join(dataset_path, 'sequences', './01/image_2'))
    os.mkdir(os.path.join(dataset_path, 'sequences', './01/velodyne'))
    os.mkdir(os.path.join(dataset_path, 'sequences', './01/velodyne_r'))

    names = sorted([name.split('.')[0] for name in os.listdir(os.path.join(sequence_00, 'image_2'))])
    train_names = names[val_len:]

    for name in train_names:
        shutil.move(
            os.path.join(dataset_path, 'sequences', '00', 'image_2', name + '.png'),
            os.path.join(dataset_path, 'sequences', '01', 'image_2', name + '.png')
        )
        shutil.move(
            os.path.join(dataset_path, 'sequences', '00', 'velodyne', name + '.bin'),
            os.path.join(dataset_path, 'sequences', '01', 'velodyne', name + '.bin')
        )
        shutil.move(
            os.path.join(dataset_path, 'sequences', '00', 'velodyne_r', name + '.bin'),
            os.path.join(dataset_path, 'sequences', '01', 'velodyne_r', name + '.bin')
        )

    a=1

if __name__ == '__main__':
    main()