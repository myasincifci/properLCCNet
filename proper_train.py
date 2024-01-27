from DatasetLidarCamera import DatasetLidarCameraKittiOdometry


def main():
    _config = {
        'checkpoints': './checkpoints/',
        'dataset': 'kitti/odom', # 'kitti/raw'
        'data_folder': './data/data_odometry_color/dataset',
        'use_reflectance': False,
        'val_sequence': 0,
        'epochs': 120,
        'BASE_LEARNING_RATE': 3e-4,  # 1e-4,
        'loss': 'combined',
        'max_t': 0.1, # 1.5, 1.0,  0.5,  0.2,  0.1
        'max_r': 1., # 20.0, 10.0, 5.0,  2.0,  1.0
        'batch_size': 32,
        'num_worker': 6,
        'network': 'Res_f1',
        'optimizer': 'adam',
        'resume': True,
        'weights': None, #'./pretrained/kitti_iter5.tar'
        'rescale_rot': 1.0,
        'rescale_transl': 2.0,
        'precision': "O0",
        'norm': 'bn',
        'dropout': 0.0,
        'max_depth': 80.,
        'weight_point_cloud': 0.5,
        'log_frequency': 10,
        'print_frequency': 50,
        'starting_epoch': -1,
    }
    
    dataset_train = DatasetLidarCameraKittiOdometry(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                  split='train', use_reflectance=_config['use_reflectance'],
                                  val_sequence=_config['val_sequence'])
    dataset_val = DatasetLidarCameraKittiOdometry(_config['data_folder'], max_r=_config['max_r'], max_t=_config['max_t'],
                                split='val', use_reflectance=_config['use_reflectance'],
                                val_sequence=_config['val_sequence'])
    
    # Training
    for epoch in _config['epochs']:
        pass

if __name__ == "__maim__":
    main