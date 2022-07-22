from src.classification.train import train
from src.classification.train import default_config

if __name__ == '__main__':
    ## train(default_config)

    default_config['logdir'] = 'cv_logs'
    for split in range(5):
        default_config['split'] = split
        train(default_config)
        print(default_config)

    print('end-t1')
    model_dict = {'model_family': 'adapt_resnet',
                  'model_name': 'resnet50',
                  'pretrained': True,
                  'w_bias': 0.25,
                  'lr': 1e-3,
                  'wd': 1e-5,
                  'use_scheduler': False,
                  }
    default_config['model'] = model_dict
    for split in range(5):
        default_config['split'] = split
        train(default_config)
        print(default_config)
    print('end-t2')

