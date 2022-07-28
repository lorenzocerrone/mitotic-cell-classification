from src.classification.train import train
from src.classification.train import default_config

if __name__ == '__main__':
    # train(default_config)
    default_config['logdir'] = 'cv_logs'
    for _ in range(5):
        for split in range(5):
            default_config['split'] = split
            print(default_config)
            train(default_config)
