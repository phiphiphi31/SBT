class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/scratch/amlt_code/save/'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/scratch/amlt_code/save/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/scratch/amlt_code/code/pretrained_networks'
        self.lasot_dir = '/mnt/data/LaSOTBenchmark'
        self.got10k_dir = '/mnt/data/got10k/train'
        self.got10k_val_dir = '/mnt/data/got10k/val'
        self.lasot_lmdb_dir = ''
        self.got10k_lmdb_dir = ''
        self.trackingnet_dir = '/mnt/data/tknet'
        self.trackingnet_lmdb_dir = ''
        self.coco_dir = '/mnt/data/coco2017'
        self.coco_lmdb_dir = '/home/v-fxi/Downloads/code/OSTrack/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/mnt/data/ImageNet'
        self.imagenet_lmdb_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
