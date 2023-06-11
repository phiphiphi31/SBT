class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/v-fxi/Downloads/code/a/ViTtrack/training_output/'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/v-fxi/Downloads/train/ViTtrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/v-fxi/Downloads/code/vittrack/pretrained_networks'
        self.lasot_dir = '/home/v-fxi/Downloads/code/vittrack/data/lasot'
        self.got10k_dir = '/home/v-fxi/Downloads/dataset/got10k/train'
        self.got10k_val_dir = '/home/v-fxi/Downloads/dataset/got10k/val'
        self.lasot_lmdb_dir = ''
        self.got10k_lmdb_dir = ''
        self.trackingnet_dir = '/home/v-fxi/Downloads/dataset/trackingnet'
        self.trackingnet_lmdb_dir = ''
        self.coco_dir = '/home/v-fxi/Downloads/dataset/coco'
        self.coco_lmdb_dir = '/home/v-fxi/Downloads/dataset/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/v-fxi/Downloads/dataset/imagenet'
        self.imagenet_lmdb_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
