from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = '/home/v-fxi/Downloads/dataset/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/v-fxi/Downloads/dataset/ITB'
    settings.lasot_extension_subset_path_path = '/home/v-fxi/Downloads/dataset/lasot_extension_subset'
    settings.lasot_lmdb_path = ''
    settings.lasot_path = '/home/v-fxi/Downloads/dataset/lasot'
    settings.network_path = '/home/v-fxi/Downloads/code/pth/vittrack'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path =  '/home/v-fxi/Downloads/dataset/OTB100'
    settings.prj_dir = '/home/v-fxi/Downloads/code/Stark'
    settings.result_plot_path = '/home/v-fxi/Downloads/pth/result/'
    settings.results_path = '/home/v-fxi/Downloads/pth/result/'    # Where to store tracking results
    settings.save_dir = '/home/v-fxi/Downloads/pth/result/'
    settings.segmentation_path = ''
    settings.tc128_path = '/home/v-fxi/Downloads/dataset/tc128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/v-fxi/Downloads/dataset/tknet'
    settings.uav_path = '/home/v-fxi/Downloads/dataset/UAV123/'
    settings.vot18_path = ''
    settings.vot22_path = '/home/v-fxi/Downloads/dataset/NOUVIT/VOT2021'
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

