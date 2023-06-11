from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = '/mnt/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/mnt/data/IFB'
    settings.lasot_extension_subset_path_path = '/mnt/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/mnt/data/lasot_lmdb'
    settings.lasot_path = '/mnt/data/LaSOTBenchmark'

    settings.network_path = '...'    # Where tracking networks are stored.
    settings.nfs_path = '/mnt/data/nfs'
    settings.otb_path = '/mnt/data/otb'
    settings.prj_dir = '/scratch/amlt_code/'
    settings.result_plot_path = '/scratch/amlt_code/save/output/plot'
    settings.results_path = '/scratch/amlt_code/save/output/tracking_results'    # Where to store tracking results
    settings.save_dir = '/scratch/amlt_code/save/output'
    settings.segmentation_path = ''
    settings.tc128_path = '/mnt/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/mnt/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/mnt/data/tknet'
    settings.uav_path = '/mnt/data/UAV123'
    settings.vot18_path = '/mnt/data/vot2018/'
    settings.vot22_path = '/mnt/data/vot2022'
    settings.vot_path = '/mnt/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

