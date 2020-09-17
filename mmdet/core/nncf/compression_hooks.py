from texttable import Texttable
from mmcv.runner.hooks.hook import Hook


class CompressionHook(Hook):
    def __init__(self, compression_ctrl=None, cfg=None):
        from nncf import NNCFConfig

        self.compression_ctrl = compression_ctrl
        # TODO: is it required now?
        # self.nncf_config = NNCFConfig(cfg.nncf_config)
        self.work_dir = cfg.work_dir

    def after_train_iter(self, runner):
        self.compression_ctrl.scheduler.step()

    def after_train_epoch(self, runner):
        self.compression_ctrl.scheduler.epoch_step()

    def before_run(self, runner):
        if runner.rank == 0:
            print_statistics(self.compression_ctrl.statistics(), runner.logger)

    def after_run(self, runner):
        pass


def print_statistics(stats, logger):
    for key, val in stats.items():
        if isinstance(val, Texttable):
            logger.info(key)
            logger.info(val.draw())
        else:
            logger.info('{}: {}'.format(key, val))
