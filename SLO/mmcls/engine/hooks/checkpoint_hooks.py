from mmengine.hooks import CheckpointHook as BaseCheckpointHook

from mmcls.registry import HOOKS


@HOOKS.register_module()
class NewCheckpointHook(BaseCheckpointHook):
    def before_train(self, runner) -> None:
        super().before_train(runner)

        self.out_dir = self.file_backend.join_path(runner.work_dir, runner.timestamp)
        runner.logger.info(f'Checkpoints will be saved to {self.out_dir}.')
