"""End-to-end test for MDLM training."""

from tests.helpers.run_sh_command import run_sh_command
from tests.conftest import RunIf


MDLM_CHECKPOINT_PATH = "/tmp/test_mdlm/dummy_checkpoints/checkpoints/last.ckpt"


@RunIf(min_gpus=1)
def test_mdlm_training():
    """Test MDLM training runs end-to-end with minimal config."""
    run_sh_command(
        [
            "-m", "discrete_diffusion",
            "experiment=mdlm",
            "data=synthetic",
            "trainer.max_steps=50",
            "wandb=null",
            "hydra.run.dir=/tmp/test_mdlm",
            "trainer.accelerator=cuda",
            "trainer.devices=1",
            "trainer.num_nodes=1",
            "loader.global_batch_size=2",
            "trainer.limit_val_batches=2",
            "model.length=16",
            # Save checkpoint for GStar test
            "callbacks.checkpoint_every_n_steps.every_n_train_steps=50",
            "callbacks.checkpoint_every_n_steps.save_last=true",
        ],
        env={"PYTHONPATH": "src"},
    )

