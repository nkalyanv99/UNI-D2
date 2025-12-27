"""End-to-end test for GStar training."""

from tests.helpers.run_sh_command import run_sh_command
from tests.conftest import RunIf

from .test_mdlm import test_mdlm_training, MDLM_CHECKPOINT_PATH


@RunIf(min_gpus=1)
def test_gstar_training():
    """Test GStar training runs end-to-end with minimal config.
    
    First runs MDLM training to obtain a checkpoint, then trains GStar
    using that checkpoint as the frozen backbone.
    """
    # First train MDLM to get a checkpoint
    test_mdlm_training()

    # Then train GStar using the MDLM checkpoint
    run_sh_command(
        [
            "-m", "discrete_diffusion",
            "experiment=gstar",
            "data=synthetic",
            "trainer.max_steps=50",
            "wandb=null",
            "hydra.run.dir=/tmp/test_gstar",
            "trainer.accelerator=cuda",
            "trainer.devices=1",
            "trainer.num_nodes=1",
            "loader.global_batch_size=2",
            "trainer.limit_val_batches=2",
            "model.length=16",
            f"training.finetune_path={MDLM_CHECKPOINT_PATH}",
            "+training.strict_load=false",  # GStar has extra layers not in MDLM
        ],
        env={"PYTHONPATH": "src"},
    )

