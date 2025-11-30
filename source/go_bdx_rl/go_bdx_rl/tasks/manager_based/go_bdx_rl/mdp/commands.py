# custom_commands.py (or similar file)

import torch
import numpy as np
from typing import Tuple, Dict

from isaaclab.utils import configclass
from isaaclab.managers import CommandTermCfg, CommandTerm

@configclass
class OneHotCommandCfg(CommandTermCfg):
    """Configuration for a custom one-hot vector command."""

    # The size of the one-hot vector (e.g., 2 for [1,0] and [0,1])
    vector_size: int = 2

    # Resampling time is set to a very large number to ensure it samples only once per episode
    resampling_time_range: Tuple[float, float] = (10000.0, 10000.0) 

    # Name of the command term, used to access it in the ObservationCfg
    command_name: str = "one_hot_encoding" 

class OneHotCommand(CommandTerm):
    """
    A custom command that samples a one-hot vector at the start of each episode.
    """

    cfg: OneHotCommandCfg

    def __init__(self, cfg: OneHotCommandCfg, env: "ManagerBasedRLEnv"):
        """Initialize the custom command."""
        super().__init__(cfg, env)
        # Allocate buffer for the command, initialized to zero
        self._commands = torch.zeros(
            (self.num_envs, self.cfg.vector_size), 
            dtype=torch.float, 
            device=self.device
        )

    def _resample(self, env_ids: torch.Tensor):
        """Resamples the commands for the given environment IDs."""
        if len(env_ids) == 0:
            return

        # Generate random indices for one-hot
        random_indices = torch.randint(
            self.cfg.vector_size, 
            (len(env_ids),), 
            device=self.device
        )
        
        # Create a new one-hot vector for the sampled environments
        one_hot = torch.zeros(
            (len(env_ids), self.cfg.vector_size), 
            dtype=torch.float, 
            device=self.device
        )
        # Set the corresponding index to 1
        one_hot.scatter_(1, random_indices.unsqueeze(1), 1.0)
        
        # Update the commands
        self._commands[env_ids] = one_hot

    def _update_heading_command(self):
        """We don't need a heading command for this, so we pass."""
        pass
        
    def compute(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return the current command vector."""
        return self._commands, {}


# Custom observation function to retrieve the command.
# This is often needed to explicitly expose the command as an ObsTerm.
def one_hot_command_term(env, command_name: str = "one_hot_encoding"):
    """
    Retrieve the generated one-hot command.
    """
    command = env.command_manager.get_command(command_name)
    return command.compute()[0]