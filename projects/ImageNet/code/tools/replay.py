
import numpy as np
import tensorflow as tf


class ReplayBuffer:
    def __init__(self, max_episodes=5, episode_length=None, verbose=False):
        assert episode_length is None or episode_length > 0

        self.set_max_episodes(max_episodes)
        self.num_episodes = 0
        self.episode_length = episode_length
        self.replay_buffer = []
        self.verbose = verbose

    def clear(self):
        num_episodes = 0
        self.replay_buffer = []
   
    def reset(self):
        self.episode_length = None
        self.set_max_episodes(None)
        self.clear()

    def get_max_episodes(self):
        return self.max_episodes

    def set_max_episodes(self, new_max_episodes):
        if new_max_episodes is None:
            new_max_episodes = 1
        assert new_max_episodes > 0
        self.max_episodes = new_max_episodes

    def _infer_episode_length(self, episode):
        self.episode_length = len(episode)
    
    def _remove_old_episodes(self):
        assert self.episode_length is not None and self.episode_length > 0
        if self.num_episodes > self.max_episodes:
            if self.verbose:
                print("Removing old episodes")
            self.replay_buffer = self.replay_buffer[self.episode_length:]
            self.num_episodes -= 1

    def update(self, new_entries):
        if self.episode_length is None:
            self._infer_episode_length(new_entries)
        
        self.replay_buffer.extend(new_entries)
        self.num_episodes += 1
        
        self._remove_old_episodes()

    def get(self):
        return self.replay_buffer

    def draw(self, num_samples=None):
        if num_samples is None:
            num_samples = self.episode_length
        if self.verbose:
            print("Drawing %s samples" % num_samples)
        assert len(self.replay_buffer) >= num_samples
        return np.array(self.replay_buffer)[np.random.choice(len(self.replay_buffer), size=num_samples, replace=False)]

