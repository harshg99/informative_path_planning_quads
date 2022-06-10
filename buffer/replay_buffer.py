import numpy as np

class ReplayBuffer:
  """Abstract base class for replay buffer."""

  def __init__(self, data_spec, capacity):
    """Initializes the replay buffer.

    Args:
      data_spec: A spec or a list/tuple/nest of specs describing
        a single item that can be stored in this buffer
      capacity: number of elements that the replay buffer can hold.
    """
    self.capacity = capacity
    self.occupied_capacity = 0
    self.buffer = {}
    self.buffer_keys = data_spec


  def init_buffer(self):
    """ Initialise the buffer"""
    for keys in self.buffer_keys:
        self.buffer[keys] = np.array([],dtype =np.object)

    self.occupied_capacity = 0

  def add_batch(self, items):
    """Adds a batch of items to the replay buffer.

    Args:
      items: Dictionary of list of transitions or trajectory buffer
    """
    trajectory_length = len(items[items.keys()[0]])
    if self.occupied_capacity + trajectory_length <= self.capacity:
        for key in items.keys():
            if key in self.buffer_keys:
                self.buffer[key].append(np.arrray(items[key],dtype = np.object))
        self.occupied_capacity += trajectory_length

    else:

        shuffled_indices = np.arange(trajectory_length)
        shuffled_indices = np.random.shuffle(shuffled_indices)

        append_index = self.capacity - self.occupied_capacity

        # Sample with replacement
        shuffle_length = trajectory_length - append_index
        replaced_indices = np.choice(np.arange(self.capacity),shuffle_length,replace = False)

        for key in items.keys():
            if key in self.buffer_keys:
                traj_partition_shuffled = np.array(items[key], dtype=np.object)[shuffled_indices]
                self.buffer[key].append(traj_partition_shuffled[:append_index])
                self.buffer[key][replaced_indices] = traj_partition_shuffled[append_index:]

  def get_next(self,sample_batch_size=None):
    """Returns an item or batch of items from the buffer."""
    if sample_batch_size>self.occupied_capacity:
        sample_indices = np.choice(np.arange(self.capacity), sample_batch_size, replace=True)
    else:
        sample_indices = np.choice(np.arange(self.capacity), sample_batch_size, replace=False)

    sampled_buffer = {}
    for key in self.buffer_keys:
        sampled_buffer[key] = self.buffer[key][sample_indices].tolist()

    return sampled_buffer

  def clear(self):
    """Resets the contents of replay buffer"""
    self.init_buffer()

class TrajReplayBuffer(ReplayBuffer):

    def __init__(self,data_spec,capacity):
        '''Initialises the replay buffer
        Args:
            data_spec: A list of keys for the replay buffer
            capacity: the number of data-points for the replay buffer
        '''
        self.capacity = capacity
        self.trajectory_buffer = {}
        self.priority_id = {}
        pass

    @property
    def capacity(self):
        return self.capacity

    @property
    def sample(self):
        pass