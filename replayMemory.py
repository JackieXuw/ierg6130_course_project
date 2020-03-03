import numpy as np
# from deeprl_hw2.core import .
class ReplayMemory:
    """Interface for replay memories.

    We have found this to be a useful interface for the replay
    memory. Feel free to add, modify or delete methods/attributes to
    this class.

    It is expected that the replay memory has implemented the
    __iter__, __getitem__, and __len__ methods.

    If you are storing raw Sample objects in your memory, then you may
    not need the end_episode method, and you may want to tweak the
    append method. This will make the sample method easy to implement
    (just ranomly draw samples saved in your memory).

    However, the above approach will waste a lot of memory (as states
    will be stored multiple times in s as next state and then s' as
    state, etc.). Depending on your machine resources you may want to
    implement a version that stores samples in a more memory efficient
    manner.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, max_size):
        """Setup memory.

        You should specify the maximum size of the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        self.max_size = max_size
        self.is_full = 0
        self.memory = np.empty(max_size).astype(object)
        self.insert_pos = 0

    def append(self, sample):
        self.memory[self.insert_pos] = sample
        if not self.is_full:
            if self.insert_pos == self.max_size - 1:
                self.is_full = 1
        self.insert_pos = (self.insert_pos + 1) % self.max_size


    #def end_episode(self, final_state, is_terminal):
    #    raise NotImplementedError('This method should be overridden')

    def sample(self, batch_size, return_index=False, indexes=None):
        if batch_size > self.max_size:
            raise IndexError('Requested batch_size is bigger than memory max_size!')
        if not self.is_full:
            if batch_size > self.insert_pos:
                raise IndexError('Requested batch_size is bigger than current memory size!')
            # elif self.insert_pos < batch_size + self.window_length:
            #     raise IndexError('Current memory have not enough states to form {} of sampleWindow'.format(batch_size))
            else:
                if type(indexes) == list:
                    if np.max(indexes) >= self.insert_pos:
                        raise IndexError('Requested index is bigger than current memory size!')
                else:
                    indexes = np.random.choice(self.insert_pos, size=batch_size, replace=False)
        else:
            if batch_size  > self.max_size:
                raise IndexError('Maximum of memory have not enough states to form {} of sampleWindow'.format(batch_size))
            if type(indexes) == list:
                if np.max(indexes) > self.max_size :
                        raise IndexError('Requested index is bigger than memory max_size !')
            else:
                indexes = np.random.choice(self.max_size, size=batch_size, replace=False)
              
        # print(indexes)
        if return_index:
            return indexes, self.memory[indexes]
        return self.memory[indexes]

    def clear(self):
        self.is_full = 0
        self.memory = np.empty(self.max_size).astype(object)
        self.insert_pos = 0
        