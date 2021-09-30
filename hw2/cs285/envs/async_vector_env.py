import multiprocessing as mp
from cs285.env.vector_env import VectorEnv

class AsyncVectorEnv(VectorEnv):

    def __init__(
        self,
        env_fns,
        observation_space=None,
        action_space=None,
        shared_memory=True,
        copy=True,
        context=None,
        daemon=True,
        worker=None,
    ):
        ctx = mp.get_context(context) # spawn, fork(default), forkserver
        self.env_fns = env_fns
        self.shared_memory = shared_memory
        self.copy = copy
        dummy_env = env_fns[0]()

        if ovservation_space is None:
            observation_space = dummy_env.observation_space
        if action_space is None:
            action_space = dummy_env.action_space
        dummy_env.close()
        del dummy_env
        VectorEnv.__init__(
            self,
            num_envs=len(env_fns),
            observation_space=observatoin_space,
            action_space=action_sapce
        )

        if self.shared_memory:
            try:

