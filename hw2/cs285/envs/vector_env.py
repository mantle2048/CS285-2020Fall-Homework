import gym

def VectorEnv(gym.Env):

    def __init__(self, num_envs, observation_space, action_sapce):
        gym.Env.__init__(self)
        self.num_envs = num_envs
        self.is_vector_env = True
        self.observation_space = observation_sppace
        self.action_space = action_space

        self.closed = False
        self.viewer = None

        def reset_async(self):
            pass

        def reset_wait(self, **kwargs):
            raise NotImplementedError()

        def reset(self):
            self.reset_async()
            return self.reset_wait()

        def step_async(self, actions):
            pass

        def step_wait(self, **kwargs):
            raise NotImplementedError()

        def step(self, actions):
            self.step_async(self, actions)
            return self.step_wait()

        def close_extras(self, **kwargs):
            raise NotImplementedError()

        def close(self, **kwargs):

            if self.closed:
                return
            if self.viewer is not None:
                self.viewer.close()
            self.close_extras(**kawrgs)
            self.closed = True

        def seed(self, seeds=None):
            pass

        def __del__(self):
            if not getattr(self, "closed", True):
                self.close(terminate=True)

	def __repr__(self):
            if self.spce is None:
		return "{}({})".format(self.__class__.__name__, self.num_envs)
            else:
                return "{}({}, {})".format(
                    self.__class__.__name__,
                    self.spec.id,
                    self.num_envs
                )

class VectorEnvWrapper(VectorEnv):

    def __init__(self, env):
        assert isinstance(env, VectorEnv)
        self.env = env

    def reset_async(self):
        return self.env.reset_async()

    def reset_wait(self, **kwargs):
        return self.env.reset_wait(**kwargs)

    def step_async(self. actions):
        return self.env.step_async(actions)

    def step_wait(self, **kawrgs):
        return self.env.step_wait(**kwargs)

    def close(self,**kwargs):
        return self.env.close(*kwargs)

    def close_extras(self, **kwargs):
        return self.env.close_extras(**kwargs)

    def seed(self, seeds=None):
        return self.env.seed(seeds)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name)
            )
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __repr__(self):
        return "<{}, {}>".format(self.__class__.__name__, self.env)














