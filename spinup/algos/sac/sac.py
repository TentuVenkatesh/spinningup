import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.sac import core
from spinup.algos.sac.core import get_vars
from spinup.utils.logx import EpochLogger
from argparse import Namespace

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, observations, act_dim, size):
        self.obs1_buffers = {}
        self.obs2_buffers = {}
        for name, space in observations.items():
            self.obs1_buffers[name] = np.zeros([size] + list(space.shape), dtype=np.float32)
            self.obs2_buffers[name] = np.zeros([size] + list(space.shape), dtype=np.float32)
        self.action_buffer = np.zeros([size] + list(act_dim), dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.done_buffer = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        for name in self.obs1_buffers.keys():
            self.obs1_buffers[name][self.ptr] = obs[name]
            self.obs2_buffers[name][self.ptr] = next_obs[name]
        self.action_buffer[self.ptr] = act
        self.reward_buffer[self.ptr] = rew
        self.done_buffer[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs1 = {}
        obs2 = {}
        for name in self.obs1_buffers.keys():
            obs1[name] = self.obs1_buffers[name][idxs]
            obs2[name] = self.obs2_buffers[name][idxs]
        return dict(obs1=obs1,
                    obs2=obs2,
                    acts=self.action_buffer[idxs],
                    rews=self.reward_buffer[idxs],
                    done=self.done_buffer[idxs])

class SAC:
    def __init__(self, sess, config, action_space, observations,
            actor_critic=core.mlp_actor_critic):
        self.sess = sess
        self.config = config
        self.action_space = action_space
        if not isinstance(observations, dict):
            # Observation space in the case of just one observation.
            observations = { 'input': observations }

        self.observations = observations
        self.act_dim = action_space.shape[0]

        self._build_graph(actor_critic)

        # Experience buffer
        self.replay_buffer = ReplayBuffer(observations=observations,
                act_dim=action_space.shape, size=self.config.replay_size)

    def _build_graph(self, actor_critic):
        # Inputs to computation graph
        self.x_ph = {}
        self.x2_ph = {}
        for name, space in self.observations.items():
            self.x_ph[name] = tf.placeholder(tf.float32, (None,) + space.shape, name='x_{}_ph'.format(name))
            self.x2_ph[name] = tf.placeholder(tf.float32, (None,) + space.shape, name='x_{}_ph'.format(name))
        self.a_ph = tf.placeholder(tf.float32, (None,) + self.action_space.shape, name='a_ph')
        self.r_ph = tf.placeholder(tf.float32, (None,), name='r_ph')
        self.d_ph = tf.placeholder(tf.float32, (None,), name='d_ph')


        log_alpha = tf.Variable(1.0, trainable=True, dtype=tf.float32)
        self.alpha = tf.exp(log_alpha)

        # Main outputs from computation graph
        with tf.variable_scope('main'):
            self.mu, self.pi, self.logp_pi, self.q1, self.q2, self.q1_pi, self.q2_pi, self.v = actor_critic(
                    self.x_ph, self.a_ph,
                    activation=tf.nn.elu,
                    action_space=self.action_space)

        # Target value network
        with tf.variable_scope('target'):
            _, _, _, _, _, _, _, self.v_targ  = actor_critic(self.x2_ph, self.a_ph,
                    activation=tf.nn.elu,
                    action_space=self.action_space)

        # Min Double-Q:
        min_q_pi = tf.minimum(self.q1_pi, self.q2_pi)

        # Targets for Q and V regression
        q_backup = tf.stop_gradient(self.r_ph + self.config.gamma*(1-self.d_ph)*self.v_targ)
        v_backup = tf.stop_gradient(min_q_pi - self.alpha * self.logp_pi)

        # Soft actor-critic losses
        pi_loss = tf.reduce_mean(self.alpha * self.logp_pi - min_q_pi)
        q1_loss = 0.5 * tf.reduce_mean((q_backup - self.q1)**2)
        q2_loss = 0.5 * tf.reduce_mean((q_backup - self.q2)**2)
        v_loss = 0.5 * tf.reduce_mean((v_backup - self.v)**2)
        value_loss = q1_loss + q2_loss + v_loss

        alpha_loss = -tf.reduce_mean(log_alpha * tf.stop_gradient(self.logp_pi + self.config.entropy_level))

        alpha_optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        minimize_alpha = alpha_optimizer.minimize(alpha_loss, var_list=[log_alpha])

        # Policy train op
        # (has to be separate from value train op, because q1_pi appears in pi_loss)
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

        # Value train op
        # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
        value_optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        value_params = get_vars('main/q') + get_vars('main/v')
        with tf.control_dependencies([train_pi_op]):
            train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

        # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        with tf.control_dependencies([train_value_op]):
            target_update = tf.group([tf.assign(v_targ, self.config.polyak*v_targ + (1-self.config.polyak)*v_main)
                                      for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # All ops to call during one training step
        self.step_ops = [pi_loss, q1_loss, q2_loss, v_loss, self.q1, self.q2, self.v, self.logp_pi,
                    train_pi_op, train_value_op, target_update, minimize_alpha]

    def act(self, observation, deterministic=False):
        act_op = self.mu if deterministic else self.pi
        feed_dict = {}
        for name, placeholder in self.x_ph.items():
            feed_dict[placeholder] = observation[name]
        return self.sess.run(act_op, feed_dict=feed_dict)[0]

    def observe(self, obs, action, reward, obs_next, done):
        self.replay_buffer.store(obs, action, reward, obs_next, done)

    def train(self):
        """Does one training iteration."""
        batch = self.replay_buffer.sample_batch(self.config.batch_size)
        feed_dict = {
                     self.a_ph: batch['acts'],
                     self.r_ph: batch['rews'],
                     self.d_ph: batch['done'],
                    }
        for name in self.observations.keys():
            obs1_ph = self.x_ph[name]
            obs2_ph = self.x2_ph[name]
            feed_dict[obs1_ph] = batch['obs1'][name]
            feed_dict[obs2_ph] = batch['obs2'][name]
        return self.sess.run(self.step_ops, feed_dict)

    def initialize(self):
        for op in self._initializer_ops():
            self.sess.run(op)

    def _initializer_ops(self):
        global_init = tf.global_variables_initializer()
        return [global_init, self._assign_targets()]

    def _assign_targets(self):
        # Initializing targets to match main variables
        return tf.group([tf.assign(v_targ, v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])


"""

Soft Actor-Critic

(With slight variations that bring it closer to TD3)

"""
def sac(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols
            for state, ``x_ph``, and action, ``a_ph``, and returns the main
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ``q1``       (batch,)          | Gives one estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and
                                           | ``pi`` for states in ``x_ph``:
                                           | q1(x, pi(x)).
            ``q2_pi``    (batch,)          | Gives the composition of ``q2`` and
                                           | ``pi`` for states in ``x_ph``:
                                           | q2(x, pi(x)).
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()

    config = Namespace(gamma=0.99,
            entropy_level=-1,
            lr=1e-3,
            batch_size=128,
            polyak=0.995,
            replay_size=100000)
    sess = tf.Session()
    sac = SAC(sess, config, env.action_space, env.observation_space)
    sac.initialize()

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': sac.x_ph['input'], 'a': sac.a_ph},
                                outputs={'mu': sac.mu, 'pi': sac.pi, 'q1': sac.q1, 'q2': sac.q2, 'v': sac.v})

    def test_agent(n=10):
        for j in range(n):
            obs, reward, done, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(done or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                obs, reward, done, _ = test_env.step(sac.act({'input': obs[None]}, deterministic=True))
                ep_ret += reward
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    obs, reward, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards,
        use the learned policy.
        """
        if t > start_steps:
            action = sac.act({'input': obs[None]})
        else:
            action = env.action_space.sample()

        # Step the env
        obs_next, reward, done, _ = env.step(action)
        ep_ret += reward
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        done = False if ep_len==max_ep_len else done

        # Store experience to replay buffer
        sac.observe({'input': obs}, action, reward, {'input': obs_next}, done)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        obs = obs_next

        if done or (ep_len == max_ep_len):
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """

            for j in range(ep_len):
                outs = sac.train()
                logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
                             LossV=outs[3], Q1Vals=outs[4], Q2Vals=outs[5],
                             VVals=outs[6], LogPi=outs[7])

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            obs, reward, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0


        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    sac(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
