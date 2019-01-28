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

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

class SAC:
    def __init__(self, sess, config, action_space, observation_space):
        self.sess = sess
        self.config = config
        self.action_space = action_space
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]

        self._build_graph()

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim,
                act_dim=self.act_dim, size=self.config.replay_size)

    def _build_graph(self):
        # Inputs to computation graph
        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = core.placeholders(
                self.obs_dim, self.act_dim, self.obs_dim, None, None)

        # Main outputs from computation graph
        with tf.variable_scope('main'):
            self.mu, self.pi, self.logp_pi, self.q1, self.q2, self.q1_pi, self.q2_pi, self.v = core.mlp_actor_critic(
                    self.x_ph, self.a_ph,
                    hidden_sizes=[32, 8],
                    activation=tf.nn.elu,
                    action_space=self.action_space)

        # Target value network
        with tf.variable_scope('target'):
            _, _, _, _, _, _, _, self.v_targ  = core.mlp_actor_critic(self.x2_ph, self.a_ph,
                    hidden_sizes=[32, 8],
                    activation=tf.nn.elu,
                    action_space=self.action_space)

        # Min Double-Q:
        min_q_pi = tf.minimum(self.q1_pi, self.q2_pi)

        # Targets for Q and V regression
        q_backup = tf.stop_gradient(self.r_ph + self.config.gamma*(1-self.d_ph)*self.v_targ)
        v_backup = tf.stop_gradient(min_q_pi - self.config.alpha * self.logp_pi)

        # Soft actor-critic losses
        pi_loss = tf.reduce_mean(self.config.alpha * self.logp_pi - self.q1_pi)
        q1_loss = 0.5 * tf.reduce_mean((q_backup - self.q1)**2)
        q2_loss = 0.5 * tf.reduce_mean((q_backup - self.q2)**2)
        v_loss = 0.5 * tf.reduce_mean((v_backup - self.v)**2)
        value_loss = q1_loss + q2_loss + v_loss

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
                    train_pi_op, train_value_op, target_update]

    def act(self, observation, deterministic=False):
        act_op = self.mu if deterministic else self.pi
        return self.sess.run(act_op, feed_dict={self.x_ph: observation.reshape(1,-1)})[0]

    def observe(self, obs, action, reward, obs_next, done):
        self.replay_buffer.store(obs, action, reward, obs_next, done)

    def train(self):
        """Does one training iteration."""
        batch = self.replay_buffer.sample_batch(self.config.batch_size)
        feed_dict = {self.x_ph: batch['obs1'],
                     self.x2_ph: batch['obs2'],
                     self.a_ph: batch['acts'],
                     self.r_ph: batch['rews'],
                     self.d_ph: batch['done'],
                    }
        return self.sess.run(self.step_ops, feed_dict)

    def initialize(self):
        for op in self._initializer_ops():
            self.sess.run(op)

    def _initializer_ops(self):
        global_init = tf.global_variables_initializer()
        # Initializing targets to match main variables
        target_init = tf.group([tf.assign(v_targ, v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])
        return [global_init, target_init]


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
            alpha=0.2,
            lr=1e-3,
            batch_size=128,
            polyak=0.995,
            replay_size=100000)
    sess = tf.Session()
    sac = SAC(sess, config, env.action_space, env.observation_space)
    sac.initialize()

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': sac.x_ph, 'a': sac.a_ph},
                                outputs={'mu': sac.mu, 'pi': sac.pi, 'q1': sac.q1, 'q2': sac.q2, 'v': sac.v})

    def test_agent(n=10):
        for j in range(n):
            obs, reward, done, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(done or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                obs, reward, done, _ = test_env.step(sac.act(obs, deterministic=True))
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
            action = sac.act(obs)
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
        sac.observe(obs, action, reward, obs_next, done)

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
