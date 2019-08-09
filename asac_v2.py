import numpy as np
import tensorflow as tf
import time
import core
from core import get_vars
import BaxterEnv
from BaxterEnv import baxter
import rospy


class Alpha:
    def __init__(self, alpha_start=0.2, alpha_end=1e-2, delta=0.001):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha = alpha_start
        self.delta = delta
    def __call__(self, ret=None):
        return self.alpha

    def update_alpha(self, Adv, increase):
        mean_A_square = np.maximum(np.mean(Adv*Adv), 2*self.alpha**2*self.delta)
        delta_alpha = self.alpha**2 * np.sqrt(2*self.delta/mean_A_square)
        delta_alpha = delta_alpha if delta_alpha < self.alpha / 2 else self.alpha / 2
        if increase and self.alpha + delta_alpha < 1.0:
            self.alpha += delta_alpha
        elif not increase and self.alpha > self.alpha_end:
            self.alpha -= delta_alpha

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])



def asac_v2(actor_critic=core.mlp_actor_critic, seed=0, ac_kwargs=dict(),
            steps_per_epoch=5000, epochs=200, replay_size=int(1e6), gamma=0.99,
            polyak=0.995, lr=0.001, alpha_start=0.2, batch_size=100, start_steps=10000,
            max_ep_len=1000, logger_kwargs=dict(), save_freq=1, loss_threshold=0.0001,
            delta=0.02, sample_step=2000):

    alpha = Alpha(alpha_start=alpha_start, delta=delta)
    alpha_t = alpha()

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = baxter()
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = 0.1

    # Share information about action space with policy architecture

    # Inputs to computation graph
    #x_ph, a_ph, x2_ph, r_ph, d_ph, ret_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None, None)
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)
    alpha_ph = core.scale_holder()
    # Main outputs from computation graph
    #R, R_next = return_estimate(x_ph, x2_ph, **ac_kwargs)
    with tf.variable_scope('main'):
        mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v, Q, Q_pi, R = actor_critic(x_ph, a_ph, **ac_kwargs)
    # Target value network
    with tf.variable_scope('target'):
        _,_,_,_,_,_,_,v_targ, _, _, R_targ = actor_critic(x2_ph, a_ph, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in
                       ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main/Q', 'main/R', 'main'])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t v: %d, \t Q: %d, \t R: %d, \t total: %d\n')%var_counts)
    # Min Double-Q:
    min_q_pi = tf.minimum(q1_pi, q2_pi)

    # Targets for Q and V regression
    q_backup = tf.stop_gradient(r_ph + gamma*(1 - d_ph)*v_targ)
    v_backup = tf.stop_gradient(min_q_pi - alpha_ph *logp_pi)
    Q_backup = tf.stop_gradient(r_ph + gamma*(1 - d_ph)*R_targ)
    R_backup = tf.stop_gradient(Q_pi)
    adv = Q_pi - R
    dQ = Q_backup * (R - Q)

    pi_loss = tf.reduce_mean(alpha_ph * logp_pi - q1_pi)
    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
    v_loss = 0.5 * tf.reduce_mean((v_backup - v)**2)
    Q_loss = 0.5*tf.reduce_mean((Q_backup - Q)**2)
    R_loss = 0.5*tf.reduce_mean((R_backup - R)**2)
    value_loss = q1_loss + q2_loss + v_loss + Q_loss + R_loss
    # Policy train op
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    value_params = get_vars('main/q') + get_vars('main/v') + get_vars('main/Q') + get_vars('main/R')
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)
    """
    R_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_R_op = R_optimizer.minimize(R_loss, var_list=get_vars('R'))
    """
    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # All ops to call during one training step
    step_ops = [pi_loss, q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi,
                train_pi_op, train_value_op, target_update, R_loss, Q_loss, v_targ]

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                            for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Setup model saving

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1, -1)})

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    obs1_epi = np.zeros([2*max_ep_len, obs_dim], dtype=np.float32)
    obs2_epi = np.zeros([2*max_ep_len, obs_dim], dtype=np.float32)
    act_epi = np.zeros([2*max_ep_len, act_dim], dtype=np.float32)
    rew_epi = np.zeros([2*max_ep_len], dtype=np.float32)
    done_epi = np.zeros([2*max_ep_len], dtype=np.float32)
    ptr_epi = 0
    alpha_update = False
    epi_num = 0
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        if t > start_steps:
            a = get_action(o["feature"])
        else:
            a = 0.1 - np.random.sample(act_dim)*0.2
        # Step the env
        o2, r = env.step(a)
        ep_ret += r
        ep_len += 1
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o["feature"], a, r, o2["feature"], d)
        obs1_epi[ptr_epi] = o["feature"]
        obs2_epi[ptr_epi] = o2["feature"]
        act_epi[ptr_epi] = a
        rew_epi[ptr_epi] = r
        done_epi[ptr_epi] = d
        ptr_epi += 1

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2
        if d or (ep_len == max_ep_len):
            epi_num += 1
            print("epi : {}, alpha : {}, return : {}".format(epi_num, alpha_t, ep_ret))
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            """
            rew_epi[ptr_epi] = sess.run(R, feed_dict={x_ph: [o]})[0]
            rets_epi = scipy.signal.lfilter([1], [1, float(-gamma)], rew_epi[::-1], axis=0)[::-1]
            rets_epi = rets_epi[:-1]
            """
            """
            v_epi = sess.run(R, feed_dict={x_ph: obs_epi})
            q_epi, adv_epi = sess.run([Q, adv], feed_dict={x_ph: obs_epi[:-1], a_ph: act_epi})
            rets_epi = rew_epi + gamma*v_epi[1:]
            if t > start_steps:
                alpha.update_alpha(adv_epi, np.mean(rets_epi*(v_epi[:-1]-q_epi)) > 0)
                alpha_t = alpha()
            print("{} {}".format(np.mean(rets_epi*(v_epi[:-1]-q_epi)), alpha_t))
            """
            if ptr_epi >= max_ep_len:
                feed_dict = {x_ph: obs1_epi[:ptr_epi],
                             x2_ph: obs2_epi[:ptr_epi],
                             a_ph: act_epi[:ptr_epi],
                             r_ph: rew_epi[:ptr_epi],
                             d_ph: done_epi[:ptr_epi]}
                adv_epi, Q_epi, R_epi = sess.run([adv, Q, R], feed_dict)
                R_next_epi = sess.run(R, feed_dict={x_ph: obs2_epi[:ptr_epi]})
                dQ_epi = (rew_epi[:ptr_epi] + gamma*(1-done_epi[:ptr_epi])*R_next_epi) * (R_epi - Q_epi)
                """
                ret_epi = np.zeros([ptr_epi], dtype=np.float32)
                for i in np.arange(ptr_epi)[::-1]:
                    if i == ptr_epi - 1:
                        R_next_epi = sess.run(R, feed_dict={x_ph: [obs2_epi[i]]})[0]
                        ret_epi[i] = rew_epi[i] + gamma*(1 - done_epi[i])*R_next_epi
                    else:
                        ret_epi[i] = rew_epi[i] + gamma*(1 - done_epi[i])*ret_epi[i+1]
                dQ_epi = ret_epi * (R_epi - Q_epi)
                """
                if t > start_steps:
                    alpha.update_alpha(adv_epi, np.mean(dQ_epi) > 0)
                    alpha_t = alpha()
                    print("{} {}".format(np.mean(dQ_epi), alpha_t))
                obs1_epi = np.zeros([max_ep_len*2, obs_dim], dtype=np.float32)
                obs2_epi = np.zeros([max_ep_len*2, obs_dim], dtype=np.float32)
                act_epi = np.zeros([max_ep_len*2, act_dim], dtype=np.float32)
                rew_epi = np.zeros([max_ep_len*2], dtype=np.float32)
                done_epi = np.zeros([max_ep_len*2], dtype=np.float32)
                ptr_epi = 0
            """
            batch = replay_buffer.sample_batch(1000)
            feed_dict = {x_ph: batch['obs1'],
                         x2_ph: batch['obs2'],
                         a_ph: batch['acts'],
                         r_ph: batch['rews'],
                         d_ph: batch['done'],
                         alpha_ph: alpha_t}
            dQ_epi = sess.run(dQ, feed_dict)
            """
            for j in range(ep_len):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done'],
                             alpha_ph: alpha_t
                            }
                outs = sess.run(step_ops, feed_dict)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=[400, 300])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='asac')
    parser.add_argument('--alpha_start', type=float, default=0.2)
    parser.add_argument('--delta', type=float, default=0.01)
    parser.add_argument('--sample_step', type=int, default=2000)
    parser.add_argument('--loss_threshold', type=float, default=0.0001)
    parser.add_argument('--max_ep_len', type=int, default=1000)
    args = parser.parse_args()

    asac_v2(actor_critic=core.mlp_actor_critic, polyak=args.polyak,
            ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
            gamma=args.gamma, seed=args.seed, epochs=args.epochs,
            alpha_start=args.alpha_start, delta=args.delta,
            sample_step=args.sample_step, loss_threshold=args.loss_threshold, lr=args.lr)
