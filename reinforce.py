from typing import Iterable
import numpy as np
import tensorflow as tf
from gym.wrappers.monitoring.video_recorder import VideoRecorder

class ReplayMemory:
    def __init__(self,config):
        self.config = config
        self.actions = np.empty((self.config.mem_size),dtype=np.int32)
        self.rewards = np.empty((self.config.mem_size),dtype=np.int32)
        self.observations = np.empty((self.config.mem_size,self.config.obs_dims),dtype=np.int32)


class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        # TODO: implement here
        self.state_dims = state_dims
        self.obs_dims = int(state_dims/2)
        self.num_actions = num_actions
        self.alpha = alpha
        beta1 = 0.9
        beta2 = 0.999
        self.X = tf.placeholder(tf.float32, [None, self.obs_dims])
        self.Y = tf.placeholder(tf.int32, [None,1])
        self.ret = tf.placeholder(tf.float32,[None,1])
        
        layer1 = tf.layers.dense(self.X,32,activation=tf.nn.relu)
        layer2 = tf.layers.dense(layer1,32,activation=tf.nn.relu)
        self.yhat = tf.layers.dense(layer2,self.num_actions)
        self.actSel = tf.multinomial(logits=self.yhat,num_samples=1)
        loss_fn = tf.reduce_mean(self.ret*tf.losses.sparse_softmax_cross_entropy(logits= self.yhat,labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha, beta1=beta1, beta2=beta2)
        self.train_net = optimizer.minimize(loss_fn)
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)

        # Tips for TF users: You will need a function that collects the probability of action taken
        # actions; i.e. you need something like
        #
            # pi(.|s_t) = tf.constant([[.3,.6,.1], [.4,.4,.2]])
            # a_t = tf.constant([1, 2])
            # pi(a_t|s_t) =  [.6,.2]
        #
        # To implement this, you need a tf.gather_nd operation. You can use implement this by,
        #
            # tf.gather_nd(pi,tf.stack([tf.range(tf.shape(a_t)[0]),a_t],axis=1)),
        # assuming len(pi) == len(a_t) == batch_size

    def __call__(self,s) -> int:
        # TODO: implement this method
        pred = self.sess.run(self.actSel,feed_dict={self.X:s.reshape(-1,self.obs_dims)})
        return pred[0][0]


    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        # TODO: implement this method
        ret = np.array(gamma_t*delta).reshape(-1,1)
        self.sess.run(self.train_net,feed_dict={self.X:s.reshape(-1,self.obs_dims),self.Y:np.array([[a]],dtype=int),self.ret:ret})

    

    
class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        # TODO: implement here
        self.state_dims = state_dims
        self.obs_dims = int(state_dims/2)
        self.alpha = alpha
        beta1 = 0.9
        beta2 = 0.999
        self.X = tf.placeholder("float", [None, self.obs_dims])
        self.Y = tf.placeholder("float", [None, 1])
        layer1 = tf.layers.dense(self.X,32,activation=tf.nn.relu)
        layer2 = tf.layers.dense(layer1,32,activation=tf.nn.relu)
        self.yhat = tf.layers.dense(layer2,1)
        loss_fn = 0.5 * tf.losses.mean_squared_error(self.Y, self.yhat)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha, beta1=beta1, beta2=beta2)
        self.train_net = optimizer.minimize(loss_fn)
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)

    def __call__(self,s) -> float:
        # TODO: implement this method
        pred = self.sess.run(self.yhat,feed_dict={self.X:s.reshape(-1,self.obs_dims)})
        return pred[0][0]

    def update(self,s,G):
        # TODO: implement this method
        self.sess.run(self.train_net,feed_dict={self.X:s.reshape(-1,self.obs_dims),self.Y:np.array([[G]])})
        return None



def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    run:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    # Video Path (dummy)
    video_path = 'videos/reinforce_run_ep_.mp4'

    G_0 = []
    obs_mask = np.array([[1,0,0,0],[0,0,1,0]])
    for e_i in range(num_episodes):
        # if G_0:print("{} - {}".format(e_i,max(G_0)))
        s = env.reset()
        z = np.matmul(obs_mask,s)
        done = False
        traj = []
        # Video recording setup #
        if e_i == 10 or e_i == 100 or e_i == 900:
            video_path = 'videos/reinforce_run{}_ep{}_.mp4'.format(run, e_i)
            video_recorder = VideoRecorder(env, video_path, enabled=video_path is not None)
            # env.unwrapped.render()
        else:
            video_recorder = VideoRecorder(env, video_path, enabled=False)
        while not done:
            a = pi(z)
            # if a ==1: print("A1")
            s_prime, r_t, done, _ = env.step(a)
            # capture frame
            video_recorder.capture_frame()
            # Mask
            z_prime = np.matmul(obs_mask, s_prime)
            traj.append((z,a,r_t,z_prime))
            z = z_prime
        for i,t_tup in enumerate(traj):
            G = sum([gamma**j*s_tup[2] for j,s_tup in enumerate(traj[i:])])
            if i==0: G_0.append(G)
            delta = G-V(t_tup[0])
            V.update(t_tup[0],G)
            pi.update(t_tup[0],t_tup[1],gamma**i,delta)
        video_recorder.close()

    return  G_0, pi

