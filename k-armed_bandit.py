import numpy as np
import matplotlib.pyplot as plt


def generate_action_vals(k, mean=0, var=1):
    """
    Generate action values for k actions according to a normal (Gaussian)
    distribution with specified mean and variance.
    Parameters
    ----------
    k : int
        number of possible actions (k-armed bandit problem)
    mean : float, optional
        mean for normal distribution action values are sampled from
    var : float, optional
        variance for the normal distribution action values are sampled from
    Returns
    ----------
    action_vals : ndarray
        array containing generated action values
    Examples
    ----------
    >>> generate_action_vals(10)
    array([ 1.18227058,  0.47557831, -0.59932958, -0.31229652,  0.23363505,
       -1.23951961,  0.21776762,  1.00188744,  0.53214417, -0.54525233])
    >>> generate_action_vals(10, mean=5, var=0.1)
    array([4.39850609, 4.75175417, 5.23473596, 4.58301931, 5.12304734,
       5.08534613, 5.54636491, 5.08127449, 4.51014831, 5.24290104])
    """
    action_vals = np.random.normal(size=k, loc=mean, scale=np.sqrt(var))
    return action_vals


def give_reward(q, a):
    return np.random.normal(loc=q[a])

def run_simulation(k, eps, steps=1000, runs=2000):
	'''
	Run a simulation for k possible actions using an epsilon-greedy algorithm
	and incremental updates. Executes 'steps' actions for each run. After each
	run, the action values are re-initialized.
	'''
    
    avg_rewards = np.zeros(steps)
    
    for i in range(runs):
        
        q = generate_action_vals(k)
        
        Q = np.zeros(k)
        N = np.zeros(k)
        
        for j in range(steps):
            
            z = np.random.random()
            if z < eps:
                A = np.random.randint(0, k)
            else:
                A = np.argmax(Q)
                
            R = give_reward(q, A)
            N[A] += 1
            Q[A] += (1/N[A])*(R - Q[A])
            
            avg_rewards[j] += (1/(i+1))*(R - avg_rewards[j])
            
    return avg_rewards
    
if __name__ == "__main__":
    
    epsilon = [0, 0.01, 0.1]
    eps_greedy_rewards = []
    for eps in epsilon:
        eps_greedy_rewards.append(run_simulation(k=10, eps=eps))
        
    plt.figure()
    for i in range(len(epsilon)):
        plt.plot(eps_greedy_rewards[i], label='eps = {}'.format(epsilon[i]))
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.show()
        