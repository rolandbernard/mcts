
# AlphaZero algorithm

This document gives an explanation for the AlphaZero algorithm.

## Overview

AlphaZero is a reinforcement learning algorithm, that is able to achieve good performance for many
tasks learning only from self-play and without the need for human knowledge <sup>[(1)](#f1)</sup>
<sup>[(2)](#f2)</sup>.

The core of the AlphaZero algorithm is a deep neural network $f$ for approximating from an input
state $s$ two values $(p, v) = f_\theta(s)$. A policy $p$ and an expected value/game outcome $v$. The
policy and value estimates are used in a Monte Carlo tree search as priors and evaluation functions
respectively. The policy $p$ is trained in such a way that it should approximate the resulting
visit counts of running MCTS from the root node $s$.

The learned policy network can also be used standalone, without MCTS <sup>[(1)](#f1)</sup>. In that
scenario AlphaZero is used only as an efficient way to train such a network using only self-play.
This can be useful for cases in which not much time is available for making decisions, since it
requires only a single network evaluation.

The AlphaZero algorithm consists of three main parts:
* _Search_ The search algorithm, that is used during play and uses Monte Carlo tree search for the best
action to use at each step. The search algorithm is also used during self-play for generating
training data.
* _Self-play_ During self-play, the latest network weights are used for playing games, where actions
for both players are decided using the search algorithm. The games are saved and later used for
training.
* _Training_ The weights of the neural network are constantly updated using the generated self-play
data. 

## Model architecture

While it is possible to train two separate networks, one for the policy prediction and one for the
value prediction, the AlphaZero paper suggests using a network with a large common input and two
separate output heads. This architecture promises to be more efficient, since a large part of the
network must be evaluated only once. Additionally, this architecture might have advantages for
training, encouraging hidden representations that are useful for both value and policy estimation.

The deep network used for AlphaZero is inspired by networks for computer vision task. It uses a
number of residual blocks, containing each two 3 by 3 convolutional layers followed by batch
normalization and rectified non-linearity. The common parts of the network contain a single
convolutional layer followed by a number of these residual blocks. Then for each of the value and
policy head, it uses a convolutional layer followed by one or two fully connected layers.


## Search algorithm

The search algorithm employed by AlphaZero is a variation of the Monte Carlo tree search. Each edge
of the search tree now additionally stores a prior probability $P(s, a)$, that is used for guiding
the search using a variant of the PUCT algorithm.

* _Selection_ Selection is performed using a PUCT algorithm, that uses the prior probability as part
of the exploration term. The selection starts from the root of the tree and selects at each step the
action that maximizes the upper confidence bound formula.
* _Expansion_ Once a leaf node is reached, the network is used to evaluate the game state
represented by the leaf $(p, v) = f_\theta(s)$. Then the child nodes are created, and the prior
probability for each edge is set to the probability associated with the action in the policy
prediction $p$. $P(s, a) = p_a$
* _Backpropagation_ The value $v$ predicted by the network, is then used for backpropagation along
the search path. No rollout is performed by AlphaZero, only the value given by the network is used
as the simulation result.

### PUCT formula

The PUCT formula proposed by the AlphaZero paper differs from the ones previously discussed
<sup>[(3)](#f3)</sup>. At each step of selection the action $a \in A_s$ is selected my maximizing
the following:
$$
a = \argmax_{a \in A_s} \left( Q(s, a) + c_\text{putc} P(s, a) \frac{\sqrt{N(s)}}{1 + N(s, a)} \right)
$$
where $Q(s, a)$ is the approximate value of taking action $a$ from state $s$ based on the previous
rounds; $N(s, a)$ is the number of times the action $a$ has been selected in previous rounds; and
$N(s) = \sum_{a \in A_s} N(s, a)$ is the number of times the parent node (representing state $s$)
has been selected in previous rounds. The constant $c_\text{utc}$ controls the balances between
the exploitation and exploration terms.


## Self-Play

During self-play, the latest network weights are used for playing games. Each action is decided by
applying the search algorithm with some fixed amount of simulations. The result of each search is a
probability distribution $\pi$. The probabilities depend on the visit counts of the edges from the
root node. The action that is taken is then sampled from this probability distribution. Once the
game has finished, the game outcome $z$ is determined. For each step in the game, the tuple $(s, \pi, z)$
is saved to be used by the training process.

By using MCTS we obtain action probabilities $\pi$ that are usually much stronger that the raw
network outputs $p$. In this way, MCTS acts as a powerful policy improvement operator. The self-play
with search may similarly be viewed as a powerful policy evaluation operator.


## Training

The training runs concurrently to the self-play process. Random games are sampled from the most
recent self-play games, and $(s, \pi, z)$ is sampled for a random position in each game. The deep
network parameters $\theta$ are that trained using stochastic gradient descent (or some other
optimizer) such that the predictions $(p, v) = f_\theta(s)$ more closely match the self-play
data $(\pi, z)$.

Every few training steps, the weights are saved and made available to the self-play process.
Self-play and training run concurrently, such that self-play data improves the network predictions,
and the improved network improves the self-play quality. This self-improvement cycle then ideally
leads to stronger and stronger play.


## MuZero

MuZero is an extension of the AlphaZero algorithm that in addition to learning the network for value
and policy predictions, also trains a network for encoding the dynamics of the environment
<sup>[(4)](#f4)</sup>. In this algorithm the player is only given the root state and available
actions, it is not given any information on the rules of the game. All game mechanics that are
important for successful planning must therefore be learned by the self-play and training process by
observation of the environment.

This algorithm makes the following changes with respect to AlphaZero:
* There are two additional function: the representation function $s' = h(s)$, that transforms the
state given to the agent into an internal hidden state; and a dynamics function $(s', r) = g(s, a)$,
that gives the next hidden state and expected reward for applying an action.
* At the start of each search, the state is transformed into a hidden state. Then transition between
internal nodes of the search tree is made using the dynamics function.
* During training, not only a single state, but a sequence of successive states is sampled from each
game. This allows the training of the dynamics function.


## References

<b id="f1">(1)</b> Silver, David, et al. "Mastering the game of go without human knowledge." nature 550.7676 (2017):
354-359.

<b id="f2">(2)</b> Silver, David, et al. "Mastering chess and shogi by self-play with a general reinforcement
learning algorithm." arXiv preprint arXiv:1712.01815 (2017).

<b id="f3">(3)</b> Rosin, Christopher D. "Multi-armed bandits with episode context." Annals of Mathematics and
Artificial Intelligence 61.3 (2011): 203-230.

<b id="f4">(4)</b> Schrittwieser, Julian, et al. "Mastering atari, go, chess and shogi by planning with a learned
model." Nature 588.7839 (2020): 604-609.
