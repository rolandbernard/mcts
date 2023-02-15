
# Monte Carlo tree search

This document gives an explanation of the Monte Carlo tree search (MCTS) algorithm. It covers the
use case, intuition, simple implementation, and some of its properties.

## Introduction
### Search

The Monte Carlo tree search algorithm is a heuristic search algorithm. For many problems it is
possible to use search to make decisions. Often however the search space is too large to be searched
completely, and an agent must use algorithms that can decide on the next action by observing only a
small part of the search space. Monte Carlo tree search is one such algorithm that tries to prune a
large part of the search tree by focusing on the most promising sequences of actions. 

### Board games

Monte Carlo Search Tree has been extensively used in software for playing board games. For this
document I am therefore going to focus mainly on deterministic zero-sum alternating Markov games,
such as Chess or Go.

For these sorts of games, there is a state space $S$; an action space $A_s$ defining the legal moves
for any given state $s \in S$; and a state transition function $f(s, a)$ defining the successor
state reached from a state $s \in S$ by selecting action $a \in A_s$. 

The exists an optimal value function $v^*(s)$ that determines the outcome for the current player in
state $s$ if both players play optimally <sup>[(1)](#f1)</sup>. If the state $s$ is terminal, the
games rules define this value. Otherwise, we can apply recursion:
$$
v^*(s) =
\begin{cases}
    r(s),  & \text{if $s$ is terminal} \\
    \max_{a \in A_s} -v^*(f(s, a)), & \text{otherwise}
\end{cases}
$$

The value function can be evaluated recursively, by so-called minimax search. However, many games
are too large to be fully explored, and the search must be truncated. This can be achieved by using
an approximate value function $v(s) \approx v^*(s)$.

Another alternative to minimax search is the use of Monte Carlo tree search.


## Simple MCTS
### Algorithm overview

The Monte Carlo tree search algorithm consists of four main parts. The algorithm operates on a
search tree and keeps track of statistics associated to each node, which are used by the different
parts.

Each round of the algorithm starts with the _selection_ of a leaf node. This may be followed by
_expanding_ the selected node, adding child nodes for some or all possible actions. After selection
and expansion, the _rollout_, step is performed and then, at the end of each round, the result of
the rollout is _backpropagated_ from the leaf node up to the root of the search tree.

For each edge $(s, a)$ in the search tree, the algorithm keeps the following statistics: $Q(s, a)$
representing the current estimate for the expected outcome; and $N(s, a)$ representing the number of
times the edge has been traversed during backpropagation.

### Selection

The selection step is an essential part of the algorithm. Along a search path starting from the
root, children are selected based on the stored statistics, until a leaf node is reached.
An important part for the MCTS is that the selection of child nodes is biased such that more
promising parts of the search tree are selected more often. This is the main way in which the
algorithm is able to prune large parts of the search space that are less promising.

There exist multiple ways of selecting among the child nodes, but an often used option is UCT.

#### Upper Confidence Bound

UCT (Upper Confidence Bound 1 for Trees) is a variation of the UCB1 algorithm originally designed
for solving the multi-armed bandit problem <sup>[(2)](#f2)</sup> <sup>[(3)](#f3)</sup>. The
algorithm consists in using the UCB1 formula for each of the internal nodes of the search tree to
select the action to take next. At each internal node representing state $s$ we select among the
actions $A_s$ the action $a$ that maximizes the upper confidence bound given by:
$$
a = \argmax_{a \in A_s} \left( Q(s, a) + c_\text{utc} \sqrt{\frac{ln N(s)}{N(s, a)}} \right)
$$
where $Q(s, a)$ is the approximate value of taking action $a$ from state $s$ based on the previous
rounds; $N(s, a)$ is the number of times the action $a$ has been selected in previous rounds; and
$N(s) = \sum_{a \in A_s} N(s, a)$ is the number of times the parent node (representing state $s$)
has been selected in previous rounds.

The constant $c_\text{utc}$ controls the balances between the exploitation and exploration terms of
the formula, with higher values yielding more exploration. The original paper uses $c_\text{utc} =
\sqrt{2}$ but in practice different values are used by different implementations.

### Expansion

After a leaf node has been selected, it may be expanded. In this step, child nodes are added for
some or all of the legal actions. Expansion can be performed in every round or after the number of
performed rollouts surpasses some threshold. The statistics for the new edges are initialized to
zero: $Q(s, a) = N(s, a) = 0$.

### Rollout

The rollout consists in completing one random playout from the leaf node. The rollout can consist
simply of taking random actions until the game is finished and the final value of the game has been
decided.

### Backpropagation

After the value of the rollout has been determined, the statistics for all edges along the search
path have to be updated. The visit count must be increased for all edges on the path, and the
expected value estimate must be adjusted. How the update is performed can vary depending on the
problem. For the board game example, one must mark the rollout as a win for every second node and as
a loss for the others.

### Finishing the search

After executing a number of rounds, either a fixed number or until the agent runs out of time, a
decision must be made about which action to take. The decision is made based on the statistics for
the actions from the root node. It can be made either based on the estimated value of the visit
counts. Selecting the next action based on the visit count is generally preferred, as they are more
stable <sup>[(1)](#f1)</sup> and most of the time, given the way selection is performed, the maximum
visit count and value estimate coincide.

For making the decision about the next action, the relevant parts of the search tree can be reused
to improve efficiency.

### Advantages

The principal advantage of MCTS is that is able to prune large parts of the search space by focusing
mostly on the most promising actions. This makes MCTS particularly useful for problems with very
large state spaces.

The simple MCTS algorithm also does not require any domain specific evaluation functions, and can be
applied when provided only with the rules of the game.

### Disadvantages

The major disadvantage of MCTS is that some parts of the search space might look superficially weak,
but lead to a subtle strategy that would be preferable (or vice-versa). MCTS might miss such
lines of play because the relevant parts of the search space are not explored.
How prevalent such states are, depends mainly on the problem that is being solved.


## MCTS improvements
### Selection using predictor

If we have a predictor that can gives us an estimate for the value of actions, that predictor can be
used to guide the selection step of the MCTS algorithm to initially favor those actions that are
predicted to be stronger. During node expansion, the predictor will be used to associate some prior
to each action, and that value will then be used in the exploration term of the selection criteria.
Using it in the exploration term ensures that MCTS will still self-correct if the initial values
observed during simulation differ from those predicted, but if the predictions are somewhat
accurate, they can significantly improve the efficiency.
There exist different forms of PUCT (Predictor + UCT) algorithms based on the basic UCT algorithm
that can be used for the selection to account for the use of a predictor <sup>[(4)](#f4)</sup>.

Such a predictor can either use handcrafted heuristics or be trained using machine learning, such as
deep learning. The function can for example be trained to predict the actions taken by human player
using supervised learning, as was used by the original AlphaGo project <sup>[(1)](#f1)</sup>.
Alternatively, the function can be trained by employing reinforcement learning, as it is done in the
AlphaZero algorithm<sup>[(5)](#f5)</sup> <sup>[(6)](#f6)</sup>.

### Rollout

The rollout is used in MCTS as a basic estimate of the value of a given state. Changing the rollout
mechanism can, depending on the domain, significantly impact the performance of the search.

#### Stronger rollout policy

One option is to use some better policy during the rollout, instead of choosing actions uniformly at
random, they can be chosen by using some other probability distribution based on domain specific
heuristic. These policies can be build using hand-crafted heuristics, or again, learned with machine
learning. Generally we want the rollout heuristics to be fast to compute, as they must be evaluated
many times for each simulation.
For example, the original AlphaGo program used a policy network trained using supervised learning to
predict actions taken by expert players and used that as the rollout policy <sup>[(1)](#f1)</sup>.

#### Evaluation function

Another alternative is to replace the payouts with a single evaluation function. Instead of playing
the game to the end, we use a heuristic function to predict the outcome of the game. This method has
the advantage of not requiring to play games to the end, which can have efficiency advantages for
long games. However, to use this approach it is necessary to have a relatively good evaluation
function.
As before, this approach allows for both manually created heuristics and those learned by some
machine learning techniques, either supervised or reinforcement learning.

This technique can also be combined with random playouts, by using a weighted combination of the
playout results and evaluation function, as was done in the original AlphaGo version
<sup>[(1)](#f1)</sup>.

### Parallelism

MCTS can be run concurrently on multiple threads. This can be done in multiple possible ways<sup>[(7)](#f7)</sup>:
* _Leaf Parallelization_ For each rollout step, we can perform many simulations in parallel and combine the results at the
end. With this approach the rest of the algorithm can be kept on a single thread.
* _Root Parallelization_ Multiple threads can build independent trees, and the final decision on which action to take can
then be made by combining the results of the different trees.
* _Tree Parallelization_ Multiple threads can operate on the same tree, and protect concurrent access using locking, or
atomic operations. For this approach virtual loss can be added to the nodes during the selection
phase, to discourage multiple threads form exploding at the same time the same search path. This
virtual loss must then be removed again at the backpropagation step. This is the method presented in
the AlphaGo and AlphaZero papers <sup>[(1)](#f1)</sup> <sup>[(6)](#f6)</sup>.


## References

<b id="f1">(1)</b> Silver, David, et al. "Mastering the game of Go with deep neural networks and tree search." nature
529.7587 (2016): 484-489.

<b id="f2">(2)</b> Auer, Peter, Nicolo Cesa-Bianchi, and Paul Fischer. "Finite-time analysis of the
multiarmed bandit problem." Machine learning 47 (2002): 235-256

<b id="f3">(3)</b> Kocsis, Levente, and Csaba Szepesvári. "Bandit based monte-carlo planning." Machine Learning: ECML
2006: 17th European Conference on Machine Learning Berlin, Germany, September 18-22, 2006
Proceedings 17. Springer Berlin Heidelberg, 2006.

<b id="f4">(4)</b> Rosin, Christopher D. "Multi-armed bandits with episode context." Annals of Mathematics and
Artificial Intelligence 61.3 (2011): 203-230.

<b id="f5">(5)</b> Silver, David, et al. "Mastering the game of go without human knowledge." nature 550.7676 (2017):
354-359.

<b id="f6">(6)</b> Silver, David, et al. "Mastering chess and shogi by self-play with a general reinforcement
learning algorithm." arXiv preprint arXiv:1712.01815 (2017).

<b id="f7">(7)</b> Chaslot, Guillaume MJ -B., Mark HM Winands, and H. Jaap van Den Herik. "Parallel
monte-carlo tree search." Computers and Games: 6th International Conference, CG 2008, Beijing,
China, September 29-October 1, 2008. Proceedings 6. Springer Berlin Heidelberg, 2008.
