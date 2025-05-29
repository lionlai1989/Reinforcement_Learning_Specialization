# Fundamentals of Reinforcement Learning

## M1: Welcome to the Course!

## M2: Introduction to Reinforcement Learning

### Key Concepts

Let's define some fundamental notation:
- Let $A_t$ be the action selected at time step $t$
- Let $R_t$ be the corresponding reward at time step $t$

The value of an arbitrary action $a$, denoted as $q_*(a)$, is the expected reward given that action $a$ is selected:

$$q_*(a) \doteq \mathbb{E}[R_t \mid A_t = a]$$

## References
- Reinforcement Learning: An introduction (Second Edition) by Richard S. Sutton and Andrew G. Barto