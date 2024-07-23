# It all started with the perceptron
Building a neural network can literally be done in a few lines of code. However, this is with the help of fabulous libraries that take away all of the heavy lifting for you. Building something from scratch arms you with the knowledge you need to understand how these things work. Let's start with a simple example, a single perceptron neural network.

We'll also look at XOR a little later and the problems this presents. 

## What is our goal?
The aim is to predict the logical outcome for two bits using either an AND or an OR operation. The table shows the outcome for each combination of A and B, firstly using AND and then using OR. We also show XOR (exclusive OR) - more on this later.

| A | B | AND | OR | XOR |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | 1 | 1 |
| 1 | 0 | 0 | 1 | 1 |
| 1 | 1 | 1 | 1 | 0 |

- The outcome of and AND operation is 1 only when A and B are both 1.
- The outcome of and OR operation is 1 when either A or B is 1.
- The outcome of and XOR operation is 1 when only when A and B are different to each other.

## What does our perceptron look like?
The [`perceptron.py`](perceptron.py)
