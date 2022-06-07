# online-machine-research

## Introduction
The **Online Machine Learning** is a service that could apply on streaming data pipeline.  
Streaming technique is using widely in various topic, or more precisely, every raw data acquisition is working streamly, e.g. thermometer reading, vibration detection from tools in manufacture industries, for clicking data extracted from Web-services, or financial transaction data. 
What if we try to analysis those streaming data from source directly right away that data created.
But the integration of machine technique and Data Streaming pipeline is still an open question nowadays.

Different from traditional Machine Learning processes. The Online Machine Learning can ingress row-level data, so called `Event`. Incrementally learning by real-time event data, model can be updated when new data coming. Thus, the meaning of _**Online**_ can be thought as **_ON-the-data-pipeLINE_**.

## Motivation
Technically speaking, every data sources is collecting streamly, e.g. thermometer. Let's concluded the basic properties for this type of data.

* streaming data can be generated anytime and accessed by AP server.
* Technically, streaming is considered as never-ending source
* New event is more worthy than historical data due to that revealing real-time situations.

Based on aforementioned properties, build up an incrementally-training ML model can leverage steaming data application. To maximize the ability of streaming event application. 

## Streaming Data

## The Model
The core model is running the _**Hoeffding Tree**_, which can take [Mining High-Speed Data Streams](https://homes.cs.washington.edu/~pedrod/papers/kdd00.pdf) as reference. To put it in a nutshell, It is a tree based model and applies _**Hoeffding Inequality**_ while processing streaming data. The properties of streaming data: 
* Never-ending
* Events coming in any time  

The streaming data is infinite. And, we need to do some trick before applies ML onto an infinitely large dataset. i.e. sampling.
The Hoeffding Inequality provide the minimum number of observations needed while sampling with restrict accuracy error tolerance. e.g.

> if set tolerance error $\epsilon \equiv$ ( Observation - Ground True), is restricted at 1%. what is relative statistic I need while sampling?
 

## Mathematics

To explain why Hoeffding Inequality is applied in this case, let us check what Markov Inequality is.
The Markov inequality:

$$\mathbb{P}(X\geq a)\leq \frac{\mathbb{E}(X)}{a} \tag{1}$$

tailing us for non-negative random variable set $\mathbf{X}: \mathcal{x} \subset \mathbb{R} \wedge\forall x > 0$ and we know the expection value of the random variable set $\mathbb{E}(X)$, for any distribution $X$ could be, the probability of $x$ given a random value greater than $a$ is smaller than $\frac{\mathbb{E}(X)}{a}$. For intuitivly thinkng, if we given a extremly large number $a$ which is far away from expective value and cause $\frac{\mathbb{E}(X)}{a}\sim 0$, thus a random number $x$ almost imposible to exceed it.  


The Chebychev inequality:
$$\mathbb{P}(\mid X - \mu \mid \geq c) \leq \frac{\sigma^{2}}{c^{2}}$$

The Hoeffding Inequality:

$$\mathbb{P}[ \mid\nu-\mu\mid > \epsilon ] \leq 2 exp(-2\epsilon^{2}N)$$


The ν stands for observed value with N sampling, and μ for exact value if calculating whole population is possible. The left-hand side of inequality represents of probability of separation between ν and μ exceeds from error tolerance ε. This probability would be decreasing exponentially while number of sample token increase while shows in the right-hand side of equation.