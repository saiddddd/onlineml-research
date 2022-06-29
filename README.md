# online-machine-research

## Introduction
The **Online Machine Learning** is a service that could apply on streaming data pipeline.  
Streaming technique is using widely on various topics. More precisely, every raw data acquisition is streamly reading, e.g. sensor reading, transaction events, services log ... etc. 
What if we try to analysis those streaming data from source directly right away that data created.
But how to integrate streaming data pipeline together with ML platform is still an open question nowadays.

Different from traditional Machine Learning processes. The Online Machine Learning can ingress row-level data, so called `Event`. Incrementally learning by real-time event data, model can be updated when new data coming. Thus, the meaning of _**Online**_ can be thought as **_ON-the-data-pipeLINE_**.

## Motivation
Technically speaking, every data sources is collecting streamly. Let's concluded the basic properties for this type of data.

* streaming data can be generated anytime and accessed by AP server.
* Technically, streaming is considered as never-ending source
* New event is more worthy than historical data due to that revealing real-time situations.

Based on aforementioned properties, build up an incrementally-training ML model can leverage steaming data application. To maximize the ability of streaming event application. 

## Streaming Data Pipeline

The high-level picture of streaming data pipeline can be roughly decomposed into three parts. The upstream data source and downstream applications, while logical implementation executing by streaming engine in the middle. 


The concrete architecture designs by most of the companies, The Streaming Engine is powered by Apache Spark (or Flink is an alternative options), Integrated by message queue to decouple streaming engine from upstream services and downstream services is the best choose, e.g. Kafka.

[Imgur](https://i.imgur.com/GM3IIUK.png)

Streaming can be considered as end-less features, thus, the size streaming dataset us various based on the event flux and time period of data taking. An infinity large dataset emerge if data acquisition running as a non-stop service. 
under streaming regime, the real-time data provide profit if ML system can extract information from them.

## Online Machine Learning

Let ML model training processing ingress real-time event data, the system can be design in different architecture. Instead of loading whole dataset into main memory at the same time. Model can ingress row-level data, so called, Event. Caching by specific designed Model data structure, integral statistics for model training control by `Hoeffding Inequality` to make sure statistics at time T is enough to go through one model training process. After that, those event data observation can be removed from memory and no need to keep them from the disk, the strategy called `one-pass learning`.

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