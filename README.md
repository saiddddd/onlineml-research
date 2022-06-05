# online-machine-research

## Introduction
The Online Machine Learning is a service that could apply on streaming data pipeline.  
Data Streaming technique is using widely in various topic. But the integration of machine technique and Data Streaming pipeline is still an open question nowadays.

Different from traditional Machine Learning processes. The Online Machine Learning can ingress row-level data, so called `Event`. Incrementally learning by real-time event data, model can be updated when new data coming. Thus, the meaning of _**Online**_ can be thought as **_On-the-data-pipeline_**.

## Motivation
Technically speaking, every data sources is collecting streamly, e.g. thermometer. Let's concluded the basic properties for this type of data.

* streaming data can be generated anytime and accessed by AP server.
* Technically, streaming is considered as never-ending source
* New event is more worthy than historical data due to that revealing real-time situations.

Based on aforementioned properties, build up an incrementally-training ML model can leverage steaming data application. To maximize the ability of streaming event application. 


## The Model
The core model is running the _**Hoeffding Tree**_, which can take [Mining High-Speed Data Streams](https://homes.cs.washington.edu/~pedrod/papers/kdd00.pdf) as reference. To put it in a nutshell, It is a tree based model and applies _**Hoeffding Inequality**_ while processing streaming data. The properties of streaming data: 
* Never-ending
* Events coming in any time  

The streaming data is infinite. And, we need to do some trick before applies ML onto an infinitely large dataset. i.e. sampling.
The Hoeffding Inequality provide the minimum number of observations needed while sampling with restrict accuracy error tolerance. e.g.

> if set tolerance error ε≡( Observation - Ground True), is restricted at 1%. what is relative statistic I need while sampling?
 

## Mathematics

To explain what Hoeffding inequality going here, introduction of Markov inequality is needed.


The Markov inequality:
![image](https://miro.medium.com/max/516/1*NZLj_hozqZgd7RmN_Vq0Jw.png)

The Chebychev inequality:
![image](https://miro.medium.com/max/460/1*sL22-svOh3JzY9esWwDcxA.png)

The Hoeffding Inequality:

![image](https://cdn-images-1.medium.com/max/800/1*PcKZv4ITALkgmj1Amd9jgA.png#center)

The ν stands for observed value with N sampling, and μ for exact value if calculating whole population is possible. The left-hand side of inequality represents of probability of separation between ν and μ exceeds from error tolerance ε. This probability would be decreasing exponentially while number of sample token increase while shows in the right-hand side of equation.