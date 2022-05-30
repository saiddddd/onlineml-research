# online-machine-research

## Introduction
The Online Machine Learning is a service that apply on streaming data pipeline.  
Different from traditional Machine Learning processes. The Online Machine Learning can ingress row-level data, so call `Event`. Training by real-time event data, model can be updated incrementally based on incoming data. Thus, the meaning of _**Online**_ can be thought as **_On-the-data-pipeline_**.


## The Model
The core model is running the _**Hoeffding Tree**_, which can take [Mining High-Speed Data Streams](https://homes.cs.washington.edu/~pedrod/papers/kdd00.pdf) as reference. To put it in a nutshell, It is a tree based model and applies _**Hoeffding Inequality**_ while processing streaming data. The properties of streaming data: 
* Never-ending
* Events coming in any time  

The streaming data is infinite. And, we need to do some trick before applies ML onto an infinitely large dataset. i.e. sampling.
The Hoeffding Inequality provide the minimum number of observations needed while sampling with restrict accuracy error tolerance. e.g.

> if set tolerance error ε≡( Observation - Ground True), is restricted at 1%. what is relative statistic I need while sampling?
 
The Hoeffding Inequality:
> ℙ[ | ν - μ | > ε] ⩽ 2 × exp(-2ε^2Ν)
 
