TL4ICF_DEMO is a project aimming to repeoduce the example of transfer learning in this [paper](https://ieeexplore.ieee.org/document/8932676?arnumber=8932676).

- [Introduction](#introduction)
- [Generate data](#generate-data)
- [Train Model](#train-model)
- [Result](#result)


# Introduction 
Just like the article says:
> Computer simulations of complex physical systems are often modeled at varying levels of fidelity. Computationally inexpensive, low fidelity models are used to explore vast design spaces for optimal settings, and expensive high fidelity models might be used in interesting regions of design space to compute predictions of planned experiments. The high fidelity simulations are often more accurate and reliable than the approximate models, but the expense of running the simulation often prevents their use in large parameter scans. It might be possible to create models that emulate high fidelity simulations with reduced computational cost with TL.

In other words, **Transfer learning** from this perspectsive is more like **Pre-Train and Fine tune**. To demonstrate the utility of TL, namely the hierarchical TL in the article, we consider a unlinear function:
$$f(x)=xe^{ax}$$
To use neutral network to learn this function, it's easy to generate data and then establish a simple net. However, this may not work when the dataset is not so large enough.

In the following chapters, we introduce the method that pre-train the net on the low-fidelity database, which can be generate to fully modify the model, and then fine tune it on the sparse and high-fidelity database. We also compare the effects of different model.

# Generate data
Since we want to learn the function form low-fidelity database, we consider the taylor expansion of it:
$$f(x)=xe^{ax}=x+ax^2+\cdots$$

In "DATA" folder, there are three codes to generate the database of $f_{low}(x)=x$, $f_{high}(x)=x+ax^2$, $f_{exp}(x)=xe^{ax}$ seperately. The size of database is dependent on your model and paramater space, and we recommend the following parameter combinations:
|Model name|Description|Learning Path|Size of Database|
|---|---|---|---|
|Exp|Experiment|$xe^{ax}$|100|
|Low2Exp|Low to Experiment|$x\Rightarrow xe^{ax}$|100 $\Rightarrow$ 25|
|High2Exp|High to Experiment|$x+ax^2\Rightarrow xe^{ax}$|100 $\Rightarrow$ 25|
|Low2High2Exp|Low to High then Experiment|$x\Rightarrow x+ax^2\Rightarrow xe^{ax}$|100 $\Rightarrow$ 50 $\Rightarrow$ 25|

with the parameter space:
$$a\in[0,1],x\in [-1,1]$$

We also provide a simple code to visualise the data distribution of your database named "plot".

> It should be paid attention that the database of each model is not same, not only in size but also in concrete distribution. The sample method is random sampling.

# Train Model
The following folders:
- exp_model
- low2exp_model
- high2exp_model
- low2high2exp_model
contain the corresponding codes. Take low2high2exp_model as example:
- train_low.py is used to train the pre_train model from low-fodelity database;
- train_low_high_exp.py is used to find tune the model from high_fidelity and authentic database;
- low_high_exp_logs_{time} can be opened by tensorboard to visiualise the error-epoch change;
- {model}.pth is the model file;
- plot_reg_low_high_exp.py is used to plot the regreesion figure which will be save to the "fig" folder.

> Be careful that there are two same code named train_low.py in low2exp_model and low2high2exp_model, and each of them produces the same model file named low_model.pth simutaneously in the two folders.

# Result
The eight figures will be saved in the fig folders and there exists a code name plot_table.py to combine them.

The variance of each model of the existing model file:
|Model name|Variance|Epochs|
|---|---|---|
|Exp|5.23e-4|2000|
|Low2Exp|3.86e-3|1000&5000|
|High2Exp|4.01e-3|1000&4000|
|Low2High2Exp|1.45e-3|1000&3000&3000|

It's intuitive because the fourth model constains the second most information. We can also predict that if we train exp_model with only 25 piece of data, the exp_model cannot compete with the Low2High2Exp_model. Actually the variance is 1e-2.
