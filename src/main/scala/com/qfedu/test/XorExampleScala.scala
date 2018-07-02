package com.qfedu.test

import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.{Layer, OptimizationAlgorithm}
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.distribution.UniformDistribution
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.deeplearning4j.nn.conf.layers.OutputLayer.Builder
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
object XorExampleScala {
  def main(args: Array[String]): Unit = {
    // 定义四行两列的零矩阵
    val input = Nd4j.zeros(4,2)
    val labels: INDArray = Nd4j.zeros(4,2)
    // 矩阵第一行赋值
    input.putScalar(Array[Int](0, 0),0)
    input.putScalar(Array[Int](0, 1), 0)
    labels.putScalar(Array[Int](0, 0), 1)
    labels.putScalar(Array[Int](0, 1), 0)

    // when first input=1 and second input=0
    input.putScalar(Array[Int](1, 0), 1)
    input.putScalar(Array[Int](1, 1), 0)
    // then xor is true, therefore the second output neuron fires
    labels.putScalar(Array[Int](1, 0), 0)
    labels.putScalar(Array[Int](1, 1), 1)

    // same as above
    input.putScalar(Array[Int](2, 0), 0)
    input.putScalar(Array[Int](2, 1), 1)
    labels.putScalar(Array[Int](2, 0), 0)
    labels.putScalar(Array[Int](2, 1), 1)

    // when both inputs fire, xor is false again - the first output should
    // fire
    input.putScalar(Array[Int](3, 0), 1)
    input.putScalar(Array[Int](3, 1), 1)
    labels.putScalar(Array[Int](3, 0), 1)
    labels.putScalar(Array[Int](3, 1), 0)

    // 构造数据
    val ds: DataSet = new DataSet(input,labels)
    // 设置模型
    val builder: NeuralNetConfiguration.Builder = new NeuralNetConfiguration.Builder()
    // 迭代次数
    builder.iterations(10000)
    // 学习率
    builder.learningRate(0.1)
    // fixed seed for the random generator, so any run of this program
    // brings the same results - may not work if you do something like
    // ds.shuffle()
    builder.seed(123)

    builder.useDropConnect(false)

    builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)

    builder.biasInit(0)

    builder.miniBatch(false)

    val listbuilder = builder.list()
    //添加隐藏层
    val hiddenbuilder = new DenseLayer.Builder()
    // two input connections - simultaneously defines the number of input
    // neurons, because it's the first non-input-layer
    hiddenbuilder.nIn(2)
    // number of outgooing connections, nOut simultaneously defines the
    // number of neurons in this layer
    hiddenbuilder.nOut(4)
    //激活函数
    hiddenbuilder.activation(Activation.SIGMOID)
    //参数初始化
    hiddenbuilder.weightInit(WeightInit.DISTRIBUTION)
    hiddenbuilder.dist(new UniformDistribution(0, 1))

    listbuilder.layer(0,hiddenbuilder.build())

    val outputLayerBuilder=new Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)

    // must be the same amout as neurons in the layer before
    outputLayerBuilder.nIn(4)
    // two neurons in this layer
    outputLayerBuilder.nOut(2)
    outputLayerBuilder.activation(Activation.SOFTMAX)
    outputLayerBuilder.weightInit(WeightInit.DISTRIBUTION)
    outputLayerBuilder.dist(new UniformDistribution(0, 1))
    listbuilder.layer(1, outputLayerBuilder.build)

    // no pretrain phase for this network
    listbuilder.pretrain(false)

    listbuilder.backprop(true)

    val conf = listbuilder.build()

    val net = new MultiLayerNetwork(conf)

    net.init()

    net.setListeners(new ScoreIterationListener(100))


    val layers: Array[Layer] = net.getLayers

    var totalNumParams=0

    for(i<- 0 until(layers.size)){
      val nparams = layers(i).numParams()
      println("Number of parameters in layer " + i + ": " + nparams)
      totalNumParams+=nparams
    }

    println("Total number of network parameters: " + totalNumParams)

    // here the actual learning takes place,训练模型
    net.fit(ds)

    // create output for every training sample
    val output = net.output(ds.getFeatureMatrix)
    println(output)

    // let Evaluation prints stats how often the right output had the
    // highest value
    val eval = new Evaluation(2)
    eval.eval(ds.getLabels, output)//评估模型
    println(eval.stats)

  }
}
