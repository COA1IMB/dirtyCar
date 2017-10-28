import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.Random;

public class Classifier {

   public static void main(String[] args) throws Exception {
      // image information
      // 28 * 28 grayscale
      // grayscale implies single channel
      int height = 100;
      int width = 100;
      int channels = 1;
      int rngseed = 123;
      Random randNumGen = new Random(rngseed);
      int batchSize = 1;
      int outputNum = 2;

      // Define the File Paths
      File trainData = new File("C:\\Development\\dirtClassifier\\src\\main\\resources\\cars");
      File testData = new File("C:\\Development\\dirtClassifier\\src\\main\\resources\\testCars");

      // Define the FileSplit(PATH, ALLOWED FORMATS,random)

      FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS,randNumGen);
      FileSplit test = new FileSplit(testData,NativeImageLoader.ALLOWED_FORMATS,randNumGen);

      // Extract the parent path as the image label

      ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

      ImageRecordReader recordReaderTrain = new ImageRecordReader(height,width,channels,labelMaker);
      ImageRecordReader recordReaderTest = new ImageRecordReader(height,width,channels,labelMaker);

      // Initialize the record reader
      // add a listener, to extract the name

      recordReaderTrain.initialize(train);
      recordReaderTrain.setListeners(new LogRecordListener());

      recordReaderTest.initialize(test);
      recordReaderTest.setListeners(new LogRecordListener());

      // DataSet Iterator

      DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReaderTrain,batchSize,1,outputNum);
      DataSetIterator testIter = new RecordReaderDataSetIterator(recordReaderTrain,batchSize,1,outputNum);

      // Scale pixel values to 0-1

      DataNormalization scaler = new ImagePreProcessingScaler(0,1);
      scaler.fit(dataIter);
      dataIter.setPreProcessor(scaler);

      DataNormalization scaler2 = new ImagePreProcessingScaler(0,1);
      scaler2.fit(testIter);
      testIter.setPreProcessor(scaler2);

      EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
         .epochTerminationConditions(new ScoreImprovementEpochTerminationCondition(10))
         .epochTerminationConditions(new MaxEpochsTerminationCondition(2))
         .scoreCalculator(new DataSetLossCalculator(testIter, true))
         .evaluateEveryNEpochs(1)
         .modelSaver(new LocalFileModelSaver("."))
         .build();

      EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, getNetworkConfig(), testIter);
      EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

      File locationToSave = new File("NeuralNetwork.zip");
      org.deeplearning4j.nn.api.Model bestModel = result.getBestModel();

      try {
         ModelSerializer.writeModel(bestModel, locationToSave, true);
      } catch (Exception e) {

      }
      recallDirtyness(testIter);
   }

   static MultiLayerConfiguration getNetworkConfig() {
      return new   NeuralNetConfiguration.Builder()
         .seed(128)
         .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
         .iterations(1)
         .learningRate(0.001)
         .updater(Updater.NESTEROVS).momentum(0.9)
         .regularization(true).l2(1e-4)
         .list()
         .layer(0, new DenseLayer.Builder()
            .nIn(100 * 100)
            .nOut(1000)
            .activation("relu")
            .weightInit(WeightInit.XAVIER)
            .build())
         .layer(1, new DenseLayer.Builder()
            .nIn(1000)
            .nOut(1000)
            .activation("relu")
            .weightInit(WeightInit.XAVIER)
            .build())
         .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nIn(1000)
            .nOut(2)
            .activation("softmax")
            .weightInit(WeightInit.XAVIER)
            .build())
         .pretrain(true).backprop(true)
         .setInputType(InputType.convolutional(100,100,1))
         .build();
   }

   public static String recallDirtyness(DataSetIterator testIter){
      testIter.reset();
      String distribution = null;
      MultiLayerNetwork model = null;

      try {
         model = ModelSerializer.restoreMultiLayerNetwork("NeuralNetwork.zip");
      } catch (Exception e) {
      }

      Evaluation eval = new Evaluation(2);

      while(testIter.hasNext()){
         DataSet next = testIter.next();
         INDArray output = model.output(next.getFeatureMatrix());
         eval.eval(next.getLabels(),output);
      }

      System.out.println(eval.stats());

      return distribution;
   }
}
