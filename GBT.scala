import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{HashingTF, NGram, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.{GBTRegressor, LinearRegression, RandomForestRegressor}

import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, StringType, StructType}
import org.apache.spark.unsafe.types.UTF8String

import scala.collection.mutable.ArrayBuffer



object GBT {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)


    val ss = SparkSession.builder().master("local[8]").appName("regression").getOrCreate()
    val inputFile = "train.csv"
    val inputFile2 = "product_descriptions.csv"
    val inputeFile3="attributes.csv"

    val descDataDF=ss.read.option("header","true").csv(inputeFile3)


    descDataDF.registerTempTable("data")
    import ss.implicits._
    val ds=descDataDF.groupBy($"product_uid").agg(GroupConcat($"name"))
      .withColumnRenamed("groupconcat$(name)","names")
    ds.printSchema()
    ds.take(20).foreach(println)
    val ds1=descDataDF.groupBy($"product_uid").agg(GroupConcat($"value"))
      .withColumnRenamed("groupconcat$(value)","values")
    ds1.printSchema()
    ds1.take(20).foreach(println)
    val df3=ds.join(ds1,Seq("product_uid"))
    df3.printSchema()
    df3.registerTempTable("data2")
    val outputDF=ss.sqlContext.sql("SELECT product_uid, CONCAT(names,' ',values ) FROM data2")
      .withColumnRenamed("concat(names,  , values)","attributes")
    println("etoimaze sximaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    outputDF.printSchema()
    outputDF.take(10).foreach(println)

    //===================================================================================

    val data2 = ss.read.option("header", "true").csv(inputFile)

    val descDataDF2=ss.read.option("header","true").csv(inputFile2)


    val data2DF=data2.join(descDataDF2,Seq("product_uid"))

    data2DF.printSchema()

    import ss.implicits._
    data2DF.registerTempTable("data")

    val outputDF2=ss.sqlContext.sql("SELECT id,product_uid, relevance, CONCAT(product_title,' ', search_term,' ',product_description) FROM data")
      .withColumnRenamed("concat(product_title,  , search_term,  , product_description)","featuresCol")
      .withColumnRenamed("relevance","labels")

    //outputDF2.take(10).foreach(println)

    outputDF2.printSchema()

    //val descDataDF2=descDataDF.groupBy("product_uid")
    //println(descDataDF2)


    //=====================================================================================================
    //=================================== outputDF + outputDF2 ============================================


    println("Join outputDF and outputDF2")

    val outputDF3=outputDF.join(outputDF2,Seq("product_uid"))

    //outputDF3.take(10).foreach(println)
    outputDF3.printSchema()
    outputDF3.registerTempTable("twoDFs")

    //=====================================================================================================
    //======================================= concat 2 dfs ================================================

    val finalResul=ss.sqlContext.sql("SELECT id, product_uid, labels, CONCAT(featuresCol,' ', attributes) FROM twoDFs")
      .withColumnRenamed("concat(featuresCol,  , attributes)","featuresCol")

    finalResul.take(10).foreach(println)
    finalResul.printSchema()

    //=====================================================================================================
    //======================================= Linear Regression ===========================================


    val udf_toDouble=udf((s:String)=>s.toDouble)
    // Convert label from string to double

    val newDF=finalResul.select($"featuresCol",udf_toDouble($"labels")).withColumnRenamed("UDF(labels)","label")
    println("newDF schema:")
    newDF.printSchema()

    //sample the file RAM error
    val sampledDF = newDF.sample(true,0.01)

    //======================================================= CONCAT TEST DATA =============================================
    val testdata = ss.read.option("header", "true").csv("test.csv")

    val descriptionDF=ss.read.option("header","true").csv("product_descriptions.csv")

    descriptionDF.printSchema()
    //enwsi test.csv me description
    val testdataDF=testdata.join(descriptionDF,Seq("product_uid"))

    testdataDF.printSchema()

    testdataDF.createOrReplaceTempView("testdata_table")

    val testDescDF=ss.sqlContext.sql("SELECT id,product_uid, CONCAT(product_title,' ', search_term,' ',product_description) FROM testdata_table")
      .withColumnRenamed("concat(product_title,  , search_term,  , product_description)","featuresCol")

    testDescDF.printSchema()
    testDescDF.take(10).foreach(println)
    //telos enwsis test.csv me description

    //enwsi stilwn tou Attributes
    val attributesDF=ss.read.option("header","true").csv("attributes.csv")

    attributesDF.createOrReplaceTempView("attributes_table")
    import ss.implicits._
    val nameDf=attributesDF.groupBy($"product_uid").agg(GroupConcat($"name"))
      .withColumnRenamed("groupconcat$(name)","names")
    nameDf.printSchema()
    nameDf.take(20).foreach(println)

    val valueDf=attributesDF.groupBy($"product_uid").agg(GroupConcat($"value"))
      .withColumnRenamed("groupconcat$(value)","values")
    valueDf.printSchema()
    valueDf.take(20).foreach(println)

    val attributesJoinedDF=nameDf.join(valueDf,Seq("product_uid"))
    attributesJoinedDF.printSchema()
    attributesJoinedDF.createOrReplaceTempView("attributesJoined_table")

    val finalAttributesDF=ss.sqlContext.sql("SELECT product_uid, CONCAT(names,' ',values ) FROM attributesJoined_table")
      .withColumnRenamed("concat(names,  , values)","attributes")
    println("printing schema of joined Attributes dataframe")
    finalAttributesDF.printSchema()
    finalAttributesDF.take(10).foreach(println)
    //telos enwsis stilwn tou Attributes


    //enwsi arxeioy Atrributes kai testData

    val testDescAttrDF=testDescDF.join(finalAttributesDF,Seq("product_uid"))
    testDescAttrDF.createOrReplaceTempView("testDescAttr_table")
    testDescAttrDF.printSchema()
    testDescAttrDF.take(10).foreach(println)
    //telos join
    val finalTestDF=ss.sqlContext.sql("SELECT id,product_uid, CONCAT(featuresCol,' ', attributes) FROM testDescAttr_table")
      .withColumnRenamed("concat(featuresCol,  , attributes)","featuresCol")
val sampledFinalTest=finalTestDF.sample(true,0.01)
    finalTestDF.take(10).foreach(println)
    finalTestDF.printSchema()
    println("telos enwsis olwn twn arxeiwn me to test.csv")
    //===================TELOS ENWSIS TESTDATAAA==============================

    //=============================================================================================================

    sampledDF.take(20).foreach(println)
    println("sampled schema:")
    sampledDF.printSchema()

    println("BEFORE TRAINING")
    val tokenizer=new Tokenizer()
      .setInputCol("featuresCol")
      .setOutputCol("ptWords")


    val sremover= new StopWordsRemover()
      .setInputCol("ptWords")
      .setOutputCol("filtered")


    val ngrams=new NGram()
      //.setN(2)
      .setInputCol("filtered")
      .setOutputCol("ngrams")


    /*val stemmed = new Stemmer()
      .setInputCol("ngrams")
      .setOutputCol("stemmed")
      .setLanguage("English")*/

    val stemmed = new Stemmer()
      .setInputCol("ngrams")
      .setOutputCol("stemmed")

    println("Hashing the words...")
    println()

    val hashingTF=new HashingTF()
      .setInputCol("stemmed")
      .setOutputCol("features")
      .setNumFeatures(10000)

    println("Taking a sample of 10%..")
    println()
    //newDF.sample(true,0.1)


    val gbt = new GBTRegressor()
        .setLabelCol("label")
        .setFeaturesCol("features")
        //.setMaxIter(10)

      println("Creating pipeline...")
      println()

      // Chain indexer and GBT in a Pipeline.
      val pipeline = new Pipeline()
        .setStages(Array(tokenizer,sremover,ngrams,stemmed,hashingTF, gbt)) //stemmed

    /* println("Spliting data into training and test datasets..")
     println()
     // Split the data into training and test sets (30% held out for testing)
     val Array(trainingData, testData) = sampledDF.randomSplit(Array(0.7, 0.3), seed = 1234L)*/


    println("Data Splited!")

    val lrModel=pipeline.fit(sampledDF)

    println("pipeline fited ok!")

    val predictions = lrModel.transform(sampledFinalTest).select("id","prediction"/*",label"*/)

    println("test data transfored ok!")


    println("Printing predictions...")
    println()

    println("predictions schema:")
    predictions.printSchema()
    predictions.show(10)

    println("Writing file predictions...")
    println()

    predictions.coalesce(1)
      .write.format("csv")
      .option("header","true").save("predictions3.csv")
    println("File ok!")
    println("Spark Stop!")



    /*
      // Select (prediction, true label) and compute test error.
      val evaluator = new RegressionEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("mse")
      val mse = evaluator.evaluate(predictions)

      val evaluator2 = new RegressionEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("rmse")
      val rmse = evaluator2.evaluate(predictions)

      println("******************************************************************")
      println("Mean Squared Error (MSE) on test data = " + mse)
      println("Root Mean Squared Error (RMSE) on test data = " +rmse)
      println("******************************************************************")

*/
  }


  object GroupConcat extends UserDefinedAggregateFunction {
    def inputSchema = new StructType().add("x", StringType)
    def bufferSchema = new StructType().add("buff", ArrayType(StringType))
    def dataType = StringType
    def deterministic = true

    def initialize(buffer: MutableAggregationBuffer) = {
      buffer.update(0, ArrayBuffer.empty[String])
    }

    def update(buffer: MutableAggregationBuffer, input: Row) = {
      if (!input.isNullAt(0))
        buffer.update(0, buffer.getSeq[String](0) :+ input.getString(0))
    }

    def merge(buffer1: MutableAggregationBuffer, buffer2: Row) = {
      buffer1.update(0, buffer1.getSeq[String](0) ++ buffer2.getSeq[String](0))
    }

    def evaluate(buffer: Row) = UTF8String.fromString(
      buffer.getSeq[String](0).mkString(","))
  }
}
