import LinearRegression.GroupConcat
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{HashingTF, NGram, StopWordsRemover, Tokenizer}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf

object RandomForest {
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



    val rfr=new RandomForestRegressor()
      .setLabelCol("label")


    val pipeline = new Pipeline()
      .setStages(Array(tokenizer,sremover,ngrams,stemmed,hashingTF, rfr)) //stemmed

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
      .option("header","true").save("predictions4.csv")

    println("File ok!")
    println("Spark Stop!")
    val evaluator=new RegressionEvaluator()
      .setLabelCol("label")
     // .setMetricName("mse")


    val evaluator2=new RegressionEvaluator()
      .setLabelCol("label")
      //.setMetricName("rmse")


    //val rmse=evaluator2.evaluate(predictions)

    //val rsquared=evaluator.evaluate(predictions)

    val paramGrid=new ParamGridBuilder()
      .addGrid(rfr.numTrees,Array(20,30,40))
      .addGrid(rfr.maxDepth, Array(1,2,3,4,5))
      .build()

    val cv= new CrossValidator()
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(evaluator)
      .setNumFolds(3)


    val cvModel=cv.fit(sampledDF)

    cvModel.getEstimatorParamMaps
      .zip(cvModel.avgMetrics)
      .maxBy(_._2)

   // val rsquaredCV=evaluator.evaluate(cvModel.transform(test))

  /*  println("=================================================================================================")
    println("MSE on test data = "+rsquared)
    println("=================================================================================================")


    println("=================================================================================================")
    println("RMSE on test data = "+rmse)
    println("=================================================================================================")
 */ }
}
