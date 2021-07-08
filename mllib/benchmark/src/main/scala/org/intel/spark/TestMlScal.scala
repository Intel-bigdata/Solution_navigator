package org.intel.spark

import org.apache.spark.ml.linalg.BLAS.scal
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.Vectors

object TestMlScal {
  def main(args: Array[String]): Unit = {
    def testScal(vectorSize: Int, vectorElementsX: Int, vectorElementsY: Int): Unit ={
      val alpha = 3.14
      val beta = 6.21
      val xs = VectorGenerator.generateVector(vectorSize, vectorElementsX)
      val yd = VectorGenerator.generateVector(vectorSize, vectorElementsY)
      val sx = Vectors.dense(xs).toSparse
      val dy = Vectors.dense(yd).asInstanceOf[DenseVector]
      for (a <- 0 to 10000) {
        scal(alpha, sx)
      }
      println(s"Current dense vector size is: ${vectorSize}.")
      println("scal: sparse")
      var startTime = System.currentTimeMillis()
      println(startTime)
      for (a <- 0 to 10000000) {
        scal(alpha, sx)
      }
      var endTime = System.currentTimeMillis()
      println("duration: " + (endTime - startTime))
      for (a <- 0 to 10000) {
        scal(beta, dy)
      }
      println("scal: dense")
      startTime = System.currentTimeMillis()
      println(startTime)
      for (a <- 0 to 10000000) {
        scal(beta, dy)
      }
      endTime = System.currentTimeMillis()
      println("duration: " + (endTime - startTime))
    }
//    testScal(10000, 1000, 9000)
//    testScal(100000, 10000, 90000)
//    testScal(10, 1, 9)
//    testScal(100, 10, 90)
//    testScal(1000, 100, 900)
//    testScal(256, 25, 220)
//    testScal(128, 12, 110)
//    testScal(150, 15, 135)
//    testScal(128, 12, 110)
//    testScal(256, 24, 220)
//    testScal(512, 50, 440)
//    testScal(500, 50, 450)
//    testScal(800, 80, 720)
//    testScal(1000, 100, 900)
//    testScal(1200, 120, 1080)
//    testScal(1500, 150, 1350)
    testScal(200, 20, 180)
    testScal(300, 30, 270)
    testScal(400, 40, 360)
    testScal(500, 50, 450)
  }
}
