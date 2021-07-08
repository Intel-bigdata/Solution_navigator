package org.intel.spark

import org.apache.spark.ml.linalg.BLAS.dot
import org.apache.spark.ml.linalg.{DenseVector, Vectors}

object TestMlDot {
  def main(args: Array[String]): Unit = {
    def testDot(vectorSize: Int, vectorElementsX: Int, vectorElementsY: Int): Unit ={
      val xd = VectorGenerator.generateVector(vectorSize, vectorElementsX)
      val yd = VectorGenerator.generateVector(vectorSize, vectorElementsY)
      val dx = Vectors.dense(xd).asInstanceOf[DenseVector]
      val dy = Vectors.dense(yd).asInstanceOf[DenseVector]
      for (a <- 0 to 10000) {
        dot(dx, dy)
      }
      println(s"Current dense vector size is: ${vectorSize}")
      println("dot: dense/dense")
      val startTime = System.currentTimeMillis()
      println(startTime)
      for (a <- 0 to 10000000) {
        dot(dx, dy)
      }
      val endTime = System.currentTimeMillis()
      println("duration: " + (endTime - startTime))
    }
//    testDot(10000, 7000, 9000)
//    testDot(100000, 70000, 90000)
//    testDot(10, 7, 9)
//    testDot(100, 70, 90)
//    testDot(1000, 700, 900)
//    testDot(256, 180, 220)
//    testDot(128, 90, 110)
////    testDot(150, 105, 135)
//    testDot(128, 90, 110)
//    testDot(256, 180, 220)
//    testDot(512, 360, 440)
//    testDot(500, 350, 450)
//    testDot(800, 560, 720)
//    testDot(1000, 700, 900)
//    testDot(1200, 840, 1080)
//    testDot(1500, 1050, 1350)
    testDot(200, 140, 180)
    testDot(300, 210, 270)
    testDot(400, 280, 360)
    testDot(500, 350, 450)
  }
}
