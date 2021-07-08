package org.intel.spark

import org.apache.spark.ml.linalg.BLAS.axpy
import org.apache.spark.ml.linalg.{DenseVector, Vectors}

object TestMlAxpy {
  def main(args: Array[String]): Unit = {
    def testAxpy(vectorSize: Int, vectorElementsX: Int, vectorElementsY: Int): Unit = {
      val alpha = 0.1
      val xd = VectorGenerator.generateVector(vectorSize, vectorElementsX)
      val yd = VectorGenerator.generateVector(vectorSize, vectorElementsY)
      val dx = Vectors.dense(xd).asInstanceOf[DenseVector]
      val dy = Vectors.dense(yd).asInstanceOf[DenseVector]

      for (a <- 0 to 10000) {
        axpy(alpha, dx, dy)
      }
      println(s"Current dense vector size is: ${vectorSize}")
      println("axpy: dense/dense")
      val startTime = System.currentTimeMillis()
      println(startTime)
      for (a <- 0 to 10000000) {
        axpy(alpha, dx, dy)
      }
      val endTime = System.currentTimeMillis()
      println("duration: " + (endTime - startTime))
    }
//    testAxpy(10000, 7000, 9000)
//    testAxpy(100000, 70000, 90000)
//    testAxpy(10, 7, 9)
//    testAxpy(100, 70, 90)
//    testAxpy(1000, 700, 900)
//    testAxpy(256, 180, 220)
//    testAxpy(128, 90, 110)
//    testAxpy(256, 180, 220)

//    testAxpy(500, 350, 360)
//    testAxpy(800, 560, 720)
//    testAxpy(1000, 700, 900)
//    testAxpy(1200, 840, 1080)
//    testAxpy(1500, 1050, 1350)
    testAxpy(200, 140, 180)
    testAxpy(300, 210, 270)
    testAxpy(400, 280, 360)
    testAxpy(500, 350, 450)
  }
}
