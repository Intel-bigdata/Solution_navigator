package org.intel.spark

import org.apache.spark.ml.linalg.BLAS.dspmv
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.Vectors

object TestMlDspmv {
  def main(args: Array[String]): Unit = {
    def testDspmv(vectorSize: Int, vectorElementsX: Int, vectorElementsY: Int, vectorElementsZ: Int): Unit ={
      val n = 4
      val ad = VectorGenerator.generateVector(vectorSize, vectorElementsX)
      val xd = VectorGenerator.generateVector(vectorSize, vectorElementsY)
      val yd = VectorGenerator.generateVector(vectorSize, vectorElementsZ)
      val dx = Vectors.dense(xd).asInstanceOf[DenseVector]
      val dy = Vectors.dense(yd).asInstanceOf[DenseVector]
      val da = Vectors.dense(ad).asInstanceOf[DenseVector]

      for (a <- 0 to 10000) {
        dspmv(n, 1.0, da, dx, 0.5, dy)
      }
      println(s"Current dense vector size is: ${vectorSize}")
      println("dspmv:")
      val startTime = System.currentTimeMillis()
      println(startTime)
      for (a <- 0 to 10000000) {
        dspmv(n, 1.0, da, dx, 1.0, dy)
      }
      val endTime = System.currentTimeMillis()
      println("duration: " + (endTime - startTime))
    }
//    testDspmv(10000, 8000, 7000, 9000)
//    testDspmv(100000, 80000, 70000, 90000)
//    testDspmv(10, 8, 7, 9)
//    testDspmv(100, 80, 70, 90)
//    testDspmv(256, 200, 180, 220)
//    testDspmv(1000, 800, 700, 900)
//    testDspmv(128, 100, 90, 110)
//    testDspmv(150, 120, 105, 135)
//    testDspmv(128, 100, 90, 110)
//    testDspmv(256, 200, 180, 220)
//    testDspmv(512, 400, 360, 440)
    testDspmv(500, 400, 350, 450)
    testDspmv(800, 640, 560, 720)
    testDspmv(1000, 800, 700, 900)
    testDspmv(1200, 960, 840, 1080)
    testDspmv(1500, 1200, 1050, 1350)
  }
}
