package org.intel.spark

import scala.util.Random

object VectorGenerator {
  def generateVector(size:Int,elementsNo:Int): Array[Double] = {
    val elementsList = new Array[Double](size)
    var nonZeroSize = elementsNo
    var i = 0
    for(i<-0 to size-1){
      elementsList(i) = 0
    }
    var j = 0
    for(j<-0 to size-1 ){
      if(nonZeroSize >0){
        val tmpValue = (new Random).nextDouble()
        val index = (new Random).nextInt(size)
        elementsList(index) = tmpValue
        if(tmpValue!=0){
          nonZeroSize -= 1
        }
      }
    }
    return elementsList
  }
}
