# RDD Cache K-Means User Guide

## Prerequisites
Before configuring in Spark to use Optane PMem cache, you need to make sure the following:

Optane PMem hardwares are installed, formatted and mounted correctly on every cluster worker node with AppDirect mode(refer [guide](https://software.intel.com/en-us/articles/quick-start-guide-configure-intel-optane-dc-persistent-memory-on-linux)). You will get a mounted directory to use if you have done this. Usually, the Optane PMem on each socket will be mounted as a directory. For example, on a two sockets system, we may get two mounted directories named `/mnt/pmem0` and `/mnt/pmem1`.

Here is a quick guide to enable AppDirect mode for two socket system. Please run the below commands by superuser.
```
ipmctl create -goal PersistentMemoryType=AppDirect
reboot
ndctl create-namespace -m fsdax -r region0
ndctl create-namespace -m fsdax -r region1
mkfs.ext4 /dev/pmem0
mkfs.ext4 /dev/pmem1
mount -o dax /dev/pmem0 /mnt/pmem0
mount -o dax /dev/pmem1 /mnt/pmem1
```

Please validate Optane PMem mount in DAX mode like below. You should find **DAX** listed in mode list for each PMEM entry like below:
```
mount | grep dax 
```

The output should looks like as below:
```
/dev/pmem0 on /mnt/pmem0 type ext4 (rw,relatime,dax)
/dev/pmem1 on /mnt/pmem1 type ext4 (rw,relatime,dax) 
```

[Memkind](http://memkind.github.io/memkind/) library must be installed on every cluster worker node. Please use the latest Memkind version. You can compile Memkind based on your system and put the file to `/lib64/` directory in each worker node in cluster. Memkind library depends on libnuma at the runtime. You need to make sure libnuma already exists in worker node system. To build memkind lib from source, you can:
```
git clone https://github.com/memkind/memkind
cd memkind
./autogen.sh
./configure
make
sudo make install
sudo cp -r /usr/local/lib/* /lib64 
```

## Configure for NUMA
To achieve the optimum performance, we need to configure NUMA for binding executor to NUMA node and try access the right Optane PMem device on the same NUMA node. You need install numactl on each worker node. For example, on CentOS, run following command to install numactl.

```
yum install numactl -y
```

# Setup Hadoop 2.7.5
Please set up Hadoop following [the official website](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html#Installing_Software) and make sure it works.

It can be validated by running [Hadoop benchmark](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/Benchmarking.html).

# Install Optane PMem supported Spark
Please download the Optane PMem supported spark from [Intel-bigdata/spark] (https://github.com/Intel-bigdata/spark/tree/branch-2.4.4-oap-0.8).

Please download OAP project from [Intel-bigdata/OAP] (https://github.com/Intel-bigdata/OAP/tree/branch-0.8-spark-2.4.x).

To build the Optane PMem supported spark, you can run below commands:
```
cd ${OAP_CODE_HOME}
mvn install -Ppersistent-memory -DskipTests

cd ${SPAKR_CODE_HOME}
./dev/make-distribution.sh --name custom-spark --tgz -Phadoop-2.7 -Phive -Phive-thriftserver -Pyarn
```
After the building process run to the end, a tarball named spark-2.4.4-bin-custom-spark.tgz will generated under spark source code directory. Please unzip it.
```
tar -zxvf spark-2.4.4-bin-custom-spark.tgz
```

Configure the environment variables for spark. Please add below command at the end of /home/${USER_NAME}/.bashrc.
```
export SPARK_HOME=${OPTANE_PMEM_SUPPORTED_SPARK_PATH}
export HADOOP_HOME=${YOUR_HADOOP_PATH}
export HADOOP_CONF_DIR=${HADOOP_HOME}/etc/hadoop 
```
Assume you have setup HADOOP_HOME and HADOOP_CONF_DIR when installing hadoop. Spark need also these two configuration.

# Setup HiBench Benchmark Suite
Please download HiBench from [HiBench webpage](https://github.com/intel-hadoop/HiBench.git). You can run the following command to build HiBench.
```
mvn -Psparkbench -Dspark=2.4 -Dscala=2.11 clean package
```

Please follow the guide [run-sparkbench] (https://github.com/Intel-bigdata/HiBench/blob/master/docs/run-sparkbench.md) to configure the hadoop.conf and spark.conf in HiBench.

## Running Spark Kmeans
See [https://github.com/Intel-bigdata/HiBench/blob/master/docs/run-sparkbench.md] 

Additional configurations to enabling Optane PMem using HiBench. Please add following configurations in `conf/spark.conf`
```
spark.memory.pmem.initial.path /mnt/pmem0,/mnt/pmem1
spark.memory.pmem.initial.size ${OPTANE_PMEM_CAPACITY_FOR_SINGLE_SOCKET}
spark.yarn.numa.enabled true
spark.yarn.numa.num 2
```
Specify spark storage level in `conf/workloads/ml/kmeans.conf
```
hibench.kmeans.storage.level PMEM_AND_DISK
```
In this case, the machine has two numa node which can be checked by "numactl --hardware" and these two numa node is associate with /mnt/pmem0 and /mnt/pmem1 for each. 

For a different number of numa nodes, You shoud try to setup Optane PMem AppDirect mode with more namespace, see the previous chapter and link.

If you want to run other storage level like OFF_HEAP. Please remove previous Optane PMem related configurations and follow the spark official doc [Which Storage Level to Choose] (https://spark.apache.org/docs/latest/rdd-programming-guide.html#which-storage-level-to-choose)

You can add the spark conf which specify to other storage level in `conf/spark.conf` and specify spark storage level in `conf/workloads/ml/kmeans.conf`.


To change the input data size, you can set `hibench.scale.profile` in `conf/hibench.conf`. Available values are tiny, small, large, huge, gigantic and bigdata. The definition of these profiles can be found in the workload's conf file `conf/workloads/ml/kmeans.conf`
 
To run kmeans workload：
```
bin/workloads/ml/kmeans/prepare/prepare.sh
bin/workloads/ml/kmeans/spark/run.sh
```

