pynn
======
介紹
-------
NTU CSIE 5052 的Homework One的代码 <br>
实验结果记录在exp_records.db


預安裝
-------
[Python 3.5.1](https://www.python.org/) <br>
[numpy](https://github.com/numpy/numpy)	<br>
[matplotlib](https://github.com/matplotlib/matplotlib) <br>

pynn目录
--------
hw2 [Link to Header](#hw2)
hw1 [Link to Header](#hw1)


#hw2
-------
### 执行
python -m unittest hw2/hw2.py

### 查询结果
均在hw2文件夾下，一個文件是```exp_records.db```, 文件夾是```exp_figures``` 

### 用法
-------

#### 配置SOMNN
``` Python
attRate = 0.0001
repRate = 1

# each tuple in form of: (activationsfun, paramater)
# if item is int, then repeat the former item
af_types = [("purelin", 1), ("tanh", 1), 10]
layers = [2, 8, 7, 6, 5, 4, 3, 2, 1]

nnConfig = {
	"attRate": attRate,
	"repRate": repRate,
	"af_types": af_types,
	"layers": layers
}
```


#### 初始化NN
``` Python
# The first af should usually be common.input
NN = somnn.init(nnConfig)
```

#### 初始化Graph
``` Python
nnplot.iniGraph(NN, 1)
nnplot.drawData(np.loadtxt(dataFileName))
```
#### 初始化SQLite
``` Python
nnSQLite.iniSQLite("exp_records.db")
```

#### 初始化訓練和測試資料
``` Python
import os
PtFilePath = os.path.join(os.path.dirname(__file__), "hw2pt.dat")
classFilePath = os.path.join(os.path.dirname(__file__), "hw2class.dat")
# for data set whose feature-table-file sperates from its catogary-file
dataSet = nnio.mergeFeatureAndClass(PtFilePath, classFilePath)
```


#### 開始訓練
``` Python
# trainLayerIndex is the layer_new in SIR-SOM algorithm
for trainLayerIndex in range(1, sizeOfLayers):
	NN.train(dataSet, trainLayerIndex)
```

### 資料庫結構
-----
#### exp_info
	exp_id	|	network_structure_json	|	initial_parameters_json	|	initial_error_rate	|	trained_error_rate	|	epochs  |	exp_category|	dataset_name   | exp_note | records_folder
	--------|:----------|---------------|---------------------------|-----------------------|-----------------------|-----------|
		id  |  神經網路結構 				| 初始的結果  		| 初始的錯誤率  |  訓練后錯誤率  |  dataset用了多少遍 | 訓練的類別 | 資料集名稱 | 訓練的注釋 | 結果儲存位置



#hw1
-------
### 执行
python -m unittest hw1/hw1.py

## 

#### 若要畫某一個exp_id的neuron分割
python -m unittest hw1-load-data.py

#### 若要查看對於輸入，exp_id的NN反應的每層layer的輸出
修改 hw1-load-data.py 執行 showOutputs <br>
python -m unittest hw1-load-data.py

### 用法
-------

#### import相關包
``` Python
from pynn import bpnn
from pynn import nn
from pynn import common
from pynn import nnio
from pynn import nnplot
from pynn import nnSQLite
```


#### 設置activate function
``` Python
def YourFunction(x):
	# do whatever you want
	return result
```
##### 如果導數是關於x的函數
``` Python
# if the derivative is in form of x
def DerivativeRelatedToX(x):
	# do whatever you want
	return result
YourActivationFunction = nn.ActivationFunction(YourFunction, False, DerivativeRelatedToX)
```


##### 如果導數是關於y的函數
``` Python
# if the derivative is in form of y
def DerivativeRelatedToY(y):
	# do whatever you want
	return result
YourActivationFunction = nn.ActivationFunction(YourFunction, True, DerivativeRelatedToY)
```

#### 初始化NN
``` Python
# The first af should usually be common.input
afs = [common.input, common.ac_tanh(1), common.ac_tanh(1), common.ac_tanh(1)]
layers = [2, 8, 6, 1]
NN = bpnn.init(alpha, layers, afs)
```

#### 初始化Graph
``` Python
nnplot.iniGraph(NN, 1)
nnplot.drawData(np.loadtxt(dataFileName))
```
#### 初始化SQLite
``` Python
nnSQLite.iniSQLite("exp_records.db")
```

#### 初始化訓練和測試資料
``` Python
trainingData, testData = nnio.readTrainingAndTestData(filePath, numberOfCategory, rowsOfTrainingData)
inputs = trainingData["inputs"]
outputs = trainingData["outputs"]
test_inputs = testData["inputs"]
test_outputs = testData["outputs"]
```


#### 開始訓練
``` Python
for i in range(0, len(inputs)):
			inputVector = np.transpose(np.mat(inputs[i]))
			output = NN.forward(inputVector)
			teacher = np.transpose(np.mat(outputs[i]))
			errorVector = teacher - output
			NN.backPropagation(errorVector)
```

### 資料庫結構
-----
#### exp_info
	exp_id	|	alpha	|	err_rate	|	layer_info	|	exp_category	|	exp_note
	--------|:----------|:--------------|---------------|-------------------|--------------
		id  |  學習速率 | 實驗的錯誤率  | 每層神經數量  |  實驗的分類/名稱  |  自定義的描述
#### exp_weight
	exp_weight_id	|	exp_id	 |	   weight_seq	 |	  weight_str
	----------------|:-----------|:------------------|--------------
	exp_weight_id   | 對應的實驗 |  是第幾層的weight | 字符串化的weight
#### exp_bias
	exp_bias_id	|	exp_id	 |	  bias_seq	   |	bias_str
	------------|:-----------|:-----------------|--------------
	exp_bias_id | 對應的實驗 |  是第幾層的bias  | 字符串化的bias


