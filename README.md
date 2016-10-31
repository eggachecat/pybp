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

pynn-lib介紹
--------
留空

测试
-------
## 执行
python hw1.py

## 

### 若要畫某一個exp_id的neuron分割
python hw1-load-data.py

### 若要查看對於輸入，exp_id的NN反應的每層layer的輸出
修改 hw1-load-data.py 執行 showOutputs <br>
python hw1-load-data.py

用法
-------

## import相關包
``` Python
from pynn import bpnn
from pynn import nn
from pynn import common
from pynn import nnio
from pynn import nnplot
from pynn import nnSQLite
```


## 設置activate function
``` Python
def YourFunction(x):
	# do whatever you want
	return result
```
### 如果導數是關於x的函數
``` Python
# if the derivative is in form of x
def DerivativeRelatedToX(x):
	# do whatever you want
	return result
YourActivationFunction = nn.ActivationFunction(YourFunction, False, DerivativeRelatedToX)
```


### 如果導數是關於y的函數
``` Python
# if the derivative is in form of y
def DerivativeRelatedToY(y):
	# do whatever you want
	return result
YourActivationFunction = nn.ActivationFunction(YourFunction, True, DerivativeRelatedToY)
```

## 初始化NN
``` Python
# The first af should usually be common.input
afs = [common.input, common.ac_tanh(1), common.ac_tanh(1), common.ac_tanh(1)]
layers = [2, 8, 6, 1]
NN = bpnn.init(alpha, layers, afs)
```

## 初始化Graph
``` Python
nnplot.iniGraph(NN, 1)
nnplot.drawData(np.loadtxt(dataFileName))
```
## 初始化SQLite
``` Python
nnSQLite.iniSQLite("exp_records.db")
```

## 初始化訓練和測試資料
``` Python
trainingData, testData = nnio.readTrainingAndTestData(filePath, numberOfCategory, rowsOfTrainingData)
inputs = trainingData["inputs"]
outputs = trainingData["outputs"]
test_inputs = testData["inputs"]
test_outputs = testData["outputs"]
```


## 開始訓練
``` Python
for i in range(0, len(inputs)):
			inputVector = np.transpose(np.mat(inputs[i]))
			output = NN.forward(inputVector)
			teacher = np.transpose(np.mat(outputs[i]))
			errorVector = teacher - output
			NN.backPropagation(errorVector)
```

資料庫結構
-----
## exp_info
	exp_id	|	alpha	|	err_rate	|	layer_info	|	exp_category	|	exp_note
	--------|:----------|:--------------|---------------|-------------------|--------------
		id  |  學習速率 | 實驗的錯誤率  | 每層神經數量  |  實驗的分類/名稱  |  自定義的描述
## exp_weight
	exp_weight_id	|	exp_id	 |	   weight_seq	 |	  weight_str
	----------------|:-----------|:------------------|--------------
	exp_weight_id   | 對應的實驗 |  是第幾層的weight | 字符串化的weight
## exp_bias
	exp_bias_id	|	exp_id	 |	  bias_seq	   |	bias_str
	------------|:-----------|:-----------------|--------------
	exp_bias_id | 對應的實驗 |  是第幾層的bias  | 字符串化的bias
