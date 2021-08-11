# 求导公式
## １、模型末端使用sigmoid作为输出，并使用cross entropy作为Loss Function的
## sigmoid 原始公式:
## $y=\frac{1}{1+e^{-x}}$
## sigmoid 求导:
## $y^{'}=-1\frac{1}{(1+e^{-x})^2}e^{-x}(-1) =\frac{e^{-x}}{(1+e^{-x})^2}=\frac{1}{1+e^{-x}}\frac{e^{-x}+1-1}{1+e^{-x}}=\frac{1}{1+e^{-x}}(1-\frac{1}{1+e^{-x}})=p_i(1-p_i)$
#
## cross entropy 原始公式:
## $Loss=-y_ilogp_i-(1-y_i)log(1-p_i)$
## 其中$y_i\in(0,1)$，这里$y_i$是真实值，$p_i$是模型预测值，也就是模型最后一层输出＋sigmoid的结果，这里注意一下和下面softmax的区别，因为sigmoid的输出只有１个，一般根据训练的batch大小为Ｎ×１，所以Loss函数如上所示，需要判断$y_i$是０还是１，但是softmax输出有ｍ个（m为类别个数），一个批次的输出为Ｎ×ｍ，所以此时Loss函数如下:$Loss={\sum_{j=0}^{n}{-y_ilogp_i}}$
## cross entropy 求导:
## $Loss^{'}=\begin{cases}-1/p_i,&y_i=１ \\ 1/(1-p_i),&y_i=0 \end{cases}$
#
## sigmoid+cross entropy　合在一起为
## $\frac{dLoss}{dx} = \frac{dLoss}{dp_i}\frac{dpi}{dx} \\ = \begin{cases}-1/p_i*p_i(1-p_i),&y_i=１ \\ 1/(1-p_i)*p_i(1-p_i),&y_i=0 \end{cases} \\ = \begin{cases} p_i-1,&y_i=１ \\ p_i,&y_i=0 \\ = p_i-y_i,&(y_i=0,y_i=1) \end{cases}$ 
#

## 进一步的，上面求导公式中的ｘ为模型最后一层的输出，常见的情况是最后一层往往是一个Linear Classifier，比如pytorch常见的nn.Linear(input_channels, classes_num)，这里classes_num为预测的类型数量，我们设　$x = X_{input}W$, 那么如果要根据某一次前向推测的Loss计算 $W$ 的梯度，则我们只需要计算 $\frac{dx}{dW}$ 即可，那么根据链式求导法则，我们有：
## $\frac{dLoss}{dW} = \frac{dLoss}{dp_i}\frac{dpi}{dx}\frac{dx}{dＷ} \\ = X_{input}.T*(p_i-y_i)$   此即为倒数最后一个Linear层的梯度 
## 使用公式 $W = W - lr*\frac{dLoss}{dW}$ 即可更新最后一层参数$W$，这里 $lr$ 为学习率
#

## ２、模型末端使用softmax作为输出，并使用cross entropy作为Loss Function的