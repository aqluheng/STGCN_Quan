### 使用Pytorch1.7.1静态量化工具记录
##### 基本用法
1. 在待量化模型中,手动加入quant与dequant操作
2. model.qconfig 可以指定量化方法,torch.quantization.default_qconfig为minMax量化,torch.quantization.get_default_qconfig('fbgemm')为直方图量化
3. 使用torch.quantization.prepare 进行准备(插入observer)
4. 使用模型进行inference,由observer记录每一层weight的最大值最小值,记录Activation的最大值最小值
5. inference多个batch后,使用torch.quantization.convert 进行转换,将模型的conv转为quantized_conv等等
6. 可以由torch.jit.save(torch.jit.script(model),path)保存,并model=torch.jit.load(path)读取
7. 量化后的模型由quantCPU这个后端模型执行,很多的操作并不支持

##### 踩坑
1. nn.BatchNorm1d,quantCPU不支持, 使用nn.BatchNorm2d代替 
    bn1(x) === bn2(x.unsqueeze(-1)).squeeze(-1)

2. torch.bmm,quantCPU不支持,使用dequant,quant包住

3. +-*/ quantCPU不支持,需要由FloatFunctional.add替代
    self.first_function = nn.quantized.FloatFunctional()
    x = self.first_function.add(self.tcn(x), res)

##### 疑问
1. 每一层的权重假设由0-1量化到0-255,activation该如何做

