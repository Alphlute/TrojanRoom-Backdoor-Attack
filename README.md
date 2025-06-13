# TrojanRoom-Backdoor-Attack
参考TrojanRoom攻击框架实现的一个简单的CNN语音指令识别模型后门攻击

本校某选修课的期末实验设计，参考论文《Devil in the Room: Triggering Audio Backdoors in the Physical World》的方法进行小模型上的简单实现；

参考文献：[Chen M, Xu X, Lu L, Ba Z, Lin F, Ren K. Devil in the Room: Triggering Audio Backdoors in the Physical World[J]. Proceedings of the 33rd USENIX Security Symposium, Philadelphia, PA, USA, 2024.](https://www.usenix.org/conference/usenixsecurity24/presentation/chen-meng)

<font color=dark>**本项目相关文件与代码仅用于网络安全与人工智能相关学习研究使用**</font>

## 使用方法
1. `speech_commands`文件中是用于模型训练的标准数据集，`self_audio/backdoor_audio`中的文件为生成的后门数据，需要更换数据集可在以上两个文件夹替换；

2. `RIR_generator`中的`ESS.py`为使用ESS方法生成RIR触发器的程序，需要先运行一次生成`ess.wav`，自录制得到一个`recorded.wav`文件，而后第二次运行`ESS.py`程序即可生成`rir.wav`文件作为后门触发器；

3. 对正常数据进行后门触发器植入需使用`backdoor_generator.py`程序（注意修改输入输出的文件目录）；

4. 模型的训练使用`train_rir.py`，根据样本量大小建议调整 $learning-  rate$ 与 $epoch$ 数；

5. 训练好的模型保存为 `speech_model_with_backdoor_f.pth`，可使用`model_test.py`进行测试。