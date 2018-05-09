天池平台-美年健康AI大赛—双高疾病风险预测

data目录：
1. meinian_round1_train_20180408.csv
2. meinian_round1_test_b_20180505.csv
3. meinian_round1_data_part1_20180408.txt
4. meinian_round1_data_part2_20180408.txt
	其中1,3,4文件作为生成训练特征及预测结果，2文件提供测试预测vid
	两个特征文件 data_part1 和 data_part2：每个文件第一行是字段名，之后每一行代表某个指标的检查结果（指标含义已脱敏）。每个文件各包含3个字段，分别表示病人id、体检项目id 和体检结果，部分字段在部分人群中有缺失。其中，体检项目id字段，数值相同表示体检的项目相同。 体检结果字段有数值和字符型，部分结果以非结构化文本形式提供。
	标签文件 train.csv：是训练数据的答案：包含六个字段，第一个字段为病人id，与上述特征文件的病人id有对应关系，之后五个字段依次为收缩压、舒张压、甘油三酯、高密度脂蛋白胆固醇和低密度脂蛋白胆固醇。



评估指标
https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.6285427cxDT64J&raceId=231654 



code目录：
main文件为整个系统运行文件，data_code_final文件中进行数据提取，数据清洗，以及数据特征转换。预测选用xgboost模型，文件xgbt_test1、xgbt_test2、xgbt_test3、xgbt_test4、xgbt_test5分别预测指标'收缩压'、'舒张压'、'甘油三酯'、'高密度脂蛋白胆固醇'和'低密度脂蛋白胆固醇'。number_v1_50sub.csv文件为提前根据50个样本进行人工选择得出的特征集合。


submit目录：
生成预测文件所存储的路径。



运行：
运行main文件，生成预测文件在submit文件夹。最终线上评测指标为0.03。（如果选择更多特征，效果应该会好一些，只选择了数值特征，可加入简单文字特征。）
