# @Time : 2023/6/18 13:13
# @Author : yichen
import os.path

file_name = "20230626_230823"
task = 'T1'
Res_task = ["T0", "T1", "T2", "T3", "T4"]
Res_task.remove(task)
print(Res_task)
input_file = os.path.join("/home/chenqiongpu/SLO/SLO/work_dirs/SLO", file_name,
                          file_name + '.log')
save_path = os.path.join("/home/chenqiongpu/SLO/SLO/work_dirs/SLO", file_name,
                         file_name + "_" + task + '.log')
if __name__ == '__main__':
    filtered_lines = []
    with open(input_file, 'r') as input_file:
        lines = input_file.readlines()
        for line in lines:
            # print(line)  # Add this line
            if "mmengine - INFO - Exp name" in line:
                continue
            elif "mmengine - INFO - Epoch(val)" in line:
                continue
            elif "mmengine - INFO - Saving" in line:
                continue
            elif any("mmengine - INFO - train  " + task + " precision" in line for task in Res_task):
                continue
            elif any("mmengine - INFO - val    " + task + " precision" in line for task in Res_task):
                continue
            elif any("mmengine - INFO - test   " + task + " precision" in line for task in Res_task):
                continue
            elif "mmengine - INFO - Epoch(train)" in line:
                info = line.split()
                filtered_line = ""
                for part in info:
                    if 'loss' in part:
                        break
                    filtered_line = filtered_line+" "+part
                for index, part in enumerate(info):
                    if f"{task}_loss" in part:
                        filtered_line = filtered_line+" "+part+" "+info[index+1]
                filtered_lines.append(filtered_line[1:] + "\n")
            else:
                filtered_lines.append(line)

    with open(save_path, 'w') as output_file:
        output_file.writelines(filtered_lines)
