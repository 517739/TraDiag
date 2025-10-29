import json
import os


def extract_fault_fields(input_file: str, output_file: str) -> None:
    """
    读取包含JSON记录的文本文件，提取指定字段后保存到新文件

    Args:
        input_file: 输入文件路径（groundtruth.txt）
        output_file: 输出文件路径（提取后的数据保存路径）
    """
    # 定义需要保留的字段（顺序可根据需求调整）
    target_fields = [
        "fault_category",
        "fault_type",
        "instance_type",
        "instance",
        "start_time",
        "end_time"
    ]

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在，请检查路径！")
        return

    # 读取输入文件并处理每条记录
    with open(input_file, "r", encoding="utf-8") as f_in, \
            open(output_file, "w", encoding="utf-8") as f_out:

        line_num = 0  # 记录行号，便于定位错误
        for line in f_in:
            line_num += 1
            # 去除行首尾空白字符（避免换行符、空格影响JSON解析）
            clean_line = line.strip()
            if not clean_line:  # 跳过空行
                continue

            try:
                # 解析JSON记录
                fault_record = json.loads(clean_line)

                # 提取目标字段：若字段不存在，赋值为None（避免KeyError）
                extracted_record = {field: fault_record.get(field, None) for field in target_fields}

                # 将提取后的记录转为JSON字符串并写入输出文件（每行一条）
                json.dump(extracted_record, f_out, ensure_ascii=False)
                f_out.write("\n")  # 换行分隔每条记录

            except json.JSONDecodeError as e:
                # 捕获JSON解析错误，提示具体行号便于排查
                print(f"警告：第 {line_num} 行JSON格式异常，已跳过该记录。错误信息：{str(e)}")
            except Exception as e:
                # 捕获其他未知错误
                print(f"警告：第 {line_num} 行处理失败，已跳过该记录。错误信息：{str(e)}")

    print(f"处理完成！提取后的数据已保存到：{os.path.abspath(output_file)}")


# ------------------- 配置文件路径 -------------------
# 输入文件：groundtruth.txt 的路径（需根据实际文件位置修改）
INPUT_FILE = "../Data/groundtruth.txt"
# 输出文件：提取后的数据保存路径（可自定义）
OUTPUT_FILE = "../Data/new_gdt.txt"
# ---------------------------------------------------

# 执行提取操作
if __name__ == "__main__":
    extract_fault_fields(INPUT_FILE, OUTPUT_FILE)