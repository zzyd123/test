import json

input_file_path = '/raid/zyd/new/CoRe-main/gpt2-large-5723-COT-generator-result/10-08_train-5723-AddSub-COT-50-generator_solution.json'
output_file_path = '/raid/zyd/new/CoRe-main/gpt2-large-5723-COT-generator-result/10-08_train-5723-AddSub-COT-50-generator_solution1.json'


# 打开输入JSON文件和输出JSON文件
with open(input_file_path, 'r', encoding='utf-8') as input_file, \
        open(output_file_path, 'w', encoding='utf-8') as output_file:

    # 遍历每一行数据
    for line in input_file:
        try:
            # 从一行数据中加载JSON字典
            item = json.loads(line)

            # 获取"solution"字段
            solution = item.get('solution', '')

            # 删除多余的双引号 `""`，只保留一个
            cleaned_solution = solution.replace('<|endoftext|>', '')

            # 更新字典中的"solution"字段
            item['solution'] = cleaned_solution
            
            solution = item.get('solution', '')
            solution = solution + '<|endoftext|>'
            item['solution'] = solution
            

            # 将处理后的字典写回输出文件
            output_file.write(json.dumps(item, ensure_ascii=False) + '\n')

        except json.JSONDecodeError:
            print(f"无法解析的行: {line}")

print("Done")