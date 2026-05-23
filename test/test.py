import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()
a='adawd'

print(os.getenv("OPENAI_API_KEY"))

# 示例：假设系统有环境变量 MY_KEY=system_val，.env里 MY_KEY=dotenv_val
print(os.getenv("MINIO_ACCESS_KEY"))