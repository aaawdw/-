# test/fastapi/upload.py
import datetime
import uuid
from typing import Optional

from fastapi import FastAPI, File,UploadFile,HTTPException
from fastapi.responses import JSONResponse
import os

from pydantic import BaseModel, Field

app = FastAPI()

UPLOAD_FOLDER = r'D:\uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_TYPES = ["image/jpeg", "image/png", "image/gif"]
CHUNK_SIZE = 1024 * 1024  # 1MB分块

# ========== Pydantic 模型定义 ==========

class UploadResponseData(BaseModel):
    """上传成功后的数据"""
    filename: str = Field(..., description="原始文件名")
    saved_filename: str = Field(..., description="保存的文件名")
    content_type: str = Field(..., description="文件类型")
    file_size: int = Field(..., description="文件大小（字节）")
    save_path: str = Field(..., description="保存路径")
    remark: Optional[str] = Field(default=None, description="备注信息")


class UploadResponse(BaseModel):
    """统一响应格式"""
    code: int = Field(default=0, description="状态码：0-成功，其他-失败")
    msg: str = Field(default="success", description="响应消息")
    data: Optional[UploadResponseData] = Field(default=None, description="响应数据")

# ========== 辅助函数 ==========

def generate_unique_filename(original_filename: str) -> str:
    """生成唯一文件名，避免文件覆盖"""
    ext = os.path.splitext(original_filename)[1]  # 获取扩展名
    unique_name = f"{uuid.uuid4().hex}{ext}"
    return unique_name


def validate_file_type(content_type: str, allowed_types: list) -> bool:
    """验证文件类型"""
    return content_type in allowed_types

# ========== API 接口 ==========
@app.post("/upload",summary="单个文件上传接口", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(
    ..., # 必填参数
    description="需要上传的文件（支持图片/文档等)",
    alias="upload_file", #前端参数的名称，默认文件名是file
    media_type="application/octet-stream"), #接受任何类型的文件（图片、文档、视频等）
    remark:str = None #可选参数
):
    """
    文件上传接口

    - 支持图片格式：JPEG, PNG, GIF
    - 大文件分块读取，避免内存溢出
    - 自动生成唯一文件名，防止文件覆盖
   """

    try:
        # 1. 校验文件类型
        if not validate_file_type(file.content_type, ALLOWED_TYPES):
            raise HTTPException(
                status_code=400,
                detail=f"仅支持上传 {', '.join(ALLOWED_TYPES)} 类型的文件，当前文件类型：{file.content_type}"
            )

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        # 2. 生成唯一文件名（避免文件覆盖）
        original_filename = file.filename
        saved_filename = generate_unique_filename(original_filename)
        file_path = os.path.join(UPLOAD_FOLDER, saved_filename)

        # 3. 分块读取并保存文件
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)

        # 4. 构建响应数据
        response_data = UploadResponseData(
            filename=original_filename,
            saved_filename=saved_filename,
            content_type=file.content_type,
            file_size=file.size,
            save_path=file_path,
            remark=remark
        )

        # 5. 返回 Pydantic 模型（FastAPI 会自动序列化）
        return UploadResponse(
            code=0,
            msg="文件上传成功",
            data=response_data
        )

    except Exception as e:
        # 异常捕获：返回友好的错误信息
        raise HTTPException(
            status_code=500,
            detail=f"文件上传失败：{str(e)}"
        )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)