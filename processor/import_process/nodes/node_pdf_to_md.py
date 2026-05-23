# processor/import_processor/nodes/node_pdf_to_md.py
import json
import logging
import shutil
import time
import zipfile
from pathlib import Path

import requests

from conf.mineru_config import mineru_config
from processor.import_process.base import BaseNode, setup_logging
from processor.import_process.exceptions import StateFieldError, FileProcessingError, ConfigurationError, \
    PdfConversionError
from processor.import_process.state import ImportGraphState


class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = "node_pdf_to_md"

    def _step_1_validate_paths(self, state: ImportGraphState):
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(field_name='pdf_path',expected_type=str)

        file_dir=state.get("file_dir")
        if not file_dir:
            raise StateFieldError(field_name='file_dir',expected_type=str)

        pdf_path_obj=Path(pdf_path)
        file_dir_obj=Path(file_dir)

        if not pdf_path_obj.exists():
            raise FileProcessingError(message=f'PDF文件{pdf_path_obj.name}不存在')

        if not file_dir_obj.exists():
            self.logger.info(f'输出目录不存在，自动创建:{file_dir_obj.absolute()}')
            file_dir_obj.mkdir(parents=True, exist_ok=True)

        return pdf_path_obj,file_dir_obj

    def _step_2_upload_and_poll(self, pdf_path_obj: Path):

        if not mineru_config.base_url:
            raise ConfigurationError('MinerU配置缺失：请在.env文件中正确配置MINERU_BASE_URL 参数')
        if not mineru_config.api_token:
            raise ConfigurationError("MinerU配置缺失：请在 .env 文件中正确配置 MINERU_API_TOKEN 参数")

        token=mineru_config.api_token
        url=f'{mineru_config.base_url}/file-urls/batch'
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }

        response = requests.post(url=url, headers=header, json=data)

        if response.status_code != 200:
            raise PdfConversionError(message=f"获取上传链接响应失败：状态码：{response.status_code}，响应结果：{response}")

        result=response.json()
        if result.get('code')!=0:
            raise PdfConversionError(f'获取上传链接失败：返回数据：{result.get("message")}')

        singed_url=result['data']['file_urls'][0]
        batch_id=result['data']['batch_id']

        with open(pdf_path_obj,'rb') as f:
            res_upload=requests.put(url=singed_url,data=f)
            if res_upload.status_code!=200:
                raise PdfConversionError(f"文件上传失败：状态码：{res_upload.status_code}，响应结果：{res_upload}")

            self.logger.info('文件上传成功!')

        poll_url=f'{mineru_config.base_url}/extract-results/batch/{batch_id}'

        start_time = time.time()
        timeout_seconds=600
        poll_interval=3
        self.logger.info(f'【任务轮询】最大超时：{timeout_seconds}，batch_id:{batch_id}')

        while True:
            elapsed_time=time.time()-start_time
            if elapsed_time>timeout_seconds:
                raise TimeoutError(f"【任务轮询】超时！任务处理超{timeout_seconds}秒，batch_id：{batch_id}")
            try:
                res_poll=requests.get(url=poll_url,headers=header,timeout=10)
            except Exception as e:
                self.logger.warning(f'【任务轮询】网络请求异常，{poll_interval}秒后重试：{str(e)},batch_id:{batch_id}')
                time.sleep(poll_interval)
                continue

            if res_poll.status_code!=200:
                raise PdfConversionError(f"【任务轮询】HTTP请求失败，状态码：{res_poll.status_code}，响应内容：{res_poll}")

            poll_data=res_poll.json()
            if poll_data['code']!=0:
                raise PdfConversionError(f"【任务轮询】业务错误，返回数据：{poll_data}")
            extract_results=poll_data['data']['extract_result']

            result_item=extract_results[0]
            data_state=result_item['state']

            if data_state=='done':
                self.logger.info(f"【任务轮询】解析任务完成！总耗时{int(elapsed_time)}s，bactch_id：{batch_id}")

                full_zip_url=result_item['full_zip_url']
                return full_zip_url
                self.logger.info(f"【任务轮询】返回ZIP包下载链接：{full_zip_url}，bactch_id：{batch_id}")
            elif data_state=='failed':
                err_msg = result_item.get("err_msg", "未知错误，无具体信息")
                raise PdfConversionError(f"【任务轮询】解析任务失败！batch_id：{batch_id}，错误信息：{err_msg}")
            else:
                self.logger.info(
                    f"【任务轮询】处理中... 已耗时{int(elapsed_time)}s，状态：{data_state}， batch_id：{batch_id}")
                time.sleep(poll_interval)

    def _step_3_download_and_extract(self, zip_url: str, output_dir_obj: Path, pdf_stem: str) -> str:
        self.logger.info(f"【ZIP下载】开始下载ZIP包：{zip_url} ...")
        response=requests.get(zip_url)

        if response.status_code!=200:
            raise RuntimeError(f"【ZIP下载】ZIP包下载失败：状态码：{response.status_code}，响应结果：{response}")

        zip_save_path=output_dir_obj/f'{pdf_stem}_result.zip'
        with open(zip_save_path, 'wb') as f:
            f.write(response.content)
        self.logger.info(f"【ZIP下载】ZIP包下载成功：保存路径：{zip_save_path}")

        extract_target_dir=output_dir_obj/pdf_stem
        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir)
        self.logger.info(f"【ZIP解压】已清空旧的解压目录：{extract_target_dir}")

        extract_target_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"【ZIP解压】开始解压ZIP包：{output_dir_obj} ...")
        with zipfile.ZipFile(zip_save_path,'r')as zip_file_obj:
            zip_file_obj.extractall(extract_target_dir)
        self.logger.info(f"【ZIP解压】ZIP解压完成，解压目录：{extract_target_dir}")

        self.logger.info(f"【MD重命名】找到MinerU生成的full.md文件")
        target_md_file=extract_target_dir/'full.md'
        self.logger.info(f"【MD重命名】开始将full.md文件进行重命名")
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        target_md_file.rename(new_md_path)
        self.logger.info(f"【MD重命名】重命名成功，文件名：{pdf_stem}.md")

        return str(new_md_path.absolute())


    def process(self, state: ImportGraphState):
        """
        :param state: `pdf_path`、`file_dir`
        :return: `md_path`、`md_content`
        """

        # 步骤1：校验PDF路径和输出目录
        pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)

        # 步骤2：上传PDF至MinerU并轮询解析结果
        zip_url = self._step_2_upload_and_poll(pdf_path_obj)

        # 步骤3：下载ZIP包并提取MD文件
        md_path = self._step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)

        # 步骤4：读取md的内容
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 步骤5：更新state状态
        state["md_path"] = str(md_path)
        state["md_content"] = md_content

        return state
if __name__ == '__main__':
    setup_logging()
    init_state = {
        "pdf_path": r"D:\work\doc\hak180产品安全手册.pdf",
        "file_dir": r"D:\output"
    }
    node_pdf_to_md = NodePDFToMD()
    result = node_pdf_to_md(init_state)

    # logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))