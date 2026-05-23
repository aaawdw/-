import json
import logging
from pathlib import Path


from processor.import_process.base import BaseNode,setup_logging
from processor.import_process.exceptions import StateFieldError, FileProcessingError, ValidationError
from processor.import_process.state import ImportGraphState


class NodeEntry(BaseNode):
    """
    入口节点：任务分发
    """

    name = "node_entry"

    def process(self, state: ImportGraphState):
        import_file_path=state.get("import_file_path")

        if not import_file_path:
            raise StateFieldError(field_name='import_file_path',expected_type=str)

        import_file_path_obj=Path(import_file_path)

        if not import_file_path_obj.exists():
            raise FileProcessingError(message=f'文件{import_file_path_obj.name}不存在')
        if import_file_path_obj.suffix=='.pdf':
            state['is_pdf_read_enabled'] = True
        elif import_file_path_obj.suffix=='.md':
            state['is_md_read_enabled'] = True
        else:
            raise ValidationError(message=f'该文件的后缀格式{import_file_path_obj.suffix}不支持')
        state['file_title']=import_file_path_obj.stem
        return state

if __name__ == '__main__':
    setup_logging()
    init_state={'import_file_path':'D:\work\doc\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册.pdf',}
    node_entry=NodeEntry()
    res=node_entry(init_state)
    logging.getLogger().info(json.dumps(res,ensure_ascii=False,indent=4))