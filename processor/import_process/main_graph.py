import json

from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.import_process.nodes.node_bge_embedding import NodeBGEEmbedding
from processor.import_process.nodes.node_document_split import NodeDocumentSplit
from processor.import_process.nodes.node_entry import NodeEntry
from processor.import_process.nodes.node_import_milvus import NodeImportMilvus
from processor.import_process.nodes.node_item_name_recognition import NodeItemNameRecognition
from processor.import_process.nodes.node_md_img import NodeMDImg
from processor.import_process.nodes.node_pdf_to_md import NodePDFToMD
from processor.import_process.state import ImportGraphState

class KBImportWorkflow:
    def __init__(self,config=None):
        self._complied_graph = None

    @property
    def graph(self):
        if self._complied_graph is None:
            self._complied_graph = self.build_graph()
        return self._complied_graph
    @staticmethod
    def route_after_entry(state: ImportGraphState) -> str:
        if state.get('is_pdf_read_enabled'):
            return 'node_pdf_to_md'
        elif state.get('is_md_read_enabled'):
            return 'node_md_img'
        else:
            return END
    def build_graph(self):
        graph=StateGraph(ImportGraphState)
        graph.add_node("node_entry", NodeEntry())
        graph.add_node("node_pdf_to_md", NodePDFToMD())
        graph.add_node("node_md_img", NodeMDImg())
        graph.add_node("node_document_split", NodeDocumentSplit())
        graph.add_node("node_item_name_recognition", NodeItemNameRecognition())
        graph.add_node("node_bge_embedding", NodeBGEEmbedding())
        graph.add_node("node_import_milvus", NodeImportMilvus())

        graph.set_entry_point('node_entry')

        graph.add_conditional_edges(
            'node_entry',
            self.route_after_entry,
            {
                "node_md_img": "node_md_img",
                "node_pdf_to_md": "node_pdf_to_md",
                END: END
            }
        )

        graph.add_edge("node_pdf_to_md", "node_md_img")
        graph.add_edge("node_md_img", "node_document_split")
        graph.add_edge("node_document_split", "node_item_name_recognition")
        graph.add_edge("node_item_name_recognition", "node_bge_embedding")
        graph.add_edge("node_bge_embedding", "node_import_milvus")
        graph.add_edge("node_import_milvus", END)

        return graph.compile()
    def run(self,state: ImportGraphState,stream=False):
        self.graph.get_graph().print_ascii()
        if stream:
            return self.graph.get_stream(state,stream_mode='values')
        else:
            return self.graph.invoke(state)
if __name__ == '__main__':
    a=KBImportWorkflow()
    a.run({'import_file_path':'D:\work\doc\H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf',"file_dir": r"D:\output",
           'pdf_path':'D:\work\doc\H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf'})