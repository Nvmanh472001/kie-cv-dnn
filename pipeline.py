import utils.utilities as utility
from modules.classification.layoutlm_v3 import LayoutLMv3Cls
import streamlit as st


def pipeline_handle(image_ndarray, image_pillow):
    args = utility.parser_args()
    configs = utility.get_config_from_yaml(args.yaml_dir)

    from tool.infer.predict_system import TextSystem
    text_system = TextSystem(configs)

    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=False, lang='en')
    [ocr_result] = ocr.ocr(image_ndarray, cls=False)

    cls = LayoutLMv3Cls(weights_path="checkpoints/classification")
    img_res = cls(image=image_pillow, ocr_result=ocr_result)
    st.session_state.image = img_res
