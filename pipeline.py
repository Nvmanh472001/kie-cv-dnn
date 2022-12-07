import utils.utilities as utility
from modules.classification.layoutlm_v3 import LayoutLMv3Cls


def pipeline_handle(image):
    args = utility.parser_args()
    configs = utility.get_config_from_yaml(args.yaml_dir)

    from tool.infer.predict_system import TextSystem
    text_system = TextSystem(configs)

    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=False, lang='en')
    [ocr_result] = ocr.ocr(image, cls=False)

    cls = LayoutLMv3Cls(weights_path="checkpoints/classification")
    img_res = cls(image_path=img_path, ocr_result=ocr_result)
    return img_res
