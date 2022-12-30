import utils.utilities as utility
import streamlit as st
import cv2

def pipeline_handle(img):
    args = utility.parser_args()
    configs = utility.get_config_from_yaml(args.yaml_dir)

    from tools.infer.predict_system import TextSystem
    text_system = TextSystem(configs)

    img_res = text_system(img)
    st.session_state.image = img_res
    return img_res
    
    
    
if __name__ == "__main__":
    img_path = "./img_12.jpg"
    img = cv2.imread(img_path)
    img_res = pipeline_handle(img)
    filename = 'savedImage.jpg'
    img_res.save(filename)