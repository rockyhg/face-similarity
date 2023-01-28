import time

import numpy as np
import streamlit as st
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw

mtcnn = MTCNN(select_largest=True)
resnet = InceptionResnetV1(pretrained="vggface2").eval()


def detect_face(img):
    box, prob = mtcnn.detect(img)
    box = box.squeeze()
    draw = ImageDraw.Draw(img)
    draw.rectangle(box, outline=(255, 0, 0), width=4)
    return img


def feature_vector(img) -> np.ndarray:
    """画像の特徴量ベクトルを取得"""
    img_cropped = mtcnn(img)
    feature_vector = resnet(img_cropped.unsqueeze(0))
    feature_vector_np = feature_vector.squeeze().to("cpu").detach().numpy().copy()
    return feature_vector_np


def cosine_similarity(a, b):
    """2つのベクトル間のコサイン類似度を計算 -
    コサイン類似度(A, B) = A・B / ||A|| ||B||"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    st.title("Face Similarity")
    st.caption("顔画像の類似度を測定します。")

    col1, col2 = st.columns(2)

    with col1:
        image_file1 = st.file_uploader("Image 1", type=["jpg", "jpeg"])
        if image_file1:
            img1 = Image.open(image_file1)
            st.image(
                detect_face(img1),
                caption="Uploaded Image",
                width=160,
                use_column_width=False,
            )

    with col2:
        image_file2 = st.file_uploader("Image 2", type=["jpg", "jpeg"])

        if image_file2:
            img2 = Image.open(image_file2)
            st.image(
                detect_face(img2),
                caption="Uploaded Image",
                width=160,
                use_column_width=False,
            )

    col1, col2, col3, col4 = st.columns(4)

    with col2:
        st.write("")
        if (image_file1 is not None) and (image_file2 is not None):
            button = st.button("Calc Similarity")
        else:
            button = st.button("Calc Similarity", disabled=True)

    with col3:
        st.write("")
        if button:
            with st.spinner("Calculating ..."):
                time.sleep(1)
                fv1 = feature_vector(img1)
                fv2 = feature_vector(img2)
                similarity = cosine_similarity(fv1, fv2)
            st.metric("Cosine Similarity", str(round(similarity, 3)), "max 1.0", delta_color="off")

    st.write("")

    with st.expander("【参考】計算方法"):
        st.markdown(
            "[facenet-pytorch](https://www.kaggle.com/code/timesler/guide-to-mtcnn-in-facenet-pytorch/notebook#Bounding-boxes-and-facial-landmarks) を使用"
        )
        st.code("from facenet_pytorch import MTCNN, InceptionResnetV1")

        st.markdown("1. アップロードされた画像ファイルから顔部分を切り出す")
        st.code(
            "mtcnn = MTCNN(select_largest=True)\nbox, prob = mtcnn.detect(img)",
            language="python",
        )

        st.markdown("2. 顔画像から特徴量を抽出する")
        st.code(
            "resnet = InceptionResnetV1(pretrained='vggface2').eval()\nfeature_vector = resnet(img_cropped.unsqueeze(0))",
            language="python",
        )

        st.markdown("3. 抽出した特徴量同士のコサイン類似度*を計算する")
        st.code(
            "import numpy as np\nnp.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))",
            language="python",
        )
        st.markdown("")
        st.markdown("**コサイン類似度:**")
        st.markdown("> 2つのベクトルが「どのくらい似ているか」という類似性を表す尺度で、具体的には2つのベクトルがなす角のコサイン値のこと。1なら「似ている」を、-1なら「似ていない」を意味する。")
        st.markdown("https://atmarkit.itmedia.co.jp/ait/articles/2112/08/news020.html")


if __name__ == "__main__":
    main()
