import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IVA Image Preprocessing Dashboard",
    page_icon="🖼️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🖼️ Image Preprocessing Dashboard")

st.markdown(
    """
    **IVA Assignment | Spatial Domain Methods | Gradient Operators**

    This application demonstrates image preprocessing techniques
    using OpenCV, including noise suppression, gradient operators,
    and Canny edge detection.
    """
)

st.divider()


# ============================================================
# OPERATOR / KERNEL MATRICES
# ============================================================

KERNELS = {

    "Mean Filter": [
        ["1/9", "1/9", "1/9"],
        ["1/9", "1/9", "1/9"],
        ["1/9", "1/9", "1/9"]
    ],

    "Gaussian Filter": [
        ["1", "2", "1"],
        ["2", "4", "2"],
        ["1", "2", "1"]
    ],

    "Laplacian": [
        ["0", "1", "0"],
        ["1", "-4", "1"],
        ["0", "1", "0"]
    ],

    "Sobel X": [
        ["-1", "0", "1"],
        ["-2", "0", "2"],
        ["-1", "0", "1"]
    ],

    "Sobel Y": [
        ["-1", "-2", "-1"],
        ["0", "0", "0"],
        ["1", "2", "1"]
    ],

    "Prewitt X": [
        ["-1", "0", "1"],
        ["-1", "0", "1"],
        ["-1", "0", "1"]
    ],

    "Prewitt Y": [
        ["-1", "-1", "-1"],
        ["0", "0", "0"],
        ["1", "1", "1"]
    ]
}


# ============================================================
# SESSION STATE
# ============================================================

if "image" not in st.session_state:
    st.session_state.image = None

if "processed" not in st.session_state:
    st.session_state.processed = None

if "operation" not in st.session_state:
    st.session_state.operation = "No operation selected"


# ============================================================
# RESET FUNCTION
# ============================================================

def reset_app():

    st.session_state.image = None
    st.session_state.processed = None
    st.session_state.operation = "No operation selected"


# ============================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================

def process_image(image, operation):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # MEAN FILTER
    # ========================================================

    if operation == "Mean Filter":

        processed = cv2.blur(
            gray,
            (5, 5)
        )

        name = "Mean Filter"


    # ========================================================
    # MEDIAN FILTER
    # ========================================================

    elif operation == "Median Filter":

        processed = cv2.medianBlur(
            gray,
            5
        )

        name = "Median Filter"


    # ========================================================
    # GAUSSIAN FILTER
    # ========================================================

    elif operation == "Gaussian Filter":

        processed = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        name = "Gaussian Filter"


    # ========================================================
    # LAPLACIAN
    # ========================================================

    elif operation == "Laplacian":

        result = cv2.Laplacian(
            gray,
            cv2.CV_64F
        )

        processed = cv2.convertScaleAbs(
            result
        )

        name = "Laplacian Gradient Operator"


    # ========================================================
    # SOBEL
    # ========================================================

    elif operation == "Sobel":

        sobel_x = cv2.Sobel(
            gray,
            cv2.CV_64F,
            1,
            0,
            ksize=3
        )

        sobel_y = cv2.Sobel(
            gray,
            cv2.CV_64F,
            0,
            1,
            ksize=3
        )

        magnitude = cv2.magnitude(
            np.float32(sobel_x),
            np.float32(sobel_y)
        )

        processed = cv2.convertScaleAbs(
            magnitude
        )

        name = "Sobel Gradient Operator"


    # ========================================================
    # PREWITT
    # ========================================================

    elif operation == "Prewitt":

        kernel_x = np.array(
            [
                [-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]
            ],
            dtype=np.float32
        )

        kernel_y = np.array(
            [
                [-1, -1, -1],
                [0, 0, 0],
                [1, 1, 1]
            ],
            dtype=np.float32
        )

        prewitt_x = cv2.filter2D(
            gray,
            cv2.CV_32F,
            kernel_x
        )

        prewitt_y = cv2.filter2D(
            gray,
            cv2.CV_32F,
            kernel_y
        )

        magnitude = cv2.magnitude(
            prewitt_x,
            prewitt_y
        )

        processed = cv2.convertScaleAbs(
            magnitude
        )

        name = "Prewitt Gradient Operator"


    # ========================================================
    # CANNY EDGE DETECTION
    # ========================================================

    elif operation == "Canny Edge Detection":

        processed = cv2.Canny(
            gray,
            100,
            200
        )

        name = "Canny Edge Detection"


    # ========================================================
    # GRAYSCALE
    # ========================================================

    elif operation == "Grayscale":

        processed = gray

        name = "Grayscale Conversion"


    else:

        processed = None
        name = "No operation selected"


    return processed, name


# ============================================================
# UPLOAD IMAGE
# ============================================================

st.header("📤 Upload Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Maximum file size: 50 MB"
)


# ============================================================
# HANDLE UPLOADED IMAGE
# ============================================================

if uploaded_file is not None:

    if uploaded_file.size > 50 * 1024 * 1024:

        st.error(
            "❌ File is larger than 50 MB."
        )

    else:

        file_bytes = np.asarray(
            bytearray(
                uploaded_file.getvalue()
            ),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if image is not None:

            # Store image
            st.session_state.image = image

            # Reset processing when a new image is uploaded
            st.session_state.processed = None
            st.session_state.operation = "No operation selected"

            st.success(
                "Image uploaded successfully! ✅"
            )


# ============================================================
# IF IMAGE EXISTS
# ============================================================

if st.session_state.image is not None:

    image = st.session_state.image


    # ========================================================
    # IMAGE COMPARISON
    # ========================================================

    st.divider()

    st.header("🖼️ Image Comparison")

    col1, col2 = st.columns(2)


    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    with col1:

        st.subheader("Original Image")

        original_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            original_rgb,
            use_container_width=True
        )


    # ========================================================
    # PROCESSED IMAGE
    # ========================================================

    with col2:

        st.subheader("Processed Image")

        if st.session_state.processed is not None:

            st.image(
                st.session_state.processed,
                use_container_width=True
            )

            st.success(
                "Current Processing: "
                + st.session_state.operation
            )

        else:

            st.info(
                "Choose an image processing technique below."
            )


    # ========================================================
    # SPATIAL DOMAIN METHODS
    # ========================================================

    st.divider()

    st.header("🔹 Spatial Domain Methods")

    st.write(
        """
        Spatial domain methods operate directly on image pixels.
        They are commonly used for noise suppression and image
        enhancement.
        """
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        if st.button(
            "Mean Filter",
            use_container_width=True
        ):

            processed, name = process_image(
                image,
                "Mean Filter"
            )

            st.session_state.processed = processed
            st.session_state.operation = name

            st.rerun()


    with col2:

        if st.button(
            "Median Filter",
            use_container_width=True
        ):

            processed, name = process_image(
                image,
                "Median Filter"
            )

            st.session_state.processed = processed
            st.session_state.operation = name

            st.rerun()


    with col3:

        if st.button(
            "Gaussian Filter",
            use_container_width=True
        ):

            processed, name = process_image(
                image,
                "Gaussian Filter"
            )

            st.session_state.processed = processed
            st.session_state.operation = name

            st.rerun()


    # ========================================================
    # GRADIENT OPERATORS
    # ========================================================

    st.divider()

    st.header("📐 Gradient Operators")

    st.write(
        """
        Gradient operators detect intensity changes and help
        identify edges in an image.
        """
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        if st.button(
            "Laplacian",
            use_container_width=True
        ):

            processed, name = process_image(
                image,
                "Laplacian"
            )

            st.session_state.processed = processed
            st.session_state.operation = name

            st.rerun()


    with col2:

        if st.button(
            "Sobel",
            use_container_width=True
        ):

            processed, name = process_image(
                image,
                "Sobel"
            )

            st.session_state.processed = processed
            st.session_state.operation = name

            st.rerun()


    with col3:

        if st.button(
            "Prewitt",
            use_container_width=True
        ):

            processed, name = process_image(
                image,
                "Prewitt"
            )

            st.session_state.processed = processed
            st.session_state.operation = name

            st.rerun()


    # ========================================================
    # CANNY EDGE DETECTION
    # ========================================================

    st.divider()

    st.header("✨ Edge Detection")

    if st.button(
        "Canny Edge Detection",
        use_container_width=True
    ):

        processed, name = process_image(
            image,
            "Canny Edge Detection"
        )

        st.session_state.processed = processed
        st.session_state.operation = name

        st.rerun()


    # ========================================================
    # BASIC PROCESSING
    # ========================================================

    st.divider()

    st.header("⚫ Basic Processing")

    if st.button(
        "Grayscale",
        use_container_width=True
    ):

        processed, name = process_image(
            image,
            "Grayscale"
        )

        st.session_state.processed = processed
        st.session_state.operation = name

        st.rerun()


    # ========================================================
    # CURRENT OPERATION
    # ========================================================

    st.divider()

    st.header("✅ Current Processing")

    st.info(
        st.session_state.operation
    )


    # ========================================================
    # OPERATOR MATRICES
    # ========================================================

    st.divider()

    st.header("🧮 Operator / Kernel Matrices")

    selected_operation = st.session_state.operation


    if selected_operation == "Laplacian Gradient Operator":

        st.subheader("Laplacian Matrix")

        st.table(
            KERNELS["Laplacian"]
        )


    elif selected_operation == "Sobel Gradient Operator":

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Sobel X")

            st.table(
                KERNELS["Sobel X"]
            )

        with col2:

            st.subheader("Sobel Y")

            st.table(
                KERNELS["Sobel Y"]
            )


    elif selected_operation == "Prewitt Gradient Operator":

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Prewitt X")

            st.table(
                KERNELS["Prewitt X"]
            )

        with col2:

            st.subheader("Prewitt Y")

            st.table(
                KERNELS["Prewitt Y"]
            )


    elif selected_operation in KERNELS:

        st.subheader(
            selected_operation
        )

        st.table(
            KERNELS[selected_operation]
        )


    else:

        st.info(
            "Select Mean, Gaussian, Laplacian, Sobel or Prewitt "
            "to display its operator matrix."
        )


    # ========================================================
    # ALL MATRICES
    # ========================================================

    with st.expander(
        "📚 View All Operator / Kernel Matrices"
    ):

        for name, matrix in KERNELS.items():

            st.subheader(name)

            st.table(matrix)


    # ========================================================
    # PIXEL MATRIX
    # ========================================================

    st.divider()

    st.header("🔢 Image Pixel Matrix")

    st.write(
        "Sample 10 × 10 grayscale pixel matrix:"
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    small = cv2.resize(
        gray,
        (10, 10)
    )

    st.dataframe(
        small,
        use_container_width=True
    )


    # ========================================================
    # IMAGE INFORMATION
    # ========================================================

    st.divider()

    st.header("📊 Image Information")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Width",
            image.shape[1]
        )

    with c2:

        st.metric(
            "Height",
            image.shape[0]
        )

    with c3:

        st.metric(
            "Channels",
            image.shape[2]
        )


    # ========================================================
    # DOWNLOAD PROCESSED IMAGE
    # ========================================================

    if st.session_state.processed is not None:

        st.divider()

        st.header("⬇️ Download Processed Image")

        processed_image = st.session_state.processed

        success, encoded_image = cv2.imencode(
            ".png",
            processed_image
        )

        if success:

            st.download_button(
                label="⬇️ Download Processed Image",
                data=encoded_image.tobytes(),
                file_name="processed_image.png",
                mime="image/png",
                use_container_width=True
            )


    # ========================================================
    # RESET
    # ========================================================

    st.divider()

    if st.button(
        "🔄 Reset Application",
        use_container_width=True
    ):

        reset_app()

        st.rerun()


# ============================================================
# NO IMAGE
# ============================================================

else:

    st.info(
        "👆 Upload an image to start image preprocessing."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "IVA Assignment | Image Preprocessing using Spatial Domain Methods"
)