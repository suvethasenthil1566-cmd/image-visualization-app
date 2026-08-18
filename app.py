from flask import Flask, request, render_template_string, redirect, url_for
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Maximum upload size = 50 MB
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

# Store current image in server memory
current_image = None
current_processed = None
current_operation = "No operation selected"


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
# HTML + CSS
# ============================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

<title>IVA Image Preprocessing Dashboard</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #eef2f7;
    color: #1e293b;
}

.header {
    background: #172554;
    color: white;
    padding: 30px;
    text-align: center;
}

.header h1 {
    margin: 0;
    font-size: 32px;
}

.header p {
    margin-top: 10px;
}

.container {
    width: 92%;
    max-width: 1250px;
    margin: 25px auto;
}

.card {
    background: white;
    padding: 25px;
    margin-bottom: 22px;
    border-radius: 15px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.08);
}

.card h2 {
    color: #172554;
    margin-top: 0;
}

.upload-box {
    border: 2px dashed #64748b;
    padding: 30px;
    text-align: center;
    border-radius: 12px;
}

input[type=file] {
    margin: 15px;
}

.button-group {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
}

button {
    background: #2563eb;
    color: white;
    border: none;
    padding: 12px 20px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
}

button:hover {
    background: #1d4ed8;
}

.reset {
    background: #dc2626;
}

.reset:hover {
    background: #b91c1c;
}

.image-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 25px;
}

.image-box {
    text-align: center;
}

.image-box img {
    width: 100%;
    max-height: 450px;
    object-fit: contain;
    border: 2px solid #cbd5e1;
    border-radius: 12px;
    background: #f8fafc;
}

.matrix-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 20px;
}

.matrix-card {
    background: #f8fafc;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dbeafe;
}

.matrix-card h3 {
    color: #1e3a8a;
}

table {
    border-collapse: collapse;
    margin: auto;
}

td {
    border: 1px solid #64748b;
    padding: 10px 15px;
    text-align: center;
    background: white;
    font-weight: bold;
}

pre {
    background: #111827;
    color: #22c55e;
    padding: 18px;
    border-radius: 10px;
    overflow-x: auto;
}

.status {
    margin-top: 15px;
    padding: 12px;
    background: #dcfce7;
    color: #166534;
    border-radius: 8px;
}

@media(max-width: 750px) {

    .image-container {
        grid-template-columns: 1fr;
    }

}

</style>

</head>


<body>


<div class="header">

<h1>
Image Preprocessing Dashboard
</h1>

<p>
IVA Assignment | Spatial Domain Methods | Gradient Operators
</p>

</div>


<div class="container">


<!-- =====================================================
UPLOAD IMAGE
===================================================== -->

<div class="card">

<h2>📤 Upload Image</h2>

<div class="upload-box">

<form method="POST"
      action="/upload"
      enctype="multipart/form-data">

<input
    type="file"
    name="image"
    accept="image/*"
    required
>

<br>

<button type="submit">
Upload Image
</button>

</form>

{% if uploaded %}

<div class="status">
Image uploaded successfully!
</div>

{% endif %}

</div>

</div>


{% if uploaded %}


<!-- =====================================================
IMAGE DISPLAY
===================================================== -->

<div class="card">

<h2>🖼️ Image Comparison</h2>

<div class="image-container">


<div class="image-box">

<h3>Original Image</h3>

<img src="{{ original_url }}">

</div>


<div class="image-box">

<h3>Processed Image</h3>

{% if processed_url %}

<img src="{{ processed_url }}">

{% else %}

<p>
Choose an image processing technique.
</p>

{% endif %}

</div>


</div>

</div>


<!-- =====================================================
SPATIAL DOMAIN
===================================================== -->

<div class="card">

<h2>🔹 Spatial Domain Methods</h2>

<p>
These filters operate directly on the pixels of the image
to reduce noise.
</p>

<div class="button-group">


<form method="POST" action="/process">

<input type="hidden"
       name="operation"
       value="mean">

<button>
Mean Filter
</button>

</form>


<form method="POST" action="/process">

<input type="hidden"
       name="operation"
       value="median">

<button>
Median Filter
</button>

</form>


<form method="POST" action="/process">

<input type="hidden"
       name="operation"
       value="gaussian">

<button>
Gaussian Filter
</button>

</form>


</div>

</div>


<!-- =====================================================
GRADIENT OPERATORS
===================================================== -->

<div class="card">

<h2>📐 Gradient Operators</h2>

<div class="button-group">


<form method="POST" action="/process">

<input type="hidden"
       name="operation"
       value="laplacian">

<button>
Laplacian
</button>

</form>


<form method="POST" action="/process">

<input type="hidden"
       name="operation"
       value="sobel">

<button>
Sobel
</button>

</form>


<form method="POST" action="/process">

<input type="hidden"
       name="operation"
       value="prewitt">

<button>
Prewitt
</button>

</form>


</div>

</div>


<!-- =====================================================
CANNY
===================================================== -->

<div class="card">

<h2>✨ Edge Detection</h2>

<form method="POST" action="/process">

<input type="hidden"
       name="operation"
       value="canny">

<button>
Canny Edge Detection
</button>

</form>

</div>


<!-- =====================================================
GRAYSCALE
===================================================== -->

<div class="card">

<h2>⚫ Basic Processing</h2>

<form method="POST" action="/process">

<input type="hidden"
       name="operation"
       value="grayscale">

<button>
Grayscale
</button>

</form>

</div>


<!-- =====================================================
OPERATOR MATRICES
===================================================== -->

<div class="card">

<h2>🧮 Operator / Kernel Matrices</h2>

<div class="matrix-container">


{% for name, matrix in kernels.items() %}

<div class="matrix-card">

<h3>
{{ name }}
</h3>

<table>

{% for row in matrix %}

<tr>

{% for value in row %}

<td>
{{ value }}
</td>

{% endfor %}

</tr>

{% endfor %}

</table>

</div>

{% endfor %}


</div>

</div>


<!-- =====================================================
PIXEL MATRIX
===================================================== -->

<div class="card">

<h2>🔢 Image Pixel Matrix</h2>

<p>
Sample 10 × 10 grayscale pixel matrix:
</p>

<pre>{{ matrix }}</pre>

</div>


<!-- =====================================================
CURRENT OPERATION
===================================================== -->

<div class="card">

<h2>✅ Current Processing</h2>

<h3>
{{ operation }}
</h3>

</div>


{% endif %}


</div>

</body>

</html>

"""


# ============================================================
# CONVERT IMAGE TO BASE64
# ============================================================

def image_to_base64(image):

    success, buffer = cv2.imencode(".jpg", image)

    if not success:
        return ""

    encoded = base64.b64encode(buffer).decode("utf-8")

    return "data:image/jpeg;base64," + encoded


# ============================================================
# PIXEL MATRIX
# ============================================================

def get_pixel_matrix(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    small = cv2.resize(
        gray,
        (10, 10)
    )

    return str(small)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    global current_image
    global current_processed
    global current_operation

    if current_image is None:

        return render_template_string(
            HTML,
            uploaded=False,
            original_url=None,
            processed_url=None,
            matrix=None,
            operation=None,
            kernels=KERNELS
        )

    matrix = get_pixel_matrix(
        current_image
    )

    return render_template_string(
        HTML,

        uploaded=True,

        original_url=image_to_base64(
            current_image
        ),

        processed_url=(
            image_to_base64(current_processed)
            if current_processed is not None
            else None
        ),

        matrix=matrix,

        operation=current_operation,

        kernels=KERNELS
    )


# ============================================================
# UPLOAD
# ============================================================

@app.route("/upload", methods=["POST"])
def upload():

    global current_image
    global current_processed
    global current_operation

    file = request.files.get("image")

    if file is not None:

        data = file.read()

        array = np.frombuffer(
            data,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR
        )

        if image is not None:

            current_image = image

            current_processed = None

            current_operation = \
                "No operation selected"

    return redirect(
        url_for("home")
    )


# ============================================================
# PROCESS IMAGE
# ============================================================

@app.route("/process", methods=["POST"])
def process():

    global current_image
    global current_processed
    global current_operation

    if current_image is None:

        return redirect(
            url_for("home")
        )

    operation = request.form.get(
        "operation"
    )

    gray = cv2.cvtColor(
        current_image,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # MEAN FILTER
    # ========================================================

    if operation == "mean":

        current_processed = cv2.blur(
            gray,
            (5, 5)
        )

        current_operation = \
            "Mean Filter"


    # ========================================================
    # MEDIAN FILTER
    # ========================================================

    elif operation == "median":

        current_processed = cv2.medianBlur(
            gray,
            5
        )

        current_operation = \
            "Median Filter"


    # ========================================================
    # GAUSSIAN FILTER
    # ========================================================

    elif operation == "gaussian":

        current_processed = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        current_operation = \
            "Gaussian Filter"


    # ========================================================
    # LAPLACIAN
    # ========================================================

    elif operation == "laplacian":

        result = cv2.Laplacian(
            gray,
            cv2.CV_64F
        )

        current_processed = cv2.convertScaleAbs(
            result
        )

        current_operation = \
            "Laplacian Gradient Operator"


    # ========================================================
    # SOBEL
    # ========================================================

    elif operation == "sobel":

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

        current_processed = cv2.convertScaleAbs(
            magnitude
        )

        current_operation = \
            "Sobel Gradient Operator"


    # ========================================================
    # PREWITT
    # ========================================================

    elif operation == "prewitt":

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
            -1,
            kernel_x
        )

        prewitt_y = cv2.filter2D(
            gray,
            -1,
            kernel_y
        )

        current_processed = cv2.addWeighted(
            prewitt_x,
            0.5,
            prewitt_y,
            0.5,
            0
        )

        current_operation = \
            "Prewitt Gradient Operator"


    # ========================================================
    # CANNY
    # ========================================================

    elif operation == "canny":

        current_processed = cv2.Canny(
            gray,
            100,
            200
        )

        current_operation = \
            "Canny Edge Detection"


    # ========================================================
    # GRAYSCALE
    # ========================================================

    elif operation == "grayscale":

        current_processed = gray

        current_operation = \
            "Grayscale Conversion"


    return redirect(
        url_for("home")
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    import webbrowser
    from threading import Timer

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:5000")

    Timer(1.5, open_browser).start()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )