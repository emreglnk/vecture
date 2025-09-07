from fastapi import FastAPI, Query, Response
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

app = FastAPI()

def render_dot_image(vec, label=""):
    if vec.size != 1536:
        raise ValueError("Vector length must be 1536.")
    mat = vec.reshape(32, 48)

    vals = mat.astype(float)
    vals = np.clip(vals, np.percentile(vals, 1), np.percentile(vals, 99))
    norm = (vals - vals.min()) / (vals.max() - vals.min())
    sizes = 0 + norm * (200 - 10) * .3

    yy, xx = np.indices((32, 48))
    x = xx.ravel()
    y = yy.ravel()
    s = sizes.ravel()

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="black")
    ax.set_facecolor("black")
    ax.scatter(x, y, s=s, c="#FFD400", edgecolors="none")
    ax.set_xlim(-0.5, 47.5)
    ax.set_ylim(31.5, -0.5)
    ax.axis("off")
    fig.subplots_adjust(bottom=0.12)
    fig.text(0.5, 0.04, label, color="white", ha="center", va="center", fontsize=12)

    buf = BytesIO()
    plt.savefig(buf, format="jpeg", dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    buf.seek(0)
    return buf

@app.get("/render")
def render_from_url(url: str = Query(...), label: str = Query("")):
    try:
        r = requests.get(url)
        r.raise_for_status()
        arr = np.load(BytesIO(r.content))

        if arr.ndim == 1:
            vec = arr
        elif arr.ndim == 2 and arr.shape[1] == 1536:
            vec = arr[0]
        else:
            return {"error": "Invalid .npy format. Must be a 1536-length vector or (N, 1536)."}

        img_buf = render_dot_image(vec, label)
        return Response(content=img_buf.read(), media_type="image/jpeg")

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rendervector:app", host="0.0.0.0", port=8003)
