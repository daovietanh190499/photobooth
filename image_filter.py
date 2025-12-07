"""
photo_editor_full.py
Gradio Photo Editor - upgraded

Dependencies:
    pip install gradio pillow numpy matplotlib

Run:
    python photo_editor_full.py
"""

import gradio as gr
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import matplotlib.colors as mc
import io
import os
import time
from typing import Tuple

# -------------------------
# Utility helpers
# -------------------------
def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img).astype(np.float32) / 255.0

def np_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def save_image_temp(img: Image.Image, prefix="edited") -> str:
    os.makedirs("/tmp/photo_editor", exist_ok=True)
    ts = int(time.time() * 1000)
    path = f"/tmp/photo_editor/{prefix}_{ts}.jpg"
    img.save(path, quality=95)
    return path

# -------------------------
# Color conversions (RGB <-> HSV)
# -------------------------
def rgb_to_hsv_np(img_np: np.ndarray) -> np.ndarray:
    flat = img_np.reshape(-1, 3)
    hsv_flat = mc.rgb_to_hsv(flat)
    return hsv_flat.reshape(img_np.shape)

def hsv_to_rgb_np(hsv_np: np.ndarray) -> np.ndarray:
    flat = hsv_np.reshape(-1, 3)
    rgb_flat = mc.hsv_to_rgb(flat)
    return rgb_flat.reshape(hsv_np.shape)

# -------------------------
# Image adjustments
# -------------------------
def adjust_brightness_np(img_np: np.ndarray, amount: float) -> np.ndarray:
    # amount: -1..+2 (additive)
    return np.clip(img_np + amount, 0, 1)

def adjust_contrast_np(img_np: np.ndarray, factor: float) -> np.ndarray:
    # factor: 0..4
    lum = 0.299*img_np[...,0] + 0.587*img_np[...,1] + 0.114*img_np[...,2]
    mean = np.mean(lum)
    out = (img_np - mean) * factor + mean
    return np.clip(out, 0, 1)

def adjust_saturation_np(img_np: np.ndarray, factor: float) -> np.ndarray:
    hsv = rgb_to_hsv_np(img_np)
    hsv[...,1] = np.clip(hsv[...,1] * factor, 0, 1)
    return hsv_to_rgb_np(hsv)

def adjust_exposure_np(img_np: np.ndarray, ev: float) -> np.ndarray:
    # ev in stops (-3..+3)
    factor = 2.0 ** ev
    return np.clip(img_np * factor, 0, 1)

def adjust_temperature_np(img_np: np.ndarray, temp: float) -> np.ndarray:
    # temp -1..+1
    r_scale = 1.0 + 0.35 * np.clip(temp, 0, 1)
    b_scale = 1.0 + 0.35 * np.clip(-temp, 0, 1)
    g_scale = 1.0 + 0.08 * temp
    out = img_np.copy()
    out[...,0] *= r_scale
    out[...,1] *= g_scale
    out[...,2] *= b_scale
    return np.clip(out, 0, 1)

def adjust_hue_np(img_np: np.ndarray, deg: float) -> np.ndarray:
    if abs(deg) < 1e-6:
        return img_np
    hsv = rgb_to_hsv_np(img_np)
    hsv[...,0] = (hsv[...,0] + deg/360.0) % 1.0
    return hsv_to_rgb_np(hsv)

def adjust_shadows_highlights_np(img_np: np.ndarray, shadows_gain: float, highlights_gain: float) -> np.ndarray:
    lum = 0.299*img_np[...,0] + 0.587*img_np[...,1] + 0.114*img_np[...,2]
    shadow_mask = np.clip((0.5 - lum) / 0.5, 0, 1)
    highlight_mask = np.clip((lum - 0.5) / 0.5, 0, 1)
    out = img_np.copy()
    out = out * (1 + (shadows_gain - 1.0) * shadow_mask[...,None])
    out = out * (1 + (highlights_gain - 1.0) * (1 - highlight_mask[...,None]))
    return np.clip(out, 0, 1)

def apply_vignette_np(img_np: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0: 
        return img_np
    h, w = img_np.shape[:2]
    y = np.linspace(-1,1,h)[:,None]
    x = np.linspace(-1,1,w)[None,:]
    radius = np.sqrt(x*x + y*y)
    mask = 1.0 - np.clip((radius - 0.5) / 0.8, 0, 1)
    vign = 1.0 - strength * (1.0 - mask)
    return img_np * vign[...,None]

def apply_fade_np(img_np: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img_np
    gray = np.ones_like(img_np) * 0.5
    mixed = img_np*(1-amount) + gray*amount
    gamma = 1.0 - 0.35*amount
    return np.clip(np.power(np.clip(mixed,0,1), gamma), 0, 1)

def apply_grain_np(img_np: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img_np
    noise = np.random.normal(loc=0.0, scale=amount*0.08, size=img_np.shape)
    out = img_np + noise
    return np.clip(out, 0, 1)

def apply_blur_pil(img: Image.Image, radius: float) -> Image.Image:
    if radius <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_sharpen_pil(img: Image.Image, strength: float) -> Image.Image:
    if strength <= 0:
        return img
    # PIL's SHARPEN is binary; use UnsharpMask for strength
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=150*strength, threshold=3))

# -------------------------
# Tone curve: simple LUT via 3 control points
# -------------------------
def make_tone_lut(shadow_out: int, mid_out: int, high_out: int) -> np.ndarray:
    """
    shadow_out, mid_out, high_out: 0..255 mapping for input points [0, 128, 255]
    returns LUT of size 256.
    We'll perform linear interpolation across three segments.
    """
    xs = np.array([0, 128, 255], dtype=np.float32)
    ys = np.array([shadow_out, mid_out, high_out], dtype=np.float32)
    xq = np.arange(256, dtype=np.float32)
    lut = np.interp(xq, xs, ys)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut

def apply_tone_curve_pil(img: Image.Image, lut: np.ndarray) -> Image.Image:
    if lut is None:
        return img
    # Apply same LUT to each channel
    return Image.merge("RGB", [img.getchannel(c).point(lut.tolist()) for c in ("R","G","B")])

# -------------------------
# Selective color by hue range adjustments
# -------------------------
def selective_color_np(img_np: np.ndarray, hue_center: float, hue_range: float, sat_mult: float, val_mult: float) -> np.ndarray:
    """
    hue_center: 0..360
    hue_range: 0..180 (half width)
    sat_mult: multiply saturation
    val_mult: multiply value (brightness)
    """
    hsv = rgb_to_hsv_np(img_np)
    # hue in [0..1], convert
    hue_deg = (hsv[...,0] * 360.0)
    # compute mask
    diff = np.abs(((hue_deg - hue_center + 180) % 360) - 180)
    mask = np.clip(1 - (diff / hue_range), 0, 1) if hue_range > 0 else np.zeros_like(diff)
    mask = mask[...,None]
    hsv[...,1] = np.clip(hsv[...,1] * (1 + (sat_mult-1.0)*mask[...,0]), 0, 1)
    hsv[...,2] = np.clip(hsv[...,2] * (1 + (val_mult-1.0)*mask[...,0]), 0, 1)
    return hsv_to_rgb_np(hsv)

# -------------------------
# Full pipeline
# -------------------------
def process_image_pipeline(
    pil_img: Image.Image,
    brightness: float,
    contrast: float,
    saturation: float,
    exposure: float,
    highlights: float,
    shadows: float,
    temperature: float,
    hue: float,
    fade: float,
    vignette: float,
    blur_radius: float,
    sharpen_strength: float,
    grain: float,
    tone_sh: int,
    tone_mid: int,
    tone_hi: int,
    selective_hue: float,
    selective_range: float,
    selective_sat_mult: float,
    selective_val_mult: float,
    preset: str
) -> Image.Image:
    img = ensure_rgb(pil_img.copy())
    original = img.copy()

    # Apply preset base tweaks
    if preset and preset != "None":
        if preset == "Vibrant":
            saturation *= 1.18
            contrast *= 1.08
        elif preset == "Vintage":
            temperature -= 0.18
            fade = min(1.0, fade + 0.22)
            saturation *= 0.9
            contrast *= 0.92
        elif preset == "Cold Blue":
            temperature -= 0.32
            saturation *= 0.95
        elif preset == "Warm Film":
            temperature += 0.25
            fade = min(1.0, fade + 0.08)
            saturation *= 1.05
        elif preset == "Cinematic":
            contrast *= 1.15
            saturation *= 0.95
            vignette = max(vignette, 0.18)

    # PIL operations for blur/sharpen first (pixel-space)
    if blur_radius > 0:
        img = apply_blur_pil(img, blur_radius)
    if sharpen_strength > 0:
        img = apply_sharpen_pil(img, sharpen_strength)

    # Convert to np for color ops
    np_img = pil_to_np(img)

    # Exposure
    np_img = adjust_exposure_np(np_img, exposure)
    # Brightness (additive)
    np_img = adjust_brightness_np(np_img, brightness)
    # Contrast
    np_img = adjust_contrast_np(np_img, contrast)
    # Saturation
    np_img = adjust_saturation_np(np_img, saturation)
    # Shadows / Highlights
    np_img = adjust_shadows_highlights_np(np_img, shadows_gain=shadows+1.0, highlights_gain=1.0-highlights*0.8 if highlights>0 else 1.0)
    # Temperature
    np_img = adjust_temperature_np(np_img, temperature)
    # Hue rotation
    np_img = adjust_hue_np(np_img, hue)
    # Selective color
    if selective_range > 0 and (selective_sat_mult != 1.0 or selective_val_mult != 1.0):
        np_img = selective_color_np(np_img, hue_center=selective_hue, hue_range=selective_range, sat_mult=selective_sat_mult, val_mult=selective_val_mult)
    # Fade
    np_img = apply_fade_np(np_img, fade)
    # Grain
    np_img = apply_grain_np(np_img, grain)
    # Vignette
    np_img = apply_vignette_np(np_img, vignette)

    img = np_to_pil(np_img)

    # Tone Curve (LUT)
    if not (tone_sh == 0 and tone_mid == 128 and tone_hi == 255):
        lut = make_tone_lut(tone_sh, tone_mid, tone_hi)
        img = apply_tone_curve_pil(img, lut)

    return img

# -------------------------
# Gradio app: history state management
# -------------------------
def init_state():
    return {"undo": [], "redo": [], "current": None, "original": None}

def push_history(state: dict, pil_img: Image.Image):
    # push current to undo, set current
    if state["current"] is not None:
        state["undo"].append(state["current"].copy())
    state["current"] = pil_img.copy()
    # limit history length
    if len(state["undo"]) > 25:
        state["undo"].pop(0)
    # clear redo on new action
    state["redo"] = []
    return state

def do_undo(state: dict):
    if not state["undo"]:
        return state["current"], state
    last = state["undo"].pop()
    if state["current"] is not None:
        state["redo"].append(state["current"].copy())
    state["current"] = last.copy()
    return state["current"], state

def do_redo(state: dict):
    if not state["redo"]:
        return state["current"], state
    nxt = state["redo"].pop()
    if state["current"] is not None:
        state["undo"].append(state["current"].copy())
    state["current"] = nxt.copy()
    return state["current"], state

# -------------------------
# Gradio callbacks
# -------------------------
def on_upload(image, state):
    # image: PIL
    if image is None:
        return None, state
    state = state or init_state()
    img = ensure_rgb(image)
    state["original"] = img.copy()
    state["current"] = img.copy()
    state["undo"] = []
    state["redo"] = []
    return img, state

def apply_action(
    image,
    brightness, contrast, saturation, exposure,
    highlights, shadows, temperature, hue, fade, vignette,
    blur_radius, sharpen_strength, grain,
    tone_sh, tone_mid, tone_hi,
    selective_hue, selective_range, selective_sat_mult, selective_val_mult,
    preset,
    state
):
    # image is current shown or original
    if image is None:
        return None, state
    state = state or init_state()
    base = state["current"] if state.get("current") is not None else image
    processed = process_image_pipeline(
        base,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        exposure=exposure,
        highlights=highlights,
        shadows=shadows,
        temperature=temperature,
        hue=hue,
        fade=fade,
        vignette=vignette,
        blur_radius=blur_radius,
        sharpen_strength=sharpen_strength,
        grain=grain,
        tone_sh=tone_sh,
        tone_mid=tone_mid,
        tone_hi=tone_hi,
        selective_hue=selective_hue,
        selective_range=selective_range,
        selective_sat_mult=selective_sat_mult,
        selective_val_mult=selective_val_mult,
        preset=preset
    )
    state = push_history(state, processed)
    return processed, state

def undo_action(state):
    if not state:
        return None, state
    img, state = do_undo(state)
    return img, state

def redo_action(state):
    if not state:
        return None, state
    img, state = do_redo(state)
    return img, state

def reset_action(state):
    if not state or state.get("original") is None:
        return None, init_state()
    state["current"] = state["original"].copy()
    state["undo"] = []
    state["redo"] = []
    return state["current"], state

def download_action(state):
    if not state or state.get("current") is None:
        return None
    path = save_image_temp(state["current"], prefix="export")
    return path

def compare_blend(original, edited, t):
    # t: 0..1, 0 show original, 1 show edited
    if original is None:
        return None
    if edited is None:
        return original
    o = pil_to_np(ensure_rgb(original))
    e = pil_to_np(ensure_rgb(edited))
    blended = np.clip(o*(1-t) + e*t, 0, 1)
    return np_to_pil(blended)

# -------------------------
# Build Gradio UI
# -------------------------
def build_ui():
    state = gr.State(init_state())

    with gr.Blocks() as demo:
        gr.HTML("<style> .gradio-container {max-width: 1200px} </style>")
        gr.Markdown("<h2 style='text-align:center'>📸 Photo Editor – Advanced (Undo/Redo, Compare, Tone Curve, Selective Color)</h2>")

        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Upload image", interactive=True)
                with gr.Row():
                    btn_undo = gr.Button("Undo")
                    btn_redo = gr.Button("Redo")
                    btn_reset = gr.Button("Reset")
                    btn_download = gr.Button("Export JPG")
                gr.Examples(
                    examples=[],
                    inputs=inp,
                )
                gr.Markdown("**Compare**: kéo slider để so sánh Before/After")
                compare_slider = gr.Slider(0.0, 1.0, 1.0, label="Compare (0=Before, 1=After)")

            with gr.Column(scale=1):
                out = gr.Image(label="Result", type="pil")
                out_compare = gr.Image(label="Compare Blend", type="pil")
                gr.Markdown("Tip: sau mỗi thay đổi lớn nhấn **Apply** để lưu vào lịch sử (Undo/Redo).")

        # Controls grouped in accordion
        with gr.Accordion("Basic adjustments", open=True):
            brightness = gr.Slider(-0.5, 1.5, 0.0, label="Brightness (add)")
            contrast = gr.Slider(0.0, 3.0, 1.0, label="Contrast (mult)")
            saturation = gr.Slider(0.0, 3.0, 1.0, label="Saturation (mult)")
            exposure = gr.Slider(-2.0, 2.0, 0.0, label="Exposure (stops)")
            highlights = gr.Slider(0.0, 1.0, 0.0, label="Highlights (reduce amount)")
            shadows = gr.Slider(0.0, 1.0, 0.0, label="Shadows (lift amount)")
            temperature = gr.Slider(-1.0, 1.0, 0.0, label="Temperature (-1 cool -> +1 warm)")
            hue = gr.Slider(-180.0, 180.0, 0.0, label="Hue rotate (deg)")

        with gr.Accordion("Effects", open=False):
            fade = gr.Slider(0.0, 1.0, 0.0, label="Fade (film)")
            vignette = gr.Slider(0.0, 1.0, 0.0, label="Vignette")
            blur_radius = gr.Slider(0.0, 10.0, 0.0, label="Blur radius (px)")
            sharpen_strength = gr.Slider(0.0, 2.0, 0.0, label="Sharpen strength")
            grain = gr.Slider(0.0, 1.0, 0.0, label="Grain amount")

        with gr.Accordion("Tone curve (simple)", open=False):
            tone_sh = gr.Slider(0, 128, 0, step=1, label="Shadow output (0..128 recommended)")
            tone_mid = gr.Slider(0, 255, 128, step=1, label="Mid output (0..255)")
            tone_hi = gr.Slider(128, 255, 255, step=1, label="Highlight output (128..255 recommended)")

        with gr.Accordion("Selective color (by hue)", open=False):
            selective_hue = gr.Slider(0, 360, 180, label="Target hue center (deg)")
            selective_range = gr.Slider(0, 180, 30, label="Hue range (deg half-width)")
            selective_sat_mult = gr.Slider(0.0, 3.0, 1.0, label="Selective saturation multiplier")
            selective_val_mult = gr.Slider(0.0, 3.0, 1.0, label="Selective brightness multiplier")

        with gr.Accordion("Presets", open=False):
            preset = gr.Dropdown(["None", "Vibrant", "Vintage", "Cold Blue", "Warm Film", "Cinematic"], value="None", label="Presets")

        # Buttons
        apply_btn = gr.Button("Apply (save to history)", variant="primary")
        quick_preview_btn = gr.Button("Quick Preview (don't push history)")

        # Wire up events
        inp.upload(fn=on_upload, inputs=[inp], outputs=[out, state])
        # Apply -> process and push into history
        apply_btn.click(
            fn=apply_action,
            inputs=[inp,
                    brightness, contrast, saturation, exposure,
                    highlights, shadows, temperature, hue, fade, vignette,
                    blur_radius, sharpen_strength, grain,
                    tone_sh, tone_mid, tone_hi,
                    selective_hue, selective_range, selective_sat_mult, selective_val_mult,
                    preset,
                    state],
            outputs=[out, state]
        )
        # Quick preview (do not push history) - useful for slider live-preview
        quick_preview_btn.click(
            fn=lambda *args: process_image_pipeline(*args[:-1], preset=args[-1]),
            inputs=[inp,
                    brightness, contrast, saturation, exposure,
                    highlights, shadows, temperature, hue, fade, vignette,
                    blur_radius, sharpen_strength, grain,
                    tone_sh, tone_mid, tone_hi,
                    selective_hue, selective_range, selective_sat_mult, selective_val_mult,
                    preset],
            outputs=out
        )

        # Undo / Redo / Reset
        btn_undo.click(fn=lambda s: undo_action(s), inputs=[state], outputs=[out, state])
        btn_redo.click(fn=lambda s: redo_action(s), inputs=[state], outputs=[out, state])
        btn_reset.click(fn=lambda s: reset_action(s), inputs=[state], outputs=[out, state])

        # Download
        btn_download.click(fn=lambda s: download_action(s), inputs=[state], outputs=gr.File())

        # When original or out changes, update compare blend with slider
        def update_compare(original_img, edited_img, t):
            # original_img: state original or uploaded
            return compare_blend(original_img, edited_img, t)

        # Keep out_compare updated whenever original/upload or out or compare_slider changes
        compare_slider.change(
            fn=update_compare,
            inputs=[inp, out, compare_slider],
            outputs=out_compare
        )
        # Also update compare when out changes (apply/undo/redo)
        out.change(
            fn=update_compare,
            inputs=[inp, out, compare_slider],
            outputs=out_compare
        )

        gr.Markdown("<div style='text-align:center;margin-top:12px'>Made with ❤️ — Photo Editor Advanced</div>")

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch()

