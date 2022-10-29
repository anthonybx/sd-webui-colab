import math
import os
import sys
import traceback

import numpy as np
from PIL import Image, ImageOps, ImageChops

from modules import devices
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, state
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html
import modules.images as images
import modules.scripts


def process_batch(p, input_dir, output_dir, args):
    processing.fix_seed(p)

    images = [file for file in [os.path.join(input_dir, x) for x in os.listdir(input_dir)] if os.path.isfile(file)]

    print(f"Will process {len(images)} images, creating {p.n_iter * p.batch_size} new images for each.")

    save_normally = output_dir == ''

    p.do_not_save_grid = True
    p.do_not_save_samples = not save_normally

    state.job_count = len(images) * p.n_iter

    for i, image in enumerate(images):
        state.job = f"{i+1} out of {len(images)}"
        if state.skipped:
            state.skipped = False

        if state.interrupted:
            break

        img = Image.open(image)
        p.init_images = [img] * p.batch_size

        proc = modules.scripts.scripts_img2img.run(p, *args)
        if proc is None:
            proc = process_images(p)

        for n, processed_image in enumerate(proc.images):
            filename = os.path.basename(image)

            if n > 0:
                left, right = os.path.splitext(filename)
                filename = f"{left}-{n}{right}"

            if not save_normally:
                processed_image.save(os.path.join(output_dir, filename))


def img2img(mode: int, prompt: str, negative_prompt: str, prompt_style: str, prompt_style2: str, init_img, init_img_with_mask, init_img_inpaint, init_mask_inpaint, mask_mode, steps: int, sampler_index: int, mask_blur: int, inpainting_fill: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, denoising_strength: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, seed_enable_extras: bool, height: int, width: int, resize_mode: int, inpaint_full_res: bool, inpaint_full_res_padding: int, inpainting_mask_invert: int, img2img_batch_input_dir: str, img2img_batch_output_dir: str, *args):
    is_inpaint = mode == 1
    is_batch = mode == 2

    if is_inpaint:
        if mask_mode == 0:
            image = init_img_with_mask['image']
            mask = init_img_with_mask['mask']
            alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
            mask = ImageChops.lighter(alpha_mask, mask.convert('L')).convert('L')
            image = image.convert('RGB')
        else:
            image = init_img_inpaint
            mask = init_mask_inpaint
    else:
        image = init_img
        mask = None

    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'
    prompt = prompt.lower()    
    prompt = prompt.replace("ramzan kadyrov", "man")
    prompt = prompt.replace("ramzan", "man")
    prompt = prompt.replace("kadyrov", "man")
    prompt = prompt.replace("missile", "flower")
    prompt = prompt.replace("nuclear", "flower")
    prompt = prompt.replace("bomb", "flower")
    prompt = prompt.replace("nuke", "flower")
    prompt = prompt.replace("war", "love")
    prompt = prompt.replace("путин", "dictator")
    prompt = prompt.replace("putine", "dictator")
    prompt = prompt.replace("poutine", "dictator")
    prompt = prompt.replace("putin", "dictator")
    prompt = prompt.replace("poutin", "dictator")
    prompt = prompt.replace("little girl", "woman")
    prompt = prompt.replace("little boy", "man")
    prompt = prompt.replace("5 year old", "adult")
    prompt = prompt.replace("6 year old", "adult")
    prompt = prompt.replace("7 year old", "adult")
    prompt = prompt.replace("8 year old", "adult")
    prompt = prompt.replace("9 year old", "adult")
    prompt = prompt.replace("10 year old", "adult")
    prompt = prompt.replace("11 year old", "adult")
    prompt = prompt.replace("12 year old", "adult")
    prompt = prompt.replace("13 year old", "adult")
    prompt = prompt.replace("14 year old", "adult")
    prompt = prompt.replace("15 year old", "adult")
    prompt = prompt.replace("16 year old", "adult")
    prompt = prompt.replace("17 year old", "adult")
    if ("year" in prompt) and ("old" in prompt) and ("boy" in prompt):
      steps = 1
    if ("year" in prompt) and ("old" in prompt) and ("girl" in prompt):
      steps = 1
    if ("young" in prompt) and ("girl" in prompt):
      prompt = "a pig"
      steps = 1
    if ("young" in prompt) and ("boy" in prompt):
      prompt = "a pig"
      steps = 1
    if ("little" in prompt) and ("girl" in prompt):
      prompt = "a pig"
      steps = 1
    if ("little" in prompt) and ("boy" in prompt):
      prompt = "a pig"
      steps = 1
    if ("young" in prompt) and ("loli" in prompt):
      prompt = "a pig"
      steps = 1
    if ("nude" in prompt) and ("young" in prompt):
      prompt = "a pig"
      steps = 1
    if ("naked" in prompt) and ("young" in prompt):
      prompt = "a pig"
      steps = 1
    if ("slutty" in prompt) and ("young" in prompt):
      prompt = "a pig"
      steps = 1
    if ("chubby" in prompt):
      steps = 1
      prompt = "a pig"
    if ("fat" in prompt) and ("woman" in prompt):
      steps = 1
    if ("fat" in prompt) and ("girl" in prompt):
      steps = 1
    if ("obese" in prompt):
      steps = 1
    if ("loli" in prompt):
      prompt = "a pig"
      steps = 1
    if ("plus-size" in prompt):
      prompt = "a pig"
      steps = 1
    if ("dragoness" in prompt):
      prompt = "a pig"
      steps = 1
    if ("bbw" in prompt):
      prompt = "a pig"
      steps = 1
    if ("chibi" in prompt) and ("girl" in prompt):
      prompt = "a pig"
      steps = 1
    if ("fat" in prompt) and ("overweight" in prompt):
      prompt = "a pig"
      steps = 1
    if ("teen" in prompt) and ("boy" in prompt):
      prompt = "a pig"
      steps = 1
    if ("teen" in prompt) and ("girl" in prompt):
      prompt = "a pig"
      steps = 1
    if ("bodybuilder" in prompt) and ("female" in prompt):
      prompt = "a man imprisoned"
      steps = 1
    if ("bodybuilder" in prompt) and ("girl" in prompt):
      prompt = "a man imprisoned"
      steps = 1
    if ("bodybuilder" in prompt) and ("feminine" in prompt):
      prompt = "a man imprisoned"
      steps = 1
    if ("bodybuilder" in prompt) and ("woman" in prompt):
      prompt = "a stupid guy imprisoned in jail,he is sad because his life is trash"
    prompt = prompt.replace("overweight", "beautiful")
    prompt = prompt.replace("sergei kuzhugetovich shoigu", "man")
    if ("sergei" in prompt) and ("shoigu" in prompt):
      prompt = "slavia ukraine"
    if ("kuzhugetovich" in prompt):
      prompt = "slavia ukraine"
    if ("boy" in prompt) and ("cute" in prompt):
      prompt = "a man imprisoned in jail"
      steps = 1
    if ("boy" in prompt) and ("handsome" in prompt):
      prompt = "a man imprisoned in jail"
      steps = 1
    if ("shemale" in prompt) or ("hermaphrodite" in prompt):
      prompt = "a man imprisoned in jail"
      steps = 1
    if ("muscular" in prompt) and ("girl" in prompt):
      prompt = "a man imprisoned in jail"
      steps = 1
    if ("muscular" in prompt) and ("woman" in prompt):
      prompt = "a man imprisoned in jail"
      steps = 1
    if ("bodybuild" in prompt) and ("girl" in prompt):
      prompt = "a man imprisoned in jail"
      steps = 1
    if ("bodybuild" in prompt) and ("woman" in prompt):
      prompt = "a man imprisoned in jail"
      steps = 1
    if ("strong" in prompt) and ("girl" in prompt):
      prompt = "a man imprisoned in jail"
      steps = 1
    if ("strong" in prompt) and ("woman" in prompt):
      prompt = "a man imprisoned in jail"
      steps = 1
    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=[prompt_style, prompt_style2],
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        seed_enable_extras=seed_enable_extras,
        sampler_index=sampler_index,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=restore_faces,
        tiling=tiling,
        init_images=[image],
        mask=mask,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        inpaint_full_res=inpaint_full_res,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
    )

    if shared.cmd_opts.enable_console_prompts:
        print(f"\nimg2img: {prompt}", file=shared.progress_print_out)

    p.extra_generation_params["Mask blur"] = mask_blur

    if is_batch:
        assert not shared.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"

        process_batch(p, img2img_batch_input_dir, img2img_batch_output_dir, args)

        processed = Processed(p, [], p.seed, "")
    else:
        processed = modules.scripts.scripts_img2img.run(p, *args)
        if processed is None:
            processed = process_images(p)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images, generation_info_js, plaintext_to_html(processed.info)
