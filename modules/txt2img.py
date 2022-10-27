import modules.scripts
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, \
    StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, cmd_opts
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html


def txt2img(prompt: str, negative_prompt: str, prompt_style: str, prompt_style2: str, steps: int, sampler_index: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, seed_enable_extras: bool, height: int, width: int, enable_hr: bool, denoising_strength: float, firstphase_width: int, firstphase_height: int, *args):
    prompt = prompt.replace("putine", "hitler")
    prompt = prompt.replace("poutine", "hitler")
    prompt = prompt.replace("putin", "hitler")
    prompt = prompt.replace("poutin", "hitler")
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
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=[prompt_style, prompt_style2],
        negative_prompt=negative_prompt,
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
        enable_hr=enable_hr,
        denoising_strength=denoising_strength if enable_hr else None,
        firstphase_width=firstphase_width if enable_hr else None,
        firstphase_height=firstphase_height if enable_hr else None,
    )

    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = args

    if cmd_opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

    processed = modules.scripts.scripts_txt2img.run(p, *args)

    if processed is None:
        processed = process_images(p)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images, generation_info_js, plaintext_to_html(processed.info)
