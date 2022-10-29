import modules.scripts
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, cmd_opts
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html


def txt2img(prompt: str, negative_prompt: str, prompt_style: str, prompt_style2: str, steps: int, sampler_index: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, seed_enable_extras: bool, height: int, width: int, enable_hr: bool, scale_latent: bool, denoising_strength: float, *args):
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
        scale_latent=scale_latent if enable_hr else None,
        denoising_strength=denoising_strength if enable_hr else None,
    )

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
