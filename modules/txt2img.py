import modules.scripts
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, \
    StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, cmd_opts
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html

def txt2img(prompt: str, negative_prompt: str, prompt_style: str, prompt_style2: str, steps: int, sampler_index: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, seed_enable_extras: bool, height: int, width: int, enable_hr: bool, denoising_strength: float, firstphase_width: int, firstphase_height: int, *args):
    prompt = prompt.lower()    
    if ("ramzan kadyrov" in prompt) or ("ramzan" in prompt) or ("kadyrov" in prompt):
      steps = 1
    if ("missile" in prompt) or ("nuclear" in prompt) or ("bomb" in prompt) or ("nuke" in prompt):
      steps = 1
    if ("путин" in prompt) or ("putine" in prompt) or ("poutine" in prompt) or ("putin" in prompt) or ("poutin" in prompt):
      steps = 1
    if ("little girl" in prompt) or ("little boy" in prompt) or ("poutine" in prompt) or ("poutin" in prompt):
      steps = 1
    if ("year" in prompt) and ("old" in prompt) and ("boy" in prompt):
      steps = 1
    if ("year" in prompt) and ("old" in prompt) and ("girl" in prompt):
      steps = 1
    if ("young" in prompt) and ("girl" in prompt):
      steps = 1
    if ("young" in prompt) and ("boy" in prompt):
      steps = 1
    if ("little" in prompt) and ("girl" in prompt):
      steps = 1
    if ("little" in prompt) and ("boy" in prompt):
      steps = 1
    if ("young" in prompt) and ("loli" in prompt):
      steps = 1
    if ("nude" in prompt) and ("young" in prompt):
      steps = 1
    if ("naked" in prompt) and ("young" in prompt):
      steps = 1
    if ("slutty" in prompt) and ("young" in prompt):
      steps = 1
    if ("chubby" in prompt):
      prompt = "a pig"
    if ("fat" in prompt) and ("woman" in prompt):
      steps = 1
    if ("fat" in prompt) and ("girl" in prompt):
      steps = 1
    if ("obese" in prompt):
      steps = 1
    if ("loli" in prompt):
      steps = 1
    if ("plus-size" in prompt):
      steps = 1
    if ("bbw" in prompt):
      steps = 1
    if ("chibi" in prompt) and ("girl" in prompt):
      steps = 1
    if ("fat" in prompt) and ("overweight" in prompt):
      steps = 1
    if ("teen" in prompt) and ("boy" in prompt):
      steps = 1
    if ("teen" in prompt) and ("girl" in prompt):
      steps = 1
    if ("bodybuilder" in prompt) and ("female" in prompt):
      steps = 1
    if ("bodybuilder" in prompt) and ("girl" in prompt):
      steps = 1
    if ("bodybuilder" in prompt) and ("feminine" in prompt):
      steps = 1
    if ("bodybuilder" in prompt) and ("woman" in prompt):
      steps = 1
    if ("sergei kuzhugetovich shoigu" in prompt):
      steps = 1
    if ("sergei" in prompt) and ("shoigu" in prompt):
      steps = 1
    if ("kuzhugetovich" in prompt):
      steps = 1
    if ("boy" in prompt) and ("cute" in prompt):
      steps = 1
    if ("boy" in prompt) and ("handsome" in prompt):
      steps = 1
    if ("shemale" in prompt) or ("hermaphrodite" in prompt):
      steps = 1
    if ("muscular" in prompt) and ("girl" in prompt):
      steps = 1
    if ("muscular" in prompt) and ("woman" in prompt):
      steps = 1
    if ("bodybuild" in prompt) and ("girl" in prompt):
      steps = 1
    if ("bodybuild" in prompt) and ("woman" in prompt):
      steps = 1
    if ("strong" in prompt) and ("girl" in prompt):
      steps = 1
    if ("strong" in prompt) and ("woman" in prompt):
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
