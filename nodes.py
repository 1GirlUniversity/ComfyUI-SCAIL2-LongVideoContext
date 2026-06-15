"""SCAIL-2 Long Video Context nodes for ComfyUI.

This module exposes two model patch nodes:
- SCAIL2SmartLongVideoContext
- SCAIL2LongVideoContext

They rely on ComfyUI's built-in comfy.context_windows support.
"""

__version__ = "0.1.0"

import logging


def _to_4n_plus_1(n: int) -> int:
    n = int(n)
    if n < 1:
        return 1
    return n - ((n - 1) % 4)


def _pixel_frames_to_latent_frames(frames: int) -> int:
    frames = int(frames)
    return max(((frames - 1) // 4) + 1, 1)


def _pick_fuse_default(cw):
    fuse_options = list(cw.ContextFuseMethods.LIST_STATIC)

    if "pyramid" in fuse_options:
        return "pyramid"

    if "overlap-linear" in fuse_options:
        return "overlap-linear"

    if "overlap" in fuse_options:
        return "overlap"

    if hasattr(cw.ContextFuseMethods, "PYRAMID"):
        return cw.ContextFuseMethods.PYRAMID

    return fuse_options[0]


def _clamp_pair(total_frames: int, segment_len: int, segment_overlap: int):
    total_frames = _to_4n_plus_1(total_frames)
    segment_len = _to_4n_plus_1(segment_len)
    segment_len = max(1, min(segment_len, total_frames))
    segment_overlap = max(0, int(segment_overlap))

    if segment_overlap >= segment_len:
        segment_overlap = max(segment_len - 4, 0)

    return segment_len, segment_overlap


def _auto_pick(total_frames: int, strategy: str):
    """
    Escolha automática baseada no length real do vídeo carregado.

    auto_smooth_rtx6000:
    Prioriza continuidade e suavização de costura em RTX PRO 6000 96GB.
    Usa janelas maiores e overlaps grandes em vídeo de 15s+.

    auto_safe:
    Menos pesado; útil se auto_smooth der OOM ou ficar lento demais.

    force_official_81_5:
    Replica o padrão oficial segment_len=81 / segment_overlap=5.
    """

    total_frames = _to_4n_plus_1(total_frames)

    if strategy == "force_official_81_5":
        if total_frames <= 81:
            return total_frames, 0, "official_81_5_single_window"
        return 81, 5, "official_81_5"

    if strategy == "auto_safe":
        if total_frames <= 161:
            return total_frames, 0, "safe_single_pass"
        if total_frames <= 321:
            return 121, 17, "safe_121_17"
        if total_frames <= 481:
            return 121, 25, "safe_121_25"
        return 81, 13, "safe_81_13"

    if strategy == "auto_quality_rtx6000":
        if total_frames <= 161:
            return total_frames, 0, "quality_single_pass_161_or_less"
        if total_frames <= 241:
            return total_frames, 0, "quality_single_pass_241_or_less"
        if total_frames <= 481:
            return 161, 25, "quality_161_25"
        if total_frames <= 721:
            return 161, 33, "quality_161_33"
        return 121, 25, "quality_very_long_121_25"

    # auto_smooth_rtx6000
    # Heurística voltada para diminuir transição visível entre janelas.
    # Para vídeo ~15s em force_rate=16, length tende a ~237,
    # então cai em 205/73.
    if total_frames <= 161:
        return total_frames, 0, "smooth_single_pass_161_or_less"

    if total_frames <= 241:
        return 205, 73, "smooth_205_73"

    if total_frames <= 321:
        return 241, 81, "smooth_241_81"

    if total_frames <= 481:
        return 241, 81, "smooth_241_81_medium"

    if total_frames <= 721:
        return 205, 73, "smooth_205_73_long"

    return 161, 65, "smooth_very_long_161_65"


def _apply_context(
    model,
    segment_len,
    segment_overlap,
    context_schedule,
    context_stride,
    fuse_method,
    freenoise,
    causal_window_fix,
    cond_retain_index_list,
):
    import comfy.context_windows as cw

    segment_len = _to_4n_plus_1(segment_len)
    segment_overlap = int(segment_overlap)

    if segment_len < 1:
        segment_len = 1

    if segment_overlap < 0:
        segment_overlap = 0

    if segment_overlap >= segment_len:
        segment_overlap = max(segment_len - 4, 0)

    latent_context_length = _pixel_frames_to_latent_frames(segment_len)

    if segment_overlap <= 0:
        latent_context_overlap = 0
    else:
        latent_context_overlap = _pixel_frames_to_latent_frames(segment_overlap)

    if latent_context_overlap >= latent_context_length:
        latent_context_overlap = max(latent_context_length - 1, 0)

    patched_model = model.clone()

    patched_model.model_options["context_handler"] = cw.IndexListContextHandler(
        context_schedule=cw.get_matching_context_schedule(context_schedule),
        fuse_method=cw.get_matching_fuse_method(fuse_method),
        context_length=latent_context_length,
        context_overlap=latent_context_overlap,
        context_stride=int(context_stride),
        closed_loop=False,
        dim=2,
        freenoise=bool(freenoise),

        # Essencial para SCAIL-2:
        # mantém o índice 0 da condição original em cada janela.
        cond_retain_index_list=str(cond_retain_index_list),

        split_conds_to_windows=False,
        causal_window_fix=bool(causal_window_fix),
    )

    cw.create_prepare_sampling_wrapper(patched_model)

    if freenoise:
        cw.create_sampler_sample_wrapper(patched_model)

    return patched_model, latent_context_length, latent_context_overlap


class SCAIL2SmartLongVideoContext:
    """
    SCAIL-2 Smart Smooth Long Video Context.

    Recebe total_frames do workflow e escolhe automaticamente:
    - segment_len
    - segment_overlap

    Use no workflow:
    VHS_VideoInfoLoaded frame_count
    -> ComfyMathExpression: a - ((a - 1) % 4)
    -> WanSCAILToVideo length
    -> este node total_frames

    Default:
    - strategy = auto_smooth_rtx6000
    """

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.context_windows as cw

        fuse_options = list(cw.ContextFuseMethods.LIST_STATIC)
        default_fuse = _pick_fuse_default(cw)

        return {
            "required": {
                "model": ("MODEL",),

                "total_frames": (
                    "INT",
                    {
                        "default": 237,
                        "min": 1,
                        "max": 99999,
                        "step": 4,
                    },
                ),

                "strategy": (
                    [
                        "auto_smooth_rtx6000",
                        "auto_safe",
                        "auto_quality_rtx6000",
                        "force_official_81_5",
                        "manual",
                    ],
                    {
                        "default": "auto_smooth_rtx6000",
                    },
                ),

                "manual_segment_len": (
                    "INT",
                    {
                        "default": 205,
                        "min": 1,
                        "max": 99999,
                        "step": 4,
                    },
                ),

                "manual_segment_overlap": (
                    "INT",
                    {
                        "default": 73,
                        "min": 0,
                        "max": 99999,
                        "step": 4,
                    },
                ),

                "context_schedule": (
                    [
                        cw.ContextSchedules.STATIC_STANDARD,
                        cw.ContextSchedules.UNIFORM_STANDARD,
                        cw.ContextSchedules.UNIFORM_LOOPED,
                        cw.ContextSchedules.BATCHED,
                    ],
                    {
                        "default": cw.ContextSchedules.STATIC_STANDARD,
                    },
                ),

                "context_stride": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 32,
                        "step": 1,
                    },
                ),

                "fuse_method": (
                    fuse_options,
                    {
                        "default": default_fuse,
                    },
                ),

                "freenoise": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),

                "causal_window_fix": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),

                "bypass_when_single_pass": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),

                "cond_retain_index_list": (
                    "STRING",
                    {
                        "default": "0",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "model",
        "chosen_segment_len",
        "chosen_segment_overlap",
        "latent_context_length",
        "latent_context_overlap",
        "decision",
    )
    FUNCTION = "apply"
    CATEGORY = "model_patches/scail2"

    def apply(
        self,
        model,
        total_frames,
        strategy,
        manual_segment_len,
        manual_segment_overlap,
        context_schedule,
        context_stride,
        fuse_method,
        freenoise,
        causal_window_fix,
        bypass_when_single_pass,
        cond_retain_index_list,
    ):
        total_frames = _to_4n_plus_1(total_frames)

        if strategy == "manual":
            segment_len, segment_overlap = _clamp_pair(
                total_frames,
                manual_segment_len,
                manual_segment_overlap,
            )
            decision = "manual"
        else:
            segment_len, segment_overlap, decision = _auto_pick(total_frames, strategy)
            segment_len, segment_overlap = _clamp_pair(
                total_frames,
                segment_len,
                segment_overlap,
            )

        latent_context_length = _pixel_frames_to_latent_frames(segment_len)
        latent_context_overlap = (
            0 if segment_overlap <= 0 else _pixel_frames_to_latent_frames(segment_overlap)
        )

        if latent_context_overlap >= latent_context_length:
            latent_context_overlap = max(latent_context_length - 1, 0)

        # Para vídeo curto demais, deixa single-pass real.
        # Para 15s em 16fps (~237 frames), NÃO cai aqui.
        if bypass_when_single_pass and segment_len >= total_frames and segment_overlap == 0:
            msg = (
                f"BYPASS single-pass: total_frames={total_frames}, "
                f"segment_len={segment_len}, segment_overlap=0. "
                f"Nenhum context window aplicado."
            )
            logging.info("SCAIL-2 Smart Smooth Long Context: " + msg)
            return (
                model,
                int(segment_len),
                int(segment_overlap),
                int(latent_context_length),
                int(latent_context_overlap),
                msg,
            )

        patched_model, latent_context_length, latent_context_overlap = _apply_context(
            model=model,
            segment_len=segment_len,
            segment_overlap=segment_overlap,
            context_schedule=context_schedule,
            context_stride=context_stride,
            fuse_method=fuse_method,
            freenoise=freenoise,
            causal_window_fix=causal_window_fix,
            cond_retain_index_list=cond_retain_index_list,
        )

        msg = (
            f"{decision}: total_frames={total_frames}, "
            f"segment_len={segment_len}, segment_overlap={segment_overlap}, "
            f"latent_context_length={latent_context_length}, "
            f"latent_context_overlap={latent_context_overlap}, "
            f"cond_retain_index_list={cond_retain_index_list}, "
            f"schedule={context_schedule}, fuse={fuse_method}"
        )

        logging.info("SCAIL-2 Smart Smooth Long Context aplicado: " + msg)

        return (
            patched_model,
            int(segment_len),
            int(segment_overlap),
            int(latent_context_length),
            int(latent_context_overlap),
            msg,
        )


class SCAIL2LongVideoContext:
    """
    Node manual mantido por compatibilidade.

    Use se quiser informar manualmente:
    - segment_len
    - segment_overlap
    """

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.context_windows as cw

        fuse_options = list(cw.ContextFuseMethods.LIST_STATIC)
        default_fuse = _pick_fuse_default(cw)

        return {
            "required": {
                "model": ("MODEL",),

                "segment_len": (
                    "INT",
                    {
                        "default": 205,
                        "min": 1,
                        "max": 99999,
                        "step": 4,
                    },
                ),

                "segment_overlap": (
                    "INT",
                    {
                        "default": 73,
                        "min": 0,
                        "max": 99999,
                        "step": 4,
                    },
                ),

                "context_schedule": (
                    [
                        cw.ContextSchedules.STATIC_STANDARD,
                        cw.ContextSchedules.UNIFORM_STANDARD,
                        cw.ContextSchedules.UNIFORM_LOOPED,
                        cw.ContextSchedules.BATCHED,
                    ],
                    {
                        "default": cw.ContextSchedules.STATIC_STANDARD,
                    },
                ),

                "context_stride": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 32,
                        "step": 1,
                    },
                ),

                "fuse_method": (
                    fuse_options,
                    {
                        "default": default_fuse,
                    },
                ),

                "freenoise": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),

                "causal_window_fix": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),

                "cond_retain_index_list": (
                    "STRING",
                    {
                        "default": "0",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "model_patches/scail2"

    def apply(
        self,
        model,
        segment_len,
        segment_overlap,
        context_schedule,
        context_stride,
        fuse_method,
        freenoise,
        causal_window_fix,
        cond_retain_index_list,
    ):
        patched_model, latent_context_length, latent_context_overlap = _apply_context(
            model=model,
            segment_len=segment_len,
            segment_overlap=segment_overlap,
            context_schedule=context_schedule,
            context_stride=context_stride,
            fuse_method=fuse_method,
            freenoise=freenoise,
            causal_window_fix=causal_window_fix,
            cond_retain_index_list=cond_retain_index_list,
        )

        logging.info(
            "SCAIL-2 Long Video Context aplicado: "
            f"segment_len={segment_len}, segment_overlap={segment_overlap}, "
            f"latent_context_length={latent_context_length}, "
            f"latent_context_overlap={latent_context_overlap}, "
            f"cond_retain_index_list={cond_retain_index_list}"
        )

        return (patched_model,)


NODE_CLASS_MAPPINGS = {
    "SCAIL2SmartLongVideoContext": SCAIL2SmartLongVideoContext,
    "SCAIL2LongVideoContext": SCAIL2LongVideoContext,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAIL2SmartLongVideoContext": "SCAIL-2 Smart Long Video Context",
    "SCAIL2LongVideoContext": "SCAIL-2 Long Video Context",
}