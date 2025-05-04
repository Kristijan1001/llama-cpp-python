from __future__ import annotations

import os
from ctypes import (
    c_bool,
    c_char_p,
    c_int,
    c_uint,
    c_int32,
    c_uint32,
    c_uint8,
    c_float,
    c_void_p,
    POINTER,
    _Pointer,  # type: ignore
    Structure,
)
import pathlib
from typing import (
    Union,
    NewType,
    Optional,
    TYPE_CHECKING,
)

import llama_cpp.llama_cpp as llama_cpp

from llama_cpp._ctypes_extensions import (
    load_shared_library,
    ctypes_function_for_shared_library,
)

if TYPE_CHECKING:
    from llama_cpp._ctypes_extensions import (
        CtypesArray,
    )


# Specify the base name of the shared library to load
_libllava_base_name = "llava"
_libllava_override_path = os.environ.get("LLAVA_CPP_LIB")
_libllava_base_path = pathlib.Path(os.path.abspath(os.path.dirname(__file__))) / "lib" if _libllava_override_path is None else pathlib.Path()

# Load the library
_libllava = load_shared_library(_libllava_base_name, _libllava_base_path)

ctypes_function = ctypes_function_for_shared_library(_libllava)


################################################
# llava.h
################################################

# struct clip_ctx;
clip_ctx_p = NewType("clip_ctx_p", int)
clip_ctx_p_ctypes = c_void_p

# struct clip_image_u8;
clip_image_u8_p = NewType("clip_image_u8_p", int)
clip_image_u8_p_ctypes = c_void_p


# struct llava_image_embed {
#     float * embed;
#     int n_image_pos;
# };
class llava_image_embed(Structure):
    _fields_ = [
        ("embed", POINTER(c_float)),
        ("n_image_pos", c_int),
    ]


# /** sanity check for clip <-> llava embed size match */
# LLAVA_API bool llava_validate_embed_size(const llama_context * ctx_llama, const clip_ctx * ctx_clip);
@ctypes_function(
    "llava_validate_embed_size",
    [llama_cpp.llama_context_p_ctypes, clip_ctx_p_ctypes],
    c_bool,
)
def llava_validate_embed_size(
    ctx_llama: llama_cpp.llama_context_p, ctx_clip: clip_ctx_p, /
) -> bool:
    ...

# LLAVA_API bool llava_image_embed_make_with_clip_img(struct clip_ctx * ctx_clip, int n_threads, const struct clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out);
@ctypes_function(
    "llava_image_embed_make_with_clip_img",
    [clip_ctx_p_ctypes, c_int, clip_image_u8_p_ctypes, POINTER(POINTER(c_float)), POINTER(c_int)],
    c_bool
)
def llava_image_embed_make_with_clip_img(
    ctx_clip: clip_ctx_p,
    n_threads: int,
    img: clip_image_u8_p,
    image_embd_out: _Pointer[_Pointer[c_float]],
    n_img_pos_out: _Pointer[c_int],
    /,
) -> bool:
    ...


# /** build an image embed from image file bytes */
# LLAVA_API struct llava_image_embed * llava_image_embed_make_with_bytes(struct clip_ctx * ctx_clip, int n_threads, const unsigned char * image_bytes, int image_bytes_length);
@ctypes_function(
    "llava_image_embed_make_with_bytes",
    [clip_ctx_p_ctypes, c_int, c_char_p, c_int],
    POINTER(llava_image_embed),
)
def llava_image_embed_make_with_bytes(
    ctx_clip: clip_ctx_p,
    n_threads: c_int,
    image_bytes: c_char_p,
    image_bytes_length: c_int,
    /,
) -> "_Pointer[llava_image_embed]":
    ...


# /** build an image embed from a path to an image filename */
# LLAVA_API struct llava_image_embed * llava_image_embed_make_with_filename(struct clip_ctx * ctx_clip, int n_threads, const char * image_path);
@ctypes_function(
    "llava_image_embed_make_with_filename",
    [clip_ctx_p_ctypes, c_int, c_char_p],
    POINTER(llava_image_embed),
)
def llava_image_embed_make_with_filename(
    ctx_clip: clip_ctx_p, n_threads: c_int, image_path: c_char_p, /
) -> "_Pointer[llava_image_embed]":
    ...


# LLAVA_API void llava_image_embed_free(struct llava_image_embed * embed);
# /** free an embedding made with llava_image_embed_make_* */
@ctypes_function("llava_image_embed_free", [POINTER(llava_image_embed)], None)
def llava_image_embed_free(embed: "_Pointer[llava_image_embed]", /):
    ...


# /** write the image represented by embed into the llama context with batch size n_batch, starting at context pos n_past. on completion, n_past points to the next position in the context after the image embed. */
# LLAVA_API bool llava_eval_image_embed(struct llama_context * ctx_llama, const struct llava_image_embed * embed, int n_batch, int * n_past);
@ctypes_function(
    "llava_eval_image_embed",
    [
        llama_cpp.llama_context_p_ctypes,
        POINTER(llava_image_embed),
        c_int,
        POINTER(c_int),
    ],
    c_bool,
)
def llava_eval_image_embed(
    ctx_llama: llama_cpp.llama_context_p,
    embed: "_Pointer[llava_image_embed]",
    n_batch: Union[c_int, int],
    n_past: "_Pointer[c_int]",
    /,
) -> bool:
    ...


################################################
# clip.h
################################################

# struct clip_image_u8_batch;
clip_image_u8_batch_p = NewType("clip_image_u8_batch_p", int)
clip_image_u8_batch_p_ctypes = c_void_p

# struct clip_image_f32;
clip_image_f32_p = NewType("clip_image_f32_p", int)
clip_image_f32_p_ctypes = c_void_p

# struct clip_image_f32_batch;
clip_image_f32_batch_p = NewType("clip_image_f32_batch_p", int)
clip_image_f32_batch_p_ctypes = c_void_p

# struct ggml_tensor;
ggml_tensor_p = NewType("ggml_tensor_p", int)
ggml_tensor_p_ctypes = c_void_p

# struct clip_image_size {
#     int width;
#     int height;
# };
class clip_image_size(Structure):
    if TYPE_CHECKING:
        width: int
        height: int

    _fields_ = [
        ("width", c_int),
        ("height", c_int),
    ]

clip_image_size_p = NewType("clip_image_size_p", int)
clip_image_size_p_ctypes = POINTER(clip_image_size)

#  struct clip_context_params {
#      bool use_gpu;
#      enum ggml_log_level verbosity;
#  };
class clip_context_params(Structure):
    if TYPE_CHECKING:
        use_gpu: bool
        ggml_log_level: int

    _fields_ = [
        ("use_gpu", c_bool),
        ("ggml_log_level", c_int),
    ]


# /** load mmproj model */
# // deprecated, use clip_init
# CLIP_API struct clip_ctx * clip_model_load    (const char * fname, int verbosity);
@ctypes_function("clip_model_load", [c_char_p, c_int], clip_ctx_p_ctypes)
def clip_model_load(
    fname: bytes, verbosity: Union[c_int, int], /
) -> Optional[clip_ctx_p]:
    ...

# CLIP_API struct clip_ctx * clip_init(const char * fname, struct clip_context_params ctx_params);
@ctypes_function("clip_init", [c_char_p, clip_context_params], clip_ctx_p_ctypes)
def clip_init(
    fname: bytes, ctx_params: clip_context_params, /
) -> Optional[clip_ctx_p]:
    ...

# /** free mmproj model */
# CLIP_API void clip_free(struct clip_ctx * ctx);
@ctypes_function("clip_free", [clip_ctx_p_ctypes], None)
def clip_free(ctx: clip_ctx_p, /):
    ...

# CLIP_API size_t clip_embd_nbytes(const struct clip_ctx * ctx);
@ctypes_function("clip_embd_nbytes", [clip_ctx_p_ctypes], c_int32)
def clip_embd_nbytes(ctx: clip_ctx_p) -> int:
    ...

# CLIP_API size_t clip_embd_nbytes_by_img(const struct clip_ctx * ctx, int img_w, int img_h);
@ctypes_function("clip_embd_nbytes_by_img", [clip_ctx_p_ctypes, c_int, c_int], c_int)
def clip_embd_nbytes_by_img(
    ctx: clip_ctx_p,
    img_w: int,
    img_h: int, /
) -> int:
    ...

# CLIP_API int32_t clip_get_image_size (const struct clip_ctx * ctx);
@ctypes_function("clip_get_image_size", [clip_ctx_p_ctypes], c_int32)
def clip_get_image_size(ctx: clip_ctx_p) -> int:
    ...

# CLIP_API int32_t clip_get_patch_size (const struct clip_ctx * ctx);
@ctypes_function("clip_get_patch_size", [clip_ctx_p_ctypes], c_int32)
def clip_get_patch_size(ctx: clip_ctx_p) -> int:
    ...

# CLIP_API int32_t clip_get_hidden_size(const struct clip_ctx * ctx);
@ctypes_function("clip_get_hidden_size", [clip_ctx_p_ctypes], c_int32)
def clip_get_hidden_size(ctx: clip_ctx_p) -> int:
    ...

# // TODO: should be enum, not string
# CLIP_API const char * clip_patch_merge_type(const struct clip_ctx * ctx);
@ctypes_function("clip_patch_merge_type", [clip_ctx_p_ctypes], c_char_p)
def clip_patch_merge_type(ctx: clip_ctx_p) -> bytes:
    ...

# CLIP_API const int32_t * clip_image_grid(const struct clip_ctx * ctx);
@ctypes_function("clip_image_grid", [clip_ctx_p_ctypes], c_int32)
def clip_image_grid(ctx: clip_ctx_p) -> int:
    ...

# CLIP_API size_t get_clip_image_grid_size(const struct clip_ctx * ctx);
@ctypes_function("get_clip_image_grid_size", [clip_ctx_p_ctypes], c_uint)
def get_clip_image_grid_size(ctx: clip_ctx_p) -> c_uint:
    ...

# CLIP_API int clip_n_output_tokens(const struct clip_ctx * ctx, struct clip_image_f32 * img);
@ctypes_function("clip_n_output_tokens", [clip_ctx_p_ctypes, clip_image_f32_p_ctypes], c_int)
def clip_n_output_tokens(
    ctx: clip_ctx_p,
    img: clip_image_f32_p, /
) -> int:
    ...

# // for M-RoPE, this will be the number of token positions in X and Y directions
# // for other models, X will be the total number of tokens and Y will be 1
# CLIP_API int clip_n_output_tokens_x(const struct clip_ctx * ctx, struct clip_image_f32 * img);
@ctypes_function("clip_n_output_tokens_x", [clip_ctx_p_ctypes, clip_image_f32_p_ctypes], c_int)
def clip_n_output_tokens_x(
    ctx: clip_ctx_p,
    img: clip_image_f32_p, /
) -> int:
    ...

# CLIP_API int clip_n_output_tokens_y(const struct clip_ctx * ctx, struct clip_image_f32 * img);
@ctypes_function("clip_n_output_tokens_y", [clip_ctx_p_ctypes, clip_image_f32_p_ctypes], c_int)
def clip_n_output_tokens_y(
    ctx: clip_ctx_p,
    img: clip_image_f32_p, /
) -> int:
    ...

# // this should be equal to the embedding dimension of the text model
# CLIP_API int clip_n_mmproj_embd(const struct clip_ctx * ctx);
@ctypes_function("clip_n_mmproj_embd", [clip_ctx_p_ctypes], c_int32)
def clip_n_mmproj_embd(ctx: clip_ctx_p) -> int:
    ...

# CLIP_API int clip_uhd_num_image_embeds_col(struct clip_ctx * ctx_clip);
@ctypes_function("clip_uhd_num_image_embeds_col", [clip_ctx_p_ctypes], c_int32)
def clip_uhd_num_image_embeds_col(ctx_clip: clip_ctx_p) -> int:
    ...

# CLIP_API void clip_add_load_image_size(struct clip_ctx * ctx_clip, struct clip_image_size * load_image_size);
@ctypes_function("clip_add_load_image_size", [clip_ctx_p_ctypes, clip_image_size_p_ctypes], None)
def clip_add_load_image_size(ctx_clip: clip_ctx_p, load_image_size: clip_image_size_p, /):
    ...

# CLIP_API struct clip_image_size * clip_get_load_image_size(struct clip_ctx * ctx_clip);
@ctypes_function("clip_get_load_image_size", [clip_ctx_p_ctypes], clip_image_size_p_ctypes)
def clip_get_load_image_size(ctx_clip: clip_ctx_p) -> Optional[clip_image_size_p]:
    ...

# CLIP_API struct clip_image_size * clip_image_size_init();
@ctypes_function("clip_image_size_init", [], clip_image_size_p_ctypes)
def clip_image_size_init() -> Optional[clip_image_size_p]:
    ...

# CLIP_API struct clip_image_u8 * clip_image_u8_init ();
@ctypes_function("clip_image_u8_init", [], clip_image_u8_p_ctypes)
def clip_image_u8_init() -> Optional[clip_image_u8_p]:
    ...

# CLIP_API struct clip_image_f32 * clip_image_f32_init();
@ctypes_function("clip_image_f32_init", [], clip_image_f32_p_ctypes)
def clip_image_f32_init() -> Optional[clip_image_f32_p]:
    ...

# CLIP_API struct clip_image_f32_batch * clip_image_f32_batch_init(); // only used by libllava
@ctypes_function("clip_image_f32_batch_init", [], clip_image_f32_batch_p_ctypes)
def clip_image_f32_batch_init() -> Optional[clip_image_f32_batch_p]:
    ...

# // nx, ny are the output image dimensions
# CLIP_API unsigned char * clip_image_u8_get_data(struct clip_image_u8 * img, uint32_t * nx, uint32_t * ny);
@ctypes_function("clip_image_u8_get_data", [clip_image_u8_p_ctypes, c_uint32, c_uint32], c_char_p)
def clip_image_u8_get_data(
    img: clip_image_u8_p,
    nx: c_uint32,
    ny: c_uint32, /
) -> bytes:
    ...

# CLIP_API void clip_image_size_free (struct clip_image_size * img_size);
@ctypes_function("clip_image_size_free", [clip_image_size_p_ctypes], None)
def clip_image_size_free(img_size: clip_image_size_p):
    ...

# CLIP_API void clip_image_u8_free (struct clip_image_u8  * img);
@ctypes_function("clip_image_u8_free", [clip_image_u8_p_ctypes], None)
def clip_image_u8_free(img: clip_image_u8_p):
    ...

# CLIP_API void clip_image_f32_free(struct clip_image_f32 * img);
@ctypes_function("clip_image_f32_free", [clip_image_f32_p_ctypes], None)
def clip_image_f32_free(img: clip_image_f32_p):
    ...

# CLIP_API void clip_image_u8_batch_free (struct clip_image_u8_batch  * batch);
@ctypes_function("clip_image_u8_batch_free", [clip_image_u8_batch_p_ctypes], None)
def clip_image_u8_batch_free(batch: clip_image_u8_batch_p):
    ...

# CLIP_API void clip_image_f32_batch_free(struct clip_image_f32_batch * batch);
@ctypes_function("clip_image_f32_batch_free", [clip_image_f32_batch_p_ctypes], None)
def clip_image_f32_batch_free(batch: clip_image_f32_batch_p):
    ...

# // use for accessing underlay data of clip_image_f32_batch
# CLIP_API size_t clip_image_f32_batch_n_images(const struct clip_image_f32_batch * batch); // equivalent to batch->size()
@ctypes_function("clip_image_f32_batch_n_images", [clip_image_f32_batch_p_ctypes], c_uint)
def clip_image_f32_batch_n_images(batch: clip_image_f32_batch_p) -> int:
    ...

# CLIP_API size_t clip_image_f32_batch_nx(const struct clip_image_f32_batch * batch, int idx); // equivalent to batch[idx]->nx
@ctypes_function("clip_image_f32_batch_nx", [clip_image_f32_batch_p_ctypes, c_int], c_uint)
def clip_image_f32_batch_nx(
    batch: clip_image_f32_batch_p,
    idx: int, /
) -> int:
    ...

# CLIP_API size_t clip_image_f32_batch_ny(const struct clip_image_f32_batch * batch, int idx); // equivalent to batch[idx]->ny
@ctypes_function("clip_image_f32_batch_nx", [clip_image_f32_batch_p_ctypes, c_int], c_uint)
def clip_image_f32_batch_nx(
    batch: clip_image_f32_batch_p,
    idx: int, /
) -> int:
    ...

# CLIP_API struct clip_image_f32 * clip_image_f32_get_img(const struct clip_image_f32_batch * batch, int idx); // equivalent to batch[idx]->data
@ctypes_function("clip_image_f32_get_img", [clip_image_f32_batch_p_ctypes, c_int], clip_image_f32_p_ctypes)
def clip_image_f32_get_img(
    batch: clip_image_f32_batch_p,
    idx: int, /
) -> Optional[clip_image_f32_p]:
    ...

# /**
#  * Build image from pixels decoded by other libraries instead of stb_image.h for better performance.
#  * The memory layout is RGBRGBRGB..., input buffer length must be 3*nx*ny bytes
#  */
# CLIP_API void clip_build_img_from_pixels(const unsigned char * rgb_pixels, int nx, int ny, struct clip_image_u8 * img);
@ctypes_function("clip_build_img_from_pixels", [c_char_p, c_int, c_int, clip_image_u8_p_ctypes], c_bool)
def clip_build_img_from_pixels(
    rgb_pixels: bytes,
    nx: int,
    ny: int,
    img: clip_image_u8_p, /
) -> bool:
    ...

# CLIP_API bool clip_image_load_from_file(const char * fname, struct clip_image_u8 * img);
@ctypes_function("clip_image_load_from_file", [c_char_p, clip_image_u8_p_ctypes], c_bool)
def clip_image_load_from_file(
    fname: bytes,
    img: clip_image_u8_p, /
) -> bool:
    ...

# /** interpret bytes as an image file with length bytes_length, and use the result to populate img */
# CLIP_API bool clip_image_load_from_bytes(const unsigned char * bytes, size_t bytes_length, struct clip_image_u8 * img);
@ctypes_function("clip_image_load_from_bytes", [c_char_p, c_uint, clip_image_u8_p_ctypes], c_bool)
def clip_image_load_from_bytes(
    bytes: bytes,
    bytes_length: c_uint,
    img: clip_image_u8_p, /
) -> bool:
    ...

# /** preprocess img and store the result in res_imgs, pad_to_square may be overridden to false depending on model configuration */
# CLIP_API bool clip_image_preprocess(struct clip_ctx * ctx, const struct clip_image_u8 * img, struct clip_image_f32_batch * res_imgs );
@ctypes_function("clip_image_preprocess", [clip_ctx_p_ctypes, clip_image_u8_p_ctypes, clip_image_f32_batch_p_ctypes], c_bool)
def clip_image_preprocess(
    ctx: clip_ctx_p,
    img: clip_image_u8_p,
    res_imgs: clip_image_f32_batch_p, /
) -> bool:
    ...

# CLIP_API struct ggml_tensor * clip_get_newline_tensor(const struct clip_ctx * ctx);
@ctypes_function("clip_get_newline_tensor", [clip_ctx_p_ctypes], ggml_tensor_p_ctypes)
def clip_get_newline_tensor(
    ctx: clip_ctx_p, /
) -> Optional[ggml_tensor_p]:
    ...

# CLIP_API bool clip_image_encode      (struct clip_ctx * ctx, int n_threads, struct clip_image_f32 * img, float * vec);
@ctypes_function("clip_image_encode", [clip_ctx_p_ctypes, c_int, clip_image_f32_p_ctypes, CtypesArray[c_float]], c_bool)
def clip_image_encode(
    ctx: clip_ctx_p,
    n_threads: int,
    img: clip_image_f32_p,
    vec: _Pointer[c_float], /
) -> bool:
    ...

# CLIP_API bool clip_image_batch_encode(struct clip_ctx * ctx, int n_threads, const struct clip_image_f32_batch * imgs, float * vec);
@ctypes_function("clip_image_batch_encode", [clip_ctx_p_ctypes, c_int, clip_image_f32_batch_p_ctypes, CtypesArray[c_float]], c_bool)
def clip_image_batch_encode(
    ctx: clip_ctx_p,
    n_threads: int,
    imgs: clip_image_f32_batch_p,
    vec: _Pointer[c_float], /
) -> bool:
    ...

# CLIP_API bool clip_model_quantize(const char * fname_inp, const char * fname_out, int itype);
@ctypes_function("clip_model_quantize", [c_char_p, c_char_p, c_int], c_bool)
def clip_model_quantize(
    fname_inp: bytes,
    fname_out: bytes,
    itype: int, /
) -> bool:
    ...

# CLIP_API int clip_is_minicpmv(const struct clip_ctx * ctx);
@ctypes_function("clip_is_minicpmv", [clip_ctx_p_ctypes], c_int32)
def clip_is_minicpmv(ctx: clip_ctx_p) -> int:
    ...

# CLIP_API bool clip_is_glm(const struct clip_ctx * ctx);
@ctypes_function("clip_is_glm", [clip_ctx_p_ctypes], c_bool)
def clip_is_glm(ctx: clip_ctx_p) -> bool:
    ...

# CLIP_API bool clip_is_qwen2vl(const struct clip_ctx * ctx);
@ctypes_function("clip_is_qwen2vl", [clip_ctx_p_ctypes], c_bool)
def clip_is_qwen2vl(ctx: clip_ctx_p) -> bool:
    ...

# CLIP_API bool clip_is_llava(const struct clip_ctx * ctx);
@ctypes_function("clip_is_llava", [clip_ctx_p_ctypes], c_bool)
def clip_is_llava(ctx: clip_ctx_p) -> bool:
    ...

# CLIP_API bool clip_is_gemma3(const struct clip_ctx * ctx);
@ctypes_function("clip_is_gemma3", [clip_ctx_p_ctypes], c_bool)
def clip_is_gemma3(ctx: clip_ctx_p) -> bool:
    ...

# CLIP_API bool clip_encode_float_image (struct clip_ctx * ctx, int n_threads, float * img, int h, int w, float * vec);
@ctypes_function("clip_n_output_tokens", [clip_ctx_p_ctypes, c_int, CtypesArray[c_float], c_int, c_int, CtypesArray[c_float]], c_bool)
def clip_n_output_tokens(
    ctx: clip_ctx_p,
    n_threads: int,
    img: _Pointer[float],
    h: int,
    w: int,
    vec: _Pointer[float],/
) -> bool:
    ...
