import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' #使用镜像站下载模型

os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'hf_download')
HF_TOKEN = None

import lib_omost.memory_management as memory_management
import uuid

import torch
import numpy as np
import gradio as gr
import tempfile

gradio_temp_dir = os.path.join(tempfile.gettempdir(), 'gradio')
os.makedirs(gradio_temp_dir, exist_ok=True)

from threading import Thread

# Phi3 Hijack
from transformers.models.phi3.modeling_phi3 import Phi3PreTrainedModel

Phi3PreTrainedModel._supports_sdpa = True

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionXLImg2ImgPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from lib_omost.pipeline import StableDiffusionXLOmostPipeline
from chat_interface import ChatInterface
from transformers.generation.stopping_criteria import StoppingCriteriaList

import lib_omost.canvas as omost_canvas

# 定义当前目录下 models/checkpoints 文件夹的路径
models_dir = os.path.join(os.getcwd(), 'models/checkpoints')
#如果没有发现这个文件夹，则创建一个 models/checkpoints文件夹    
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# sdxl_name是本地模型名称
sdxl_name = 'RealVisXL_V4.0'

# 检查model_path是否为当前目录下models文件夹中的一个.safetensors文件
model_path = os.path.join(models_dir, sdxl_name + '.safetensors')

if not os.path.isfile(model_path):
    print(f"Downloading {sdxl_name} from https://huggingface.co/SG161222/RealVisXL_V4.0/resolve/main/RealVisXL_V4.0.safetensors")  
    print(f"Downloading {sdxl_name} from https://hf-mirror.com/SG161222/RealVisXL_V4.0/resolve/main/RealVisXL_V4.0.safetensors")  
    #抛出异常
    raise FileNotFoundError(f"{sdxl_name}.safetensors not found in {models_dir}")

# 文件存在，加载模型并创建pipeline
temp_pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(model_path, torch_dtype=torch.float16)

# 从pipeline中获取组件
tokenizer = temp_pipeline.tokenizer
tokenizer_2 = temp_pipeline.tokenizer_2
text_encoder = temp_pipeline.text_encoder
text_encoder_2 = temp_pipeline.text_encoder_2
# 转换text_encoder_2为ClipTextModel
text_encoder_2 = CLIPTextModel(config=text_encoder_2.config)
vae = temp_pipeline.vae
unet = temp_pipeline.unet

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

pipeline = StableDiffusionXLOmostPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    unet=unet,
    scheduler=None,  # We completely give up diffusers sampling system and use A1111's method
)
memory_management.unload_all_models([text_encoder, text_encoder_2, vae, unet])

# LLM

# llm_name = 'lllyasviel/omost-phi-3-mini-128k-8bits'
llm_name = 'lllyasviel/omost-llama-3-8b-4bits'
# llm_name = 'lllyasviel/omost-dolphin-2.9-llama3-8b-4bits'

llm_model = AutoModelForCausalLM.from_pretrained(
    llm_name,
    torch_dtype=torch.bfloat16,  # This is computation type, not load/memory type. The loading quant type is baked in config.
    token=HF_TOKEN,
    device_map="auto"  # This will load model to gpu with an offload system
)

llm_tokenizer = AutoTokenizer.from_pretrained(
    llm_name,
    token=HF_TOKEN
)

memory_management.unload_all_models(llm_model)


def process_seed(seed_string):
    # 尝试将字符串转换为整数
    try:
        seed = int(seed_string)
    except ValueError:
        raise ValueError(f"The seed string '{seed_string}' is not a valid integer.")
    # 处理转换后的整数
    if seed == -1:
        # 如果是 -1，重新生成一个随机整数
        seed = np.random.randint(0, 2**31 - 1)
        #打印生成的随机种子
        print(f"Random seed: {seed}")
    elif not (0 <= seed <= 2**31 - 1):
        # 如果整数不在 0 到 2**31 - 1 范围内，抛出异常
        raise ValueError(f"The seed value '{seed}' is out of the valid range for int32 [0, {2**31 - 1}].")    
    return seed

@torch.inference_mode()
def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def chat_fn(message: str, history: list, seed:int, temperature: float, top_p: float, max_new_tokens: int) -> str:
    seed = process_seed(seed)
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    conversation = [{"role": "system", "content": omost_canvas.system_prompt}]

    for user, assistant in history:
        if isinstance(user, str) and isinstance(assistant, str):
            if len(user) > 0 and len(assistant) > 0:
                conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])

    conversation.append({"role": "user", "content": message})

    memory_management.load_models_to_gpu(llm_model)

    input_ids = llm_tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True).to(llm_model.device)

    streamer = TextIteratorStreamer(llm_tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    def interactive_stopping_criteria(*args, **kwargs) -> bool:
        if getattr(streamer, 'user_interrupted', False):
            print('User stopped generation')
            return True
        else:
            return False

    stopping_criteria = StoppingCriteriaList([interactive_stopping_criteria])

    def interrupter():
        streamer.user_interrupted = True
        return

    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        stopping_criteria=stopping_criteria,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    if temperature == 0:
        generate_kwargs['do_sample'] = False

    Thread(target=llm_model.generate, kwargs=generate_kwargs).start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        # print(outputs)
        yield "".join(outputs), interrupter

    #打印完成信息
    print('Chat finished')

    return


@torch.inference_mode()
def post_chat(history):
    canvas_outputs = None

    try:
        if history:
            history = [(user, assistant) for user, assistant in history if isinstance(user, str) and isinstance(assistant, str)]
            last_assistant = history[-1][1] if len(history) > 0 else None
            canvas = omost_canvas.Canvas.from_bot_response(last_assistant)
            canvas_outputs = canvas.process()
    except Exception as e:
        print('Last assistant response is not valid canvas:', e)

    return canvas_outputs, gr.update(visible=canvas_outputs is not None), gr.update(interactive=len(history) > 0)


@torch.inference_mode()
def diffusion_fn(chatbot, canvas_outputs, num_samples, seed, image_width, image_height,
                 highres_scale, steps, cfg, highres_steps, highres_denoise, negative_prompt):

    use_initial_latent = False
    eps = 0.05 

    image_width, image_height = int(image_width // 64) * 64, int(image_height // 64) * 64

    seed = process_seed(seed)

    rng = torch.Generator(device=memory_management.gpu).manual_seed(seed)

    memory_management.load_models_to_gpu([text_encoder, text_encoder_2])

    positive_cond, positive_pooler, negative_cond, negative_pooler = pipeline.all_conds_from_canvas(canvas_outputs, negative_prompt)

    if use_initial_latent:
        memory_management.load_models_to_gpu([vae])
        initial_latent = torch.from_numpy(canvas_outputs['initial_latent'])[None].movedim(-1, 1) / 127.5 - 1.0
        initial_latent_blur = 40
        initial_latent = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(initial_latent, (initial_latent_blur,) * 4, mode='reflect'),
            kernel_size=(initial_latent_blur * 2 + 1,) * 2, stride=(1, 1))
        initial_latent = torch.nn.functional.interpolate(initial_latent, (image_height, image_width))
        initial_latent = initial_latent.to(dtype=vae.dtype, device=vae.device)
        initial_latent = vae.encode(initial_latent).latent_dist.mode() * vae.config.scaling_factor
    else:
        initial_latent = torch.zeros(size=(num_samples, 4, image_height // 8, image_width // 8), dtype=torch.float32)

    memory_management.load_models_to_gpu([unet])

    initial_latent = initial_latent.to(dtype=unet.dtype, device=unet.device)

    latents = pipeline(
        initial_latent=initial_latent,
        strength=1.0,
        num_inference_steps=int(steps),
        batch_size=num_samples,
        prompt_embeds=positive_cond,
        negative_prompt_embeds=negative_cond,
        pooled_prompt_embeds=positive_pooler,
        negative_pooled_prompt_embeds=negative_pooler,
        generator=rng,
        guidance_scale=float(cfg),
    ).images

    memory_management.load_models_to_gpu([vae])
    latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
    pixels = vae.decode(latents).sample
    B, C, H, W = pixels.shape
    pixels = pytorch2numpy(pixels)

    if highres_scale > 1.0 + eps:
        pixels = [
            resize_without_crop(
                image=p,
                target_width=int(round(W * highres_scale / 64.0) * 64),
                target_height=int(round(H * highres_scale / 64.0) * 64)
            ) for p in pixels
        ]

        pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
        latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor

        memory_management.load_models_to_gpu([unet])
        latents = latents.to(device=unet.device, dtype=unet.dtype)

        latents = pipeline(
            initial_latent=latents,
            strength=highres_denoise,
            num_inference_steps=highres_steps,
            batch_size=num_samples,
            prompt_embeds=positive_cond,
            negative_prompt_embeds=negative_cond,
            pooled_prompt_embeds=positive_pooler,
            negative_pooled_prompt_embeds=negative_pooler,
            generator=rng,
            guidance_scale=float(cfg),
        ).images

        memory_management.load_models_to_gpu([vae])
        latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
        pixels = vae.decode(latents).sample
        pixels = pytorch2numpy(pixels)

    for i in range(len(pixels)):
        unique_hex = uuid.uuid4().hex
        image_path = os.path.join(gradio_temp_dir, f"{unique_hex}_{i}.png")
        image = Image.fromarray(pixels[i])
        image.save(image_path)
        chatbot = chatbot + [(None, (image_path, 'image'))]

    return chatbot


css = '''
code {white-space: pre-wrap !important;}
.gradio-container {max-width: none !important;}
.outer_parent {flex: 1;}
.inner_parent {flex: 1;}
footer {display: none !important; visibility: hidden !important;}
.translucent {display: none !important; visibility: hidden !important;}
'''

from gradio.themes.utils import colors

with gr.Blocks(
        fill_height=True, css=css,
        theme=gr.themes.Default(primary_hue=colors.blue, secondary_hue=colors.cyan, neutral_hue=colors.gray)
) as demo:
    with gr.Row(elem_classes='outer_parent'):
        with gr.Column(scale=25):
            with gr.Row():
                clear_btn = gr.Button("➕ 新建对话", variant="secondary", size="sm", min_width=60)
                retry_btn = gr.Button("重试", variant="secondary", size="sm", min_width=60, visible=False)
                undo_btn = gr.Button("✏️️ 编辑最近一次输入", variant="secondary", size="sm", min_width=60, interactive=False)

            seed = gr.Number(label="随机种子", value=-1, precision=0)

            with gr.Accordion(open=True, label='语言模型'):
                with gr.Group():
                    with gr.Row():
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            step=0.01,
                            value=0.6,
                            label="随机性调节")
                        top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.9,
                            label="核心词采样")
                    max_new_tokens = gr.Slider(
                        minimum=128,
                        maximum=4096,
                        step=1,
                        value=4096,
                        label="最大新词元（Tokens）数")
            with gr.Accordion(open=True, label='图像扩散模型'):
                with gr.Group():
                    with gr.Row():
                        image_width = gr.Slider(label="图像宽度", minimum=256, maximum=2048, value=896, step=64)
                        image_height = gr.Slider(label="图像高度", minimum=256, maximum=2048, value=1152, step=64)

                    with gr.Row():
                        num_samples = gr.Slider(label="出图数量", minimum=1, maximum=12, value=1, step=1)
                        steps = gr.Slider(label="采样步数", minimum=1, maximum=100, value=25, step=1)

            with gr.Accordion(open=False, label='高级设置'):
                cfg = gr.Slider(label="提示引导系数 CFG", minimum=1.0, maximum=32.0, value=5.0, step=0.01)
                highres_scale = gr.Slider(label="高清修复放大倍数（1为禁用）", minimum=1.0, maximum=2.0, value=1.0, step=0.01)
                highres_steps = gr.Slider(label="高清修复步数", minimum=1, maximum=100, value=20, step=1)
                highres_denoise = gr.Slider(label="高清修复降噪强度", minimum=0.1, maximum=1.0, value=0.4, step=0.01)
                n_prompt = gr.Textbox(label="反向提示词", value='lowres, bad anatomy, bad hands, cropped, worst quality')

            render_button = gr.Button("渲染图像！", size='lg', variant="primary", visible=False)

            examples = gr.Dataset(
                samples=[
                    ['generate an image of the fierce battle of warriors and a dragon'],
                    ['change the dragon to a dinosaur']
                ],
                components=[gr.Textbox(visible=False)],
                label='提示词快捷列表'
            )
        with gr.Column(scale=75, elem_classes='inner_parent'):
            canvas_state = gr.State(None)
            chatbot = gr.Chatbot(label='Omost', scale=1, show_copy_button=True, layout="panel", render=False)
            chatInterface = ChatInterface(
                fn=chat_fn,
                post_fn=post_chat,
                post_fn_kwargs=dict(inputs=[chatbot], outputs=[canvas_state, render_button, undo_btn]),
                pre_fn=lambda: gr.update(visible=False),
                pre_fn_kwargs=dict(outputs=[render_button]),
                chatbot=chatbot,
                retry_btn=retry_btn,
                undo_btn=undo_btn,
                clear_btn=clear_btn,
                additional_inputs=[seed, temperature, top_p, max_new_tokens],
                examples=examples
            )

    render_button.click(
        fn=diffusion_fn, inputs=[
            chatInterface.chatbot, canvas_state,
            num_samples, seed, image_width, image_height, highres_scale,
            steps, cfg, highres_steps, highres_denoise, n_prompt
        ], outputs=[chatInterface.chatbot]).then(
        fn=lambda x: x, inputs=[
            chatInterface.chatbot
        ], outputs=[chatInterface.chatbot_state])

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, server_name='0.0.0.0')
