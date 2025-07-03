from gradio import Blocks, Row, Column, Accordion, Markdown, Textbox, Button, Audio
import gradio as gr
import subprocess
import uuid
import datetime
import shutil
import threading
import time
import os


with gr.Blocks(title="AudiobookAgent 配音演示", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📖 Role-Playing-TTS 一键配音演示

        请按需填写下列输入项：

        - ✅ 若仅填写 text，将完全由模型执行对话脚本生成 + 配音脚本生成 + 音频合成；
        - ⚙️ 若填写 对话脚本内容，将跳过对话脚本生成；模型直接进行配音脚本生成+音频合成；
        - 🚀 若填写 配音脚本内容，将跳过对话脚本生成和配音脚本生成，模型直接合成音频；
        """,
    )

    with gr.Row(variant="panel"):
        with gr.Column(scale=1, min_width=300):
            with gr.Accordion("📥 输入参数（点击展开）", open=True):
                text_input = gr.Textbox(
                    label="📝 text (主题)",
                    lines=4,
                    placeholder="例如：请你演绎一下唐僧师徒四人取经的故事。",
                )
                step1_input = gr.Textbox(
                    label="⚙️ 对话脚本内容",
                    lines=6,
                    placeholder="可选：将对话脚本内容粘贴于此，由大模型分析生成相应的配音脚本。\n注意：请确保对话脚本内容格式正确，否则可能导致配音脚本生成失败。",
                )
                step2_input = gr.Textbox(
                    label="🚀 配音脚本内容",
                    lines=8,
                    placeholder="可选：将配音脚本内容粘贴于此，可跳过所有分析步骤，直接合成。\n注意：请确保配音脚本内容格式正确，否则可能导致音频生成失败。",
                )

            run_button = gr.Button("▶️ 生成并播放", variant="primary")

        with gr.Column(scale=2, min_width=500):
            status_text = gr.Markdown("✨ **状态：** 一切就绪，请点击生成按钮。")
            log_output = gr.Textbox(
                label="📜 实时日志",
                lines=18,
                interactive=False,
                show_copy_button=True,
                placeholder="此处将实时显示 pipeline 运行日志...",
            )
            audio_output = gr.Audio(
                label="🎧 最终音频",
                interactive=False,
                type="filepath",
            )


    # 更新状态提示 & 运行按钮行为
    def wrapped_pipeline(*args):
        # 1. 更新状态为"运行中"
        yield {status_text: "⏳ **状态：** 正在生成，请稍候...", log_output: None, audio_output: None}

        final_log = ""
        final_audio = None
        try:
            # 2. 流式传输日志
            for log, audio in run_pipeline(*args):
                final_log = log
                final_audio = audio
                yield {
                    status_text: "⏳ **状态：** 正在生成，请稍候...",
                    log_output: final_log,
                    audio_output: final_audio,
                }

            # 3. 根据最终结果更新状态
            if final_audio and os.path.exists(final_audio):
                yield {
                    status_text: "✅ **状态：** 生成成功！",
                    log_output: final_log,
                    audio_output: final_audio,
                }
            elif final_log.strip().endswith("脚本运行失败") or "return code" in final_log:
                 yield {
                    status_text: "⚠️ **状态：** 生成失败，请检查日志。",
                    log_output: final_log,
                    audio_output: None,
                }
            else:
                 yield {
                    status_text: "⚠️ **状态：** 运行成功，但未找到音频文件。",
                    log_output: final_log,
                    audio_output: None,
                }
        except Exception as e:
            import traceback
            error_message = f"{final_log}\n\n🔥 **程序运行异常**: {e}\n\n**Traceback:**\n{traceback.format_exc()}"
            yield {
                status_text: "💥 **状态：** 运行出错！",
                log_output: error_message,
                audio_output: None,
            }

    run_button.click(
        fn=wrapped_pipeline,
        inputs=[text_input, step1_input, step2_input],
        outputs=[status_text, log_output, audio_output],
    )


def run_pipeline(text: str, step1_content: str, step2_content: str):
    """包装 AudiobookAgent/pipeline.py，实时流式返回日志并最终给出音频路径。

    参数说明：
        text:            传给 --text 的主题内容（可为空）。
        step1_content:   若提供则写入临时文件并作为 --step1_file（优先生效）。
        step2_content:   若提供则写入临时文件并作为 --step2_file（最高优先级）。

    生成器产出：
        Tuple[str, str]: (实时日志字符串, 最终音频文件路径或 None)。
    """
    # 生成一个唯一的输出目录名称
    unique_id = uuid.uuid4().hex
    output_path = os.path.join("./output_gradio", f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_id}")
    os.makedirs(output_path, exist_ok=True)

    try:
        # 构建 pipeline.py 命令
        cmd = ["python", "-u", "pipeline.py"]

        step1_file_path = None
        step2_file_path = None

        # 处理 step2_content (最高优先级)
        if step2_content:
            step2_file_path = os.path.join(output_path, "step2_input.jsonl")
            with open(step2_file_path, "w", encoding="utf-8") as f:
                f.write(step2_content)
            cmd.extend(["--step2", step2_file_path])
        # 处理 step1_content (次高优先级)
        elif step1_content:
            step1_file_path = os.path.join(output_path, "step1_input.json")
            with open(step1_file_path, "w", encoding="utf-8") as f:
                f.write(step1_content)
            cmd.extend(["--step1", step1_file_path])
        # 处理 text (最低优先级，如果 step1/step2_content 都未提供)
        elif text:
            cmd.extend(["--text", text])
        else:
            # 如果所有输入都为空，则返回错误或提示用户
            yield "🛑 错误：请至少提供 'text', '对话脚本内容' 或 '配音脚本内容' 中的一项！", None
            return

        cmd.extend(["--output_path", output_path])

        # 启动子进程，合并 stdout/stderr 以便统一展示
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',
        )

        log_buffer = ""
        audio_path = None # 初始化为 None

        # 实时读取并向前端流式输出
        assert proc.stdout is not None
        for line in proc.stdout:
            log_buffer += line
            yield log_buffer, None # 流式传输日志时，音频路径为 None

        # 等待进程结束
        proc.wait()

        # 确保临时文件被清理（这部分保留，因为 output_path 需要延时清理）
        if step1_file_path and os.path.exists(step1_file_path):
            os.remove(step1_file_path)
        if step2_file_path and os.path.exists(step2_file_path):
            os.remove(step2_file_path)

        if proc.returncode != 0:
            log_buffer += f"\n\n--- 脚本运行失败：return code {proc.returncode} ---"
            yield log_buffer, None
            return

        # 组装最终音频路径
        final_audio_path = os.path.join(output_path, "audio", "final_mix.wav")
        if os.path.exists(final_audio_path):
            audio_path = final_audio_path
            log_buffer += "\n\n--- 音频生成成功 ---"
        else:
            log_buffer += "\n\n--- 脚本运行完毕，但未在预期路径找到音频文件 ---"
            
        # 最终成功：同时输出完整日志与音频路径
        yield log_buffer, audio_path
    finally:
        # 不管成功与否，都启动后台线程延迟清理临时目录
        threading.Thread(target=_delayed_cleanup, args=(output_path, 30), daemon=True).start()


def _delayed_cleanup(path: str, delay: int = 30):
    """延时删除目录，确保 Gradio 已完成文件缓存。"""
    time.sleep(delay)
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception as e:
        # 打印但不干扰主流程
        print(f"[cleanup] 删除 {path} 失败：{e}")


if __name__ == "__main__":
    demo.launch()
