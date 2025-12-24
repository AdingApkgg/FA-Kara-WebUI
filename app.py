#!/usr/bin/env python3
"""
FA-Kara WebUI - 基于歌词文本和人声音频的自动打轴工具
"""

import gradio as gr
import tempfile
import shutil
import os
import sys
import bisect
import time

# Add FA-Kara to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'FA-Kara'))

import librosa
import numpy as np

import align
import haruraw2norm as hn
import lrcfmt
import norm2ass
from norm2lrc import (
    process_main, process_ruby, process_rlf,
    non_silent_head_adjust, split_long_segments,
    parse_time_to_hundredths, format_hundredths_to_time_str
)


def non_silent_recog(audio_file, sr=None, frame_second=1, threspct=10, thresrto=0.1):
    """识别非静音片段"""
    frame_length = int(sr * frame_second)
    hop_length = frame_length // 2
    energy = librosa.feature.rms(y=audio_file, frame_length=frame_length, hop_length=hop_length)[0]
    threshold = np.percentile(energy, 100 - threspct) * thresrto
    non_silent_frames = energy > threshold
    times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)
    segments = []
    start = None
    for i, (t, active) in enumerate(zip(times, non_silent_frames)):
        if active and start is None:
            start = max(t - frame_second / 4, 0)
        elif not active and start is not None:
            segments.append((start, t + frame_second / 4))
            start = None
    if start is not None:
        segments.append((start, times[-1]))
    return segments


def process_lyrics(
    audio_file,
    lyrics_text,
    audio_speed: float = 1.0,
    sokuon_split: bool = False,
    hatsuon_split: bool = True,
    tail_correct: int = 3,
    silent_window: float = 0.8,
    tail_thres_pct: float = 10,
    tail_thres_ratio: float = 0.1,
    ruby_offset: int = -150,
    bpm: float = 60,
    beats_per_bar: int = 3,
    language: str = "jaen",
    txt_format: str = "hrh",
    chars_per_line: int = 0,
    progress=gr.Progress()
):
    """处理歌词和音频，生成时间轴文件"""
    
    if audio_file is None:
        raise gr.Error("请上传音频文件！")
    
    if not lyrics_text or not lyrics_text.strip():
        raise gr.Error("请输入歌词文本！")
    
    progress(0.1, desc="正在加载文件...")
    
    # 处理歌词文本
    result_list = []
    lines = lyrics_text.strip().split('\n')
    
    if txt_format == 'uta':
        lines = lrcfmt.utat_process(lyrics_text)
    
    for line in lines:
        if txt_format == 'moe':
            line = lrcfmt.moeg_process_line(line)
        if line.strip():
            result_list.extend(hn.process_haruhi_line(line, language, int(sokuon_split), int(hatsuon_split)))
    
    if not result_list:
        raise gr.Error("歌词解析失败，请检查格式！")
    
    if result_list[-1]['orig'] != '\n':
        result_list.append({'orig': '\n', 'type': 0, 'pron': ''})
    
    # 尾音处理 (tail_correct == 1 or 2)
    if tail_correct == 1:
        for i in range(len(result_list)):
            if result_list[i]['type'] == 0:
                try:
                    if result_list[i - 1].get('pron') and result_list[i - 1]['type'] != 0:
                        pre_vowel = result_list[i - 1]['pron'][-1]
                        post_consonant = ''
                        if i < len(result_list) - 1:
                            post_i = i + 1
                            while post_i < len(result_list):
                                if 'pron' in result_list[post_i] and len(result_list[post_i]['pron']) >= 1:
                                    post_consonant = result_list[post_i]['pron'][0]
                                    break
                                else:
                                    post_i += 1
                        if pre_vowel != post_consonant and post_consonant not in ('a', 'e', 'i', 'o', 'u'):
                            result_list[i]['pron'] = pre_vowel + 'h'
                except:
                    continue
    elif tail_correct == 2:
        for i in range(len(result_list)):
            if result_list[i]['type'] == 0:
                try:
                    if len(result_list[i - 1]['pron']) >= 1 and result_list[i - 1]['type'] != 0:
                        result_list[i]['pron'] = result_list[i - 1]['pron'][-1] + 'h'
                except:
                    continue
    
    progress(0.2, desc="正在分析歌词...")
    
    # 构建对齐 tokens
    alignment_tokens = []
    token_to_index_map = {}
    for i, item in enumerate(result_list):
        if 'pron' in item and item['pron']:
            alignment_tokens.append(item['pron'])
            token_to_index_map[len(alignment_tokens) - 1] = i
    
    progress(0.3, desc="正在加载音频...")
    
    # 加载音频
    audio_file_data, sr = librosa.load(audio_file, sr=None)
    non_silent_ranges = non_silent_recog(audio_file_data, sr, silent_window, tail_thres_pct, tail_thres_ratio)
    
    progress(0.4, desc="正在进行对齐推理...")
    
    # 对齐处理
    if audio_speed == 1:
        alignment_results = align.align_audio_with_text(audio_file_data, alignment_tokens, non_silent_ranges, sr)
    else:
        y_processed = librosa.effects.time_stretch(audio_file_data, rate=audio_speed)
        alignment_results = align.align_audio_with_text(y_processed, alignment_tokens, non_silent_ranges, sr, audio_speed)
    
    progress(0.7, desc="正在生成时间轴...")
    
    # 映射结果
    for i, result in enumerate(alignment_results):
        if i in token_to_index_map:
            original_index = token_to_index_map[i]
            result_list[original_index]['start'] = result['start']
            result_list[original_index]['end'] = result['end']
    
    result_list = non_silent_head_adjust(result_list, non_silent_ranges)
    
    # tail_correct == 3 处理
    if tail_correct == 3:
        ns_small = non_silent_recog(audio_file_data, sr, 0.02, tail_thres_pct, tail_thres_ratio)
        ns_ends = [int(np.ceil(ns_end * 100)) for _, ns_end in ns_small]
        for i in range(len(result_list) - 1):
            if result_list[i]['type'] != 0 and result_list[i + 1]['type'] == 0:
                current_end = parse_time_to_hundredths(result_list[i]['end'])
                next_ind = i + 2
                next_start = np.inf
                while next_ind < len(result_list):
                    if 'start' in result_list[next_ind]:
                        next_start = parse_time_to_hundredths(result_list[next_ind]['start'])
                        break
                    next_ind += 1
                left_index = bisect.bisect_left(ns_ends, current_end)
                right_index = bisect.bisect_left(ns_ends, next_start)
                if left_index < right_index and left_index < len(ns_ends):
                    result_list[i]['end'] = format_hundredths_to_time_str(ns_ends[left_index])
                else:
                    interval_covered = False
                    for nss_start, nss_end in ns_small:
                        if int(nss_start * 100) > current_end:
                            break
                        if int(nss_start * 100) <= current_end and int(np.ceil(nss_end * 100)) >= next_start:
                            interval_covered = True
                            break
                    if interval_covered:
                        result_list[i]['end'] = format_hundredths_to_time_str(max(next_start - 2, current_end))
    
    if chars_per_line > 0:
        split_long_segments(result_list, max_length=chars_per_line)
    
    progress(0.9, desc="正在保存文件...")
    
    # 生成输出文件
    output_dir = tempfile.mkdtemp()
    
    # Ruby LRC
    main_output = process_main(result_list, ruby_offset, bpm, beats_per_bar)
    ruby_output = process_ruby(result_list)
    ruby_lrc_path = os.path.join(output_dir, 'output_ruby.lrc')
    with open(ruby_lrc_path, 'w', encoding='utf-8') as f:
        f.write(f"{main_output}\n{ruby_output}")
    
    # RLF LRC
    rlf_output = process_rlf(result_list)
    rlf_lrc_path = os.path.join(output_dir, 'output_rlf.lrc')
    with open(rlf_lrc_path, 'w', encoding='utf-8') as f:
        f.write(rlf_output)
    
    # ASS
    ass_output = norm2ass.process_norm2assV2(result_list)
    ass_head = '''[Script Info]
ScriptType: v4.00+
YCbCr Matrix: TV.601
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Source Han Serif,71,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1.99999,1.99999,2,11,11,101,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
    ass_path = os.path.join(output_dir, 'output.ass')
    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write(ass_head + ass_output)
    
    progress(1.0, desc="处理完成！")
    
    return ruby_lrc_path, rlf_lrc_path, ass_path, "✅ 处理完成！"


# 自定义 CSS - 赛博朋克 x 日式卡拉OK 风格
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=M+PLUS+Rounded+1c:wght@400;500;700&family=Orbitron:wght@400;700&display=swap');

/* 全局重置 - 覆盖 Gradio 主题变量 */
:root, .dark, .light, [data-theme="dark"], [data-theme="light"] {
    --body-background-fill: #0a0a0f !important;
    --background-fill-primary: rgba(15, 15, 25, 0.95) !important;
    --background-fill-secondary: rgba(20, 20, 35, 0.9) !important;
    --block-background-fill: rgba(18, 18, 28, 0.95) !important;
    --block-border-color: rgba(255, 0, 255, 0.25) !important;
    --block-label-background-fill: transparent !important;
    --block-label-text-color: #b0b0c0 !important;
    --block-title-text-color: #00ffff !important;
    --body-text-color: #e0e0e8 !important;
    --body-text-color-subdued: #8080a0 !important;
    --input-background-fill: rgba(5, 5, 15, 0.8) !important;
    --input-border-color: rgba(0, 255, 255, 0.3) !important;
    --input-placeholder-color: #505070 !important;
    --button-primary-background-fill: linear-gradient(135deg, #ff00ff 0%, #8000ff 100%) !important;
    --button-primary-background-fill-hover: linear-gradient(135deg, #ff40ff 0%, #a020ff 100%) !important;
    --button-primary-text-color: #ffffff !important;
    --button-secondary-background-fill: rgba(0, 255, 255, 0.15) !important;
    --button-secondary-border-color: rgba(0, 255, 255, 0.4) !important;
    --slider-color: #ff00ff !important;
    --checkbox-background-color-selected: #ff00ff !important;
    --checkbox-border-color-selected: #ff00ff !important;
    --shadow-drop: 0 0 15px rgba(255, 0, 255, 0.15) !important;
    --shadow-drop-lg: 0 0 30px rgba(255, 0, 255, 0.2) !important;
    --color-accent: #ff00ff !important;
    --color-accent-soft: rgba(255, 0, 255, 0.2) !important;
    --neutral-50: #0a0a0f !important;
    --neutral-100: #12121a !important;
    --neutral-200: #1a1a28 !important;
    --neutral-300: #252535 !important;
    --neutral-400: #404060 !important;
    --neutral-500: #606080 !important;
    --neutral-600: #8080a0 !important;
    --neutral-700: #a0a0c0 !important;
    --neutral-800: #c0c0d8 !important;
    --neutral-900: #e0e0f0 !important;
    --neutral-950: #f0f0ff !important;
}

* {
    transition: all 0.15s ease;
}

/* 主容器背景 */
.gradio-container, .main, .contain, body, html {
    font-family: 'M PLUS Rounded 1c', -apple-system, sans-serif !important;
    background: 
        radial-gradient(ellipse at top left, rgba(255, 0, 128, 0.12) 0%, transparent 50%),
        radial-gradient(ellipse at bottom right, rgba(0, 255, 255, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse at center, rgba(128, 0, 255, 0.06) 0%, transparent 70%),
        linear-gradient(180deg, #0a0a0f 0%, #10101a 50%, #0a0a12 100%) !important;
    min-height: 100vh;
}

/* 所有面板和区块 */
.gr-group, .gr-box, .gr-form, .gr-panel, .block, .wrap, .contain,
div[class*="block"], div[class*="panel"], div[class*="group"] {
    background: rgba(15, 15, 25, 0.85) !important;
    border-color: rgba(255, 0, 255, 0.2) !important;
}

/* 霓虹标题 */
.neon-header {
    text-align: center;
    padding: 2rem 0 1rem;
    position: relative;
}

.neon-title {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 3.2rem !important;
    font-weight: 700 !important;
    color: #fff !important;
    text-shadow: 
        0 0 10px #ff00ff,
        0 0 20px #ff00ff,
        0 0 40px #ff00ff,
        0 0 80px #ff00ff;
    letter-spacing: 0.15em;
    margin: 0 !important;
    animation: neon-flicker 4s infinite alternate;
}

.neon-subtitle {
    font-size: 1rem;
    color: #00ffff !important;
    text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff;
    letter-spacing: 0.3em;
    margin-top: 0.5rem;
    opacity: 0.9;
}

@keyframes neon-flicker {
    0%, 18%, 22%, 25%, 53%, 57%, 100% {
        text-shadow: 
            0 0 10px #ff00ff,
            0 0 20px #ff00ff,
            0 0 40px #ff00ff,
            0 0 80px #ff00ff;
    }
    20%, 24%, 55% {
        text-shadow: 0 0 5px #ff00ff, 0 0 10px #ff00ff;
    }
}

/* 卡片容器 */
.card-container {
    background: linear-gradient(145deg, rgba(18, 18, 30, 0.95) 0%, rgba(25, 18, 35, 0.9) 100%) !important;
    border: 1px solid rgba(255, 0, 255, 0.25) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    margin: 0.5rem !important;
    box-shadow: 
        0 0 25px rgba(255, 0, 255, 0.08),
        inset 0 0 80px rgba(0, 0, 0, 0.4) !important;
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
}

.card-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #ff00ff, #00ffff, transparent);
    animation: border-flow 4s linear infinite;
}

@keyframes border-flow {
    to { left: 100%; }
}

/* 区块标题 */
.section-title {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    color: #00ffff !important;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    margin-bottom: 1rem !important;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid transparent;
    border-image: linear-gradient(90deg, #00ffff, #ff00ff, transparent) 1;
    text-shadow: 0 0 12px rgba(0, 255, 255, 0.6);
}

/* 输入框样式 */
textarea, input[type="text"], input[type="number"], .gr-textbox textarea, .gr-textbox input {
    background: rgba(5, 5, 15, 0.9) !important;
    border: 1px solid rgba(0, 255, 255, 0.25) !important;
    border-radius: 8px !important;
    color: #e0e0e8 !important;
    font-family: 'M PLUS Rounded 1c', monospace !important;
}

textarea:focus, input:focus {
    border-color: #00ffff !important;
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.25) !important;
    outline: none !important;
}

/* 音频上传区域 */
.gr-audio, div[data-testid="audio"], .audio-container {
    background: rgba(10, 10, 20, 0.7) !important;
    border: 2px dashed rgba(255, 0, 255, 0.35) !important;
    border-radius: 12px !important;
}

.gr-audio:hover, div[data-testid="audio"]:hover {
    border-color: #ff00ff !important;
    box-shadow: 0 0 25px rgba(255, 0, 255, 0.25) !important;
}

/* 滑块样式 */
input[type="range"] {
    accent-color: #ff00ff !important;
}

input[type="range"]::-webkit-slider-thumb {
    background: #ff00ff !important;
    box-shadow: 0 0 10px #ff00ff !important;
}

/* 下拉框 */
select, .gr-dropdown, div[data-testid="dropdown"] {
    background: rgba(5, 5, 15, 0.9) !important;
    border: 1px solid rgba(255, 0, 255, 0.25) !important;
    border-radius: 8px !important;
    color: #e0e0e8 !important;
}

/* 复选框 */
input[type="checkbox"]:checked {
    background: #ff00ff !important;
    border-color: #ff00ff !important;
}

/* 主按钮 - 霓虹效果 */
.primary-btn, button.primary-btn, .gr-button.primary-btn {
    background: linear-gradient(135deg, #ff00ff 0%, #8000ff 50%, #00ffff 100%) !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 1rem 3rem !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #fff !important;
    box-shadow: 
        0 0 25px rgba(255, 0, 255, 0.5),
        0 0 50px rgba(128, 0, 255, 0.25) !important;
    position: relative;
    overflow: hidden;
    cursor: pointer;
}

.primary-btn:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 
        0 0 35px rgba(255, 0, 255, 0.7),
        0 0 70px rgba(128, 0, 255, 0.4),
        0 12px 40px rgba(0, 0, 0, 0.4) !important;
}

/* 文件下载区域 */
.gr-file, div[data-testid="file"] {
    background: rgba(10, 10, 20, 0.7) !important;
    border: 1px solid rgba(0, 255, 255, 0.25) !important;
    border-radius: 10px !important;
}

.gr-file:hover, div[data-testid="file"]:hover {
    border-color: #00ffff !important;
    box-shadow: 0 0 18px rgba(0, 255, 255, 0.2) !important;
}

/* 状态显示 */
.status-box, .status-box textarea {
    background: rgba(0, 255, 128, 0.08) !important;
    border: 1px solid rgba(0, 255, 128, 0.35) !important;
    border-radius: 8px !important;
    color: #00ff80 !important;
    text-shadow: 0 0 8px rgba(0, 255, 128, 0.4);
}

/* 标签文字 */
label, .gr-block-label, span.block-label {
    color: #a0a0b8 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}

/* 信息提示文字 */
.gr-info, .info-text, span[data-testid="info"] {
    color: #6060a0 !important;
    font-size: 0.78rem !important;
}

/* 页脚 */
.footer-section {
    text-align: center;
    padding: 2rem 1rem;
    color: #404060 !important;
    font-size: 0.85rem;
}

.footer-section a {
    color: #00ffff !important;
    text-decoration: none;
    text-shadow: 0 0 6px rgba(0, 255, 255, 0.5);
}

.footer-section a:hover {
    text-shadow: 0 0 18px rgba(0, 255, 255, 0.9);
}

/* 分隔装饰线 */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 0, 255, 0.4), rgba(0, 255, 255, 0.4), transparent);
    margin: 1.5rem 0;
}

/* 滚动条 */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: rgba(10, 10, 20, 0.5);
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #ff00ff, #00ffff);
    border-radius: 3px;
}

/* Accordion 折叠面板 */
.gr-accordion, div[data-testid="accordion"] {
    background: rgba(15, 15, 28, 0.9) !important;
    border: 1px solid rgba(128, 0, 255, 0.25) !important;
    border-radius: 12px !important;
}

.gr-accordion summary, .gr-accordion button {
    color: #c0c0d8 !important;
}

/* 响应式优化 */
@media (max-width: 768px) {
    .neon-title {
        font-size: 2rem !important;
        letter-spacing: 0.08em;
    }
    .card-container {
        padding: 1rem !important;
        margin: 0.25rem !important;
    }
}

/* 隐藏 Gradio 的主题切换图标 (如有) */
.dark-mode-toggle, [aria-label="Toggle dark mode"] {
    display: none !important;
}

/* 强制覆盖 Light 模式 - 保持深色主题 */
.light, [data-theme="light"], body.light {
    --body-background-fill: #0a0a0f !important;
    --background-fill-primary: rgba(15, 15, 25, 0.95) !important;
    --background-fill-secondary: rgba(20, 20, 35, 0.9) !important;
    --block-background-fill: rgba(18, 18, 28, 0.95) !important;
    --panel-background-fill: rgba(15, 15, 25, 0.95) !important;
    --table-even-background-fill: rgba(20, 20, 35, 0.5) !important;
    --table-odd-background-fill: rgba(15, 15, 25, 0.5) !important;
    --block-border-color: rgba(255, 0, 255, 0.25) !important;
    --border-color-primary: rgba(255, 0, 255, 0.25) !important;
    --border-color-accent: rgba(0, 255, 255, 0.3) !important;
    --body-text-color: #e0e0e8 !important;
    --body-text-color-subdued: #8080a0 !important;
    --block-label-text-color: #b0b0c0 !important;
    --block-title-text-color: #00ffff !important;
    --input-background-fill: rgba(5, 5, 15, 0.8) !important;
    --input-border-color: rgba(0, 255, 255, 0.3) !important;
    --neutral-50: #0a0a0f !important;
    --neutral-100: #12121a !important;
    --neutral-200: #1a1a28 !important;
    --neutral-300: #252535 !important;
    --neutral-400: #404060 !important;
    --neutral-500: #606080 !important;
    --neutral-600: #8080a0 !important;
    --neutral-700: #a0a0c0 !important;
    --neutral-800: #c0c0d8 !important;
    --neutral-900: #e0e0f0 !important;
    --neutral-950: #f0f0ff !important;
    color-scheme: dark !important;
}

/* 设置弹窗样式 */
div[role="dialog"], .modal, .settings-modal {
    background: rgba(15, 15, 25, 0.98) !important;
    border: 1px solid rgba(255, 0, 255, 0.3) !important;
    color: #e0e0e8 !important;
}

/* Row 容器背景修复 */
.gr-row, .row, div[class*="row"] {
    background: transparent !important;
}

/* Column 容器背景修复 */
.gr-column, .column, div[class*="column"] {
    background: transparent !important;
}
"""

# 创建 Gradio 界面 - 强制暗色主题
dark_theme = gr.themes.Default(
    primary_hue=gr.themes.colors.fuchsia,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
)

with gr.Blocks(
    title="FA-Kara WebUI",
    css=custom_css,
    theme=dark_theme,
) as demo:
    
    # 强制暗色模式 + 霓虹标题
    gr.HTML("""
        <script>
        (function() {
            // 强制暗色模式
            document.documentElement.classList.add('dark');
            document.documentElement.classList.remove('light');
            document.documentElement.setAttribute('data-theme', 'dark');
            document.body.classList.add('dark');
            document.body.classList.remove('light');
            localStorage.setItem('theme', 'dark');
            
            // 监听并阻止主题切换
            const observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    if (mutation.attributeName === 'class' || mutation.attributeName === 'data-theme') {
                        const el = mutation.target;
                        if (el.classList.contains('light')) {
                            el.classList.remove('light');
                            el.classList.add('dark');
                        }
                        if (el.getAttribute('data-theme') === 'light') {
                            el.setAttribute('data-theme', 'dark');
                        }
                    }
                });
            });
            observer.observe(document.documentElement, { attributes: true });
            observer.observe(document.body, { attributes: true });
        })();
        </script>
        <div class="neon-header">
            <h1 class="neon-title">FA-KARA</h1>
            <p class="neon-subtitle">カラオケ・リリック・シンク</p>
        </div>
    """)
    
    with gr.Row(equal_height=True):
        # 左侧 - 输入区域
        with gr.Column(scale=1):
            gr.HTML('<div class="card-container">')
            gr.HTML('<div class="section-title">🎵 音频输入</div>')
            
            audio_input = gr.Audio(
                label="上传人声音频",
                type="filepath",
                sources=["upload"],
                show_label=True,
            )
            
            gr.HTML('<div class="divider"></div>')
            gr.HTML('<div class="section-title">📝 歌词文本</div>')
            
            lyrics_input = gr.Textbox(
                label="",
                placeholder="在此粘贴歌词（支持振假名格式）\n\n示例：\n{阻|はば}むものは{無|な}い\n{身|み}{勝|かっ}{手|て}に More love!",
                lines=10,
                max_lines=15,
                show_label=False,
            )
            
            with gr.Row():
                language = gr.Dropdown(
                    label="语言",
                    choices=[("日语+英语", "jaen"), ("仅日语", "ja")],
                    value="jaen",
                    scale=1,
                )
                txt_format = gr.Dropdown(
                    label="格式",
                    choices=[
                        ("春日向け", "hrh"),
                        ("utaten", "uta"),
                        ("萌娘百科", "moe"),
                    ],
                    value="hrh",
                    scale=1,
                )
            gr.HTML('</div>')
        
        # 右侧 - 设置与输出
        with gr.Column(scale=1):
            gr.HTML('<div class="card-container">')
            gr.HTML('<div class="section-title">⚡ 快速设置</div>')
            
            audio_speed = gr.Slider(
                label="音频倍速",
                minimum=0.25,
                maximum=2.0,
                value=1.0,
                step=0.05,
                info="语速快可降低此值",
            )
            
            with gr.Row():
                bpm = gr.Number(label="BPM", value=60, minimum=0, maximum=300, scale=1)
                tail_correct = gr.Dropdown(
                    label="尾音模式",
                    choices=[("推荐", 3), ("模式2", 2), ("模式1", 1), ("禁用", 0)],
                    value=3,
                    scale=1,
                )
            
            # 高级设置折叠
            with gr.Accordion("🔧 高级设置", open=False):
                with gr.Row():
                    silent_window = gr.Slider(
                        label="静音窗口(秒)", minimum=0.1, maximum=2.0, value=0.8, step=0.1
                    )
                    tail_thres_pct = gr.Slider(
                        label="阈值百分位(%)", minimum=1, maximum=50, value=10, step=1
                    )
                
                with gr.Row():
                    tail_thres_ratio = gr.Slider(
                        label="阈值比例", minimum=0.01, maximum=0.5, value=0.1, step=0.01
                    )
                    chars_per_line = gr.Slider(
                        label="每行字数限制", minimum=0, maximum=50, value=0, step=1,
                        info="0=不限"
                    )
                
                with gr.Row():
                    sokuon_split = gr.Checkbox(label="促音拆分", value=False)
                    hatsuon_split = gr.Checkbox(label="拨音拆分", value=True)
                
                with gr.Row():
                    ruby_offset = gr.Number(label="偏移量(ms)", value=-150)
                    beats_per_bar = gr.Number(label="每小节拍数", value=3, minimum=1, maximum=8)
            
            gr.HTML('<div class="divider"></div>')
            gr.HTML('<div class="section-title">📤 输出文件</div>')
            
            status_output = gr.Textbox(
                label="状态",
                interactive=False,
                elem_classes=["status-box"],
            )
            
            with gr.Row():
                ruby_lrc_output = gr.File(label="Ruby LRC", scale=1)
                rlf_lrc_output = gr.File(label="RLF LRC", scale=1)
                ass_output = gr.File(label="ASS", scale=1)
            
            gr.HTML('</div>')
    
    # 处理按钮
    gr.HTML('<div style="display: flex; justify-content: center; margin: 1.5rem 0;">')
    process_btn = gr.Button(
        "⚡ 开始同步",
        variant="primary",
        size="lg",
        elem_classes=["primary-btn"],
    )
    gr.HTML('</div>')
    
    # 页脚
    gr.HTML("""
        <div class="footer-section">
            <p>Powered by <a href="https://github.com/moriwx/FA-Kara" target="_blank">FA-Kara</a> | MMS-FA + librosa + PyTorch</p>
        </div>
    """)
    
    process_btn.click(
        fn=process_lyrics,
        inputs=[
            audio_input,
            lyrics_input,
            audio_speed,
            sokuon_split,
            hatsuon_split,
            tail_correct,
            silent_window,
            tail_thres_pct,
            tail_thres_ratio,
            ruby_offset,
            bpm,
            beats_per_bar,
            language,
            txt_format,
            chars_per_line,
        ],
        outputs=[
            ruby_lrc_output,
            rlf_lrc_output,
            ass_output,
            status_output,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
