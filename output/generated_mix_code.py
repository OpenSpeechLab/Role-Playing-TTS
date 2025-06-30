
import os
import time
import sys
import datetime
import torch
from utils import MIX, CAT, COMPUTE_LEN, LOOP
from api import tts, audio
wav_path = "/cpfs01/user/renyiming/AudiobookAgent/output/audio"
os.makedirs(wav_path, exist_ok=True)


def function_one():
    print("🚀 开始生成音效和背景音素材")
    start_time = time.time()
    audio(prompt="Distant river flowing, gentle wind blowing", duration=6, volume=-28, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_0_Distant_river_flowing_gentle_wind.wav"))
    audio(prompt="A quick leap sound, followed by shifting gravel", duration=2, volume=-20, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_1_A_quick_leap_sound_followed.wav"))
    audio(prompt="Loud, powerful river current and crashing waves", duration=4, volume=-22, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_2_Loud_powerful_river_current_and.wav"))
    audio(prompt="Heavy body sitting down with a thud", duration=2, volume=-25, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_3_Heavy_body_sitting_down_with.wav"))
    audio(prompt="Faint distant village sounds, gentle crackling of a fire", duration=6, volume=-30, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_4_Faint_distant_village_sounds_gentle.wav"))
    audio(prompt="Footsteps on dirt path, increasing sound of rushing river water", duration=6, volume=-24, negative_prompt=" ", output_path=os.path.join(wav_path, "fg_sound_effect_5_Footsteps_on_dirt_path_increasing.wav"))
    audio(prompt="Wistful and tired journey music with a hint of mystery", volume=-38, duration=30, negative_prompt=" ", output_path=os.path.join(wav_path, "bg_music_0_Wistful_and_tired_journey_music.wav"))
    torch.cuda.empty_cache()
    audio(prompt="Ominous and anticipatory orchestral music", volume=-35, duration=30, negative_prompt=" ", output_path=os.path.join(wav_path, "bg_music_1_Ominous_and_anticipatory_orchestral_music.wav"))
    torch.cuda.empty_cache()
    end_time = time.time()
    print(f"🎉 音效和背景音素材生成完成，耗时 {end_time - start_time:.2f} 秒")

def function_two():
    print("🚀 开始生成配音")
    start_time = time.time()
    tts(tts_text="夕阳西下，晚霞如火，将万里河山镀上一层金边。唐僧师徒四人西行许久，身心俱疲，正行至一处荒凉河畔。", prompt_text="整体恐怖事件，是从几个年轻人的一场无聊的游戏开始的。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/cvd/pb.wav", speaker="旁白", volume=-15, output_path=os.path.join(wav_path, "fg_speech_0_.wav"))
    tts(tts_text="悟空，八戒，悟净，天色已晚，今日路途如此遥远，却只见这滔滔河水，不知前方可有村舍客栈歇脚？", prompt_text="阿弥陀佛，仙童一见那东西我就胆战心惊，哪里还敢偷着吃呀。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/cvd/ts.wav", speaker="唐僧", volume=-15, output_path=os.path.join(wav_path, "fg_speech_1_.wav"))
    tts(tts_text="师父莫急！俺老孙去前方探探路。这河水波澜壮阔，怕不是寻常小溪。待俺仔细瞧瞧！", prompt_text="欸，师父，你忘了他们把你吊在树上的滋味了。哼，你要是心疼这帮毛贼，就让他们还把你吊在树上荡秋千，老孙再也不管了。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/cvd/swk.wav", speaker="孙悟空", volume=-15, output_path=os.path.join(wav_path, "fg_speech_2_.wav"))
    tts(tts_text="孙悟空纵身一跃，跳上高处，眯起火眼金睛，极目远眺。只见眼前河流宽阔无边，水流湍急，波涛汹涌，隐约有股异样的气息弥漫。", prompt_text="整体恐怖事件，是从几个年轻人的一场无聊的游戏开始的。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/cvd/pb.wav", speaker="旁白", volume=-15, output_path=os.path.join(wav_path, "fg_speech_3_.wav"))
    tts(tts_text="哎呀！师父，这哪里是寻常河流？一眼望不到对岸，宽阔如海！水面宽达八百里，深不知几许！恐怕正是那通天河！", prompt_text="欸，师父，你忘了他们把你吊在树上的滋味了。哼，你要是心疼这帮毛贼，就让他们还把你吊在树上荡秋千，老孙再也不管了。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/cvd/swk.wav", speaker="孙悟空", volume=-15, output_path=os.path.join(wav_path, "fg_speech_4_.wav"))
    tts(tts_text="我的老天爷啊！八百里宽？这可如何是好？莫不是老猪要在这里做水怪了？这等大河，别说渡，看着都让人腿软。", prompt_text="你在这看着师父，我去把白龙马牵到有人家的地方买了，咱们把师父埋了呢，咱们只好是各奔东西啦。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/cvd/zbj.wav", speaker="猪八戒", volume=-15, output_path=os.path.join(wav_path, "fg_speech_5_.wav"))
    tts(tts_text="二师兄休要胡言。我们自西天取经，逢山开路，遇水搭桥。这通天河如此宽广，想必其中必有蹊跷，或有渡口，或有神灵相助。总之，师父在此，我们总能想到办法。", prompt_text="师父，大师兄对师父忠心耿耿，师父您又何必如此绝情呢。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/cvd/swj.wav", speaker="沙僧", volume=-15, output_path=os.path.join(wav_path, "fg_speech_6_.wav"))
    tts(tts_text="阿弥陀佛。善哉善哉。悟净说得是。既是天命如此，这通天河，想必正是此行一劫。只是这河水浩瀚，又无舟楫可寻，这可如何是好？", prompt_text="阿弥陀佛，仙童一见那东西我就胆战心惊，哪里还敢偷着吃呀。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/cvd/ts.wav", speaker="唐僧", volume=-15, output_path=os.path.join(wav_path, "fg_speech_7_.wav"))
    tts(tts_text="哎，又要露宿荒野了……不知道这河里有没有什么好吃的鱼虾蟹……", prompt_text="你在这看着师父，我去把白龙马牵到有人家的地方买了，咱们把师父埋了呢，咱们只好是各奔东西啦。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/cvd/zbj.wav", speaker="猪八戒", volume=-15, output_path=os.path.join(wav_path, "fg_speech_8_.wav"))
    tts(tts_text="呆子！别光想着吃！师父，俺老孙看这河边不远处，似乎有一户人家，炊烟袅袅。不如我们前去借宿，顺便打听一下这通天河的情况，看有无渡船或摆渡之人。", prompt_text="欸，师父，你忘了他们把你吊在树上的滋味了。哼，你要是心疼这帮毛贼，就让他们还把你吊在树上荡秋千，老孙再也不管了。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/cvd/swk.wav", speaker="孙悟空", volume=-15, output_path=os.path.join(wav_path, "fg_speech_9_.wav"))
    tts(tts_text="悟空此言有理。我等行走多日，也该歇歇了。走吧，去那户人家看看，或许能得知过河之法。", prompt_text="阿弥陀佛，仙童一见那东西我就胆战心惊，哪里还敢偷着吃呀。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/cvd/ts.wav", speaker="唐僧", volume=-15, output_path=os.path.join(wav_path, "fg_speech_10_.wav"))
    tts(tts_text="于是，师徒四人循着炊烟的方向，向着那户河边的人家走去。通天河的滔滔水声，在暮色中显得格外沉重，仿佛在低语着即将降临的挑战，预示着前方未知的艰难。", prompt_text="整体恐怖事件，是从几个年轻人的一场无聊的游戏开始的。", prompt_speech_path="/cpfs01/user/renyiming/AudiobookAgent/cvd/pb.wav", speaker="旁白", volume=-15, output_path=os.path.join(wav_path, "fg_speech_11_.wav"))
    end_time = time.time()
    print(f"🎉 配音生成完成，耗时 {end_time - start_time:.2f} 秒")
print("🚀 开始生成音频文件")
start_time = time.time()
function_one()
function_two()
fg_audio_wavs = []
fg_audio_lens = []
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_0_Distant_river_flowing_gentle_wind.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_0_Distant_river_flowing_gentle_wind.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_0_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_0_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_1_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_1_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_2_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_2_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_1_A_quick_leap_sound_followed.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_1_A_quick_leap_sound_followed.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_3_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_3_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_2_Loud_powerful_river_current_and.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_2_Loud_powerful_river_current_and.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_4_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_4_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_3_Heavy_body_sitting_down_with.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_3_Heavy_body_sitting_down_with.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_5_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_5_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_6_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_6_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_7_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_7_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_8_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_8_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_9_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_9_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_4_Faint_distant_village_sounds_gentle.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_4_Faint_distant_village_sounds_gentle.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_10_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_10_.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_5_Footsteps_on_dirt_path_increasing.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_5_Footsteps_on_dirt_path_increasing.wav")))

fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_11_.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_11_.wav")))

CAT(wavs=fg_audio_wavs, out_wav=os.path.join(wav_path, "foreground.wav"))
print("🚀 开始处理背景音并生成混音")

bg_audio_offsets = []
bg_audio_len = sum(fg_audio_lens[0:16])
bg_audio_offset = sum(fg_audio_lens[:0])
LOOP(os.path.join(wav_path, "bg_music_0_Wistful_and_tired_journey_music.wav"), os.path.join(wav_path, "bg_music_0_Wistful_and_tired_journey_music.wav"), bg_audio_len)
bg_audio_offsets.append(bg_audio_offset)

bg_audio_len = sum(fg_audio_lens[16:18])
bg_audio_offset = sum(fg_audio_lens[:16])
LOOP(os.path.join(wav_path, "bg_music_1_Ominous_and_anticipatory_orchestral_music.wav"), os.path.join(wav_path, "bg_music_1_Ominous_and_anticipatory_orchestral_music.wav"), bg_audio_len)
bg_audio_offsets.append(bg_audio_offset)

bg_audio_wavs = []
bg_audio_lens = []
bg_audio_wavs.append(os.path.join(wav_path, "bg_music_0_Wistful_and_tired_journey_music.wav"))
bg_audio_wavs.append(os.path.join(wav_path, "bg_music_1_Ominous_and_anticipatory_orchestral_music.wav"))
bg_audio_wav_offset_pairs = list(zip(bg_audio_wavs, bg_audio_offsets))
bg_audio_wav_offset_pairs.append((os.path.join(wav_path, "foreground.wav"), 0))
MIX(wavs=bg_audio_wav_offset_pairs, out_wav=os.path.join(wav_path, "final_mix.wav"))
end_time = time.time()
print(f"🎉 音频生成完成，耗时 {end_time - start_time:.2f} 秒")
