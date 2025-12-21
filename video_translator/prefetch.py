#!/usr/bin/env python3
"""Interactive prefetch tool using shared ManifestManager from downloader.

Run as: python -m video_translator.prefetch
"""
from __future__ import annotations
import os
import argparse
# 确保包结构正确，如果运行报错，请检查目录结构
from .model_downloader import ModelDownloader, ManifestManager

def ask_yes_no(prompt: str, default: bool = True) -> bool:
    y = 'Y' if default else 'y'
    n = 'n' if default else 'N'
    while True:
        ans = input(f"{prompt} [{y}/{n}]: ").strip().lower()
        if ans == '': return default
        if ans in ('y', 'yes'): return True
        if ans in ('n', 'no'): return False

def ask_choice_from_list(prompt: str, choices: list, default: str) -> str:
    print(prompt)
    for i, c in enumerate(choices):
        print(f"  {i+1}) {c}")
    while True:
        ans = input(f"Select (1-{len(choices)}, default {default}): ").strip()
        if ans == '': return default
        if ans in choices: return ans
        if ans.isdigit() and 0 <= int(ans)-1 < len(choices): return choices[int(ans)-1]
        print("Invalid selection.")

def main():
    parser = argparse.ArgumentParser(prog='video-translator-prefetch')
    parser.add_argument('--model-cache-dir', default=None)
    parser.add_argument('--hf-mirror', default=None)
    parser.add_argument('--proxy', default=None)
    parser.add_argument('--update', action='store_true', help='Update manifest from network')
    parser.add_argument('--preview', action='store_true', help='Show manifest content')
    args = parser.parse_args()

    root = args.model_cache_dir or os.path.abspath('./model-cache')
    os.makedirs(root, exist_ok=True)

    # 1. 实例化 ManifestManager
    # 此时它会加载本地 json，如果文件破损或格式不对，会加载默认值
    manifest = ManifestManager(root)

    # --- UPDATE MODE ---
    if args.update:
        # 这里会尝试联网，如果配置了 mirror，会自动应用
        manifest.update_from_network(proxy=args.proxy, mirror=args.hf_mirror)
        return

    # --- PREVIEW MODE ---
    if args.preview:
        print(f"\nManifest File: {manifest.filepath}")
        print(f"Last Updated: {manifest.data.get('last_updated', 'Never')}")
        
        # 这里调用了 whisper_sizes 属性
        if not hasattr(manifest, 'whisper_sizes'):
            print("Error: ManifestManager invalid (code mismatch).")
            return
            
        print("\n[Whisper Sizes Available]")
        for s in manifest.whisper_sizes:
            print(f" - {s}")
            
        print("\n[Translation Repos]")
        for r in manifest.translation_repos:
            print(f" - {r}")
        return

    # --- INTERACTIVE MODE ---
    print(f'\nInteractive prefetch (Manifest: {manifest.data.get("last_updated", "Default")})')
    
    # 再次检查属性是否存在，防止代码未更新
    if not hasattr(manifest, 'whisper_sizes'):
        print("Critical Error: ModelDownloader code is outdated. Please update model_downloader.py")
        return

    sizes = manifest.whisper_sizes
    # 安全回退
    if not sizes: 
        sizes = ['small']
        
    def_size = 'small' if 'small' in sizes else sizes[0]
    model_size = ask_choice_from_list("Choose Whisper Size:", sizes, default=def_size)

    do_openai = ask_yes_no(f'Download OpenAI Whisper ({model_size})?', default=True)
    do_faster = ask_yes_no(f'Download Faster-Whisper ({model_size})?', default=True)
    do_align = ask_yes_no('Download Alignment models?', default=True)
    
    trans_repos = manifest.translation_repos
    do_trans = False
    chosen_trans_repo = None
    
    if trans_repos:
        if ask_yes_no('Download Translation model?', default=True):
            do_trans = True
            def_trans = trans_repos[0]
            if len(trans_repos) > 1:
                chosen_trans_repo = ask_choice_from_list("Select Translation Model:", trans_repos, default=def_trans)
            else:
                chosen_trans_repo = def_trans
    else:
        print("No translation models in manifest.")

    if not any([do_openai, do_faster, do_align, do_trans]):
        print("Nothing selected.")
        return

    # 执行下载
    downloader = ModelDownloader(proxy=args.proxy, hf_mirror=args.hf_mirror)
    print(f"\nDownloading to {root}...")
    
    downloader.download_all_models(
        save_root=root,
        model_size=model_size,
        include_openai_whisper=do_openai,
        include_faster_whisper=do_faster,
        include_alignment=do_align,
        include_m2m100=do_trans,
        specific_translation_repo=chosen_trans_repo
    )
    print("\nDone.")

if __name__ == '__main__':
    main()
