#!/usr/bin/env python3
"""Front door for evaluations and recommendation runs: make wizard."""
import argparse
import os
import shlex
import subprocess
import sys

from eval import wizard as ui

ROOT = ui.ROOT


def recommendation_command(*, user_id, group_id, count, model, agent, base_url,
                           preview=True, enrich=False, no_spotify=False):
    if not user_id.strip() or user_id.startswith('-'):
        raise ValueError('Enter a nonempty Firebase user ID that does not start with a dash')
    if count < 1:
        raise ValueError('Count must be positive')
    if no_spotify and not preview:
        raise ValueError('Live delivery in the wizard requires Spotify verification')
    # env makes the printed command reusable; the shell is never invoked.
    command = ['env', f'LLM_URL={base_url}', sys.executable, str(ROOT / 'recommend.py'),
               user_id, '--count', str(count), '--model', model, '--agent', agent,
               '--group-id', group_id]  # empty explicitly overrides GROUP_ID with auto-selection
    if preview:
        command.append('--dry-run')
    if enrich:
        command.append('--enrich')
    if no_spotify:
        command.append('--no-spotify')
    return command


def recommendation_plan():
    mode = ui.choose('Recommendation mode', [
        ('preview', 'Preview — fetch reviews and generate; send nothing'),
        ('live', 'Live delivery — generate and send recommendations to the user'),
    ], ['preview'], maximum=1)[0]
    user_id = ''
    while not user_id or user_id.startswith('-'):
        user_id = ui.ask('Recipient Firebase user ID', os.getenv('FB_UID', ''))
        if not user_id or user_id.startswith('-'):
            print('Enter the actual Firebase user ID, not an eval profile name.')
    group = ui.ask('Existing group ID, or auto for the only shared group', os.getenv('GROUP_ID') or 'auto')
    group = '' if group.lower() == 'auto' else group
    print('The runner checks LLMBot and recipient membership. Multiple shared groups require an explicit ID.')
    count = ui.positive('Maximum recommendations to deliver', 1)
    base_url = ui.ask('Model server URL', os.getenv('LLM_URL') or os.getenv('OLLAMA_URL') or 'http://localhost:8092')
    default_model = os.getenv('LLM_MODEL') or os.getenv('OLLAMA_MODEL') or 'llama3'
    try:
        available = ui.discover_models(base_url, os.getenv('LLM_BACKEND', 'llamacpp').lower())
    except (OSError, ValueError, KeyError, TypeError):
        print('Model catalog unavailable; use the configured model or enter its ID.')
        available = []
    manual = ui.ask('Additional model ID (optional)')
    available = list(dict.fromkeys([default_model, *available, *([manual] if manual else [])]))
    model = ui.choose('Model', [(m, m) for m in available], [default_model], maximum=1)[0]
    registry = ui.ev.afrec_agents
    agent = ui.choose('Agent', [(a.tag, f'{a.tag} — {a.summary}') for a in registry.list_agents()],
                      [registry.selected_agent().tag], maximum=1)[0]
    enrich = ui.yes('Enable catalog enrichment for agents that support it?')
    preview = mode == 'preview'
    no_spotify = preview and not ui.yes('Verify albums with Spotify?', True)
    if not preview:
        print('Live delivery uses Spotify verification. Repeating a live run can send additional recommendations.')
    print(f'\nRecipient: {user_id}\nGroup: {group or "auto (only shared group)"}\n'
          f'Mode: {mode}\nMaximum picks: {count}\nModel: {model}\nAgent: {agent}')
    return recommendation_command(user_id=user_id, group_id=group, count=count, model=model,
                                  agent=agent, base_url=base_url, preview=preview,
                                  enrich=enrich, no_spotify=no_spotify)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Print commands only; do not fetch reviews, generate, or send')
    args = parser.parse_args()
    try:
        task = ui.choose('What do you want to do?', [
            ('eval', 'Evaluate models / agents or review judging results'),
            ('recommend', 'Generate recommendations for a real user'),
        ], ['eval'], maximum=1)[0]
        commands = ui.plan() if task == 'eval' else [recommendation_plan()]
        print('\nCommands (run from the repository):\ncd ' + shlex.quote(str(ROOT)))
        for command in commands:
            print(shlex.join(command))
        live = task == 'recommend' and '--dry-run' not in commands[0]
        prompt = 'Generate and SEND to this recipient now?' if live else 'Run these commands now?'
        if args.dry_run or not ui.yes(prompt):
            print('Commands printed only; nothing executed.')
            return 0
        for command in commands:
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode:
                return result.returncode
        return 0
    except (EOFError, KeyboardInterrupt):
        print('\nCancelled.')
        return 130
    except ValueError as exc:
        print(f'Cannot build command: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
